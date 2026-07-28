#!/usr/bin/env python3
"""Read a TensorBoard ``events.out.tfevents.*`` file with the standard library only.

Why this exists: the per-step scalars a training run writes are a *single sample*
from rank 0 (Lightning's ``self.log`` does not reduce across ranks unless asked),
recorded every ``log_every_n_steps``.  Eyeballing that in the TensorBoard UI with
smoothing cranked to 0.99 shows a trend but hides the two things that actually
matter when a loss looks noisy: how wide the per-step spread is, and whether the
spread is explained by the batch composition rather than by the model.

So this script does not re-plot the curve.  It reports, per tag and per bin of
steps, the *distribution* (median / IQR / p10-p90) instead of an EMA, and it can
correlate any two tags -- e.g. ``loss/segvggt`` against
``loss/loss_segvggt_num_matched`` -- to test "is the loss just tracking how many
GT instances the random view sample happened to contain?".

No protobuf, no tensorboard, no numpy: the TFRecord framing and the handful of
protobuf fields needed for scalar summaries are decoded inline.

Usage::

    python scripts/read_tfevents.py FILE --list
    python scripts/read_tfevents.py FILE --tag loss/segvggt --bins 20
    python scripts/read_tfevents.py FILE --tag 'loss/*' --bins 10
    python scripts/read_tfevents.py FILE --corr loss/segvggt loss/loss_segvggt_num_matched
    python scripts/read_tfevents.py FILE --tag loss/segvggt --csv out.csv
"""
from __future__ import annotations

import argparse
import fnmatch
import struct
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------- #
# protobuf wire format (only what scalar summaries need)
# ---------------------------------------------------------------------------- #
_WIRE_VARINT, _WIRE_64, _WIRE_LEN, _WIRE_32 = 0, 1, 2, 5


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _fields(buf: bytes):
    """Yield ``(field_number, wire_type, payload)`` for one protobuf message.

    ``payload`` is raw bytes for length-delimited fields, otherwise the decoded
    scalar (int for varint, bytes for the fixed-width cases).
    """
    pos, end = 0, len(buf)
    while pos < end:
        key, pos = _varint(buf, pos)
        fnum, wire = key >> 3, key & 7
        if wire == _WIRE_VARINT:
            val, pos = _varint(buf, pos)
        elif wire == _WIRE_64:
            val, pos = buf[pos:pos + 8], pos + 8
        elif wire == _WIRE_LEN:
            ln, pos = _varint(buf, pos)
            val, pos = buf[pos:pos + ln], pos + ln
        elif wire == _WIRE_32:
            val, pos = buf[pos:pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield fnum, wire, val


def _tensor_scalar(buf: bytes) -> float | None:
    """Pull a single float out of a TensorProto (newer writers use this for scalars)."""
    for fnum, _wire, val in _fields(buf):
        if fnum == 5 and len(val) >= 4:            # repeated float float_val (packed)
            return struct.unpack("<f", val[:4])[0]
        if fnum == 4 and len(val) >= 4:            # bytes tensor_content
            return struct.unpack("<f", val[:4])[0]
    return None


def _summary_values(buf: bytes):
    """Yield ``(tag, value)`` from a Summary message, skipping non-scalar entries."""
    for fnum, _wire, val in _fields(buf):
        if fnum != 1:                              # repeated Value value = 1
            continue
        tag, value = None, None
        for vf, vwire, vval in _fields(val):
            if vf == 1 and vwire == _WIRE_LEN:
                tag = vval.decode("utf-8", "replace")
            elif vf == 2 and vwire == _WIRE_32:    # float simple_value = 2
                value = struct.unpack("<f", vval)[0]
            elif vf == 8 and vwire == _WIRE_LEN:   # TensorProto tensor = 8
                value = _tensor_scalar(vval)
        if tag is not None and value is not None:
            yield tag, value


def read_events(path: str) -> dict[str, list[tuple[int, float]]]:
    """Parse a tfevents file into ``{tag: [(step, value), ...]}``.

    TFRecord framing is ``u64 length | u32 crc(length) | payload | u32 crc(payload)``.
    CRCs are skipped (crc32c is not in the stdlib); a truncated tail -- normal when a
    run was killed -- stops the scan instead of raising.
    """
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with open(path, "rb") as fh:
        blob = fh.read()

    pos, end, skipped = 0, len(blob), 0
    while pos + 12 <= end:
        (length,) = struct.unpack_from("<Q", blob, pos)
        payload_start = pos + 12
        payload_end = payload_start + length
        if payload_end + 4 > end:
            skipped += 1
            break
        payload = blob[payload_start:payload_end]
        pos = payload_end + 4

        step = 0
        summaries: list[bytes] = []
        try:
            for fnum, _wire, val in _fields(payload):
                if fnum == 2:                      # int64 step = 2
                    step = val
                elif fnum == 5:                    # Summary summary = 5
                    summaries.append(val)
            for summary in summaries:
                for tag, value in _summary_values(summary):
                    series[tag].append((step, value))
        except (ValueError, IndexError, struct.error):
            skipped += 1
            continue

    if skipped:
        print(f"# note: {skipped} record(s) skipped (truncated or unparsable tail)",
              file=sys.stderr)
    for tag in series:
        series[tag].sort(key=lambda sv: sv[0])
    return dict(series)


# ---------------------------------------------------------------------------- #
# statistics (stdlib only)
# ---------------------------------------------------------------------------- #
def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _stats(vals: list[float]) -> dict[str, float]:
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    var = sum((v - mean) ** 2 for v in s) / n if n > 1 else 0.0
    return {
        "n": n, "mean": mean, "std": var ** 0.5,
        "p10": _pct(s, 0.10), "p25": _pct(s, 0.25), "med": _pct(s, 0.50),
        "p75": _pct(s, 0.75), "p90": _pct(s, 0.90),
        "min": s[0], "max": s[-1],
    }


def _sparkline(values: list[float]) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return blocks[0] * len(values)
    return "".join(blocks[min(7, int((v - lo) / (hi - lo) * 8))] for v in values)


def report_tag(tag: str, points: list[tuple[int, float]], bins: int) -> None:
    steps = [s for s, _ in points]
    vals = [v for _, v in points]
    overall = _stats(vals)
    print(f"\n=== {tag} ===")
    print(f"points={overall['n']}  steps={steps[0]}..{steps[-1]}  "
          f"mean={overall['mean']:.4g}  std={overall['std']:.4g}  "
          f"min={overall['min']:.4g}  max={overall['max']:.4g}")

    lo_s, hi_s = steps[0], steps[-1]
    width = max((hi_s - lo_s) / bins, 1e-9)
    buckets: list[list[float]] = [[] for _ in range(bins)]
    for s, v in points:
        buckets[min(bins - 1, int((s - lo_s) / width))].append(v)

    print(f"{'step range':>17} {'n':>5} {'median':>10} {'p10':>10} {'p90':>10} "
          f"{'mean':>10} {'std':>10}")
    medians = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        st = _stats(bucket)
        medians.append(st["med"])
        rng = f"{int(lo_s + i * width)}-{int(lo_s + (i + 1) * width)}"
        print(f"{rng:>17} {st['n']:>5} {st['med']:>10.4g} {st['p10']:>10.4g} "
              f"{st['p90']:>10.4g} {st['mean']:>10.4g} {st['std']:>10.4g}")
    if len(medians) > 1:
        print(f"  median trend: {_sparkline(medians)}")


def correlate(a_pts: list[tuple[int, float]], b_pts: list[tuple[int, float]]) -> None:
    """Pearson r over steps present in both series."""
    b_map = dict(b_pts)
    pairs = [(v, b_map[s]) for s, v in a_pts if s in b_map]
    if len(pairs) < 3:
        print("  not enough shared steps to correlate")
        return
    n = len(pairs)
    ma = sum(x for x, _ in pairs) / n
    mb = sum(y for _, y in pairs) / n
    cov = sum((x - ma) * (y - mb) for x, y in pairs)
    va = sum((x - ma) ** 2 for x, _ in pairs)
    vb = sum((y - mb) ** 2 for _, y in pairs)
    r = cov / ((va * vb) ** 0.5) if va > 0 and vb > 0 else float("nan")
    print(f"  n={n}  pearson_r={r:+.4f}  r^2={r * r:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="path to events.out.tfevents.*")
    ap.add_argument("--list", action="store_true", help="list tags and exit")
    ap.add_argument("--tag", action="append", default=[],
                    help="tag or glob to report (repeatable)")
    ap.add_argument("--bins", type=int, default=20, help="step bins per tag report")
    ap.add_argument("--corr", nargs=2, metavar=("TAG_A", "TAG_B"),
                    help="Pearson correlation between two tags over shared steps")
    ap.add_argument("--csv", help="dump the selected tags to a CSV file")
    args = ap.parse_args()

    series = read_events(args.file)
    if not series:
        print("no scalar summaries found", file=sys.stderr)
        return 1

    if args.list:
        print(f"{len(series)} tags in {args.file}:")
        for tag in sorted(series):
            pts = series[tag]
            print(f"  {tag:<45} n={len(pts):<6} steps={pts[0][0]}..{pts[-1][0]}")
        return 0

    if args.corr:
        a, b = args.corr
        if a not in series or b not in series:
            print(f"missing tag: {a if a not in series else b}", file=sys.stderr)
            return 1
        print(f"corr({a}, {b}):")
        correlate(series[a], series[b])
        return 0

    patterns = args.tag or ["*"]
    selected = [t for t in sorted(series)
                if any(fnmatch.fnmatch(t, p) for p in patterns)]
    if not selected:
        print(f"no tag matched {patterns}", file=sys.stderr)
        return 1

    for tag in selected:
        report_tag(tag, series[tag], args.bins)

    if args.csv:
        all_steps = sorted({s for t in selected for s, _ in series[t]})
        maps = {t: dict(series[t]) for t in selected}
        with open(args.csv, "w") as fh:
            fh.write("step," + ",".join(selected) + "\n")
            for s in all_steps:
                row = ["" if s not in maps[t] else repr(maps[t][s]) for t in selected]
                fh.write(f"{s}," + ",".join(row) + "\n")
        print(f"\nwrote {args.csv} ({len(all_steps)} rows x {len(selected)} tags)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
