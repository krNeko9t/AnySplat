"""04 号票：量教师（VLM 伪标签）自己留了多少类内余量，并判别余量是信号还是噪声。

纯 CPU。用与 `InstasceneVlmPhysGMParser` 完全相同的量纲变换（E: log10(Pa)、
density: log10(kg/m^3)、poisson: raw），保证这里的方差比和 loss 看到的是同一个空间。
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

KEY_E = "Young's modulus (MPa)"
KEY_RHO = "Density (kg/m³)"
KEY_NU = "Poisson's ratio"

# 与 parsers.py 一致：z = (log10(max(v * si_scale, 1.0)) - mean) / std，poisson 不取 log
QUANTITIES = [
    ("youngs_modulus", KEY_E, 1e6, True),
    ("density", KEY_RHO, 1.0, True),
    ("poisson_ratio", KEY_NU, 1.0, False),
]


def transform(value, si_scale, use_log):
    v = float(value) * si_scale
    if use_log:
        return math.log10(max(v, 1.0))
    return v


def camel_words(cls: str) -> list[str]:
    """BookStack -> ['book','stack']；room:wall -> ['wall']。"""
    cls = cls.split(":")[-1]
    parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+", cls)
    return [p.lower() for p in parts if p]


# Factory 类名 -> 描述里可接受的同义词。裸子串匹配会把 "TV vs television"、
# "Pillar vs column" 误判成 grounding 错误，这张表只用来压低假阳性；
# 它压不干净（"Blanket vs comforter" 这类相邻语义仍算不一致），
# 所以"不一致率"读作 grounding 错误率的**上界**。
SYN = {
    "tv": ["television", "screen", "monitor"], "pillar": ["column", "post"],
    "lite": ["door"], "louver": ["door", "shutter", "blind"], "panel": ["door", "panel"],
    "rug": ["carpet", "mat"], "art": ["painting", "picture", "poster", "frame", "artwork", "canvas"],
    "lamp": ["lamp", "light", "luminaire", "sconce"],
    "light": ["light", "lamp", "luminaire", "fixture"],
    "container": ["pot", "planter", "vase", "container"],
    "plant": ["plant", "shrub", "foliage", "tree", "greenery"],
    "bookcase": ["shelf", "shelving", "bookcase", "bookshelf"],
    "shelf": ["shelf", "shelving", "bookcase", "rack"],
    "cabinet": ["cabinet", "cupboard", "dresser", "sideboard", "drawer"],
    "stack": ["stack", "pile", "books", "book"], "sofa": ["sofa", "couch", "settee"],
    "boulder": ["rock", "stone", "boulder"],
    "exterior": ["wall", "exterior", "outside", "facade", "building"],
    "ceiling": ["ceiling", "overhead"], "floor": ["floor", "ground", "flooring"],
    "wall": ["wall", "partition"], "cell": ["shelf", "shelving", "cubby"],
    "skirtingboard": ["skirting", "baseboard", "trim", "molding", "moulding"],
    "support": ["support", "trim", "beam"], "window": ["window", "glazing", "pane"],
    "sink": ["sink", "basin"], "bed": ["bed", "mattress"],
    "chair": ["chair", "seat", "stool"], "table": ["table", "desk"], "desk": ["desk", "table"],
    "single": [], "large": [], "simple": [],
}


def description_matches_class(cls: str, desc: str) -> bool:
    """描述里是否提到了这个 Factory 类名（或其同义词）。"""
    d = desc.lower()
    for w in camel_words(cls):
        if w in d or any(s in d for s in SYN.get(w, [])):
            return True
    return False


def variance_decomposition(groups):
    """groups: {key: [x, ...]} -> (within_frac, between_frac, n, n_groups)

    组内方差用去均值残差的总平方和 / N（即 E[Var(X|G)]），与全体方差同一分母。
    """
    n = sum(len(v) for v in groups.values())
    if n == 0:
        return None
    all_vals = [x for v in groups.values() for x in v]
    mu = sum(all_vals) / n
    total_ss = sum((x - mu) ** 2 for x in all_vals)
    within_ss = 0.0
    for v in groups.values():
        if not v:
            continue
        m = sum(v) / len(v)
        within_ss += sum((x - m) ** 2 for x in v)
    if total_ss <= 0:
        return None
    return within_ss / total_ss, 1.0 - within_ss / total_ss, n, len(groups)


def mae_baselines(groups):
    """常数预测（全局均值）与类别查表（leave-one-out 组均值）的 MAE，单位 = 变换后的量。"""
    all_vals = [x for v in groups.values() for x in v]
    n = len(all_vals)
    if n == 0:
        return None
    gmu = sum(all_vals) / n
    const_mae = sum(abs(x - gmu) for x in all_vals) / n
    lut_err = 0.0
    for v in groups.values():
        k = len(v)
        s = sum(v)
        for x in v:
            pred = (s - x) / (k - 1) if k > 1 else gmu  # 单例类退化到全局均值
            lut_err += abs(x - pred)
    return const_mae, lut_err / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--class_table", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    class_table_path = Path(args.class_table or root / "infinigen_class_table.json")
    class_table = json.loads(class_table_path.read_text())

    ann_root = root / "preprocessed/annotations/infinigen/infinigen"
    label_files = sorted(ann_root.glob("scene_*/*/Qwen3.6-27B.json"))
    print(f"[load] {len(label_files)} 个场景标签文件")

    records = []  # (cls, n_views, desc_hit, prototype, {q: value})
    n_raw = n_no_class = n_bad = 0
    for lf in label_files:
        scene_id = f"infinigen_{lf.parent.parent.name}_{lf.parent.name}"
        table = class_table.get(scene_id, {})
        for rec in json.loads(lf.read_text()):
            n_raw += 1
            inst = table.get(str(rec["id"]))
            if inst is None:
                n_no_class += 1
                continue
            resp = rec.get("response") or {}
            phys = resp.get("physical_property") or {}
            vals = {}
            ok = True
            for name, key, si, use_log in QUANTITIES:
                entry = phys.get(key)
                if not isinstance(entry, dict) or entry.get("mean") is None:
                    ok = False
                    break
                vals[name] = transform(entry["mean"], si, use_log)
            if not ok:
                n_bad += 1
                continue
            cls = inst["class"]
            desc = resp.get("object_description") or ""
            mats = resp.get("appearance_materials") or []
            proto = mats[0].get("prototype") if mats and isinstance(mats[0], dict) else None
            records.append((cls, rec.get("n_views"), description_matches_class(cls, desc), proto, vals))

    print(f"[load] 原始 {n_raw} 条，无类别 {n_no_class} 条，物性畸形 {n_bad} 条，进入统计 {len(records)} 条")

    report = {
        "n_records": len(records),
        "n_raw": n_raw,
        "n_no_class": n_no_class,
        "n_malformed": n_bad,
    }

    def group_by_class(recs, q):
        g = defaultdict(list)
        for cls, _, _, _, vals in recs:
            g[cls].append(vals[q])
        return g

    # ---- 1. 主结果：类内 / 全体方差比 ----
    print("\n=== 1. 全体：组内方差占比（组 = Factory 类名）===")
    print(f"{'量':<18}{'类内占比':>10}{'类别解释':>10}{'N':>8}{'类数':>7}"
          f"{'常数MAE':>10}{'查表MAE':>10}")
    main_tbl = {}
    for name, _, _, _ in QUANTITIES:
        g = group_by_class(records, name)
        wf, bf, n, k = variance_decomposition(g)
        cm, lm = mae_baselines(g)
        main_tbl[name] = {
            "within_frac": wf, "between_frac": bf, "n": n, "n_classes": k,
            "mae_const": cm, "mae_class_lut": lm,
        }
        print(f"{name:<18}{wf:>9.1%}{bf:>10.1%}{n:>8}{k:>7}{cm:>10.3f}{lm:>10.3f}")
    report["overall"] = main_tbl

    # ---- 2. 类规模分布 ----
    counts = sorted(((c, len(v)) for c, v in group_by_class(records, "density").items()),
                    key=lambda x: -x[1])
    singles = sum(1 for _, n in counts if n == 1)
    lt10 = sum(1 for _, n in counts if n < 10)
    print(f"\n=== 2. 类规模：{len(counts)} 类，单例类 {singles}，<10 实例的类 {lt10} ===")
    print("  Top15:", ", ".join(f"{c}={n}" for c, n in counts[:15]))
    report["class_sizes"] = {"n_classes": len(counts), "n_singleton": singles,
                             "n_lt10": lt10, "top": counts[:30]}

    # ---- 3. 判别口径 A：描述是否提到类名（grounding 一致性）----
    hit = [r for r in records if r[2]]
    miss = [r for r in records if not r[2]]
    print(f"\n=== 3. grounding 一致性：描述提到类名 {len(hit)}/{len(records)} = "
          f"{len(hit)/len(records):.1%}，不一致 {len(miss)} ===")
    print(f"{'量':<18}{'一致子集类内':>14}{'不一致子集类内':>16}")
    ground = {}
    for name, _, _, _ in QUANTITIES:
        a = variance_decomposition(group_by_class(hit, name))
        b = variance_decomposition(group_by_class(miss, name))
        ground[name] = {"match_within": a[0], "mismatch_within": b[0],
                        "n_match": a[2], "n_mismatch": b[2]}
        print(f"{name:<18}{a[0]:>13.1%}{b[0]:>16.1%}")
    report["grounding"] = {"match_rate": len(hit) / len(records), **{"per_q": ground}}

    # ---- 4. 判别口径 B：材质 prototype 能否解释类内残差 ----
    print("\n=== 4. 类内残差是否被 appearance_materials.prototype 解释 ===")
    protos = defaultdict(int)
    for r in records:
        protos[r[3]] += 1
    print("  prototype 分布（top10）：",
          ", ".join(f"{k}={v}" for k, v in sorted(protos.items(), key=lambda x: -x[1])[:10]))
    print(f"{'子集':<22}" + "".join(f"{n[:9]:>24}" for n, _, _, _ in QUANTITIES))
    print(f"{'':<22}" + "".join(f"{'类内':>8}{'类×proto':>8}{'解释份额':>8}" for _ in QUANTITIES))
    proto_tbl = {}
    for lab, sub in [("全体", records), ("一致子集", hit), ("不一致子集", miss)]:
        cells, row = [], {}
        for name, _, _, _ in QUANTITIES:
            g2 = defaultdict(list)
            for cls, _, _, proto, vals in sub:
                g2[(cls, proto)].append(vals[name])
            w1 = variance_decomposition(group_by_class(sub, name))[0]
            w2 = variance_decomposition(g2)[0]
            share = (w1 - w2) / w1 if w1 > 0 else float("nan")
            row[name] = {"within_class": w1, "within_class_proto": w2, "proto_share": share}
            cells.append(f"{w1:>7.1%}{w2:>8.1%}{share:>8.1%}")
        proto_tbl[lab] = row
        print(f"{lab:<22}" + "".join(cells))
    report["prototype"] = proto_tbl

    # ---- 5. 判别口径 C：按 n_views 分层 ----
    print("\n=== 5. 按 n_views 分层的类内方差占比 ===")
    print(f"{'n_views':<10}{'N':>8}" + "".join(f"{n:>20}" for n, _, _, _ in QUANTITIES))
    nv_tbl = {}
    for nv in sorted({r[1] for r in records}, key=lambda x: (x is None, x)):
        sub = [r for r in records if r[1] == nv]
        row = {}
        cells = []
        for name, _, _, _ in QUANTITIES:
            d = variance_decomposition(group_by_class(sub, name))
            row[name] = d[0] if d else None
            cells.append(f"{d[0]:>19.1%}" if d else f"{'-':>20}")
        nv_tbl[str(nv)] = {"n": len(sub), **row}
        print(f"{str(nv):<10}{len(sub):>8}" + "".join(cells))
    report["by_n_views"] = nv_tbl

    # ---- 6. 类内跨度最大的类（人工复核用）----
    print("\n=== 6. 类内 log10(E) 跨度最大的类（>=20 实例）===")
    gE = group_by_class(records, "youngs_modulus")
    spans = [(c, max(v) - min(v), len(v)) for c, v in gE.items() if len(v) >= 20]
    spans.sort(key=lambda x: -x[1])
    for c, s, n in spans[:12]:
        print(f"  {c:<20} span={s:.2f} dec  n={n}")
    report["top_span_classes"] = [{"class": c, "span_log10E": s, "n": n} for c, s, n in spans[:20]]

    # ---- 7. 稳健性：离群值有没有撑起方差 ----
    print("\n=== 7. 稳健性：E 截尾后重算类内占比 ===")
    Es = sorted(v["youngs_modulus"] for *_, v in records)
    lo, hi = Es[int(0.005 * (len(Es) - 1))], Es[int(0.995 * (len(Es) - 1))]
    rob = {}
    print(f"{'子集':<26}{'N':>8}" + "".join(f"{n[:9]:>11}" for n, _, _, _ in QUANTITIES))
    for lab, filt in [("全体", lambda v: True),
                      (f"E∈[p0.5,p99.5]", lambda v: lo <= v["youngs_modulus"] <= hi),
                      ("E∈[1kPa,1TPa]", lambda v: 3 <= v["youngs_modulus"] <= 12)]:
        sub = [r for r in records if filt(r[4])]
        row = {n: variance_decomposition(group_by_class(sub, n))[0] for n, _, _, _ in QUANTITIES}
        rob[lab] = {"n": len(sub), **row}
        print(f"{lab:<26}{len(sub):>8}" + "".join(f"{row[n]:>11.1%}" for n, _, _, _ in QUANTITIES))
    report["robustness"] = rob

    # ---- 8. 非物体类（room:* / Window / 门镜画）单独看，交给 08 号票 ----
    NON_OBJ = {"Window", "room:wall", "room:floor", "room:ceiling", "room:exterior",
               "PanelDoor", "LiteDoor", "LouverDoor", "Mirror", "WallArt"}
    print("\n=== 8. 非物体类 vs 其余（08 号票要的数）===")
    print(f"{'子集':<22}{'N':>8}{'描述一致率':>12}" + "".join(f"{n[:9]:>11}" for n, _, _, _ in QUANTITIES))
    nonobj_tbl = {}
    for lab, filt in [("非物体类", lambda c: c in NON_OBJ or c.startswith("room:")),
                      ("其余", lambda c: not (c in NON_OBJ or c.startswith("room:")))]:
        sub = [r for r in records if filt(r[0])]
        gr = sum(1 for r in sub if r[2]) / len(sub)
        row = {n: variance_decomposition(group_by_class(sub, n))[0] for n, _, _, _ in QUANTITIES}
        nonobj_tbl[lab] = {"n": len(sub), "grounding_match_rate": gr, **row}
        print(f"{lab:<22}{len(sub):>8}{gr:>11.1%}" + "".join(f"{row[n]:>11.1%}" for n, _, _, _ in QUANTITIES))
    # 逐类明细（前 10 大非物体类）
    print("  逐类（log10 E 的类内标准差，越大越像在乱给）：")
    gE = group_by_class(records, "youngs_modulus")
    import statistics
    rows = [(c, len(v), statistics.pstdev(v)) for c, v in gE.items() if len(v) >= 50]
    for c, n, sd in sorted(rows, key=lambda x: -x[2])[:12]:
        tag = "  <-非物体" if (c in NON_OBJ or c.startswith("room:")) else ""
        print(f"    {c:<26} n={n:<6} sd={sd:.2f}{tag}")
    report["non_object"] = nonobj_tbl

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
