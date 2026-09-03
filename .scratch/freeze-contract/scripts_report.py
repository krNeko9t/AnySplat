"""Render notes/trainable_sets.md from the per-config probe JSONs."""
import json, os, sys

ORDER = ["segvggt_finetune_agnostic", "segvggt_agnostic_phys_joint", "segvggt_physgm",
         "segvggt_scannet", "instseg_iggt", "phys_iggt", "phys_prop_iggt",
         "physgm_iggt", "physgm_dpt_iggt"]
RAW = ".scratch/freeze-contract/notes/raw"


def m(n):
    return f"{n/1e6:.3f}M"


def dt(d):
    return ", ".join(f"{k.replace('torch.','')} {m(v)}" for k, v in sorted(d.items()))


def section(r):
    o = []
    a = o.append
    a(f"### {r['experiment']}\n")
    a(f"- wrapper `{r['wrapper']}` · base lr `{r['lr']}` · "
      f"总参数 {m(r['total_params_numel'])} · **可训 {m(r['trainable_numel'])}** "
      f"({100*r['trainable_numel']/r['total_params_numel']:.1f}%)")
    a(f"- 构造期就已冻结（`setup()` 之前）：**{r['ctor_frozen_count']}** 个参数张量")
    a("")
    a("**(1) freeze_keywords 命中**\n")
    if r["freeze_keywords"]:
        a("| keyword | 命中张量 | 命中参数量 |")
        a("|---|---:|---:|")
        for kw, h in r["keyword_hits"].items():
            a(f"| `{kw}` | {h['params']} | {m(h['numel'])} |")
    else:
        a("（无 `freeze_keywords`）")
    a("")
    a("**(2) 实际 requires_grad=True 的集合（按模块 rollup）**\n")
    a("| 模块 | 张量 | 参数量 |")
    a("|---|---:|---:|")
    for k, v in sorted(r["trainable_rollup_deep"].items(), key=lambda x: -x[1]["numel"]):
        a(f"| `{k}` | {v['params']} | {m(v['numel'])} |")
    a(f"| **合计** | | **{m(r['trainable_numel'])}** |")
    a("")
    a("**(3) dtype**\n")
    a(f"- 可训集合：{dt(r['trainable_dtypes'])}")
    a(f"- 全模型：{dt(r['all_param_dtypes'])}")
    a("")
    a(f"**(4) param_groups 归属与实际 lr**（`{r['group_mode']}`）\n")
    a("| 组 | keywords | lr_multiplier | 实际 lr | 张量 | 参数量 | dtype |")
    a("|---|---|---:|---:|---:|---:|---|")
    for g in r["groups"]:
        kws = ", ".join(f"`{k}`" for k in g["kw"])
        a(f"| | {kws} | {g['mult']} | {r['lr']*g['mult']:.2e} | {g['params']} | "
          f"{m(g['numel'])} | {dt(g['dtypes']) or '—'} |")
    a("")
    return "\n".join(o)


if __name__ == "__main__":
    out = []
    for e in ORDER:
        p = f"{RAW}/{e}.json"
        if not os.path.exists(p):
            out.append(f"### {e}\n\n**（未测：probe 未产出）**\n")
            continue
        out.append(section(json.load(open(p))))
    sys.stdout.write("\n".join(out))
