"""Publication figures (vector PDF + 600 dpi PNG) in 04_manuscript_v2/figures.

Run after 07_results_tables.py:  python -I 02_code_v2/scripts/08_figures.py
"""
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bmp_early import cases as C         # noqa: E402

R = C.ROOT / "03_outputs_v2"
FIG = C.ROOT / "04_manuscript_v2" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"
COL = {"locked": "#1f5fa8", "P0": "#55534d", "P1": "#14805a", "P2": "#d08c00", "C1": "#c2588a"}
MRK = {"locked": "o", "P0": "s", "P1": "^", "P2": "D", "C1": "v"}
NAME = {"locked": "Residual-corrected first-order model", "P0": "Last observed net yield",
        "P1": "Class-completion ratio", "P2": "Lag-adjusted first-order fit",
        "C1": "Cellulose-adjusted ratio"}
CLS = [("readily", "Microcrystalline cellulose"), ("feed", "Animal feeds"), ("straw", "Wheat straw")]
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
                     "font.family": "DejaVu Sans", "pdf.fonttype": 42})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def style(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_completion():
    df, curves = pickle.load(open(R / "iis" / "iis_cases.pkl", "rb"))
    refs = df.drop_duplicates(["lab", "test", "material"])
    refs = refs[refs.status == "validated"]
    grid = np.arange(0, 41, 0.5)
    fig, axes = plt.subplots(1, 3, figsize=(7.48, 2.4), sharey=True)
    for ax, (c, title) in zip(axes, CLS):
        sub = refs[refs.cls == c]
        vals = []
        for _, r in sub.iterrows():
            tt, yy = curves[(r.lab, r.test, r.material)]
            v = np.interp(grid, tt, yy) / r.ystar
            v[grid > tt.max()] = np.nan
            vals.append(v)
        V = np.vstack(vals)
        q = np.nanpercentile(V, [10, 25, 50, 75, 90], axis=0)
        ax.fill_between(grid, q[0], q[4], color="#cde2fb", lw=0, label="10th–90th percentile")
        ax.fill_between(grid, q[1], q[3], color="#86b6ef", lw=0, label="Interquartile range")
        ax.plot(grid, q[2], color="#184f95", lw=1.6, label="Median")
        ts = sub.tstar.median()
        ax.axvline(ts, color=INK, lw=0.8, ls="--")
        ax.text(ts + 0.6 if ts < 25 else ts - 0.6, 0.06, ha="left" if ts < 25 else "right", s= f"median day at\n1% net 3 d: {ts:.0f}", fontsize=7, color=INK)
        for d in (5, 7, 10, 14, 21):
            ax.axvline(d, color=GRID, lw=0.8, zorder=0)
        m10 = q[2][grid == 10][0]
        ax.plot(10, m10, "o", ms=3.5, color="#184f95", mec="white", mew=0.6, zorder=4)
        ax.annotate(f"day 10: {100 * m10:.0f}%", (10, m10), xytext=(20.5, m10 - 0.30) if m10 > 0.8 else (13, m10 - 0.12),
                    fontsize=7, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
        ax.set_title(f"{title}\n(n = {len(sub)}; {sub.lab.nunique()} laboratories)", fontsize=8, color=INK)
        ax.set_xlabel("Test day (d)")
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 1.1)
        style(ax)
    axes[0].set_ylabel("Net SMP / reference BMP")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.17))
    save(fig, "Figure_1")


def panel(ax, letter, x=-0.10, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)


def fig_error():
    m = pd.read_csv(R / "results" / "metrics.csv")
    m = m[m.set == "IIS_LOLO_validated"]
    dls = json.load(open(R / "results" / "additional_results.json"))["between_lab_abs_log_dev_IIS"]
    fig, axes = plt.subplots(1, 3, figsize=(7.48, 2.8), sharey=True)
    offs = {k: o for k, o in zip(COL, np.linspace(-0.6, 0.6, len(COL)))}
    for i, (ax, (c, title)) in enumerate(zip(axes, CLS)):
        d = m[m.cls == c]
        for meth in COL:
            x = d[d.method == meth].sort_values("day")
            main = meth in ("locked", "P1", "P0")
            ax.errorbar(x.day + offs[meth], x.mae_log, yerr=[x.mae_log - x.ci_lo, x.ci_hi - x.mae_log],
                        color=COL[meth], marker=MRK[meth], ms=4 if main else 3, lw=1.4 if main else 0.7,
                        alpha=1.0 if main else 0.55, elinewidth=0.7, capsize=1.5, label=NAME[meth],
                        zorder=3 if main else 2)
        dl = dls[c]
        ax.axhline(dl, color=INK, lw=0.9, ls="--", zorder=1)
        ax.text(15.5 if c == "feed" else 3.8, dl * 1.07, f"$d_L$ = {dl:.3f}", fontsize=6.5, color=INK, ha="left", va="bottom")
        x0 = d[(d.method == "locked") & (d.day == 10)]
        ax.set_title(f"({'ABC'[i]}) {title}\n({int(x0.G.iloc[0])} labs; {int(x0.n_cases.iloc[0])} eligible conditions)", fontsize=8, color=INK)
        ax.set_yscale("log")
        ax.set_ylim(0.008, 2.2)
        ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1.0])
        ax.set_yticklabels(["0.01", "0.03", "0.1", "0.3", "1"])
        ax.set_xticks([5, 7, 10, 14, 21])
        ax.set_xlim(3.5, 23.5)
        ax.set_xlabel("Test day (d)")
        style(ax)
        if i == 2:
            sec = ax.secondary_yaxis("right", functions=(lambda v: v, lambda v: v))
            ticks = [0.01, 0.03, 0.1, 0.3, 1.0]
            sec.set_yticks(ticks)
            sec.set_yticklabels([f"{100 * (np.exp(t) - 1):.0f}%" for t in ticks], fontsize=7)
            sec.set_ylabel("Multiplicative deviation, exp(MALE) − 1", fontsize=7.5, color=INK)
            sec.tick_params(colors=MUTED)
            sec.spines["right"].set_visible(True)
            sec.spines["right"].set_color(MUTED)
    axes[0].set_ylabel("Mean absolute log error")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.19))
    save(fig, "Figure_2")


def fig_cellulose():
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec
    hm = pd.read_csv(R / "results" / "heatmap_lab_substrate_day10.csv", dtype={"lab": str}).set_index("lab")
    subs = ["CEL", "SA", "SB", "SC", "SD"]
    lab_names = {"CEL": "Cellulose", "SA": "Feed SA", "SB": "Feed SB", "SC": "Feed SC", "SD": "Wheat straw"}
    hm["mean_e"] = hm[subs].mean(axis=1)
    hm = hm.sort_values("mean_e")
    d = pd.read_csv(R / "results" / "cellulose_ratio_day10.csv", dtype={"lab": str}).dropna(subset=["cel_ratio"])
    d = d[np.isfinite(d.locked) & (d.locked > 0)]
    d["e"] = np.log(d.locked / d.ystar)
    fig = plt.figure(figsize=(7.48, 5.6))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.18, 1.0, 1.75], width_ratios=[1.45, 1], hspace=0.34, wspace=0.32)
    # A: laboratory x substrate heat map with cellulose strip
    a0 = fig.add_subplot(gs[0, :])
    a1 = fig.add_subplot(gs[1, :], sharex=a0)
    seq = plt.get_cmap("YlOrBr_r")
    strip = hm.cel_ratio.clip(0.2, 1.2).to_numpy()[None, :]
    im0 = a0.imshow(strip, aspect="auto", cmap=seq, vmin=0.2, vmax=1.2, interpolation="nearest")
    a0.set_yticks([0])
    a0.set_yticklabels(["Cellulose ratio"], fontsize=7)
    a0.tick_params(axis="x", bottom=False, labelbottom=False)
    for s_ in a0.spines.values():
        s_.set_visible(False)
    for j, (lab, r) in enumerate(hm.iterrows()):
        if r.share_accepted == 0:
            a0.text(j, -0.75, "×", ha="center", va="bottom", fontsize=6, color=INK)
    div = plt.get_cmap("RdBu")
    norm = mcolors.TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=0.6)
    M = hm[subs].T.to_numpy()
    im1 = a1.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=div, norm=norm, interpolation="nearest")
    a1.set_facecolor("#ecebe6")
    a1.set_yticks(range(len(subs)))
    a1.set_yticklabels([lab_names[s_] for s_ in subs], fontsize=7)
    a1.set_xticks(range(len(hm)))
    a1.set_xticklabels([])
    a1.tick_params(axis="x", length=0)
    a1.set_xlabel(f"{len(hm)} laboratories, ordered by mean signed error", fontsize=7.5, labelpad=3)
    for s_ in a1.spines.values():
        s_.set_visible(False)
    a1.set_xticks(np.arange(-0.5, len(hm), 1), minor=True)
    a1.set_yticks(np.arange(-0.5, len(subs), 1), minor=True)
    a1.grid(which="minor", color="white", lw=0.8)
    a1.tick_params(which="minor", length=0)
    panel(a0, "A", x=-0.075, y=1.3)
    cax = fig.add_axes([0.915, 0.555, 0.012, 0.22])
    cb = fig.colorbar(im1, cax=cax)
    cb.set_label("Signed log error", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cax2 = fig.add_axes([0.915, 0.83, 0.012, 0.05])
    cb2 = fig.colorbar(im0, cax=cax2, orientation="vertical")
    cb2.set_ticks([0.2, 1.2])
    cb2.ax.tick_params(labelsize=6)
    # B: scatter
    b = fig.add_subplot(gs[2, 0])
    b.axvspan(0, 0.9, color="#f6e7d3", lw=0, zorder=0)
    for st, fc, lab in (("validated", "#1f5fa8", "Accepted test"), ("outside_qc", "none", "Test failing criteria")):
        x = d[d.status == st]
        b.scatter(x.cel_ratio, x.e, s=10, facecolor=fc, edgecolor="#1f5fa8", lw=0.6, alpha=0.75, label=lab, zorder=3)
    b.axvline(0.9, color=INK, lw=0.8, ls="--")
    b.axhline(0, color=MUTED, lw=0.7)
    b.text(0.62, 0.97, "0.9 threshold", transform=b.transAxes, fontsize=6.5, color=INK, va="top")
    b.set_xlim(0, 1.5)
    b.set_ylim(-1.6, 0.95)
    b.set_xlabel("Day-10 cellulose yield / mean of other laboratories", fontsize=7.5)
    b.set_ylabel("Signed log error on day 10", fontsize=7.5)
    b.legend(frameon=False, fontsize=6.5, loc="lower right")
    panel(b, "B", x=-0.13, y=1.0)
    style(b)
    # C: group errors
    c = fig.add_subplot(gs[2, 1])
    d["grp"] = pd.cut(d.cel_ratio, [0, 0.9, 1.1, 9], labels=["< 0.9", "0.9–1.1", "> 1.1"])
    rng = np.random.default_rng(3)
    for j, col in enumerate(("locked", "P1")):
        xs, ms, lo, hi = [], [], [], []
        for i, (g, x) in enumerate(d.groupby("grp", observed=True)):
            x = x[np.isfinite(x[col]) & (x[col] > 0)]
            per = np.abs(np.log(x[col] / x.ystar)).groupby(x.lab).mean()
            bs = [rng.choice(per.values, len(per)).mean() for _ in range(2000)]
            xs.append(i + (j - 0.5) * 0.28)
            ms.append(per.mean())
            lo.append(per.mean() - np.percentile(bs, 2.5))
            hi.append(np.percentile(bs, 97.5) - per.mean())
        c.errorbar(xs, ms, yerr=[lo, hi], fmt=MRK[col], color=COL[col], ms=4.5, capsize=2, elinewidth=0.8, label=NAME[col])
    ns = d.groupby("grp", observed=True).agg(n=("e", "size"), g=("lab", "nunique"))
    c.set_xticks(range(3))
    c.set_xticklabels([f"{k}\n{r.n} cond.\n{r.g} labs" for k, r in ns.iterrows()], fontsize=6.5)
    c.set_xlabel("Day-10 cellulose yield ratio", fontsize=7.5)
    c.set_ylabel("Mean absolute log error on day 10", fontsize=7.5)
    c.set_ylim(0, 0.75)
    T = json.load(open(R / "results" / "review_checks.json"))["E_test_level"]
    c.text(0.97, 0.97, "Tests failing the cellulose criteria\n"
           f"ratio < 0.9: {T['low_other']} of {T['low_other'] + T['low_validated']}\n"
           f"ratio ≥ 0.9: {T['not_low_other']} of {T['not_low_other'] + T['not_low_validated']}",
           transform=c.transAxes, ha="right", va="top", fontsize=6.3, color=INK,
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.6))
    c.legend(frameon=False, fontsize=6.3, loc="center", bbox_to_anchor=(0.62, 0.32))
    panel(c, "C", x=-0.2, y=1.0)
    style(c)
    save(fig, "Figure_3")


def fig_external():
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec
    p = pd.read_csv(R / "il" / "il_predictions.csv", dtype={"lab": str})
    p["eval_lab"] = p.lab.str.startswith("E")
    val = p[(p.status == "validated") & (p.base != C.CEL_IL) & p.sync]
    prim = val[(val["round"] == 1) & val.eval_lab]
    other = val[~((val["round"] == 1) & val.eval_lab)]
    mats = ["Rabbit feed", "Sunflower seeds", "Wheat straw"]
    mk = dict(zip(mats, ["o", "s", "^"]))
    fig = plt.figure(figsize=(7.2, 7.6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 0.9], hspace=0.62, wspace=0.3)
    lim = (150, 750)
    for k, day in enumerate((10, 14)):
        ax = fig.add_subplot(gs[0, k])
        o = other[other.day == day]
        ax.scatter(o.ystar, o.locked, s=11, facecolor="none", edgecolor="#a9a8a0", lw=0.7, label="Other accepted conditions (round 2, reference laboratories)", zorder=1)
        d = prim[prim.day == day]
        for mat in mats:
            x = d[d.base == mat]
            ax.errorbar(x.ystar, x.locked, yerr=[x.locked - x.pi_lo, x.pi_up - x.locked], fmt=mk[mat], ms=4.5,
                        color="#1f5fa8", mec="white", mew=0.5, elinewidth=0.7, capsize=1.5, label=mat, zorder=3)
        xx = np.array(lim)
        ax.plot(xx, xx, color=INK, lw=0.8)
        ax.plot(xx, xx * 1.1, color=MUTED, lw=0.6, ls="--")
        ax.plot(xx, xx * 0.9, color=MUTED, lw=0.6, ls="--")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_title(f"Day {day}", fontsize=8, color=INK)
        e = np.abs(np.log(d.locked / d.ystar)).groupby(d.lab).mean().mean()
        cov = int(((d.ystar >= d.pi_lo) & (d.ystar <= d.pi_up)).sum())
        ax.text(0.97, 0.04, f"MALE {e:.3f}\ninside interval {cov}/{len(d)}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.3, color=INK)
        ax.set_xlabel(r"Reference BMP (NmL CH$_4$ g$^{-1}$ VS)", fontsize=7.5)
        if k == 0:
            ax.set_ylabel(r"Early BMP estimate (NmL CH$_4$ g$^{-1}$ VS)", fontsize=7.5)
        else:
            h_, l_ = ax.get_legend_handles_labels()
            fig.legend(h_, l_, frameon=False, fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.505), handletextpad=0.3)
        panel(ax, "AB"[k], x=-0.16)
        style(ax)
    # C: laboratory x material heat map, corrected model and class-completion ratio
    h = pd.read_csv(R / "results" / "il_primary_day10.csv", dtype={"lab": str})
    labs = sorted(h.lab.unique())
    ax = fig.add_subplot(gs[1, :].subgridspec(1, 3, width_ratios=[0.18, 1, 0.12])[0, 1])
    cols = [(mat, meth) for meth in ("e_P3", "e_P1") for mat in mats]
    M = np.full((len(labs), len(cols)), np.nan)
    for i, lab in enumerate(labs):
        for j, (mat, meth) in enumerate(cols):
            v = h[(h.lab == lab) & (h.base == mat)][meth]
            if len(v):
                M[i, j] = v.mean()
    norm = mcolors.TwoSlopeNorm(vmin=-0.4, vcenter=0, vmax=0.4)
    im = ax.imshow(np.ma.masked_invalid(M), cmap="RdBu", norm=norm, aspect="auto", interpolation="nearest")
    ax.set_facecolor("#ecebe6")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}".replace("-", "−"), ha="center", va="center", fontsize=6.5,
                        color="white" if abs(M[i, j]) > 0.25 else INK)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels([f"Lab {i + 1}" for i in range(len(labs))], fontsize=6.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["Rabbit\nfeed", "Sunflower\nseeds", "Wheat\nstraw"] * 2, fontsize=6.5)
    ax.axvline(2.5, color="white", lw=2.5)
    ax.text(1, -0.62, "Corrected model", ha="center", va="bottom", fontsize=7, color=INK)
    ax.text(4, -0.62, "Class-completion ratio", ha="center", va="bottom", fontsize=7, color=INK)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Signed log error, day 10", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    panel(ax, "C", x=-0.14, y=1.08)
    save(fig, "Figure_4")


if __name__ == "__main__":
    fig_cellulose()
    fig_completion()
    fig_error()
    fig_external()
    print(sorted(p.name for p in FIG.iterdir()))


def graphical_abstract():
    df, curves = pickle.load(open(R / "iis" / "iis_cases.pkl", "rb"))
    m = pd.read_csv(R / "results" / "metrics.csv")
    refs = df.drop_duplicates(["lab", "test", "material"])
    refs = refs[refs.status == "validated"]
    grid = np.arange(0, 41, 0.5)
    cols = {"readily": "#2a78d6", "feed": "#1baf7a", "straw": "#eb6834"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.2, 2.4), gridspec_kw={"width_ratios": [1.3, 1]})
    for c, title in CLS:
        sub = refs[refs.cls == c]
        V = []
        for _, r in sub.iterrows():
            tt, yy = curves[(r.lab, r.test, r.material)]
            v = np.interp(grid, tt, yy) / r.ystar
            v[grid > tt.max()] = np.nan
            V.append(v)
        a1.plot(grid, np.nanmedian(np.vstack(V), axis=0), color=cols[c], lw=1.8, label=title)
    a1.axvline(10, color=INK, lw=0.8, ls="--")
    a1.text(10.6, 0.04, "day 10", fontsize=6.5, color=INK)
    a1.set_xlabel("Test day (d)")
    a1.set_ylabel("Net SMP / reference BMP")
    a1.set_xlim(0, 40)
    a1.set_ylim(0, 1.1)
    a1.legend(frameon=False, fontsize=6.5, loc="center right", bbox_to_anchor=(1.0, 0.42))
    style(a1)
    x = []
    for i, (c, title) in enumerate(CLS):
        r = m[(m.set == "IIS_LOLO_validated") & (m.cls == c) & (m.day == 10) & (m.method == "locked")].iloc[0]
        a2.bar(i, r.mae_log, color=cols[c], width=0.6, yerr=[[r.mae_log - r.ci_lo], [r.ci_hi - r.mae_log]],
               capsize=2, error_kw={"elinewidth": 0.8})
        x.append({"readily": "Cellulose", "feed": "Feeds", "straw": "Wheat\nstraw"}[c])
    q = m[(m.set == "IIS_LOLO_outsideQC") & (m.cls == "all") & (m.day == 10) & (m.method == "locked")].iloc[0]
    a2.bar(3, q.mae_log, color="#a9a8a0", width=0.6, yerr=[[q.mae_log - q.ci_lo], [q.ci_hi - q.mae_log]],
           capsize=2, error_kw={"elinewidth": 0.8})
    a2.set_xticks(range(4))
    a2.set_xticklabels(x + ["Failing\ncellulose\ncriteria"], fontsize=6.5)
    a2.set_ylabel("Day-10 MALE, held-out laboratories")
    style(a2)
    fig.tight_layout()
    fig.savefig(FIG / "Graphical_abstract.pdf", bbox_inches="tight")
    fig.savefig(FIG / "Graphical_abstract.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    graphical_abstract()
