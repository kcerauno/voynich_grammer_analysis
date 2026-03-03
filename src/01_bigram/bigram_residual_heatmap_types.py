"""
bigram_residual_heatmap_types.py
=================================
「スロット境界・語形成ルール」の検出を目的とした
バイグラム残差ヒートマップ（単語を一意化した types 版）。

bigram_residual_heatmap.py との違い:
  - 全体・各カテゴリそれぞれで単語を set() で一意化してからバイグラムを集計する。
  - 同じ単語が何度テキストに現れても 1 回としか数えない（type 頻度）。
  - これにより、高頻度単語による遷移頻度の誇張を除き、
    語形成ルール（どの文字遷移が構造的に許容・忌避されるか）を純粋に反映する。

出力先: hypothesis/01_bigram/results/
  - bigram_residual_heatmap_types_all.png   全語彙 標準化残差ヒートマップ
  - bigram_pmi_heatmap_types_all.png        全語彙 PMIヒートマップ
  - bigram_residual_heatmap_types_{cat}.png カテゴリ別 標準化残差ヒートマップ
  - bigram_top_transitions_types.csv        上位/下位遷移ペアのCSV
  - bigram_residual_report_types.txt        テキストレポート
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib import font_manager
from collections import Counter, defaultdict
from pathlib import Path
import os

# ─── 日本語フォント設定（Windows 標準フォントを優先順で試す） ─────────────────
def _setup_jp_font():
    candidates = ["Yu Gothic", "Meiryo", "MS Gothic", "IPAexGothic", "Noto Sans CJK JP"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            print(f"  [font] Using '{name}' for Japanese text.")
            break
    else:
        print("  [font] Warning: No standard Japanese font found. Titles may show □.")
    matplotlib.rcParams["axes.unicode_minus"] = False

_setup_jp_font()

# ─── 設定 ────────────────────────────────────────────────────────────────────
DB_PATH = "data/voynich.db"
OUT_DIR = Path("hypothesis/01_bigram/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_PAIR_COUNT = 2      # types 版では総数が減るので閾値を下げる
PMI_MIN_COUNT  = 2

# ─── データ読み込み ──────────────────────────────────────────────────────────
print("Loading data from DB...")
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(
    "SELECT page, category, word FROM words_enriched WHERE word IS NOT NULL AND word != ''",
    conn,
)
conn.close()

categories = sorted(df["category"].dropna().unique().tolist())

# ─── 一意化ユーティリティ ────────────────────────────────────────────────────
def unique_words(words):
    """単語リストを一意化して返す（type 頻度に変換）"""
    return list(set(words))

# ─── バイグラム頻度行列の構築 ────────────────────────────────────────────────
def build_bigram_matrix(words):
    """
    単語リスト（一意化済みを想定）からバイグラム頻度行列を構築する。
    BOS = '^', EOS = '$' を含む。
    """
    raw = defaultdict(Counter)
    for w in words:
        if not w:
            continue
        seq = ["^"] + list(w) + ["$"]
        for a, b in zip(seq, seq[1:]):
            raw[a][b] += 1

    row_chars = sorted(raw.keys())
    col_chars_set = set()
    for cnt in raw.values():
        col_chars_set.update(cnt.keys())
    col_chars = sorted(col_chars_set)

    matrix = np.zeros((len(row_chars), len(col_chars)), dtype=float)
    for i, a in enumerate(row_chars):
        for j, b in enumerate(col_chars):
            matrix[i, j] = raw[a].get(b, 0.0)

    return pd.DataFrame(matrix, index=row_chars, columns=col_chars)

def compute_residuals(count_df):
    """頻度行列から標準化残差・PMIを計算して返す。"""
    O = count_df.values.copy()
    total = O.sum()

    row_sums = O.sum(axis=1, keepdims=True)
    col_sums = O.sum(axis=0, keepdims=True)
    E = (row_sums @ col_sums) / total

    with np.errstate(divide="ignore", invalid="ignore"):
        std_resid = np.where(E > 0, (O - E) / np.sqrt(E), 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.where(
            (O >= PMI_MIN_COUNT) & (E > 0),
            np.log2(np.where(E > 0, O * total / (row_sums @ col_sums), 1.0)),
            np.nan,
        )

    std_resid_df = pd.DataFrame(std_resid, index=count_df.index, columns=count_df.columns)
    pmi_df       = pd.DataFrame(pmi,       index=count_df.index, columns=count_df.columns)
    return std_resid_df, pmi_df

# ─── ヒートマップ描画 ─────────────────────────────────────────────────────────
def _char_label(c):
    return {"^": "BOS", "$": "EOS"}.get(c, c)

def plot_heatmap(matrix_df, title, out_path, cmap="RdBu_r", center=0,
                 vmin=None, vmax=None, annot=False, figsize=None):
    n_rows, n_cols = matrix_df.shape
    if figsize is None:
        figsize = (max(12, n_cols * 0.55), max(9, n_rows * 0.45))

    fig, ax = plt.subplots(figsize=figsize)
    mask = matrix_df.isnull()

    sns.heatmap(
        matrix_df,
        mask=mask,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.3,
        linecolor="gray",
        ax=ax,
        annot=annot,
        fmt=".1f" if annot else "",
        xticklabels=[_char_label(c) for c in matrix_df.columns],
        yticklabels=[_char_label(c) for c in matrix_df.index],
        cbar_kws={"shrink": 0.75},
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("次の文字 (b)", fontsize=11)
    ax.set_ylabel("現在の文字 (a)", fontsize=11)
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=8, rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

# ─── Top/Bottom 遷移の抽出 ────────────────────────────────────────────────────
def extract_top_transitions(count_df, std_resid_df, pmi_df, n=30):
    rows = []
    for a in count_df.index:
        for b in count_df.columns:
            o = count_df.loc[a, b]
            if o < MIN_PAIR_COUNT:
                continue
            sr = std_resid_df.loc[a, b]
            pm = pmi_df.loc[a, b] if not np.isnan(pmi_df.loc[a, b]) else None
            rows.append({
                "from": _char_label(a),
                "to":   _char_label(b),
                "type_count": int(o),
                "std_residual": round(float(sr), 3),
                "pmi": round(float(pm), 3) if pm is not None else None,
            })
    result = pd.DataFrame(rows).sort_values("std_residual", ascending=False)
    top    = result.head(n)
    bottom = result.tail(n).iloc[::-1]
    sep    = pd.DataFrame([{"from": "...", "to": "...", "type_count": None,
                            "std_residual": None, "pmi": None}])
    return pd.concat([top, sep, bottom], ignore_index=True)

# ─── テキストレポート生成 ──────────────────────────────────────────────────────
def generate_report(count_df, std_resid_df, pmi_df, n_types, label="全語彙"):
    lines = []
    lines.append(f"{'='*72}")
    lines.append(f"  バイグラム残差レポート（types版） — {label}")
    lines.append(f"{'='*72}")
    lines.append(f"  ユニーク単語数（types）: {n_types:,}")
    lines.append(f"  総バイグラム数（types）: {int(count_df.values.sum()):,}")
    lines.append(f"  ユニーク前文字: {len(count_df.index)}")
    lines.append(f"  ユニーク後文字: {len(count_df.columns)}")
    lines.append("")

    rows = []
    for a in count_df.index:
        for b in count_df.columns:
            o = count_df.loc[a, b]
            if o < MIN_PAIR_COUNT:
                continue
            sr = std_resid_df.loc[a, b]
            pm = pmi_df.loc[a, b]
            rows.append((_char_label(a), _char_label(b), int(o), float(sr),
                         float(pm) if not np.isnan(pm) else None))

    for heading, reverse in [
        ("【上位 25 遷移（構造的に強制されている遷移）】", True),
        ("【下位 25 遷移（構造的に忌避されている遷移）】", False),
    ]:
        lines.append(heading)
        lines.append(f"  {'遷移':<12} {'type数':>8} {'標準化残差':>14} {'PMI':>10}")
        lines.append("  " + "-"*50)
        rows.sort(key=lambda x: x[3], reverse=reverse)
        for a, b, o, sr, pm in rows[:25]:
            pmi_str = f"{pm:+.3f}" if pm is not None else "  N/A"
            lines.append(f"  {a} → {b:<8} {o:>8,} {sr:>+14.2f} {pmi_str:>10}")
        lines.append("")

    return "\n".join(lines)

# ════════════════════════════════════════════════════════════════════════════
# メイン処理
# ════════════════════════════════════════════════════════════════════════════

report_sections = []

# ── 1. 全語彙 ────────────────────────────────────────────────────────────────
print("\n[1/2] 全語彙（types）のバイグラム残差ヒートマップを生成中...")
all_words_types = unique_words(df["word"].tolist())
print(f"  全トークン数: {len(df):,}  →  ユニーク語数(types): {len(all_words_types):,}")

count_all      = build_bigram_matrix(all_words_types)
std_resid_all, pmi_all = compute_residuals(count_all)

# 標準化残差ヒートマップ（全語彙）
clim = float(np.nanpercentile(np.abs(std_resid_all.values), 98))
plot_heatmap(
    std_resid_all,
    title="Voynich 全語彙(types) バイグラム標準化残差ヒートマップ\n(赤=構造的に強制, 青=構造的に忌避)",
    out_path=OUT_DIR / "bigram_residual_heatmap_types_all.png",
    cmap="RdBu_r", center=0, vmin=-clim, vmax=clim,
)

# PMIヒートマップ（全語彙）
valid_pmi = pmi_all.values[~np.isnan(pmi_all.values)]
pmi_clip = float(np.nanpercentile(np.abs(valid_pmi), 95)) if len(valid_pmi) > 0 else 5.0
plot_heatmap(
    pmi_all,
    title="Voynich 全語彙(types) バイグラム PMI ヒートマップ\n(高=強い共起, 低=忌避)",
    out_path=OUT_DIR / "bigram_pmi_heatmap_types_all.png",
    cmap="PuOr", center=0, vmin=-pmi_clip, vmax=pmi_clip,
)

report_sections.append(
    generate_report(count_all, std_resid_all, pmi_all,
                    n_types=len(all_words_types), label="全語彙")
)

# Top/Bottom CSV
top_df = extract_top_transitions(count_all, std_resid_all, pmi_all, n=30)
csv_path = OUT_DIR / "bigram_top_transitions_types.csv"
top_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"  Saved: {csv_path}")

# ── 2. カテゴリ別 ─────────────────────────────────────────────────────────────
print(f"\n[2/2] カテゴリ別ヒートマップを生成中 ({len(categories)} カテゴリ)...")
for cat in categories:
    tokens_cat = df[df["category"] == cat]["word"].tolist()
    types_cat  = unique_words(tokens_cat)

    if len(types_cat) < 30:
        print(f"  Skipping '{cat}' (types={len(types_cat)} < 30)")
        continue

    print(f"  Category: {cat}  tokens={len(tokens_cat):,}  types={len(types_cat):,}")
    count_cat              = build_bigram_matrix(types_cat)
    std_resid_cat, pmi_cat = compute_residuals(count_cat)

    clim_cat  = float(np.nanpercentile(np.abs(std_resid_cat.values), 98))
    safe_name = cat.replace("/", "_").replace(" ", "_")
    plot_heatmap(
        std_resid_cat,
        title=f"Voynich [{cat}] バイグラム標準化残差ヒートマップ（types）\n"
              f"tokens={len(tokens_cat):,} / types={len(types_cat):,}  (赤=強制, 青=忌避)",
        out_path=OUT_DIR / f"bigram_residual_heatmap_types_{safe_name}.png",
        cmap="RdBu_r", center=0, vmin=-clim_cat, vmax=clim_cat,
    )

    report_sections.append(
        generate_report(count_cat, std_resid_cat, pmi_cat,
                        n_types=len(types_cat), label=f"category={cat}")
    )

# ── テキストレポート保存 ────────────────────────────────────────────────────
report_path = OUT_DIR / "bigram_residual_report_types.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(report_sections))
print(f"\nReport saved: {report_path}")

print("\n✓ 完了。出力先:", OUT_DIR.resolve())
