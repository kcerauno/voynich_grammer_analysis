"""
bigram_residual_heatmap.py
==========================
Voynich Manuscript のバイグラム残差ヒートマップを生成する。

残差の定義:
  O_ij = 実測バイグラム (i→j) の頻度
  E_ij = 独立仮定期待値 = rowsum(i) * colsum(j) / total
  標準化残差 = (O - E) / sqrt(E)
  PMI(i→j) = log2( P(i,j) / (P(i) * P(j)) )

出力先: hypothesis/01_bigram/results/
  - bigram_residual_heatmap_all.png   全データ 標準化残差ヒートマップ
  - bigram_pmi_heatmap_all.png        全データ PMIヒートマップ
  - bigram_residual_heatmap_{cat}.png カテゴリ別 標準化残差ヒートマップ
  - bigram_top_transitions.csv        上位/下位遷移ペアのCSV
  - bigram_residual_report.txt        テキストレポート
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
        # フォント名がヒットしなくてもパスから探す（日本語フォントが別名登録の場合）
        print("  [font] Warning: No standard Japanese font found. Titles may show □.")
    matplotlib.rcParams["axes.unicode_minus"] = False  # マイナス記号が□になるのを防ぐ

_setup_jp_font()
from collections import Counter, defaultdict
from pathlib import Path
import math
import os

# ─── 設定 ────────────────────────────────────────────────────────────────────
DB_PATH = "data/voynich.db"
OUT_DIR = Path("hypothesis/01_bigram/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_PAIR_COUNT = 5       # ヒートマップに表示する最低バイグラム出現回数
PMI_MIN_COUNT  = 5       # PMI計算に使う最低出現回数（ゼロ頻度の影響を抑制）

# ─── データ読み込み ──────────────────────────────────────────────────────────
print("Loading data from DB...")
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(
    "SELECT page, category, word FROM words_enriched WHERE word IS NOT NULL AND word != ''",
    conn,
)
conn.close()

all_words    = df["word"].tolist()
categories   = sorted(df["category"].dropna().unique().tolist())

# ─── ユーティリティ: バイグラム頻度行列の構築 ────────────────────────────────
def build_bigram_matrix(words):
    """
    単語リストからバイグラム頻度行列を構築する。
    BOS = '^', EOS = '$' を含む。
    Returns:
        count_df  : DataFrame (row=前文字, col=後文字, values=出現回数)
        chars     : 出現した文字一覧（BOS/EOSを含む）
    """
    raw = defaultdict(Counter)
    for w in words:
        if not w:
            continue
        seq = ["^"] + list(w) + ["$"]
        for a, b in zip(seq, seq[1:]):
            raw[a][b] += 1

    # 全文字セット（行:前文字, 列:後文字）
    row_chars = sorted(raw.keys())
    col_chars_set = set()
    for cnt in raw.values():
        col_chars_set.update(cnt.keys())
    col_chars = sorted(col_chars_set)

    # DataFrameに整形
    matrix = np.zeros((len(row_chars), len(col_chars)), dtype=float)
    for i, a in enumerate(row_chars):
        for j, b in enumerate(col_chars):
            matrix[i, j] = raw[a].get(b, 0.0)

    count_df = pd.DataFrame(matrix, index=row_chars, columns=col_chars)
    return count_df

def compute_residuals(count_df):
    """
    頻度行列から標準化残差・PMIを計算して返す。
    """
    O = count_df.values.copy()
    total = O.sum()

    row_sums = O.sum(axis=1, keepdims=True)   # 各前文字の合計
    col_sums = O.sum(axis=0, keepdims=True)   # 各後文字の合計

    # 独立仮定期待値
    E = (row_sums @ col_sums) / total

    # 標準化残差 (Pearson): (O-E)/sqrt(E)
    with np.errstate(divide="ignore", invalid="ignore"):
        std_resid = np.where(E > 0, (O - E) / np.sqrt(E), 0.0)

    # PMI: log2( (O/total) / (P(a)*P(b)) ) = log2( O*total / (rowsum*colsum) )
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
    """BOS/EOS を分かりやすいラベルに変換"""
    return {"^": "BOS", "$": "EOS"}.get(c, c)

def plot_heatmap(matrix_df, title, out_path, cmap="RdBu_r", center=0,
                 vmin=None, vmax=None, fmt=".1f", annot=False, figsize=None):
    n_rows, n_cols = matrix_df.shape
    if figsize is None:
        figsize = (max(12, n_cols * 0.55), max(9, n_rows * 0.45))

    fig, ax = plt.subplots(figsize=figsize)

    # NaN マスク（PMI でゼロ頻度のセル）
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
        fmt=fmt if annot else "",
        xticklabels=[_char_label(c) for c in matrix_df.columns],
        yticklabels=[_char_label(c) for c in matrix_df.index],
        cbar_kws={"shrink": 0.75},
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Next character (b)", fontsize=11)
    ax.set_ylabel("Current character (a)", fontsize=11)
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=8, rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Top/Bottom 遷移の抽出 ────────────────────────────────────────────────────
def extract_top_transitions(count_df, std_resid_df, pmi_df, n=30):
    """上位・下位の遷移ペアを DataFrame で返す"""
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
                "count": int(o),
                "std_residual": round(float(sr), 3),
                "pmi": round(float(pm), 3) if pm is not None else None,
            })
    result = pd.DataFrame(rows)
    result = result.sort_values("std_residual", ascending=False)
    top    = result.head(n)
    bottom = result.tail(n).iloc[::-1]
    return pd.concat([top, pd.DataFrame([{"from":"...","to":"...","count":None,"std_residual":None,"pmi":None}]), bottom], ignore_index=True)


# ─── テキストレポート生成 ──────────────────────────────────────────────────────
def generate_report(count_df, std_resid_df, pmi_df, label="全データ"):
    lines = []
    lines.append(f"{'='*72}")
    lines.append(f"  バイグラム残差レポート — {label}")
    lines.append(f"{'='*72}")
    lines.append(f"  総バイグラム数: {int(count_df.values.sum()):,}")
    lines.append(f"  ユニーク前文字: {len(count_df.index)}")
    lines.append(f"  ユニーク後文字: {len(count_df.columns)}")
    lines.append("")

    lines.append("【上位 25 遷移（標準化残差が最大 = 実測が期待値を大幅に上回る）】")
    lines.append(f"  {'遷移':<12} {'頻度':>8} {'標準化残差':>14} {'PMI':>10}")
    lines.append("  " + "-"*50)
    rows = []
    for a in count_df.index:
        for b in count_df.columns:
            o = count_df.loc[a, b]
            if o < MIN_PAIR_COUNT:
                continue
            sr = std_resid_df.loc[a, b]
            pm = pmi_df.loc[a, b]
            rows.append((_char_label(a), _char_label(b), int(o), float(sr), float(pm) if not np.isnan(pm) else None))
    rows.sort(key=lambda x: x[3], reverse=True)
    for a, b, o, sr, pm in rows[:25]:
        pmi_str = f"{pm:+.3f}" if pm is not None else "  N/A"
        lines.append(f"  {a} → {b:<8} {o:>8,} {sr:>+14.2f} {pmi_str:>10}")

    lines.append("")
    lines.append("【下位 25 遷移（標準化残差が最小 = 実測が期待値を大幅に下回る = 忌避）】")
    lines.append(f"  {'遷移':<12} {'頻度':>8} {'標準化残差':>14} {'PMI':>10}")
    lines.append("  " + "-"*50)
    rows.sort(key=lambda x: x[3])
    for a, b, o, sr, pm in rows[:25]:
        pmi_str = f"{pm:+.3f}" if pm is not None else "  N/A"
        lines.append(f"  {a} → {b:<8} {o:>8,} {sr:>+14.2f} {pmi_str:>10}")

    lines.append("")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# メイン処理
# ════════════════════════════════════════════════════════════════════════════

report_sections = []

# ── 1. 全データ ──────────────────────────────────────────────────────────────
print("\n[1/2] 全データのバイグラム残差ヒートマップを生成中...")
count_all       = build_bigram_matrix(all_words)
std_resid_all, pmi_all = compute_residuals(count_all)

# 標準化残差ヒートマップ（全体）
clim = float(np.nanpercentile(np.abs(std_resid_all.values), 98))
plot_heatmap(
    std_resid_all,
    title="Voynich 全単語 バイグラム標準化残差ヒートマップ\n(赤=実測>期待値, 青=実測<期待値)",
    out_path=OUT_DIR / "bigram_residual_heatmap_all.png",
    cmap="RdBu_r",
    center=0,
    vmin=-clim,
    vmax=clim,
)

# PMIヒートマップ（全体）
pmi_clip = float(np.nanpercentile(np.abs(pmi_all.values[~np.isnan(pmi_all.values)]), 95))
plot_heatmap(
    pmi_all,
    title="Voynich 全単語 バイグラム PMI ヒートマップ\n(高=強い共起, 低=忌避)",
    out_path=OUT_DIR / "bigram_pmi_heatmap_all.png",
    cmap="PuOr",
    center=0,
    vmin=-pmi_clip,
    vmax=pmi_clip,
)

# レポート
report_sections.append(generate_report(count_all, std_resid_all, pmi_all, label="全データ"))

# Top/Bottom CSV
top_df = extract_top_transitions(count_all, std_resid_all, pmi_all, n=30)
csv_path = OUT_DIR / "bigram_top_transitions.csv"
top_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"  Saved: {csv_path}")

# ── 2. カテゴリ別 ─────────────────────────────────────────────────────────────
print(f"\n[2/2] カテゴリ別ヒートマップを生成中 ({len(categories)} カテゴリ)...")
for cat in categories:
    words_cat = df[df["category"] == cat]["word"].tolist()
    if len(words_cat) < 50:
        print(f"  Skipping '{cat}' (word count={len(words_cat)} < 50)")
        continue

    print(f"  Category: {cat} ({len(words_cat)} words)")
    count_cat         = build_bigram_matrix(words_cat)
    std_resid_cat, pmi_cat = compute_residuals(count_cat)

    clim_cat = float(np.nanpercentile(np.abs(std_resid_cat.values), 98))
    safe_name = cat.replace("/", "_").replace(" ", "_")
    plot_heatmap(
        std_resid_cat,
        title=f"Voynich [{cat}] バイグラム標準化残差ヒートマップ ({len(words_cat):,} words)\n(赤=実測>期待値, 青=実測<期待値)",
        out_path=OUT_DIR / f"bigram_residual_heatmap_{safe_name}.png",
        cmap="RdBu_r",
        center=0,
        vmin=-clim_cat,
        vmax=clim_cat,
    )

    report_sections.append(
        generate_report(count_cat, std_resid_cat, pmi_cat, label=f"category={cat}")
    )

# ── テキストレポート保存 ────────────────────────────────────────────────────
report_path = OUT_DIR / "bigram_residual_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(report_sections))
print(f"\nReport saved: {report_path}")

print("\n✓ 完了。出力先:", OUT_DIR.resolve())
