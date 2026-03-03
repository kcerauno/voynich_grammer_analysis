"""
slot_hmm.py
===========
Voynich Manuscript スロット文法仮説の数理検証（HMM版）
idea_bigram.txt ④ の実装

設計方針:
  - 一意化（types）した単語を学習対象とする（語形成ルールが目的）
  - 観測: 文字（BOS='^', EOS='$' を含む）
  - 隠れ状態: Prefix / Core / Suffix ほか（k=2〜5 で比較）
  - 学習: Baum-Welch（多重スタート N_RESTARTS 回、対数尤度最大のモデルを採用）
  - トポロジー: Full（完全結合）と Left-to-Right（Bakis）の両方を試す

出力先: hypothesis/01_bigram/results/hmm/
  - bic_comparison.png        BICによるモデル選択グラフ
  - emission_full_k{k}.png    Full-HMM 放射確率ヒートマップ
  - transition_full_k{k}.png  Full-HMM 遷移確率ヒートマップ
  - emission_ltr_k{k}.png     L-to-R-HMM 放射確率ヒートマップ
  - transition_ltr_k{k}.png   L-to-R-HMM 遷移確率ヒートマップ
  - word_examples.txt         Viterbi 状態割り当て例（k=3 Full）
  - hmm_report.txt            テキストレポート
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib import font_manager
import warnings
warnings.filterwarnings("ignore")

# ── hmmlearn インポート（バージョン吸収） ──────────────────────────────────────
try:
    from hmmlearn.hmm import CategoricalHMM
    HMM_CLASS = CategoricalHMM
    print("  [hmm] Using CategoricalHMM")
except ImportError:
    try:
        from hmmlearn.hmm import MultinomialHMM
        HMM_CLASS = MultinomialHMM
        print("  [hmm] Using MultinomialHMM (fallback)")
    except ImportError:
        raise ImportError("hmmlearn が見つかりません。pip install hmmlearn を実行してください。")

# ── 日本語フォント設定 ─────────────────────────────────────────────────────────
def _setup_jp_font():
    candidates = ["Yu Gothic", "Meiryo", "MS Gothic", "IPAexGothic", "Noto Sans CJK JP"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            print(f"  [font] Using '{name}'")
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

_setup_jp_font()

# ── 設定 ──────────────────────────────────────────────────────────────────────
DB_PATH   = "data/voynich.db"
OUT_DIR   = Path("hypothesis/01_bigram/results/hmm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_RANGE      = [2, 3, 4, 5]      # 試す状態数
N_RESTARTS   = 10                 # Baum-Welch 多重スタート回数
N_ITER       = 100                # 1回の Baum-Welch の最大反復数
LTR_ITER     = 20                 # Left-to-Right カスタムループ回数
MIN_WORD_LEN = 2                  # 最低単語長（1文字語は3状態割り当て不能）

# ── データ読み込み・一意化 ────────────────────────────────────────────────────
print("Loading data from DB...")
conn = sqlite3.connect(DB_PATH)
df   = pd.read_sql_query(
    "SELECT word FROM words_enriched WHERE word IS NOT NULL AND word != ''", conn
)
conn.close()

all_types = sorted(set(df["word"].tolist()))
all_types = [w for w in all_types if len(w) >= MIN_WORD_LEN]
print(f"  ユニーク単語数（types, len>={MIN_WORD_LEN}）: {len(all_types):,}")

# ── 文字エンコーディング ──────────────────────────────────────────────────────
BOS, EOS = "^", "$"
raw_chars = sorted(set(c for w in all_types for c in w))
all_chars = [BOS] + raw_chars + [EOS]
char2idx  = {c: i for i, c in enumerate(all_chars)}
idx2char  = {i: c for c, i in char2idx.items()}
V         = len(all_chars)        # 語彙サイズ

def encode_words(words):
    """単語リスト → (X, lengths) の形式に変換"""
    seqs, lengths = [], []
    for w in words:
        seq = [char2idx[BOS]] + [char2idx[c] for c in w] + [char2idx[EOS]]
        seqs.extend(seq)
        lengths.append(len(seq))
    return np.array(seqs, dtype=int).reshape(-1, 1), lengths

X_all, L_all = encode_words(all_types)
print(f"  文字語彙サイズ: {V}（BOS/EOS含む）")
print(f"  総観測シンボル数: {len(X_all):,}")

# ── HMM 学習関数 ──────────────────────────────────────────────────────────────
def fit_full_hmm(k, X, lengths, n_restarts=N_RESTARTS, n_iter=N_ITER):
    """Full（完全結合）HMM を多重スタートで学習 → 最良モデルを返す"""
    best_model, best_score = None, -np.inf
    for seed in range(n_restarts):
        try:
            model = HMM_CLASS(
                n_components=k,
                n_iter=n_iter,
                tol=1e-4,
                random_state=seed,
                verbose=False,
            )
            model.fit(X, lengths)
            score = model.score(X, lengths)
            if score > best_score:
                best_score, best_model = score, model
        except Exception:
            continue
    return best_model, best_score

def fit_ltr_hmm(k, X, lengths, n_restarts=N_RESTARTS, ltr_iter=LTR_ITER):
    """
    Left-to-Right（Bakis）HMM をカスタムループで学習。
    各 Baum-Welch 反復後に下三角（後退遷移）を 0 に強制し正規化する。
    """
    best_model, best_score = None, -np.inf
    for seed in range(n_restarts):
        try:
            model = HMM_CLASS(
                n_components=k,
                n_iter=1,           # 1ステップずつ手動ループ
                tol=1e-10,
                random_state=seed,
                verbose=False,
            )
            model.fit(X, lengths)

            # 初期遷移行列を上三角に制約
            triu_mask = np.triu(np.ones((k, k)))
            model.transmat_ = model.transmat_ * triu_mask
            row_sums = model.transmat_.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            model.transmat_ /= row_sums

            # カスタム反復ループ
            for _ in range(ltr_iter):
                model.fit(X, lengths)
                # 下三角を強制 0
                model.transmat_ = model.transmat_ * triu_mask
                row_sums = model.transmat_.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                model.transmat_ /= row_sums

            score = model.score(X, lengths)
            if score > best_score:
                best_score, best_model = score, model
        except Exception:
            continue
    return best_model, best_score

# ── BIC 計算 ──────────────────────────────────────────────────────────────────
def compute_bic(model, X, lengths, k):
    """BIC = -2 * logL + n_params * ln(N)"""
    log_likelihood = model.score(X, lengths)
    N = len(X)
    # パラメータ数: 遷移 k*(k-1) + 放射 k*(V-1) + 初期 (k-1)
    n_params = k * (k - 1) + k * (V - 1) + (k - 1)
    bic = -2 * log_likelihood * N + n_params * np.log(N)
    aic = -2 * log_likelihood * N + 2 * n_params
    return bic, aic, log_likelihood

# ── ヒートマップ描画ユーティリティ ────────────────────────────────────────────
def _state_label(i, k):
    labels = {
        2: ["Prefix/Core", "Suffix"],
        3: ["Prefix", "Core", "Suffix"],
        4: ["Prefix", "Core-1", "Core-2", "Suffix"],
        5: ["Prefix", "Core-1", "Core-2", "Core-3", "Suffix"],
    }
    return labels.get(k, [f"S{j}" for j in range(k)])[i]

def plot_transition(model, k, topology, out_path):
    state_labels = [_state_label(i, k) for i in range(k)]
    A = pd.DataFrame(model.transmat_, index=state_labels, columns=state_labels)

    fig, ax = plt.subplots(figsize=(max(5, k * 1.5), max(4, k * 1.3)))
    sns.heatmap(
        A, annot=True, fmt=".3f", cmap="Blues",
        vmin=0, vmax=1, ax=ax,
        linewidths=0.5, linecolor="gray",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"遷移確率行列（{topology}, k={k}）\n※左→右構造なら上三角が高い", fontsize=13)
    ax.set_xlabel("遷移先状態", fontsize=10)
    ax.set_ylabel("遷移元状態", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

def plot_emission(model, k, topology, out_path):
    state_labels = [_state_label(i, k) for i in range(k)]
    char_labels  = [("BOS" if c == "^" else ("EOS" if c == "$" else c)) for c in all_chars]
    B = pd.DataFrame(model.emissionprob_, index=state_labels, columns=char_labels)

    fig, ax = plt.subplots(figsize=(max(14, V * 0.55), max(4, k * 1.5)))
    sns.heatmap(
        B, annot=True, fmt=".3f", cmap="YlOrRd",
        vmin=0, ax=ax,
        linewidths=0.3, linecolor="gray",
        cbar_kws={"shrink": 0.6},
    )
    ax.set_title(
        f"放射確率行列（{topology}, k={k}）\n各状態でどの文字が放射されやすいか",
        fontsize=13
    )
    ax.set_xlabel("観測文字", fontsize=10)
    ax.set_ylabel("隠れ状態", fontsize=10)
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

def plot_bic(results, out_path):
    ks = [r["k"] for r in results]
    bics_full = [r["bic_full"] for r in results]
    bics_ltr  = [r["bic_ltr"]  for r in results]

    x = np.arange(len(ks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, bics_full, width, label="Full HMM", color="steelblue", alpha=0.85)
    ax.bar(x + width/2, bics_ltr,  width, label="Left-to-Right HMM", color="coral", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_xlabel("状態数 k")
    ax.set_ylabel("BIC（小さいほど良い）")
    ax.set_title("HMM モデル選択: BIC 比較\n（白抜き矢印 ▼ が最小 BIC = 最適 k）")
    ax.legend()

    # 最小点に矢印
    min_full = min(zip(bics_full, ks), key=lambda t: t[0])
    min_ltr  = min(zip(bics_ltr, ks),  key=lambda t: t[0])
    ax.annotate("▼", xy=(min_full[1] - K_RANGE[0] - width/2, min_full[0]),
                fontsize=14, ha="center", va="bottom", color="steelblue")
    ax.annotate("▼", xy=(min_ltr[1] - K_RANGE[0] + width/2, min_ltr[0]),
                fontsize=14, ha="center", va="bottom", color="coral")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

# ── Viterbi デコード例 ────────────────────────────────────────────────────────
def decode_examples(model, k, words, n=40):
    state_labels = [_state_label(i, k) for i in range(k)]
    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"  Viterbi 状態割り当て例（k={k}, Full HMM）")
    lines.append(f"{'='*70}")

    # 頻度上位語・短語・長語をまとめて取得
    sample = sorted(words, key=len)
    sample_words = sample[:10] + sample[len(sample)//2-5:len(sample)//2+5] + sample[-10:]
    seen = set()

    for w in sample_words:
        if w in seen:
            continue
        seen.add(w)
        seq = [char2idx[BOS]] + [char2idx[c] for c in w] + [char2idx[EOS]]
        X_w = np.array(seq, dtype=int).reshape(-1, 1)
        try:
            _, state_seq = model.viterbi(X_w)
            # BOS/EOS の状態を除いた本体部分
            full_chars  = [BOS] + list(w) + [EOS]
            full_labels = [state_labels[s] for s in state_seq]
            char_disp  = " ".join(f"{c:>3}" for c in full_chars)
            state_disp = " ".join(f"{l[:3]:>3}" for l in full_labels)
            lines.append(f"  {w:<15}  chars: {char_disp}")
            lines.append(f"  {'':15}  state: {state_disp}")
            lines.append("")
        except Exception:
            continue

    return "\n".join(lines)

# ════════════════════════════════════════════════════════════════════════════
# メイン処理
# ════════════════════════════════════════════════════════════════════════════
report_lines = []
bic_results  = []
best_k3_full = None   # k=3 Full モデルを後で Viterbi 用に保持

for k in K_RANGE:
    print(f"\n{'─'*60}")
    print(f"  k = {k}")
    print(f"{'─'*60}")

    # ── Full HMM ─────────────────────────────────────────────────────────
    print(f"  [Full ] 学習中（{N_RESTARTS} スタート）...")
    model_full, _ = fit_full_hmm(k, X_all, L_all)

    if model_full is None:
        print(f"  [Full ] k={k}: 学習失敗")
        continue

    bic_full, aic_full, ll_full = compute_bic(model_full, X_all, L_all, k)
    plot_transition(model_full, k, "Full", OUT_DIR / f"transition_full_k{k}.png")
    plot_emission  (model_full, k, "Full", OUT_DIR / f"emission_full_k{k}.png")

    if k == 3:
        best_k3_full = model_full

    # ── Left-to-Right HMM ────────────────────────────────────────────────
    print(f"  [L-to-R] 学習中（{N_RESTARTS} スタート × {LTR_ITER} ループ）...")
    model_ltr, _ = fit_ltr_hmm(k, X_all, L_all)

    if model_ltr is None:
        print(f"  [L-to-R] k={k}: 学習失敗")
        bic_ltr, aic_ltr, ll_ltr = np.nan, np.nan, np.nan
    else:
        bic_ltr, aic_ltr, ll_ltr = compute_bic(model_ltr, X_all, L_all, k)
        plot_transition(model_ltr, k, "L-to-R", OUT_DIR / f"transition_ltr_k{k}.png")
        plot_emission  (model_ltr, k, "L-to-R", OUT_DIR / f"emission_ltr_k{k}.png")

    bic_results.append({
        "k": k,
        "bic_full": bic_full, "aic_full": aic_full, "ll_full": ll_full,
        "bic_ltr":  bic_ltr,  "aic_ltr":  aic_ltr,  "ll_ltr":  ll_ltr,
    })

    # ── レポートセクション ─────────────────────────────────────────────────
    section = [
        f"{'='*70}",
        f"  k = {k}",
        f"{'='*70}",
        f"  [Full HMM]",
        f"    对数尤度:  {ll_full:.2f}",
        f"    BIC:       {bic_full:.2f}",
        f"    AIC:       {aic_full:.2f}",
        f"",
        f"  遷移行列（Full）:",
    ]
    state_labels = [_state_label(i, k) for i in range(k)]
    for i, row in enumerate(model_full.transmat_):
        vals = "  ".join(f"{v:.3f}" for v in row)
        section.append(f"    {state_labels[i]:<12} → [{vals}]")

    if model_ltr:
        section += [
            f"",
            f"  [Left-to-Right HMM]",
            f"    対数尤度: {ll_ltr:.2f}",
            f"    BIC:      {bic_ltr:.2f}",
            f"    AIC:      {aic_ltr:.2f}",
            f"",
            f"  遷移行列（L-to-R）:",
        ]
        for i, row in enumerate(model_ltr.transmat_):
            vals = "  ".join(f"{v:.3f}" for v in row)
            section.append(f"    {state_labels[i]:<12} → [{vals}]")

    report_lines.append("\n".join(section))
    print(f"  BIC Full={bic_full:.1f}  L-to-R={bic_ltr:.1f}")

# ── BIC 比較グラフ ────────────────────────────────────────────────────────────
plot_bic(bic_results, OUT_DIR / "bic_comparison.png")

# ── Viterbi デコード例（k=3 Full） ────────────────────────────────────────────
if best_k3_full:
    print("\n  Viterbi デコード例を生成中...")
    example_text = decode_examples(best_k3_full, k=3, words=all_types, n=40)
    ex_path = OUT_DIR / "word_examples.txt"
    with open(ex_path, "w", encoding="utf-8") as f:
        f.write(example_text)
    print(f"  Saved: {ex_path}")

# ── テキストレポート ──────────────────────────────────────────────────────────
report_path = OUT_DIR / "hmm_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(report_lines))
print(f"\nReport saved: {report_path}")

# ── BIC サマリー ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  BIC サマリー")
print("="*60)
print(f"  {'k':>4}  {'BIC(Full)':>14}  {'BIC(L-to-R)':>14}")
print("  " + "-"*40)
for r in bic_results:
    print(f"  {r['k']:>4}  {r['bic_full']:>14.1f}  {r['bic_ltr']:>14.1f}")

best_k_full = min(bic_results, key=lambda r: r["bic_full"])
print(f"\n  最適モデル（Full）: k={best_k_full['k']}  BIC={best_k_full['bic_full']:.1f}")

print("\n✓ 完了。出力先:", OUT_DIR.resolve())
