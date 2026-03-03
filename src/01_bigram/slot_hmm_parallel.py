"""
slot_hmm_parallel.py
====================

高速化版:
  ・k を joblib で並列化
  ・多重スタート seed も並列化
  ・model.score() を使わず monitor_.history[-1] を使用
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
from joblib import Parallel, delayed
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

# ── hmmlearn ─────────────────────────────────────────────────────
try:
    from hmmlearn.hmm import CategoricalHMM
    HMM_CLASS = CategoricalHMM
    print("  [hmm] Using CategoricalHMM")
except ImportError:
    from hmmlearn.hmm import MultinomialHMM
    HMM_CLASS = MultinomialHMM
    print("  [hmm] Using MultinomialHMM (fallback)")

# ── 日本語フォント ───────────────────────────────────────────────
def _setup_jp_font():
    candidates = ["Yu Gothic", "Meiryo", "MS Gothic", "IPAexGothic", "Noto Sans CJK JP"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

_setup_jp_font()

# ── 設定 ─────────────────────────────────────────────────────────
DB_PATH   = "data/voynich.db"
OUT_DIR   = Path("hypothesis/01_bigram/results/hmm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_RANGE      = [2, 3, 4, 5]
N_RESTARTS   = 10
N_ITER       = 100
MIN_WORD_LEN = 2

N_JOBS = multiprocessing.cpu_count()
print(f"[parallel] Using {N_JOBS} cores")

# ── データ読み込み ──────────────────────────────────────────────
print("Loading data...")
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(
    "SELECT word FROM words_enriched WHERE word IS NOT NULL AND word != ''",
    conn
)
conn.close()

all_types = sorted(set(df["word"].tolist()))
all_types = [w for w in all_types if len(w) >= MIN_WORD_LEN]

print(f"  ユニーク単語数: {len(all_types):,}")

# ── 文字エンコード ─────────────────────────────────────────────
BOS, EOS = "^", "$"
raw_chars = sorted(set(c for w in all_types for c in w))
all_chars = [BOS] + raw_chars + [EOS]
char2idx  = {c: i for i, c in enumerate(all_chars)}
V         = len(all_chars)

def encode_words(words):
    seqs, lengths = [], []
    for w in words:
        seq = [char2idx[BOS]] + [char2idx[c] for c in w] + [char2idx[EOS]]
        seqs.extend(seq)
        lengths.append(len(seq))
    X = np.array(seqs, dtype=np.int16).reshape(-1, 1)
    X = np.ascontiguousarray(X)
    return X, lengths

X_all, L_all = encode_words(all_types)

print(f"  観測総数: {len(X_all):,}")
print(f"  語彙サイズ: {V}")

# ── seed並列学習（Full HMM） ───────────────────────────────────
def train_full_seed(seed, k, X, lengths):
    try:
        model = HMM_CLASS(
            n_components=k,
            n_iter=N_ITER,
            tol=1e-4,
            random_state=seed,
            verbose=False,
        )
        model.fit(X, lengths)
        score = model.monitor_.history[-1]   # score()削減
        return model, score
    except Exception:
        return None, -np.inf

def fit_full_hmm_parallel(k, X, lengths):
    results = Parallel(n_jobs=N_JOBS)(
        delayed(train_full_seed)(seed, k, X, lengths)
        for seed in range(N_RESTARTS)
    )
    results = [r for r in results if r[0] is not None]
    if not results:
        return None, -np.inf
    best_model, best_score = max(results, key=lambda x: x[1])
    return best_model, best_score

# ── BIC ─────────────────────────────────────────────────────────
def compute_bic_from_loglik(log_likelihood, k, N):
    n_params = k*(k-1) + k*(V-1) + (k-1)
    bic = -2*log_likelihood + n_params*np.log(N)
    aic = -2*log_likelihood + 2*n_params
    return bic, aic

# ── k並列処理 ───────────────────────────────────────────────────
def train_for_k(k):
    print(f"\n=== k={k} ===")
    model_full, loglik = fit_full_hmm_parallel(k, X_all, L_all)

    if model_full is None:
        print("  学習失敗")
        return None

    bic, aic = compute_bic_from_loglik(loglik, k, len(X_all))

    print(f"  logL={loglik:.2f}  BIC={bic:.2f}")

    return {
        "k": k,
        "model": model_full,
        "loglik": loglik,
        "bic": bic,
        "aic": aic
    }

# ── 並列実行（k並列） ──────────────────────────────────────────
print("\n[Parallel training for k...]")

results = Parallel(n_jobs=min(len(K_RANGE), N_JOBS))(
    delayed(train_for_k)(k) for k in K_RANGE
)

results = [r for r in results if r is not None]

# ── BIC サマリー ───────────────────────────────────────────────
print("\n" + "="*60)
print("BIC Summary")
print("="*60)
print(f"{'k':>4}  {'logL':>14}  {'BIC':>14}")
print("-"*40)

for r in results:
    print(f"{r['k']:>4}  {r['loglik']:>14.1f}  {r['bic']:>14.1f}")

best = min(results, key=lambda x: x["bic"])
print(f"\nBest model: k={best['k']}  BIC={best['bic']:.1f}")

print("\n✓ 完了")