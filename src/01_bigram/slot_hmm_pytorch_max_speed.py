"""
slot_hmm_pytorch_max_speed.py
============================
Voynich Manuscript スロット文法仮説の数理検証
【PyTorch GPU 極限高速化版: Word Batching】

- 全単語(8,059語)をパディングして行列(バッチ)化し、GPUのコアを数万個規模で同時稼働させます。
- Pythonのループ回数を「全文字数(67,546)」から「最大単語長(約15)」へ劇的に削減。
- N_RESTARTS(20試行)も同時に並列処理。
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
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] ERROR: PyTorchがインストールされていません。")
    import sys
    sys.exit(1)

def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"[PyTorch] Using device: {DEVICE}")

def _setup_jp_font():
    candidates = ["Yu Gothic", "Meiryo", "MS Gothic", "IPAexGothic", "Noto Sans CJK JP"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

_setup_jp_font()

# ── 設定 ──────────────────────────────────────────────────────────────
DB_PATH      = "data/voynich.db"
OUT_DIR      = Path("hypothesis/01_bigram/results/hmm_pt_max")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_RANGE      = [2, 3, 4, 5]
N_RESTARTS   = 20
N_ITER       = 200
LTR_ITER     = 50
TOL          = 1e-4
MIN_WORD_LEN = 2

# ── データ準備 ────────────────────────────────────────────────────────
BOS_CHAR, EOS_CHAR, PAD_CHAR = "^", "$", "_"

def prepare_batched_data(words, char2idx):
    """単語リストをパディングして (N_words, Max_Len) のテンソルにする"""
    processed = []
    for w in words:
        # BOS + word + EOS
        processed.append([char2idx[BOS_CHAR]] + [char2idx[c] for c in w] + [char2idx[EOS_CHAR]])
    
    max_len = max(len(p) for p in processed)
    X_matrix = []
    mask = []
    for p in processed:
        pad_len = max_len - len(p)
        X_matrix.append(p + [char2idx[PAD_CHAR]] * pad_len)
        # 実際の文字がある場所以外をマスクする (1.0: 有効, 0.0: 無効)
        mask.append([1.0] * len(p) + [0.0] * pad_len)
    
    return np.array(X_matrix, dtype=np.int16), np.array(mask, dtype=np.float32), max_len

# ── PyTorchによる Word-Batched HMM 実装 ─────────────────────────────
class MaxSpeedHMM_PT:
    def __init__(self, n_restarts, n_components, n_vocab, device=DEVICE):
        self.b = n_restarts # 試行バッチ (20)
        self.k = n_components
        self.v = n_vocab
        self.device = device
        
        # log_startprob: (B, K)
        # log_transmat:  (B, K, K)
        # log_emission:  (B, K, V)
        self.log_startprob = None
        self.log_transmat  = None
        self.log_emission  = None

    def _init_params(self, seed=42):
        torch.manual_seed(seed)
        s = torch.rand(self.b, self.k, device=self.device) + 0.5
        self.log_startprob = torch.log(s / s.sum(dim=1, keepdim=True))
        
        t = torch.rand(self.b, self.k, self.k, device=self.device) + 0.5
        self.log_transmat = torch.log(t / t.sum(dim=2, keepdim=True))
        
        e = torch.rand(self.b, self.k, self.v, device=self.device) + 0.5
        self.log_emission = torch.log(e / e.sum(dim=2, keepdim=True))

    def fit(self, X_pt, mask_pt, n_iter=N_ITER, tol=TOL, left_to_right=False):
        """
        X_pt: (N, T)  - N:単語数, T:最大長
        mask_pt: (N, T)
        """
        N, T = X_pt.shape
        B = self.b
        K = self.k
        
        best_logL = torch.full((B,), -float('inf'), device=self.device)

        for it in range(n_iter):
            # 1. Forward Pass (Log-domain)
            # log_alpha: (T, N, B, K) -> 大文字 T (時間軸) は Python ループ
            # ここが高速化の肝: N(8000) と B(20) を次元に組み込む
            log_alpha = torch.empty(T, N, B, K, device=self.device)
            
            # t=0: 全単語の開始
            # self.log_emission[:, :, X_pt[:, 0]] -> (B, K, N)
            # 転置が必要: (N, B, K)
            emiss_0 = self.log_emission[:, :, X_pt[:, 0]].permute(2, 0, 1)
            log_alpha[0] = self.log_startprob.unsqueeze(0) + emiss_0
            
            for t in range(1, T):
                # (N, B, K, 1) + (1, B, K, K) -> (N, B, K, K)
                prev = log_alpha[t-1].unsqueeze(3) + self.log_transmat.unsqueeze(0)
                # dim=2 は遷移元 i
                sum_prev = torch.logsumexp(prev, dim=2) # (N, B, K)
                
                emiss_t = self.log_emission[:, :, X_pt[:, t]].permute(2, 0, 1)
                log_alpha[t] = sum_prev + emiss_t
                
                # パディング（有効でない文字）については前の状態を維持するかマスクする
                # 本実装では gamma/xi 計算時に文字マスクを適用するため、alphaはそのまま進める

            # 2. Backward Pass (T, N, B, K)
            log_beta = torch.empty(T, N, B, K, device=self.device)
            log_beta[T-1] = 0.0
            
            for t in range(T-2, -1, -1):
                # (1, B, K, K) + (N, B, 1, K) + (N, B, 1, K) -> (N, B, K, K)
                # emiss(t+1) + beta(t+1)
                emiss_tp1 = self.log_emission[:, :, X_pt[:, t+1]].permute(2, 0, 1).unsqueeze(2)
                nxt = self.log_transmat.unsqueeze(0) + emiss_tp1 + log_beta[t+1].unsqueeze(2)
                # dim=3 は遷移先 j
                log_beta[t] = torch.logsumexp(nxt, dim=3)

            # 3. 尤度と Gamma / Xi の計算
            # 各単語の有効な最後 (mask の最後の 1.0) の時点の alpha * beta
            # 系列全体の有効な logL は、各系列の有効な終端の logsumexp
            # マスクを使って有効な終端の alpha を抽出
            # mask: (N, T) -> 最後の 1.0 のインデックスを取得
            last_idx = (mask_pt.sum(dim=1) - 1).long() # (N,)
            
            # gather を使って系列ごとの終了 alpha を取得
            # alpha: (T, N, B, K) -> (N, T, B, K)
            alpha_swapped = log_alpha.permute(1, 0, 2, 3)
            # 終了 alpha: (N, 1, B, K) -> squeeze -> (N, B, K)
            end_alpha = torch.gather(alpha_swapped, 1, last_idx.view(-1, 1, 1, 1).expand(-1, 1, B, K)).squeeze(1)
            
            logL_per_word = torch.logsumexp(end_alpha, dim=2) # (N, B)
            total_logL = logL_per_word.sum(dim=0) # (B,)

            if it > 0 and (total_logL - best_logL).abs().max() < tol:
                break
            best_logL = total_logL.clone()

            # --- M-step ---
            log_gamma = log_alpha + log_beta # (T, N, B, K)
            # 正規化 (T, N, B, 1)
            norm_factor = torch.logsumexp(log_gamma, dim=3, keepdim=True)
            log_gamma_norm = log_gamma - norm_factor
            # 有効な文字のみ集計するためマスク適用
            log_gamma_norm = log_gamma_norm + torch.log(mask_pt.permute(1, 0).unsqueeze(2).unsqueeze(3))

            # log_xi: (T-1, N, B, K, K)
            # alpha(t) + A + emiss(t+1) + beta(t+1)
            # t+1 の emission
            emiss_tp1_all = self.log_emission[:, :, X_pt[:, 1:]].permute(3, 2, 0, 1).unsqueeze(3) # (T-1, N, B, 1, K)
            log_xi = log_alpha[:-1].unsqueeze(4) + \
                     self.log_transmat.view(1, 1, B, K, K) + \
                     emiss_tp1_all + \
                     log_beta[1:].unsqueeze(3)
            
            # 境界マスク (T-1, N) 0.0の箇所(=パディング開始以降または系列末)を -inf に
            xi_mask = mask_pt[:, :-1] * mask_pt[:, 1:] # (N, T-1)
            log_xi = log_xi + torch.log(xi_mask.permute(1, 0).view(T-1, N, 1, 1, 1))
            # 正規化 (T-1, N, B, K, K) - (T-1, N, B, 1, 1)
            log_xi = log_xi - norm_factor[:-1].unsqueeze(4)

            # 更新 1: Start Prob (B, K) - 各系列の t=0
            new_log_start = torch.logsumexp(log_gamma_norm[0], dim=0) # sum over N
            self.log_startprob = new_log_start - torch.logsumexp(new_log_start, dim=1, keepdim=True)

            # 更新 2: Transmat (B, K, K)
            new_log_trans = torch.logsumexp(log_xi, dim=(0, 1)) # sum over T-1, N
            if left_to_right:
                m = torch.triu(torch.ones(K, K, device=self.device)).bool()
                new_log_trans[:, ~m] = -float('inf')
            self.log_transmat = new_log_trans - torch.logsumexp(new_log_trans, dim=2, keepdim=True)

            # 更新 3: Emission (B, K, V)
            # exp_gamma: (T, N, B, K) -> (B, K, T*N)
            gamma_flat = torch.exp(log_gamma_norm).permute(2, 3, 0, 1).reshape(B, K, -1)
            # X_onehot: (T*N, V) - 共通
            X_flat = X_pt.permute(1, 0).reshape(-1)
            X_onehot = torch.zeros(T*N, self.v, device=self.device)
            X_onehot.scatter_(1, X_flat.view(-1, 1), 1.0)
            
            new_emiss = torch.matmul(gamma_flat, X_onehot) # (B, K, V)
            new_log_emiss = torch.log(new_emiss + 1e-35)
            self.log_emission = new_log_emiss - torch.logsumexp(new_log_emiss, dim=2, keepdim=True)

        return best_logL

    def get_best(self, logL_pt):
        idx = torch.argmax(logL_pt).item()
        return {
            "start": torch.exp(self.log_startprob[idx]).cpu().numpy(),
            "trans": torch.exp(self.log_transmat[idx]).cpu().numpy(),
            "emiss": torch.exp(self.log_emission[idx]).cpu().numpy(),
            "logL": logL_pt[idx].item(),
            "best_log_start": self.log_startprob[idx],
            "best_log_trans": self.log_transmat[idx],
            "best_log_emiss": self.log_emission[idx]
        }

# ── 以降、描画・ユーティリティ (slot_hmm_pytorch.py と同様) ───────────
def _state_label(i, k):
    labels = {2: ["Pref/Core", "Suff"], 3: ["Pref", "Core", "Suff"], 
              4: ["Pref", "C1", "C2", "Suff"], 5: ["Pref", "C1", "C2", "C3", "Suff"]}
    return labels.get(k, [f"S{j}" for j in range(k)])[i]

def plot_res(info, k, topo, all_chars, out_dir):
    slbls = [_state_label(i, k) for i in range(k)]
    clbls = [("BOS" if c == "^" else ("EOS" if c == "$" else ("PAD" if c == "_" else c))) for c in all_chars]
    
    # Trans
    A = pd.DataFrame(info["trans"], index=slbls, columns=slbls)
    plt.figure(figsize=(5, 4)); sns.heatmap(A, annot=True, fmt=".3f", cmap="Blues"); plt.title(f"Trans {topo} k={k}")
    plt.savefig(out_dir / f"transition_{topo}_k{k}.png"); plt.close()
    
    # Emiss
    B = pd.DataFrame(info["emiss"], index=slbls, columns=clbls)
    plt.figure(figsize=(15, 4)); sns.heatmap(B, annot=True, fmt=".3f", cmap="YlOrRd"); plt.title(f"Emiss {topo} k={k}")
    plt.savefig(out_dir / f"emission_{topo}_k{k}.png"); plt.close()

if __name__ == "__main__":
    log("Starting MAX SPEED GPU HMM (Word Batching)...")
    
    conn = sqlite3.connect(DB_PATH)
    words = pd.read_sql_query("SELECT word FROM words_enriched WHERE word IS NOT NULL AND word != ''", conn)["word"].tolist()
    conn.close()
    all_types = sorted(set(words))
    all_types = [w for w in all_types if len(w) >= MIN_WORD_LEN]
    log(f"Words: {len(all_types)}")

    raw_chars = sorted(set(c for w in all_types for c in w))
    all_chars = [BOS_CHAR, EOS_CHAR, PAD_CHAR] + raw_chars
    char2idx = {c: i for i, c in enumerate(all_chars)}
    V = len(all_chars)

    X_np, mask_np, MaxLen = prepare_batched_data(all_types, char2idx)
    log(f"Max Word Len (T): {MaxLen} (Python loops will be this count)")
    X_pt = torch.tensor(X_np, dtype=torch.long, device=DEVICE)
    mask_pt = torch.tensor(mask_np, device=DEVICE)

    results = []
    best_k3_full = None

    for k in K_RANGE:
        log(f"{'='*50}\n  k = {k}\n{'='*50}")
        
        # Full
        log(f"  [Full] Training {N_RESTARTS} models in parallel...")
        model = MaxSpeedHMM_PT(N_RESTARTS, k, V, DEVICE)
        model._init_params(seed=42)
        ll = model.fit(X_pt, mask_pt, left_to_right=False)
        info = model.get_best(ll)
        log(f"  Best LogL: {info['logL']:.2f}")
        plot_res(info, k, "full", all_chars, OUT_DIR)
        if k == 3: best_k3_full = info

        # LTR
        log(f"  [LTR ] Training {N_RESTARTS} models in parallel...")
        model_ltr = MaxSpeedHMM_PT(N_RESTARTS, k, V, DEVICE)
        model_ltr._init_params(seed=77)
        ll_l = model_ltr.fit(X_pt, mask_pt, n_iter=LTR_ITER, left_to_right=True)
        info_l = model_ltr.get_best(ll_l)
        log(f"  Best LogL: {info_l['logL']:.2f}")
        plot_res(info_l, k, "ltr", all_chars, OUT_DIR)

        bic_f, _ = compute_bic(info["logL"], X_np.size, k, V)
        bic_l, _ = compute_bic(info_l["logL"], X_np.size, k, V)
        results.append({"k": k, "bic_f": bic_f, "bic_l": bic_l})

    # Summary
    log("Finalizing...")
    with open(OUT_DIR / "hmm_report.txt", "w", encoding="utf-8") as f:
        f.write("HMM Report (Max Speed GPU)\n\n")
        for r in results:
            f.write(f"k={r['k']}: BIC_Full={r['bic_f']:.1f}, BIC_LTR={r['bic_l']:.1f}\n")
    
    log(f"Done. Results in {OUT_DIR.resolve()}")
