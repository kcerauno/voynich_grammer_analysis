#!/usr/bin/env python3
"""
voynich_model_v2.py — 修正版 Voynich バイグラム言語モデル
==========================================================

修正点（元コードからの変更）
  1. 3分割評価: train(60%) / val(20%) / test(20%)
     - θ は val セットで最適化
     - 最終評価は test セットのみで実施（θ選択に未使用）
  2. Layer貢献分離
     - Layer1のみ の F1 を独立計測
     - Layer1+2（バイグラム）の F1 を独立計測
  3. 最難負例「ランダム(同文字種)」をメイン評価の軸に据える
"""

import math, random, argparse
from collections import Counter, defaultdict
from pathlib import Path

# ─── コーパス定義 ─────────────────────────────────────────────

VOYNICH_PATH = Path('/mnt/user-data/uploads/unique_word.txt')

LATIN_RAW = """
et in est non ad sed ut ab ex per de cum si vel ac ne iam nec aut hoc quo eos dum nam
modo rem ubi nos tam vir res lex deus pax lux ars vis sic ius qui quae quod haec homo
vita mors amor fides gens rex terra aqua ignis aer anima corpus mente tempus locus modus
forma causa ratio natura veritas virtus gloria fama honor studium opus labor dolor gaudium
bellum bonus malus magnus parvus altus latus longus brevis novus vetus primus secundus
tertius quartus quintus sextus septimus octavus amare habere esse posse velle nolle ire
ferre dare stare amo amas amat amamus amatis amant eram erat eramus erant fui video audio
dico facio vivo possum volo nolo malo eo fero do sto carpe diem fati fugit mea culpa
pater mater frater soror filius filia femina mulier caelum mare fons silva campus mons
flumen urbs villa omnis semper numquam nunc tum tunc saepe propter contra inter intra
extra supra infra ante post sub super pro sine trans circum ursa aquila corvus piscis
equus leo lupus vulpes canis felis mus cervus avis serpens puer puella miles nauta
agricola poeta philosophus senator consul liber libri litterae epistula verbum nomen mens
animus manus pes caput oculus auris nasus os lingua dens brachium crus ossa cutis sanguis
cibus potio panis vinum aurum argentum ferrum lignum lapis herba arbor flos fructus semen
""".split()

ITALIAN_RAW = """
il la lo le gli un una di a da in con su per tra fra e ma o se che chi cui dove quando
come questo quello questa essere avere fare dire andare vedere sapere potere volere dovere
sono sei siamo siete ho hai ha abbiamo avete hanno io tu lui lei noi voi loro mi ti ci vi
casa porta finestra tavolo sedia letto libro carta penna acqua vino pane carne pesce
frutta verdura latte caffe padre madre fratello sorella figlio figlia amico citta paese
strada via piazza chiesa scuola ospedale grande piccolo bello brutto vecchio nuovo buono
cattivo rosso blu verde giallo bianco nero alto basso lungo corto uno due tre quattro
cinque sei sette otto nove dieci primo secondo terzo tempo modo luogo cosa parte vita
morte amore pace guerra liberta giustizia verita bellezza andare venire partire tornare
entrare uscire salire scendere mangiare bere dormire parlare leggere scrivere lavorare
studiare sole luna stelle cielo terra mare montagna fiume lago bosco foresta campo pietra
fuoco aria uomo donna bambino ragazzo ragazza giorno notte mattina sera estate inverno
primavera autunno veloce lento forte debole ricco povero felice triste stanco sano malato
""".split()

ENGLISH_RAW = """
the be to of and a in that have it for not on with he as you do at this but his by from
they we say her she or an will my one all would there their what so up out if about who
get which go me when make can like time no just him know take people into year your good
some could them see other than then now look only come its over think also back after use
how our work first well way even new want because any these give day most us great man
woman child life world hand part place case week point government money children state
city play problem fact house number night water right left back front top bottom inside
outside under over through across along around before after above below between among
without against during although because unless whether while still yet already often
""".split()

def load_words(raw):
    return list(set(
        w.lower().strip(".,;:!?'\"-()") for w in raw
        if w.strip() and all(c.isalpha() for c in w.strip(".,;:!?'\"-()"))
    ))


# ─── モデルクラス ────────────────────────────────────────────

class VoynichBigramModel:
    BOS   = '^'
    EOS   = '$'
    ALPHA = 0.01

    def __init__(self):
        self.charset: frozenset = frozenset()
        self.trans: dict        = {}
        self.bos_logprob: dict  = {}
        self._raw_trans: dict   = {}

    def fit(self, words):
        self.charset = frozenset(c for w in words for c in w)
        vocab = sorted(self.charset) + [self.EOS]
        V = len(vocab)
        raw = defaultdict(Counter)
        for w in words:
            seq = [self.BOS] + list(w) + [self.EOS]
            for a, b in zip(seq, seq[1:]):
                raw[a][b] += 1
        self._raw_trans = raw
        self.trans = {}
        for ctx, cnts in raw.items():
            total = sum(cnts.values()) + self.ALPHA * V
            self.trans[ctx] = {
                c: math.log((cnts.get(c, 0) + self.ALPHA) / total)
                for c in vocab
            }
        bos_total = sum(raw[self.BOS].values()) + self.ALPHA * len(vocab)
        self.bos_logprob = {
            c: math.log((raw[self.BOS].get(c, 0) + self.ALPHA) / bos_total)
            for c in vocab
        }
        return self

    def _unk_lp(self):
        return math.log(self.ALPHA / (1.0 + self.ALPHA * (len(self.charset) + 1)))

    # --- Layer 1 のみ ---
    def pass_layer1(self, word):
        return bool(word) and all(c in self.charset for c in word)

    # --- Layer 1+2 スコア ---
    def log_prob(self, word):
        if not word: return -math.inf
        if not self.pass_layer1(word): return -math.inf
        lp = self.bos_logprob.get(word[0], self._unk_lp())
        prev = word[0]
        for c in word[1:]:
            lp += self.trans.get(prev, {}).get(c, self._unk_lp())
            prev = c
        lp += self.trans.get(prev, {}).get(self.EOS, self._unk_lp())
        return lp

    def score(self, word):
        lp = self.log_prob(word)
        return -math.inf if lp == -math.inf else lp / len(word)

    def predict(self, word, theta):
        return self.score(word) >= theta

    def generate(self, max_len=12, rng=None):
        if rng is None: rng = random.Random()
        vocab = sorted(self.charset) + [self.EOS]
        def sample(lp_dict):
            items = list(lp_dict.items())
            probs = [math.exp(lp) for _, lp in items]
            t = sum(probs); r = rng.random() * t; acc = 0.0
            for (c, _), p in zip(items, probs):
                acc += p
                if r <= acc: return c
            return items[-1][0]
        word = []
        c = sample(self.bos_logprob)
        while c != self.EOS and len(word) < max_len:
            word.append(c)
            c = sample(self.trans.get(word[-1], {k: self._unk_lp() for k in vocab}))
        return ''.join(word) if word else self.generate(max_len, rng)

    # θ チューニング（指定セットのみ使用）
    def tune_threshold(self, pos_words, neg_words, metric='f1'):
        pos_scores = [self.score(w) for w in pos_words]
        neg_scores = [self.score(w) for w in neg_words]
        all_scores = sorted(set(s for s in pos_scores + neg_scores if s > -math.inf))
        best_theta, best_val = 0.0, -1.0
        for theta in all_scores:
            tp = sum(1 for s in pos_scores if s >= theta)
            fp = sum(1 for s in neg_scores if s >= theta)
            fn = sum(1 for s in pos_scores if s < theta)
            prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0
            rec  = tp/(tp+fn) if (tp+fn) > 0 else 0.0
            f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
            val  = f1 if metric == 'f1' else prec
            if val > best_val:
                best_val, best_theta = val, theta
        return best_theta, best_val


# ─── 評価ユーティリティ ──────────────────────────────────────

def evaluate(model, pos_words, neg_words, theta, label='', show=True):
    tp = sum(1 for w in pos_words if model.predict(w, theta))
    fp = sum(1 for w in neg_words if model.predict(w, theta))
    fn = sum(1 for w in pos_words if not model.predict(w, theta))
    tn = sum(1 for w in neg_words if not model.predict(w, theta))
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn) > 0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    acc  = (tp+tn)/(tp+fp+fn+tn)
    if show:
        print(f"  {label:<38s}  Prec={prec*100:5.1f}%  Rec={rec*100:5.1f}%"
              f"  F1={f1*100:5.1f}%  Acc={acc*100:5.1f}%"
              f"  (TP={tp} FP={fp} FN={fn} TN={tn})")
    return prec, rec, f1, acc

# Layer1のみの評価（θ不要）
def evaluate_layer1(model, pos_words, neg_words, label='', show=True):
    tp = sum(1 for w in pos_words if model.pass_layer1(w))
    fp = sum(1 for w in neg_words if model.pass_layer1(w))
    fn = sum(1 for w in pos_words if not model.pass_layer1(w))
    tn = sum(1 for w in neg_words if not model.pass_layer1(w))
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn) > 0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    acc  = (tp+tn)/(tp+fp+fn+tn)
    if show:
        print(f"  {label:<38s}  Prec={prec*100:5.1f}%  Rec={rec*100:5.1f}%"
              f"  F1={f1*100:5.1f}%  Acc={acc*100:5.1f}%"
              f"  (TP={tp} FP={fp} FN={fn} TN={tn})")
    return prec, rec, f1, acc

def random_words_from_charset(charset, n=2000, seed=42):
    rng = random.Random(seed)
    chars = sorted(charset)
    lens = [4, 5, 6, 7, 8]
    return [''.join(rng.choices(chars, k=rng.choice(lens))) for _ in range(n)]


# ─── メイン ─────────────────────────────────────────────────

def main():
    # データロード
    voynich = [w.strip().replace('\r','') for w in
               VOYNICH_PATH.read_text(encoding='utf-8').splitlines() if w.strip()]
    latin   = load_words(LATIN_RAW)
    italian = load_words(ITALIAN_RAW)
    english = load_words(ENGLISH_RAW)
    natural = latin + italian + english

    # ── 3分割: train 60% / val 20% / test 20% ──────────────
    rng = random.Random(0)
    rng.shuffle(voynich)
    n = len(voynich)
    n_train = int(n * 0.60)
    n_val   = int(n * 0.20)
    train_voy = voynich[:n_train]
    val_voy   = voynich[n_train:n_train+n_val]
    test_voy  = voynich[n_train+n_val:]

    print("=" * 72)
    print("Voynich バイグラム言語モデル v2（修正版）")
    print("=" * 72)
    print(f"  総語数   : {n}")
    print(f"  train    : {len(train_voy)} 語 (60%)  ← 学習のみ")
    print(f"  val      : {len(val_voy)}  語 (20%)  ← θ選択のみ")
    print(f"  test     : {len(test_voy)}  語 (20%)  ← 最終評価のみ")
    print(f"  負例     : {len(natural)} 語 (Latin+Italian+English)")

    # 学習（train のみ使用）
    model = VoynichBigramModel()
    model.fit(train_voy)

    # 同文字種ランダム負例（最難負例：Layer1を必ず通過）
    rand_neg = random_words_from_charset(model.charset, n=2000)

    # ── θ を val セットで決定 ──────────────────────────────
    print("\n" + "=" * 72)
    print("閾値チューニング（val セット使用 ─ test は未使用）")
    print("=" * 72)
    # 自然語 vs val
    theta_nat, f1_nat = model.tune_threshold(val_voy, natural, metric='f1')
    # ランダム同文字種 vs val（最難設定）
    theta_hard, f1_hard = model.tune_threshold(val_voy, rand_neg, metric='f1')
    print(f"  θ_natural  = {theta_nat:.4f}  (val F1={f1_nat*100:.1f}%)  ← 自然語対抗")
    print(f"  θ_hard     = {theta_hard:.4f}  (val F1={f1_hard*100:.1f}%)  ← ランダム同文字種対抗")
    print(f"  ※ θ決定にtestセットは一切関与していない")

    # ════════════════════════════════════════════════════════
    # 最終評価（test セットのみ）
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("最終評価（test セットのみ／θはvalで確定済み）")
    print("=" * 72)

    # ── A. Layer 1 単独の貢献 ──────────────────────────────
    print("\n【A】Layer 1 のみ（文字集合フィルター）")
    print("-" * 72)
    evaluate_layer1(model, test_voy, latin,    "test_voy vs ラテン語")
    evaluate_layer1(model, test_voy, italian,  "test_voy vs イタリア語")
    evaluate_layer1(model, test_voy, english,  "test_voy vs 英語")
    evaluate_layer1(model, test_voy, natural,  "test_voy vs 自然語(合算)")
    evaluate_layer1(model, test_voy, rand_neg, "test_voy vs ランダム同文字種 ★")
    print("  ↑ランダム同文字種は全語がLayer1を通過するため F1=N/A に近い")

    # ── B. Layer 1+2（バイグラム）θ_natural ────────────────
    print(f"\n【B】Layer 1+2 バイグラムモデル（θ = {theta_nat:.4f}，自然語対抗）")
    print("-" * 72)
    evaluate(model, test_voy, latin,    theta_nat, "test_voy vs ラテン語")
    evaluate(model, test_voy, italian,  theta_nat, "test_voy vs イタリア語")
    evaluate(model, test_voy, english,  theta_nat, "test_voy vs 英語")
    evaluate(model, test_voy, natural,  theta_nat, "test_voy vs 自然語(合算)")
    evaluate(model, test_voy, rand_neg, theta_nat, "test_voy vs ランダム同文字種 ★")

    # ── C. Layer 1+2 θ_hard（最難設定）────────────────────
    print(f"\n【C】Layer 1+2 バイグラムモデル（θ = {theta_hard:.4f}，最難負例対抗）")
    print("-" * 72)
    evaluate(model, test_voy, latin,    theta_hard, "test_voy vs ラテン語")
    evaluate(model, test_voy, italian,  theta_hard, "test_voy vs イタリア語")
    evaluate(model, test_voy, english,  theta_hard, "test_voy vs 英語")
    evaluate(model, test_voy, natural,  theta_hard, "test_voy vs 自然語(合算)")
    evaluate(model, test_voy, rand_neg, theta_hard, "test_voy vs ランダム同文字種 ★")

    # ── D. Layer貢献の差分サマリー ──────────────────────────
    print("\n" + "=" * 72)
    print("【D】Layer 貢献サマリー（test セット, vs ランダム同文字種）")
    print("=" * 72)
    _, _, f1_L1,   _ = evaluate_layer1(model, test_voy, rand_neg, "Layer1のみ", show=False)
    _, _, f1_L12n, _ = evaluate(model, test_voy, rand_neg, theta_nat,  "Layer1+2 θ_natural", show=False)
    _, _, f1_L12h, _ = evaluate(model, test_voy, rand_neg, theta_hard, "Layer1+2 θ_hard",    show=False)
    print(f"  Layer 1 のみ                     F1 = {f1_L1*100:5.1f}%")
    print(f"  Layer 1+2  (θ_natural={theta_nat:.4f})  F1 = {f1_L12n*100:5.1f}%")
    print(f"  Layer 1+2  (θ_hard   ={theta_hard:.4f})  F1 = {f1_L12h*100:5.1f}%")
    print(f"  バイグラムの純貢献（θ_hard基準）  +{(f1_L12h-f1_L1)*100:.1f}pt")

    # ── スコア分布 ───────────────────────────────────────────
    print("\n" + "=" * 72)
    print("スコア分布（test セット）")
    print("=" * 72)
    def dist(words, label):
        scores = [model.score(w) for w in words]
        fin = [s for s in scores if s > -math.inf]
        n_inf = len(scores) - len(fin)
        if fin:
            fin_s = sorted(fin)
            med = fin_s[len(fin_s)//2]
            print(f"  {label:<38s}  n={len(scores):4d}  -inf={n_inf:4d}"
                  f"  min={min(fin):.2f}  med={med:.2f}  max={max(fin):.2f}")
    dist(test_voy, "Voynich (test)")
    dist(latin,    "ラテン語")
    dist(italian,  "イタリア語")
    dist(english,  "英語")
    dist(rand_neg, "ランダム同文字種")

    # ── 誤検出しやすい自然語 ────────────────────────────────
    print("\n" + "=" * 72)
    print(f"自然語のうちθ_natural通過（誤検出）上位")
    print("=" * 72)
    false_pos = [w for w in natural if model.predict(w, theta_nat)]
    false_pos.sort(key=lambda w: model.score(w), reverse=True)
    for w in false_pos[:15]:
        print(f"  {w:<20s}  score={model.score(w):.3f}")
    print(f"  合計 {len(false_pos)} 語 / {len(natural)} 語")

    # ── モデルサマリー ───────────────────────────────────────
    print("\n" + "=" * 72)
    print("モデルサマリー")
    print("=" * 72)
    n_params = len(model.charset) * (len(model.charset) + 1)
    print(f"  パラメータ数  : {n_params}")
    print(f"  文字種数      : {len(model.charset)}")
    print(f"  スムージング α: {VoynichBigramModel.ALPHA}")
    print(f"  θ_natural     : {theta_nat:.4f}  (val F1={f1_nat*100:.1f}%)")
    print(f"  θ_hard        : {theta_hard:.4f}  (val F1={f1_hard*100:.1f}%)")
    print(f"  評価セット    : test {len(test_voy)} 語（θ選択に未使用）")

if __name__ == '__main__':
    main()
