#!/usr/bin/env python3
"""
Voynich Manuscript - Slot Grammar Analyzer v4

v4文法の定義:
  v3 (v2スロット文法×最大2回繰り返し) に加え、
  Slot09 に 'c' を追加。

変更理由:
  - v3未マッチ343語のうち158語で残余が'c'始まり
  - 'c'の直後: s(41), k(29), t(29), h(24), o(9), p(7), f(6), y(6) など
  - 'c'単体がどのSlotにも存在しないことが根本原因
  - Slot00への追加は 'cth','ckh','cph','cfh'(Slot06) を壊す (92語破損)
  - Slot07は1語破損、Slot08は13語破損
  - Slot09 (既存: {s,d}) への追加は破損ゼロ・115語改善で最も安全
  - 'c'をSlot09末尾 ({s,d,c}) に置くことで s,d より低優先度を維持

v4文法:
  Slot 9: {s, d, c}   ← c追加 (それ以外はv2と同一)
  繰り返し上限: 2回 (v3と同じ)
"""

from pathlib import Path
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Slot grammar v4 (v2 + Slot09に'c'追加)
# ---------------------------------------------------------------------------
SLOTS_V4 = [
    ["l", "r", "o", "y", "s"],                          # Slot 0
    ["q", "s", "d", "x", "l", "r", "h"],               # Slot 1  (h追加@v2)
    ["o", "y"],                                           # Slot 2
    ["d", "r"],                                           # Slot 3
    ["t", "k", "p", "f"],                                # Slot 4
    ["ch", "sh"],                                         # Slot 5
    ["cth", "ckh", "cph", "cfh"],                        # Slot 6
    ["eee", "ee", "e", "g"],                             # Slot 7
    ["k", "t", "p", "f", "ch", "sh", "l", "r", "o", "y"],  # Slot 8
    ["s", "d", "c"],                                      # Slot 9  ★ c追加
    ["o", "a", "y"],                                      # Slot 10
    ["iii", "ii", "i"],                                   # Slot 11
    ["d", "l", "r", "m", "n"],                           # Slot 12
    ["s"],                                                # Slot 13
    ["y"],                                                # Slot 14
    ["k", "t", "p", "f", "l", "r", "o", "y"],           # Slot 15
]


def parse_greedy(word: str, slots) -> tuple[list, str]:
    pos = 0
    matched = []
    for idx, options in enumerate(slots):
        if pos >= len(word):
            break
        for opt in options:
            if word.startswith(opt, pos):
                matched.append((idx, opt))
                pos += len(opt)
                break
    return matched, word[pos:]


def is_v4_base(word: str) -> bool:
    """単語がv4スロット文法1回でマッチするか"""
    m, r = parse_greedy(word, SLOTS_V4)
    return r == "" and bool(m)


def is_v4_match(word: str) -> tuple[bool, list]:
    """
    v4マッチ判定 (最大2回繰り返し)
    Returns (matched, parts)
    """
    if is_v4_base(word):
        m, _ = parse_greedy(word, SLOTS_V4)
        return True, [(word, m)]
    for i in range(1, len(word)):
        p1, p2 = word[:i], word[i:]
        m1, r1 = parse_greedy(p1, SLOTS_V4)
        if r1 != "" or not m1:
            continue
        m2, r2 = parse_greedy(p2, SLOTS_V4)
        if r2 == "" and m2:
            return True, [(p1, m1), (p2, m2)]
    return False, []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    base = Path(__file__).parent
    words = [w.strip() for w in (base / "unique_word.txt").read_text(encoding="utf-8").splitlines() if w.strip()]

    matched_v4 = []
    unmatched_v4 = []
    match_info: dict[str, list] = {}

    for word in words:
        ok, parts = is_v4_match(word)
        if ok:
            matched_v4.append(word)
            match_info[word] = parts
        else:
            unmatched_v4.append(word)

    print("=" * 60)
    print("Slot Grammar v4 結果")
    print("=" * 60)
    print(f"総単語数     : {len(words)}")
    print(f"v4 マッチ    : {len(matched_v4)} ({len(matched_v4)/len(words)*100:.1f}%)")
    print(f"v4 未マッチ  : {len(unmatched_v4)} ({len(unmatched_v4)/len(words)*100:.1f}%)")

    # --- 新規マッチサンプル ---
    v4_base_slots = [list(s) for s in SLOTS_V4]
    v4_base_slots[9] = ["s", "d"]  # v3相当 (c除去)

    def is_v3(w):
        def iv2(x): m,r=parse_greedy(x,v4_base_slots); return r=="" and bool(m)
        if iv2(w): return True
        for i in range(1,len(w)):
            if iv2(w[:i]) and iv2(w[i:]): return True
        return False

    newly_matched = [w for w in matched_v4 if not is_v3(w)]
    print(f"\n新規マッチ (v3→v4): {len(newly_matched)}")
    print("\n--- 新規マッチ サンプル (20語) ---")
    for w in newly_matched[:20]:
        parts = match_info[w]
        if len(parts) == 1:
            slot_str = " ".join(f"[{i}:{s}]" for i, s in parts[0][1])
            print(f"  {w:22s} (1語) {slot_str}")
        else:
            p1, m1 = parts[0]; p2, m2 = parts[1]
            s1 = "|".join(s for _, s in m1); s2 = "|".join(s for _, s in m2)
            print(f"  {w:22s} [{p1}]({s1}) + [{p2}]({s2})")

    # --- 残余分析 ---
    rem_head: Counter = Counter()
    rem_full: Counter = Counter()
    for w in unmatched_v4:
        m, r = parse_greedy(w, SLOTS_V4)
        best_rem = r if r else w
        for i in range(1, len(w)):
            p1, p2 = w[:i], w[i:]
            m1, r1 = parse_greedy(p1, SLOTS_V4)
            if r1 != "" or not m1: continue
            _, r2 = parse_greedy(p2, SLOTS_V4)
            if len(r2) < len(best_rem): best_rem = r2
        if best_rem:
            rem_head[best_rem[0]] += 1
            rem_full[best_rem[:5]] += 1

    print(f"\n--- v4未マッチ: 残余先頭文字 ---")
    for ch, cnt in rem_head.most_common(12):
        print(f"  {ch!r}: {cnt}")

    print(f"\n--- v4未マッチ: 残余文字列 top20 ---")
    for rem, cnt in rem_full.most_common(20):
        print(f"  {rem!r}: {cnt}")

    # --- Slot定義サマリ ---
    print("\n" + "=" * 60)
    print("改良版 Slot Grammar v4")
    print("=" * 60)
    SLOTS_V2 = [
        ["l","r","o","y","s"], ["q","s","d","x","l","r","h"], ["o","y"],
        ["d","r"], ["t","k","p","f"], ["ch","sh"], ["cth","ckh","cph","cfh"],
        ["eee","ee","e","g"], ["k","t","p","f","ch","sh","l","r","o","y"],
        ["s","d"], ["o","a","y"], ["iii","ii","i"], ["d","l","r","m","n"],
        ["s"], ["y"], ["k","t","p","f","l","r","o","y"],
    ]
    for i, (v2, v4) in enumerate(zip(SLOTS_V2, SLOTS_V4)):
        diff = " ★変更" if v2 != v4 else ""
        added = sorted(set(v4) - set(v2))
        add_str = f"  (追加: {added})" if added else ""
        print(f"  Slot {i:2d}: {{{', '.join(v4)}}}{diff}{add_str}")
    print("  繰り返し上限: 2回")

    # --- ファイル出力 ---
    out_path = base / "unmatched_words_v4.txt"
    lines = []
    for w in unmatched_v4:
        m, r = parse_greedy(w, SLOTS_V4)
        slot_str = " ".join(f"[{i}:{s}]" for i, s in m) if m else "(none)"
        lines.append(f"{w}\t{slot_str}\tremaining={r!r}")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[出力] v4 未マッチ単語 ({len(unmatched_v4)}語) -> {out_path}")


if __name__ == "__main__":
    main()
