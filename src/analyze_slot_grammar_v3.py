#!/usr/bin/env python3
"""
Voynich Manuscript - Slot Grammar Analyzer v3

v3文法の定義:
  v2スロット文法を最大2回繰り返して生成された単語
  つまり word = part1 + part2 (part1, part2 はそれぞれv2でマッチ、part2は空でも可)
  ※ part1のみ = v2と同等, part1+part2 = 2語接合
"""

from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# Slot grammar v2
# ---------------------------------------------------------------------------
SLOTS_V2 = [
    ["l", "r", "o", "y", "s"],                          # Slot 0
    ["q", "s", "d", "x", "l", "r", "h"],               # Slot 1  ★ h追加
    ["o", "y"],                                           # Slot 2
    ["d", "r"],                                           # Slot 3
    ["t", "k", "p", "f"],                                # Slot 4
    ["ch", "sh"],                                         # Slot 5
    ["cth", "ckh", "cph", "cfh"],                        # Slot 6
    ["eee", "ee", "e", "g"],                             # Slot 7
    ["k", "t", "p", "f", "ch", "sh", "l", "r", "o", "y"],  # Slot 8
    ["s", "d"],                                           # Slot 9
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


def is_v2_match(word: str) -> bool:
    m, r = parse_greedy(word, SLOTS_V2)
    return r == "" and len(m) > 0


def is_v3_match(word: str) -> tuple[bool, list]:
    """
    v3マッチ判定。マッチした場合は分割情報を返す。
    Returns (matched, parts)
      parts: [(part_str, slots_used), ...]  長さ1または2
    """
    # 1回でマッチ
    m, r = parse_greedy(word, SLOTS_V2)
    if r == "" and m:
        return True, [(word, m)]

    # 2分割でマッチ
    for i in range(1, len(word)):
        p1, p2 = word[:i], word[i:]
        m1, r1 = parse_greedy(p1, SLOTS_V2)
        if r1 != "" or not m1:
            continue
        m2, r2 = parse_greedy(p2, SLOTS_V2)
        if r2 == "" and m2:
            return True, [(p1, m1), (p2, m2)]

    return False, []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    base = Path(__file__).parent
    words = [w.strip() for w in (base / "unique_word.txt").read_text(encoding="utf-8").splitlines() if w.strip()]

    matched_v3 = []
    unmatched_v3 = []
    match_info: dict[str, list] = {}

    for word in words:
        ok, parts = is_v3_match(word)
        if ok:
            matched_v3.append(word)
            match_info[word] = parts
        else:
            unmatched_v3.append(word)

    print(f"総単語数    : {len(words)}")
    print(f"v3 マッチ   : {len(matched_v3)} ({len(matched_v3)/len(words)*100:.1f}%)")
    print(f"v3 未マッチ : {len(unmatched_v3)} ({len(unmatched_v3)/len(words)*100:.1f}%)")

    # 残余パターン分析
    rem_head1: Counter = Counter()
    rem_full: Counter = Counter()
    for w in unmatched_v3:
        # 2分割を試みて、最も進んだ残余を記録
        best_rem = w
        m0, r0 = parse_greedy(w, SLOTS_V2)
        if r0 and len(r0) < len(best_rem):
            best_rem = r0
        for i in range(1, len(w)):
            p1, p2 = w[:i], w[i:]
            m1, r1 = parse_greedy(p1, SLOTS_V2)
            if r1 != "" or not m1:
                continue
            # p1はv2マッチ。p2のparse残余を見る
            m2, r2 = parse_greedy(p2, SLOTS_V2)
            if r2 and len(r2) < len(best_rem):
                best_rem = r2
        if best_rem:
            rem_head1[best_rem[0]] += 1
            rem_full[best_rem[:4]] += 1

    print("\n--- 残余先頭文字 (best-effort) top15 ---")
    for ch, cnt in rem_head1.most_common(15):
        print(f"  {ch!r}: {cnt}")

    print("\n--- 残余文字列先頭4文字 top20 ---")
    for rem, cnt in rem_full.most_common(20):
        print(f"  {rem!r}: {cnt}")

    # ファイル出力
    out_path = base / "unmatched_words_v3.txt"
    lines = []
    for w in unmatched_v3:
        # 1回parseの状況を記録
        m, r = parse_greedy(w, SLOTS_V2)
        slot_str = " ".join(f"[{i}:{s}]" for i, s in m) if m else "(none)"
        lines.append(f"{w}\t{slot_str}\tremaining={r!r}")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[出力] v3 未マッチ単語 ({len(unmatched_v3)}語) -> {out_path}")


if __name__ == "__main__":
    main()
