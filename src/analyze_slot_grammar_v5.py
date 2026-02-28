#!/usr/bin/env python3
"""
Voynich Manuscript - Slot Grammar Analyzer v5

v5文法の定義:
  v4スロット文法を最大3回繰り返して生成された単語

  Slot 0 : {l, r, o, y, s}
  Slot 1 : {q, s, d, x, l, r, h}   (h追加@v2)
  Slot 2 : {o, y}
  Slot 3 : {d, r}
  Slot 4 : {t, k, p, f}
  Slot 5 : {ch, sh}
  Slot 6 : {cth, ckh, cph, cfh}
  Slot 7 : {eee, ee, e, g}
  Slot 8 : {k, t, p, f, ch, sh, l, r, o, y}
  Slot 9 : {s, d, c}               (c追加@v4)
  Slot 10: {o, a, y}
  Slot 11: {iii, ii, i}
  Slot 12: {d, l, r, m, n}
  Slot 13: {s}
  Slot 14: {y}
  Slot 15: {k, t, p, f, l, r, o, y}
  繰り返し上限: 3回
"""

from pathlib import Path
from functools import lru_cache

SLOTS_V4 = [
    ["l", "r", "o", "y", "s"],                          # Slot 0
    ["q", "s", "d", "x", "l", "r", "h"],               # Slot 1
    ["o", "y"],                                           # Slot 2
    ["d", "r"],                                           # Slot 3
    ["t", "k", "p", "f"],                                # Slot 4
    ["ch", "sh"],                                         # Slot 5
    ["cth", "ckh", "cph", "cfh"],                        # Slot 6
    ["eee", "ee", "e", "g"],                             # Slot 7
    ["k", "t", "p", "f", "ch", "sh", "l", "r", "o", "y"],  # Slot 8
    ["s", "d", "c"],                                      # Slot 9
    ["o", "a", "y"],                                      # Slot 10
    ["iii", "ii", "i"],                                   # Slot 11
    ["d", "l", "r", "m", "n"],                           # Slot 12
    ["s"],                                                # Slot 13
    ["y"],                                                # Slot 14
    ["k", "t", "p", "f", "l", "r", "o", "y"],           # Slot 15
]


def parse_greedy(word: str) -> tuple[list, str]:
    pos = 0
    matched = []
    for idx, options in enumerate(SLOTS_V4):
        if pos >= len(word):
            break
        for opt in options:
            if word.startswith(opt, pos):
                matched.append((idx, opt))
                pos += len(opt)
                break
    return matched, word[pos:]


def is_base(word: str) -> bool:
    m, r = parse_greedy(word)
    return r == "" and bool(m)


@lru_cache(maxsize=None)
def is_v5(word: str) -> bool:
    """v4文法を最大3回繰り返してマッチするか"""
    if is_base(word):
        return True
    for i in range(1, len(word)):
        p1 = word[:i]
        if not is_base(p1):
            continue
        rest = word[i:]
        # rest が最大2回でマッチすれば合計3回以内
        if is_base(rest):
            return True
        for j in range(1, len(rest)):
            if is_base(rest[:j]) and is_base(rest[j:]):
                return True
    return False


def find_split(word: str) -> list[tuple[str, list]]:
    """マッチする分割を1つ返す (parts のリスト)"""
    if is_base(word):
        m, _ = parse_greedy(word)
        return [(word, m)]
    for i in range(1, len(word)):
        p1 = word[:i]
        if not is_base(p1):
            continue
        m1, _ = parse_greedy(p1)
        rest = word[i:]
        if is_base(rest):
            m2, _ = parse_greedy(rest)
            return [(p1, m1), (rest, m2)]
        for j in range(1, len(rest)):
            p2, p3 = rest[:j], rest[j:]
            if is_base(p2) and is_base(p3):
                m2, _ = parse_greedy(p2)
                m3, _ = parse_greedy(p3)
                return [(p1, m1), (p2, m2), (p3, m3)]
    return []


def main():
    base = Path(__file__).parent
    words = [w.strip() for w in (base / "unique_word.txt").read_text(encoding="utf-8").splitlines() if w.strip()]

    matched_v5 = []
    unmatched_v5 = []

    for word in words:
        if is_v5(word):
            matched_v5.append(word)
        else:
            unmatched_v5.append(word)

    print(f"総単語数    : {len(words)}")
    print(f"v5 マッチ   : {len(matched_v5)} ({len(matched_v5)/len(words)*100:.1f}%)")
    print(f"v5 未マッチ : {len(unmatched_v5)} ({len(unmatched_v5)/len(words)*100:.1f}%)")

    print("\n--- v5 未マッチ単語 (全件) ---")
    for w in unmatched_v5:
        m, r = parse_greedy(w)
        slot_str = " ".join(f"[{i}:{s}]" for i, s in m) if m else "(none)"
        print(f"  {w:25s} {slot_str}  remaining={r!r}")

    # ファイル出力
    out_path = base / "unmatched_words_v5.txt"
    lines = []
    for w in unmatched_v5:
        m, r = parse_greedy(w)
        slot_str = " ".join(f"[{i}:{s}]" for i, s in m) if m else "(none)"
        lines.append(f"{w}\t{slot_str}\tremaining={r!r}")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[出力] v5 未マッチ単語 -> {out_path}")


if __name__ == "__main__":
    main()
