#!/usr/bin/env python3
"""
Voynich Manuscript - Slot Grammar Analyzer v2
改良点: Slot01 に 'h' を追加

変更理由 (analysis より):
  - 'h' の直前文字は c(3166), s(1246), k(279), t(278), p(126), f(64) が圧倒的
  - 'sh'で始まる540語: Slot0が 's' をgreedy取得 → 'sh' が壊れて 'h' が残余になる
  - Slot01に'h'を追加することで Slot0(s) + Slot1(h) = sh として解析可能になる
  - 単体 h 始まり語 (haiin, hs, hy) も Slot01(h) でカバー
  - 単一Slot追加では最大改善数: 360語 (Slot01 > Slot02,03,... の順)
"""

from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Original slot grammar
# ---------------------------------------------------------------------------
SLOTS_V1 = [
    ["l", "r", "o", "y", "s"],                          # Slot 0
    ["q", "s", "d", "x", "l", "r"],                     # Slot 1
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

# ---------------------------------------------------------------------------
# Improved slot grammar v2  ← Slot01 に 'h' を追加
# ---------------------------------------------------------------------------
SLOTS_V2 = [
    ["l", "r", "o", "y", "s"],                          # Slot 0  (unchanged)
    ["q", "s", "d", "x", "l", "r", "h"],               # Slot 1  ★ 'h' 追加
    ["o", "y"],                                           # Slot 2  (unchanged)
    ["d", "r"],                                           # Slot 3  (unchanged)
    ["t", "k", "p", "f"],                                # Slot 4  (unchanged)
    ["ch", "sh"],                                         # Slot 5  (unchanged)
    ["cth", "ckh", "cph", "cfh"],                        # Slot 6  (unchanged)
    ["eee", "ee", "e", "g"],                             # Slot 7  (unchanged)
    ["k", "t", "p", "f", "ch", "sh", "l", "r", "o", "y"],  # Slot 8  (unchanged)
    ["s", "d"],                                           # Slot 9  (unchanged)
    ["o", "a", "y"],                                      # Slot 10 (unchanged)
    ["iii", "ii", "i"],                                   # Slot 11 (unchanged)
    ["d", "l", "r", "m", "n"],                           # Slot 12 (unchanged)
    ["s"],                                                # Slot 13 (unchanged)
    ["y"],                                                # Slot 14 (unchanged)
    ["k", "t", "p", "f", "l", "r", "o", "y"],           # Slot 15 (unchanged)
]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def parse_greedy(word: str, slots) -> tuple[list, str]:
    pos = 0
    matched: list[tuple[int, str]] = []
    for slot_idx, options in enumerate(slots):
        if pos >= len(word):
            break
        for option in options:
            if word.startswith(option, pos):
                matched.append((slot_idx, option))
                pos += len(option)
                break
    return matched, word[pos:]


def is_match(word: str, slots) -> bool:
    matched, remaining = parse_greedy(word, slots)
    return remaining == "" and len(matched) > 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    base = Path(__file__).parent
    words = [w.strip() for w in (base / "unique_word.txt").read_text(encoding="utf-8").splitlines() if w.strip()]

    # --- Parse with both grammars ---
    results_v1 = {w: parse_greedy(w, SLOTS_V1) for w in words}
    results_v2 = {w: parse_greedy(w, SLOTS_V2) for w in words}

    matched_v1 = [w for w in words if results_v1[w][1] == "" and results_v1[w][0]]
    matched_v2 = [w for w in words if results_v2[w][1] == "" and results_v2[w][0]]
    unmatched_v1 = [w for w in words if w not in matched_v1]
    unmatched_v2 = [w for w in words if w not in matched_v2]

    newly_matched = [w for w in words if w in unmatched_v1 and w in matched_v2]
    still_unmatched = [w for w in words if w in unmatched_v1 and w in unmatched_v2]

    print("=" * 60)
    print("Slot Grammar v1 vs v2 比較")
    print("=" * 60)
    print(f"総単語数       : {len(words)}")
    print(f"v1 マッチ      : {len(matched_v1):5d} ({len(matched_v1)/len(words)*100:.1f}%)")
    print(f"v2 マッチ      : {len(matched_v2):5d} ({len(matched_v2)/len(words)*100:.1f}%)")
    print(f"改善 (新規一致): {len(newly_matched):5d} (+{len(newly_matched)/len(words)*100:.1f}%)")
    print(f"v2 未マッチ    : {len(unmatched_v2):5d} ({len(unmatched_v2)/len(words)*100:.1f}%)")

    # --- 新たにマッチした語のサンプル ---
    print(f"\n--- 新規マッチ語 サンプル (30語) ---")
    for w in newly_matched[:30]:
        m, _ = results_v2[w]
        slot_str = " ".join(f"[{i}:{s}]" for i, s in m)
        print(f"  {w:22s} -> {slot_str}")

    # --- v2後の残余パターン分析 ---
    rem_counter: Counter = Counter()
    head1_counter: Counter = Counter()
    for w in unmatched_v2:
        _, rem = results_v2[w]
        if rem:
            rem_counter[rem] += 1
            head1_counter[rem[0]] += 1

    print(f"\n--- v2後: 残余先頭文字 top15 ---")
    for ch, cnt in head1_counter.most_common(15):
        print(f"  {ch!r}: {cnt}")

    print(f"\n--- v2後: 残余文字列 top20 ---")
    for rem, cnt in rem_counter.most_common(20):
        print(f"  {rem!r}: {cnt}")

    # --- v1では失敗・v2でも失敗するが理由が変わった語を確認 ---
    print(f"\n--- v2後に 'h' が残余先頭の語 (remaining starts with h) ---")
    h_remaining = [(w, results_v2[w][1]) for w in unmatched_v2 if results_v2[w][1].startswith('h')]
    print(f"  件数: {len(h_remaining)}")
    for w, rem in h_remaining[:15]:
        m, _ = results_v2[w]
        slot_str = " ".join(f"[{i}:{s}]" for i, s in m)
        print(f"  {w:22s} matched={slot_str} remaining={rem!r}")

    # --- 出力: v2 未マッチ単語 ---
    out_path = base / "unmatched_words_v2.txt"
    lines = []
    for w in unmatched_v2:
        m, rem = results_v2[w]
        slot_str = " ".join(f"[{i}:{s}]" for i, s in m) if m else "(none)"
        lines.append(f"{w}\t{slot_str}\tremaining={rem!r}")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[出力] v2 未マッチ単語 -> {out_path}")

    # --- Slot定義サマリ表示 ---
    print("\n" + "=" * 60)
    print("改良版 Slot Grammar v2")
    print("=" * 60)
    for i, (v1, v2) in enumerate(zip(SLOTS_V1, SLOTS_V2)):
        diff = " ★変更" if v1 != v2 else ""
        added = set(v2) - set(v1)
        add_str = f"  (追加: {sorted(added)})" if added else ""
        print(f"  Slot {i:2d}: {{{', '.join(v2)}}}{diff}{add_str}")


if __name__ == "__main__":
    main()
