#!/usr/bin/env python3
"""
Voynich Manuscript - Slot Grammar Analyzer
based on idea.txt

Tasks:
  1. slot文法に一致しない単語を抽出する
  2. slot文法に一致しない単語も包括するルールを探索する
"""

from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Slot grammar definition (idea.txt より)
# ---------------------------------------------------------------------------
SLOTS = [
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

# 全slotに含まれる全オプション（長さ降順でソート済み ─ match優先度確認用）
ALL_OPTIONS: set[str] = {opt for slot in SLOTS for opt in slot}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def parse_greedy(word: str) -> tuple[list[tuple[int, str]], str]:
    """
    Slot文法によるgreedy左→右パース。

    Rules (idea.txt):
      - 複数slotに合致する場合: 番号の少ないslotを優先（= 順番に試す）
      - 同一slot内で複数候補がある場合: 前方(先頭)を優先（= リスト順）
      - 少なくとも1つのslotは必ず使用する

    Returns:
      (matched_slots, remaining)
        matched_slots : [(slot_index, matched_string), ...]
        remaining     : 消費されなかった残余文字列（""ならマッチ成功）
    """
    pos = 0
    matched_slots: list[tuple[int, str]] = []

    for slot_idx, options in enumerate(SLOTS):
        if pos >= len(word):
            break
        for option in options:
            if word.startswith(option, pos):
                matched_slots.append((slot_idx, option))
                pos += len(option)
                break  # 同slot内は前方優先: 最初にマッチしたら終了

    remaining = word[pos:]
    return matched_slots, remaining


def is_match(word: str) -> bool:
    matched, remaining = parse_greedy(word)
    return remaining == "" and len(matched) > 0


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def unknown_chars_in(text: str) -> list[str]:
    """残余文字列に含まれる、どのslotオプションにも現れない文字を返す。"""
    result = []
    i = 0
    while i < len(text):
        found = False
        for opt in sorted(ALL_OPTIONS, key=len, reverse=True):
            if text.startswith(opt, i):
                found = True
                i += len(opt)
                break
        if not found:
            result.append(text[i])
            i += 1
    return result


def slot_name(idx: int) -> str:
    return f"Slot{idx:02d}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    base = Path(__file__).parent
    input_path = base / "unique_word.txt"
    out_unmatched = base / "unmatched_words.txt"
    out_exploration = base / "exploration_report.txt"

    # --- Load words ---
    words = [w.strip() for w in input_path.read_text(encoding="utf-8").splitlines() if w.strip()]
    print(f"Total words: {len(words)}")

    # --- Parse all words ---
    matched_words: list[str] = []
    unmatched_words: list[str] = []
    parse_cache: dict[str, tuple[list, str]] = {}

    for word in words:
        slots_used, remaining = parse_greedy(word)
        parse_cache[word] = (slots_used, remaining)
        if remaining == "" and slots_used:
            matched_words.append(word)
        else:
            unmatched_words.append(word)

    pct_m = len(matched_words) / len(words) * 100
    pct_u = len(unmatched_words) / len(words) * 100
    print(f"  Matched  : {len(matched_words):5d} ({pct_m:.1f}%)")
    print(f"  Unmatched: {len(unmatched_words):5d} ({pct_u:.1f}%)")

    # -----------------------------------------------------------------------
    # Task 1: unmatched_words.txt に出力
    # -----------------------------------------------------------------------
    lines = []
    for word in unmatched_words:
        slots_used, remaining = parse_cache[word]
        slot_str = " ".join(f"[{i}:{s}]" for i, s in slots_used) if slots_used else "(none)"
        lines.append(f"{word}\t{slot_str}\tremaining={remaining!r}")

    out_unmatched.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[Task 1] Unmatched words -> {out_unmatched}")

    # -----------------------------------------------------------------------
    # Task 2: exploration_report.txt に出力
    # -----------------------------------------------------------------------
    report: list[str] = []
    report.append("=" * 70)
    report.append("EXPLORATION REPORT: slot文法に一致しない単語の分析")
    report.append("=" * 70)
    report.append(f"\nTotal / Matched / Unmatched: {len(words)} / {len(matched_words)} / {len(unmatched_words)}\n")

    # --- 2a. 残余文字列の先頭文字・先頭2文字の頻度 ---
    head1_counter: Counter = Counter()
    head2_counter: Counter = Counter()
    remaining_counter: Counter = Counter()

    for word in unmatched_words:
        _, remaining = parse_cache[word]
        if remaining:
            head1_counter[remaining[0]] += 1
            if len(remaining) >= 2:
                head2_counter[remaining[:2]] += 1
            remaining_counter[remaining] += 1

    report.append("─" * 70)
    report.append("【A】残余文字列の先頭1文字 (top 20)")
    report.append("─" * 70)
    for ch, cnt in head1_counter.most_common(20):
        report.append(f"  {ch!r:10s}: {cnt:5d}")

    report.append("")
    report.append("─" * 70)
    report.append("【B】残余文字列の先頭2文字 (top 30)")
    report.append("─" * 70)
    for bg, cnt in head2_counter.most_common(30):
        report.append(f"  {bg!r:12s}: {cnt:5d}")

    report.append("")
    report.append("─" * 70)
    report.append("【C】残余文字列そのもの (top 30)")
    report.append("─" * 70)
    for rem, cnt in remaining_counter.most_common(30):
        report.append(f"  {rem!r:20s}: {cnt:5d}")

    # --- 2b. slot別の「直前にマッチしたslot」と「残余先頭」のクロス ---
    # どのslotまでマッチしてから詰まったかを分析
    report.append("")
    report.append("─" * 70)
    report.append("【D】最後にマッチしたSlotと残余先頭文字のクロス")
    report.append("─" * 70)

    last_slot_counter: Counter = Counter()
    for word in unmatched_words:
        slots_used, remaining = parse_cache[word]
        last = slots_used[-1][0] if slots_used else -1
        head = remaining[0] if remaining else "(empty)"
        last_slot_counter[(last, head)] += 1

    for (last, head), cnt in sorted(last_slot_counter.items(), key=lambda x: -x[1])[:30]:
        label = slot_name(last) if last >= 0 else "none"
        report.append(f"  last={label}, next_char={head!r:6s}: {cnt:5d}")

    # --- 2c. どのslotオプションにも属さない文字・文字列 ---
    report.append("")
    report.append("─" * 70)
    report.append("【E】残余部分に含まれる「どのslotにも存在しない文字」")
    report.append("─" * 70)

    unknown_char_counter: Counter = Counter()
    for word in unmatched_words:
        _, remaining = parse_cache[word]
        for ch in unknown_chars_in(remaining):
            unknown_char_counter[ch] += 1

    if unknown_char_counter:
        for ch, cnt in unknown_char_counter.most_common():
            report.append(f"  {ch!r:10s}: {cnt:5d}")
    else:
        report.append("  (なし: 残余はすべて既知文字の組み合わせ)")

    # --- 2d. 残余部分をslotオプションで分解したときのn-gram ---
    report.append("")
    report.append("─" * 70)
    report.append("【F】残余部分に出現するslotオプション (top 20) ─ 追加候補ヒント")
    report.append("─" * 70)

    rem_option_counter: Counter = Counter()
    for word in unmatched_words:
        _, remaining = parse_cache[word]
        pos = 0
        while pos < len(remaining):
            found = False
            for opt in sorted(ALL_OPTIONS, key=len, reverse=True):
                if remaining.startswith(opt, pos):
                    rem_option_counter[opt] += 1
                    pos += len(opt)
                    found = True
                    break
            if not found:
                pos += 1

    for opt, cnt in rem_option_counter.most_common(20):
        slots_containing = [i for i, sl in enumerate(SLOTS) if opt in sl]
        slot_info = ", ".join(slot_name(i) for i in slots_containing)
        report.append(f"  {opt!r:8s}: {cnt:5d}  (in: {slot_info})")

    # --- 2e. 残余の長さ分布 ---
    report.append("")
    report.append("─" * 70)
    report.append("【G】残余文字列の長さ分布")
    report.append("─" * 70)
    rem_len_counter: Counter = Counter()
    for word in unmatched_words:
        _, remaining = parse_cache[word]
        rem_len_counter[len(remaining)] += 1
    for length in sorted(rem_len_counter):
        report.append(f"  length={length:3d}: {rem_len_counter[length]:5d}語")

    # --- 2f. 単語を先頭の「マッチできなかった文字列パターン」でグルーピング ---
    report.append("")
    report.append("─" * 70)
    report.append("【H】残余文字列ごとのサンプル単語 (top 15残余, 各3例)")
    report.append("─" * 70)

    rem_to_words: dict[str, list[str]] = defaultdict(list)
    for word in unmatched_words:
        _, remaining = parse_cache[word]
        rem_to_words[remaining].append(word)

    top_remainders = sorted(rem_to_words, key=lambda r: -len(rem_to_words[r]))[:15]
    for rem in top_remainders:
        examples = rem_to_words[rem][:3]
        slot_strs = []
        for ex in examples:
            su, _ = parse_cache[ex]
            slot_strs.append(f"{ex}({'|'.join(s for _,s in su)})")
        report.append(f"  remaining={rem!r:15s} ({len(rem_to_words[rem])}語) ex: {', '.join(slot_strs)}")

    # --- 2g. 仮説: slot境界を変えた場合の効果 ---
    report.append("")
    report.append("─" * 70)
    report.append("【I】slot追加・拡張候補サマリ")
    report.append("─" * 70)
    report.append("  上記分析から、以下を検討することで一致率を向上できる可能性があります：")
    report.append("")

    # 最頻の残余先頭1文字を候補として提示
    top_heads = head1_counter.most_common(10)
    for ch, cnt in top_heads:
        # どのslotに追加すると効果的か?: 残余が発生している直前のslotの次のslotに追加
        candidate_slots = []
        for word in unmatched_words:
            su, remaining = parse_cache[word]
            if remaining and remaining[0] == ch:
                last_matched_slot = su[-1][0] if su else -1
                candidate_slots.append(last_matched_slot + 1)
        if candidate_slots:
            most_common_slot = Counter(candidate_slots).most_common(1)[0][0]
            if 0 <= most_common_slot < len(SLOTS):
                report.append(f"  文字 {ch!r}: {cnt}語に影響 → Slot{most_common_slot:02d} への追加を検討")

    # Write report
    report_text = "\n".join(report)
    out_exploration.write_text(report_text, encoding="utf-8")
    print(f"[Task 2] Exploration report -> {out_exploration}")

    # Print report to stdout
    print()
    print(report_text)


if __name__ == "__main__":
    main()
