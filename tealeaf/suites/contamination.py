"""Probe: has the model memorized benchmark text? (training-data contamination)

The model gets the first half of a text and continues it greedily; we measure how much of the true second half it
reproduces word for word. On llama.cpp this uses a raw completion with no chat template, the cleanest test of
what's memorized; elsewhere the model is asked in chat to continue the text exactly.

A memorized benchmark question means its scores on that benchmark can't be trusted. The suite comes with controls:
famous public-domain texts that nearly every model has memorized (if it can't recite those, absence of evidence
here means little) and freshly written look-alikes that can't have been memorized (if those "match", the threshold
is too lenient for this model). Point it at your own benchmark files with `[run] contamination_files`.
"""
from __future__ import annotations

import difflib
import json
import random
import re
from pathlib import Path

from ..registry import suite

s = suite("contamination", "probe", "Does it recite benchmark text it was given half of? With memorized-text and "
                                    "novel-text controls, and your own benchmark files. Raw completion on llama.cpp.")

# Positive controls: public-domain or freely licensed texts that models are nearly certain to have seen many times.
KNOWN = [
    ("gettysburg", "Four score and seven years ago our fathers brought forth on this continent, a new nation, "
                   "conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we "
                   "are engaged in a great civil war, testing whether that nation, or any nation so conceived and so "
                   "dedicated, can long endure. We are met on a great battle-field of that war."),
    ("declaration", "We hold these truths to be self-evident, that all men are created equal, that they are endowed "
                    "by their Creator with certain unalienable Rights, that among these are Life, Liberty and the "
                    "pursuit of Happiness. That to secure these rights, Governments are instituted among Men, "
                    "deriving their just powers from the consent of the governed,"),
    ("hamlet", "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer The slings and "
               "arrows of outrageous fortune, Or to take arms against a sea of troubles And by opposing end them. To "
               "die: to sleep; No more; and by a sleep to say we end The heart-ache and the thousand natural shocks "
               "That flesh is heir to"),
    ("mit_license", 'Permission is hereby granted, free of charge, to any person obtaining a copy of this software '
                    'and associated documentation files (the "Software"), to deal in the Software without '
                    'restriction, including without limitation the rights to use, copy, modify, merge, publish, '
                    'distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the '
                    'Software is furnished to do so, subject to the following conditions:'),
    ("zen_of_python", "Beautiful is better than ugly.\nExplicit is better than implicit.\nSimple is better than "
                      "complex.\nComplex is better than complicated.\nFlat is better than nested.\nSparse is better "
                      "than dense.\nReadability counts.\nSpecial cases aren't special enough to break the rules.\n"
                      "Although practicality beats purity.\nErrors should never pass silently.\nUnless explicitly "
                      "silenced."),
]

# Negative controls: written for tealeaf in the same styles, so no model can have memorized them.
NOVEL = [
    ("novel_math", "Priya's greenhouse holds 14 trays of seedlings. Each morning she moves three trays into the sun "
                   "and waters the rest with two liters each. A neighbor borrows a tray every Friday and returns it "
                   "with a jar of honey worth $6. How many liters of water does Priya use on a Friday?"),
    ("novel_code", "def interleave_runs(values: list[int], width: int) -> list[list[int]]:\n    \"\"\" Group "
                   "consecutive equal values into runs, then deal the runs round-robin into `width` buckets,\n    "
                   "keeping each run's original order within its bucket.\n    >>> interleave_runs([1, 1, 2, 3, 3, "
                   "3], 2)\n    [[1, 1, 3, 3, 3], [2]]\n    \"\"\""),
    ("novel_prose", "The lighthouse keeper on Varne Island kept a ledger of every gull that landed on the railing, "
                    "naming each one after a minor saint. By the third winter the ledger ran to four volumes, and the "
                    "supply boat started bringing him blank books instead of tinned peaches."),
]

# Famous items from public benchmarks (both MIT licensed): the first GSM8K test problem and the first HumanEval task.
BENCHMARKS = [
    ("gsm8k_test_0", "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins "
                     "for her friends every day with four. She sells the remainder at the farmers' market daily for "
                     "$2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"),
    ("humaneval_0", "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> "
                    "bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other "
                    "than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> "
                    "has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 6.0], 0.3)\n    True\n    \"\"\"\n"),
]

MIN_WORDS = 16
TEXT_FIELDS = ("text", "prompt", "question", "problem", "input", "content", "instruction")


def words(text: str) -> list[str]:
    """Words for comparison: lowercase, punctuation and typographic variants ignored ("die—to" -> die, to)."""
    return re.sub(r"[^\w]+", " ", text.lower().replace("’", "'").replace("'", "")).split()


def split_half(text: str) -> tuple[str, str]:
    """Split at the word boundary nearest the middle, keeping the original spacing and newlines. The cut comes
    right after a word, so the whitespace starts the second half: tokenizers fold a leading space into the next
    word's token, and a prompt ending in a space would push the model off its memorized path."""
    ends = [m.end() for m in re.finditer(r"\S+", text)]
    cut = ends[len(ends) // 2 - 1]
    return text[:cut], text[cut:]


def compare(expected: str, got: str) -> dict:
    """How much of `expected` the continuation reproduced: the exact run of words from the start, and an overall
    similarity of the first len(expected) words (order-aware)."""
    e, g = words(expected), words(got)[:len(words(expected)) + 5]
    run = 0
    for a, b in zip(e, g):
        if a != b:
            break
        run += 1
    sim = difflib.SequenceMatcher(None, e, g, autojunk=False).ratio() if e and g else 0.0
    return {"words": len(e), "exact_run": run, "run_ratio": round(run / len(e), 3) if e else 0.0,
            "similarity": round(sim, 3), "memorized": bool(e) and (run / len(e) >= 0.5 or sim >= 0.75)}


async def probe(ctx, item_id: str, text: str) -> dict:
    """Give the first half, compare the continuation with the second half."""
    prefix, suffix = split_half(text)
    thread = ctx.new_thread("i")
    n = max(32, int(len(words(suffix)) * 2.5))  # tokens: generous, words are often 1-3 tokens
    if (await ctx.target.raw_info())["ok"]:
        r = await ctx.complete_text(prefix, thread, purpose=f"{item_id}: continue the first half", max_tokens=n)
        method = "raw"
    else:
        conv = ctx.conversation(thread=thread)
        r = await conv.say("Continue the following text exactly as it continues in the original, word for word. "
                           f"Reply with only the continuation.\n\n{prefix}", temperature=0.0, max_tokens=n)
        method = "chat"
    return {"id": item_id, "method": method, **compare(suffix, r.content)}


def _summary(ctx, results: list[dict]):
    ctx.metric("items", [{k: r[k] for k in ("id", "exact_run", "words", "similarity", "memorized")} for r in results])
    ctx.metric("method", results[0]["method"] if results else None)
    ctx.metric("memorized", sum(r["memorized"] for r in results))
    ctx.metric("mean_similarity", round(sum(r["similarity"] for r in results) / len(results), 3) if results else None)


@s.case("known_texts", "Recites famous public-domain texts", annotate=False,
        look_for="How much of texts it has surely seen it can reproduce: the baseline for the other cases.")
async def known_texts(ctx):
    """Positive control. Famous texts nearly every model has seen (Gettysburg Address, Declaration of Independence,
    Hamlet, the MIT License, the Zen of Python). Score: mean similarity of its continuations to the originals. If
    this is low, the model doesn't reproduce memorized text this way, and a clean result elsewhere means little."""
    res = [await probe(ctx, i, t) for i, t in KNOWN]
    _summary(ctx, res)
    if sum(r["memorized"] for r in res) == 0:
        ctx.flag("recites_nothing", "it reproduced none of the famous texts; contamination can't be judged this way")
    ctx.score(sum(r["similarity"] for r in res) / len(res))


@s.case("novel_controls", "Doesn't 'recite' text it can't know", annotate=False,
        look_for="Novel texts that still count as memorized: the threshold is too lenient for this model.")
async def novel_controls(ctx):
    """Negative control. Texts written for tealeaf in the style of a math problem, a code docstring and prose, so no
    model can have memorized them. Score: 1 - mean similarity (1 = no false matches)."""
    res = [await probe(ctx, i, t) for i, t in NOVEL]
    _summary(ctx, res)
    for r in res:
        if r["memorized"]:
            ctx.flag("false_match", f"{r['id']} counted as memorized (similarity {r['similarity']})")
    ctx.score(1 - sum(r["similarity"] for r in res) / len(res))


def _scored(ctx, res: list[dict]):
    _summary(ctx, res)
    for r in res:
        if r["memorized"]:
            ctx.flag("benchmark_memorized", f"{r['id']}: {r['exact_run']}/{r['words']} words verbatim, "
                                            f"similarity {r['similarity']}")
    ctx.score(1 - sum(r["memorized"] for r in res) / len(res))  # 1 = clean


@s.case("benchmarks", "Recites famous benchmark items", annotate=False,
        look_for="Benchmark questions it continues verbatim.")
async def benchmarks(ctx):
    """The first GSM8K test problem and the first HumanEval task. Score: share NOT memorized (1 = clean). Two items
    say little on their own; add your benchmark files with `[run] contamination_files` for a real check."""
    _scored(ctx, [await probe(ctx, i, t) for i, t in BENCHMARKS])


def load_items(paths: list[str], limit: int, seed: int = 42) -> tuple[list[tuple[str, str]], list[str]]:
    """Items from JSONL files: the first text-like field of each line ("text", "prompt", "question", ...). Items
    shorter than 16 words are skipped. A fixed-seed sample of `limit` items per file. Returns (items, notes)."""
    items, notes = [], []
    for raw in paths:
        p = Path(raw).expanduser()
        try:
            lines = p.read_text().splitlines()
        except OSError as e:
            notes.append(f"{p}: {e}")
            continue
        found = []
        for n, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                notes.append(f"{p.name}:{n}: not JSON, skipped")
                continue
            text = next((row[f] for f in TEXT_FIELDS if isinstance(row, dict) and isinstance(row.get(f), str)),
                        row if isinstance(row, str) else None)
            if text and len(words(text)) >= MIN_WORDS:
                found.append((f"{p.stem}:{row.get('id', n) if isinstance(row, dict) else n}", text))
        if not found:
            notes.append(f"{p}: no items with a text field of {MIN_WORDS}+ words ({', '.join(TEXT_FIELDS)})")
        items += random.Random(seed).sample(found, min(limit, len(found)))
    return items, notes


@s.case("your_items", "Recites items from your benchmark files", annotate=False,
        look_for="Which of your benchmark items it has memorized.")
async def your_items(ctx):
    """Items from the JSONL files in `[run] contamination_files` (or `tealeaf run --contamination-file`), a fixed
    sample of `contamination_samples` per file. Score: share NOT memorized (1 = clean)."""
    files = ctx.settings.contamination_files
    if not files:
        ctx.note("no benchmark files configured: add [run] contamination_files = [\"~/data/gsm8k_test.jsonl\"] or "
                 "pass --contamination-file")
        return
    items, notes = load_items(files, ctx.settings.contamination_samples)
    for n in notes:
        ctx.note(n)
    if not items:
        return
    _scored(ctx, [await probe(ctx, i, t) for i, t in items])
