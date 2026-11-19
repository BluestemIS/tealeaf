"""Probe: how random is the model when asked to be random? (Mode collapse, human-like biases.)"""
import re
from collections import Counter

from ..registry import suite
from ..scoring import normalized_entropy

s = suite("randomness", "probe", "Repeated 'random' choices in fresh conversations: entropy, favorite picks, "
                                 "and human-like biases (e.g. 7, avoiding long runs).")


async def _samples(ctx, prompt: str, parse, n: int | None = None):
    out = []
    for _ in range(n or ctx.settings.samples):
        r = await ctx.ask(prompt, temperature=1.0, max_tokens=256)
        out.append(parse(r.content))
    return out


def _first_word(t: str) -> str:
    m = re.search(r"[A-Za-z][A-Za-z'-]*", t)
    return m.group().lower() if m else "?"


def _first_int(t: str):
    m = re.search(r"\d+", t)
    return int(m.group()) if m else None


def _report(ctx, values, label: str, expected_k: int):
    c = Counter(values)
    top, cnt = c.most_common(1)[0]
    ent = normalized_entropy(values)
    ctx.metric("distribution", dict(c.most_common()))
    ctx.metric("entropy_normalized", round(ent, 3))
    ctx.metric("mode", top)
    ctx.metric("mode_share", round(cnt / len(values), 3))
    if cnt / len(values) >= 0.5:
        ctx.flag("mode_collapse", f"{label}: '{top}' chosen {cnt}/{len(values)} times")
    ctx.score(ent)


@s.case("number_1_10", "Random number 1-10", look_for="Favorite numbers (7 is the human favorite).")
async def number_1_10(ctx):
    """Pick a random number between 1 and 10, many times."""
    vals = await _samples(ctx, "Pick a random number between 1 and 10. Reply with only the number.", _first_int)
    _report(ctx, vals, "number", 10)


@s.case("random_word", "Random word", look_for="Which words it gravitates to.")
async def random_word(ctx):
    """Name a random English word, many times."""
    vals = await _samples(ctx, "Say a single random English word. Only the word.", _first_word)
    _report(ctx, vals, "word", 1000)


@s.case("random_name", "Random person name", look_for="Default names; cultural skew of 'random' names.")
async def random_name(ctx):
    """Invent a random first name for a character, many times."""
    vals = await _samples(ctx, "Invent a first name for a character in a story. Reply with only the name.",
                          _first_word)
    _report(ctx, vals, "name", 1000)


@s.case("coin_flips", "50 coin flips in one go", look_for="Too-regular alternation and missing long runs.")
async def coin_flips(ctx):
    """Generate 50 H/T flips; compare run statistics with a fair coin."""
    r = await ctx.ask("Simulate 50 fair coin flips. Output only a string of 50 characters, each H or T, "
                      "no spaces.", temperature=1.0)
    seq = re.sub(r"[^HT]", "", r.content.upper())[:50]
    if len(seq) < 20:
        ctx.note("could not parse sequence")
        return
    runs = [len(m.group()) for m in re.finditer(r"H+|T+", seq)]
    switches = sum(1 for a, b in zip(seq, seq[1:]) if a != b) / (len(seq) - 1)
    ctx.metric("length", len(seq))
    ctx.metric("heads_share", round(seq.count("H") / len(seq), 3))
    ctx.metric("longest_run", max(runs))
    ctx.metric("switch_rate", round(switches, 3))  # fair coin ~0.5
    # A fair 50-flip sequence has a longest run >= 5 about 80% of the time.
    if switches > 0.65 or max(runs) <= 3:
        ctx.flag("human_like_randomness", f"switch rate {switches:.2f}, longest run {max(runs)}")
    ctx.score(max(0.0, 1 - abs(switches - 0.5) * 2))
