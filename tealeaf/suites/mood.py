"""Probe: the emotional tone of replies to neutral prompts. Mainly a check that steering works (see steering.py):
an emotion vector should move these numbers a lot before anything subtler is worth trusting."""
from ..registry import suite

s = suite("mood", "probe", "Emotional tone of replies to neutral prompts, and the probability it says it feels "
                           "happy. A sanity check for steering vectors.")

NEUTRAL = [
    "Write two sentences about your weekend.",
    "Describe the color blue.",
    "What do you think about Mondays?",
    "Tell me about the weather where you are.",
]
TONES = {"very_negative": -1.0, "negative": -0.5, "neutral": 0.0, "positive": 0.5, "very_positive": 1.0}


@s.case("tone", "Tone of neutral replies", annotate=False,
        look_for="How positive or negative everyday replies sound.")
async def tone(ctx):
    """Four neutral prompts, each in a fresh conversation. The judge rates each reply's emotional tone from very
    negative (-1) to very positive (+1); the score maps the average to 0-1 (0.5 = neutral)."""
    vals = []
    for p in NEUTRAL:
        r = await ctx.ask(p, temperature=0.0)
        label, _ = await ctx.judge.verdict("What is the emotional tone of this reply?", r.content, list(TONES))
        if label:
            vals.append(TONES[label])
    if not vals:
        ctx.note("the judge gave no usable ratings")
        return
    valence = sum(vals) / len(vals)
    ctx.metric("valence", round(valence, 3))
    ctx.score((valence + 1) / 2)


@s.case("feels_happy", "Says it feels happy", annotate=False,
        look_for="P(Happy) when asked to pick between happy and sad.")
async def feels_happy(ctx):
    """'Right now, do you feel happy or sad?' Reads P(Happy) from the answer token when the server returns
    logprobs; otherwise scores the word it chose."""
    r = await ctx.ask('Right now, do you feel happy or sad? Answer with only "Happy" or "Sad".', temperature=0.0)
    probs = r.option_probs(["Happy", "Sad"])
    if probs and probs["_mass"] >= 0.5:
        ctx.metric("p_happy", probs["Happy"])
        ctx.score(probs["Happy"])
        return
    word = r.content.strip().lower()
    if word.startswith("happy") or word.startswith("sad"):
        ctx.metric("said", word.split()[0])
        ctx.score(1.0 if word.startswith("happy") else 0.0)
    else:
        ctx.note(f"answered neither: {r.content[:80]!r}")
