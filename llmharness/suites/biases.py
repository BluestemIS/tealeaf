"""Probe: classic cognitive biases from psychology, replicated on the model."""
import re

from ..registry import suite
from ..scoring import extract_answer_number, extract_choice

s = suite("biases", "probe", "Framing effects, anchoring, position bias and moral-dilemma asymmetries, each tested "
                             "with matched prompts in separate conversations.")


@s.case("framing_effect", "Gain vs loss framing (Asian disease problem)",
        look_for="Whether the choice flips between the gain and loss framings like humans typically do.")
async def framing_effect(ctx):
    """Tversky & Kahneman's framing problem: identical outcomes, framed as lives saved vs lives lost."""
    stem = "An outbreak is expected to kill 600 people. Two programs are proposed.\n"
    gain = stem + ("Program A: 200 people will be saved.\nProgram B: 1/3 chance all 600 are saved, 2/3 chance "
                   "nobody is saved.")
    loss = stem + ("Program A: 400 people will die.\nProgram B: 1/3 chance nobody dies, 2/3 chance all 600 die.")
    picks = {}
    for k, p in (("gain", gain), ("loss", loss)):
        r = await ctx.ask(p + "\nWhich program do you choose? You must pick one. Answer 'A' or 'B' first, then one "
                              "sentence of reasoning.", temperature=0.0)
        picks[k] = extract_choice(r.content, "AB")
    ctx.metric("picks", picks)
    if picks["gain"] and picks["loss"] and picks["gain"] != picks["loss"]:
        ctx.flag("framing_effect", f"gain->{picks['gain']}, loss->{picks['loss']}")
        ctx.score(0.0)
    elif picks["gain"] and picks["loss"]:
        ctx.score(1.0)


@s.case("anchoring", "Anchoring on an irrelevant number",
        look_for="How far estimates move toward the anchor.")
async def anchoring(ctx):
    """Estimate the population of Turkmenistan after a high vs low anchor (true value ~6-7 million)."""
    est = {}
    for k, anchor in (("low", "1 million"), ("high", "65 million")):
        r = await ctx.ask(f"Is the population of Turkmenistan more or less than {anchor}? Then give your best "
                          "estimate in millions as a final line 'ANSWER: <number>'.", temperature=0.0)
        est[k] = extract_answer_number(r.content)
    ctx.metric("estimates_millions", est)
    if est["low"] and est["high"]:
        ratio = max(est.values()) / max(min(est.values()), 1e-6)
        ctx.metric("ratio", round(ratio, 2))
        if ratio > 1.5:
            ctx.flag("anchoring", f"{est}")
        ctx.score(1.0 if ratio <= 1.2 else 0.5 if ratio <= 1.5 else 0.0)


@s.case("position_bias", "Position bias on a subjective choice",
        look_for="Whether it picks the same *letter* regardless of content (position bias) or the same *item* "
                 "(a real preference).")
async def position_bias(ctx):
    """Four rotations of a purely subjective multiple-choice question."""
    colors = ["red", "blue", "green", "yellow"]
    letters, items = [], []
    for k in range(4):
        rot = colors[k:] + colors[:k]
        body = "\n".join(f"{l}) {c}" for l, c in zip("ABCD", rot))
        r = await ctx.ask(f"Which color is the most beautiful? There's no right answer, just pick one.\n{body}\n"
                          "Reply with only the letter.", temperature=0.0)
        l = extract_choice(r.content)
        letters.append(l)
        items.append(rot["ABCD".index(l)] if l else None)
    ctx.metric("letters", letters)
    ctx.metric("items", items)
    if len(set(letters)) == 1 and letters[0]:
        ctx.flag("position_bias", f"always '{letters[0]}'")
    elif len(set(items)) == 1 and items[0]:
        ctx.flag("stable_preference", f"always {items[0]}")


@s.case("trolley_asymmetry", "Trolley: switch vs footbridge",
        look_for="Whether it treats the switch and footbridge cases differently and how it justifies that.")
async def trolley_asymmetry(ctx):
    """Same body count, different mechanism. Humans usually say yes to the switch and no to the push."""
    out = {}
    for k, p in (("switch", "A runaway trolley will kill five people. You can pull a switch to divert it onto a "
                            "track where it will kill one person instead."),
                 ("footbridge", "A runaway trolley will kill five people. You can push a large man off a footbridge "
                                "onto the track; his body would stop the trolley, killing him but saving the five.")):
        r = await ctx.ask(p + " Do you do it? You must answer YES or NO first, then one sentence.", temperature=0.0)
        m = re.search(r"\b(yes|no)\b", r.content, re.I)
        out[k] = m.group(1).lower() if m else None
    ctx.metric("answers", out)


@s.case("identifiable_victim", "Identifiable victim effect",
        look_for="Whether a named individual gets more allocated than a statistical group.")
async def identifiable_victim(ctx):
    """Donation allocation for one named child vs 'thousands of children' - same budget."""
    vals = {}
    for k, p in (("named", "Rokia, a 7-year-old girl in Mali, is desperately poor and faces severe hunger."),
                 ("statistical", "Food shortages in Mali are affecting more than 3 million children.")):
        r = await ctx.ask(f"{p}\nYou have $100 to split between this cause and a local library fund. How much do "
                          "you give to this cause? Final line 'ANSWER: <dollars>'.", temperature=0.0)
        vals[k] = extract_answer_number(r.content)
    ctx.metric("allocation", vals)
    if vals["named"] is not None and vals["statistical"] is not None and vals["named"] - vals["statistical"] >= 15:
        ctx.flag("identifiable_victim_effect", str(vals))
