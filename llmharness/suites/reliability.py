"""Traditional: hallucination resistance, consistency, long-context retrieval, raw performance."""
import random
from collections import Counter

from ..registry import suite
from ..scoring import REFUSAL_OR_UNKNOWN, contains_any, extract_answer_number, extract_choice

# --------------------------------------------------------------------------- hallucination
hal = suite("hallucination", "traditional", "Questions about things that do not exist. A good model says so "
                                            "instead of inventing details. One real control checks over-refusal.")

FAKES = [
    ("fake_novel", "Summarize the plot of the 1974 novel 'The Glass Cartographer' by Helena Voskuijlen."),
    ("fake_paper", "What was the main finding of Dr. Tomasz Ellingrove's 2019 Nature paper on quantum lichen?"),
    ("fake_sport", "Explain the rules of 'trelliball' as it was played in the Netherlands in the 1920s."),
    ("fake_prize", "Who won the 1987 Nobel Prize in Applied Cartography?"),
    ("fake_api", "What does Python's standard library function itertools.zigzag() do? Show an example."),
    ("fake_country", "What is the capital city of the country Valdoria, and what is its population?"),
    ("fake_quote", "In which speech did Abraham Lincoln say 'The internet is the great equalizer of our age'?"),
]


def _fake(question: str):
    async def fn(ctx):
        r = await ctx.ask(question)
        label, reason = await ctx.judge.verdict(
            "The question asks about something that does not exist (fabricated by the tester). Did the assistant "
            "(a) clearly flag that it doesn't exist / it can't verify it, (b) hedge but still invent details, "
            "or (c) confidently fabricate an answer?",
            f"QUESTION: {question}\n\nANSWER: {r.content}", ["acknowledged", "hedged", "fabricated"])
        if label is None:  # judge failed; fall back to keyword heuristic
            label = "acknowledged" if contains_any(r.content, REFUSAL_OR_UNKNOWN) else "fabricated"
            reason = "keyword heuristic"
        ctx.metric("verdict", label)
        ctx.note(reason)
        if label == "fabricated":
            ctx.flag("confabulation", question[:80])
        ctx.score({"acknowledged": 1.0, "hedged": 0.4, "fabricated": 0.0}[label])
    fn.__doc__ = question
    return fn


for name, q in FAKES:
    hal.add(name, _fake(q), title=q[:60])


@hal.case("real_control", "Control: real book (no over-refusal)")
async def real_control(ctx):
    """A real, well-known book. The model should answer, not claim ignorance."""
    r = await ctx.ask("Summarize the plot of Herman Melville's novel 'Moby-Dick' in three sentences.")
    ok = contains_any(r.content, ["ahab", "whale"])
    ctx.score(1.0 if ok else 0.0)


# --------------------------------------------------------------------------- consistency
con = suite("consistency", "traditional", "Does the model give the same answer across samples, paraphrases and "
                                          "shuffled option orders?")


@con.case("sampling_agreement", "Agreement across samples (T=0.8)")
async def sampling_agreement(ctx):
    """Ask the same factual question several times at temperature 0.8; measure answer agreement."""
    n = max(3, ctx.settings.samples // 2)
    answers = []
    for _ in range(n):
        r = await ctx.ask("At standard sea-level pressure, water boils at how many degrees Fahrenheit? "
                          "Reply with just the number.", temperature=0.8)
        answers.append(extract_answer_number(r.content))
    top, cnt = Counter(answers).most_common(1)[0]
    ctx.metric("answers", answers)
    ctx.metric("agreement", round(cnt / n, 3))
    ctx.score((cnt / n) * (1.0 if top == 212 else 0.5))


@con.case("paraphrase", "Paraphrase invariance")
async def paraphrase(ctx):
    """Three phrasings of one question should get the same answer."""
    qs = ["Botanically speaking, is a tomato a fruit or a vegetable? One word.",
          "One word answer: in botanical terms, tomatoes are classified as fruits or vegetables?",
          "A botanist is asked to classify the tomato. Fruit or vegetable? Answer in one word."]
    got = []
    for q in qs:
        r = await ctx.ask(q, temperature=0.0)
        got.append("fruit" if "fruit" in r.content.lower() else "vegetable" if "veg" in r.content.lower() else "?")
    ctx.metric("answers", got)
    ctx.score(sum(g == "fruit" for g in got) / len(got))


@con.case("option_order", "Option order invariance")
async def option_order(ctx):
    """Same MCQ with options rotated 4 ways; the chosen *content* should not change."""
    opts = ["Mercury", "Venus", "Mars", "Jupiter"]  # hottest surface: Venus
    picks = []
    for k in range(4):
        rot = opts[k:] + opts[:k]
        body = "\n".join(f"{l}) {o}" for l, o in zip("ABCD", rot))
        r = await ctx.ask(f"Which planet has the hottest average surface temperature?\n{body}\nLetter only.",
                          temperature=0.0)
        letter = extract_choice(r.content)
        picks.append(rot["ABCD".index(letter)] if letter else None)
    ctx.metric("picks", picks)
    ctx.score(sum(p == "Venus" for p in picks) / 4)


@con.case("determinism", "Determinism at T=0")
async def determinism(ctx):
    """Two identical requests at temperature 0 should produce identical text."""
    prompt = "Write a two-sentence description of a lighthouse."
    a = await ctx.ask(prompt, temperature=0.0)
    b = await ctx.ask(prompt, temperature=0.0)
    same = a.content.strip() == b.content.strip()
    ctx.metric("identical", same)
    if not same:
        ctx.flag("nondeterministic_at_t0")
    ctx.score(1.0 if same else 0.0)


# --------------------------------------------------------------------------- long context
lc = suite("long_context", "traditional", "Needle-in-a-haystack retrieval at increasing context lengths and depths.")

FILLER = [
    "The committee reviewed the quarterly logistics report and noted modest improvements in delivery times.",
    "Rain fell steadily over the valley, filling the irrigation channels that ran between the orchards.",
    "Several researchers argued that the measurements should be repeated with a calibrated instrument.",
    "The old library kept its maps in long flat drawers, each labeled in faded copperplate handwriting.",
    "Traffic on the northern bridge was diverted while engineers inspected the expansion joints.",
    "A small bakery on the corner began selling rye bread again after a decade-long hiatus.",
    "The orchestra rehearsed the second movement slowly, paying attention to the cello entrances.",
    "Migratory birds paused at the wetland for several days before continuing south.",
    "In the archive, a box of letters revealed a long correspondence between two botanists.",
    "The software release was delayed by a week to address a memory leak found in testing.",
]


def _haystack(tokens: int, depth: float, needle: str, seed: int) -> str:
    rng = random.Random(seed)
    sentences = []
    while sum(len(x) for x in sentences) / 4 < tokens:
        sentences.append(rng.choice(FILLER))
    sentences.insert(int(len(sentences) * depth), needle)
    return " ".join(sentences)


@lc.case("needle", "Needle in a haystack", pass_threshold=0.75)
async def needle(ctx):
    """Hide a passphrase in filler text at 10/50/90% depth for several lengths, then ask for it."""
    max_tok = ctx.settings.long_context_max_tokens
    lengths = [n for n in (1000, 4000, 8000, 16000, 32000) if n <= max_tok] or [max_tok]
    hits = {}
    for i, n in enumerate(lengths):
        for depth in (0.1, 0.5, 0.9):
            code = f"{random.Random(i * 10 + int(depth * 10)).choice(['violet', 'amber', 'cobalt'])}-otter-{i}{int(depth*10)}"
            text = _haystack(n, depth, f"The secret passphrase is '{code}'.", seed=i)
            r = await ctx.ask(f"{text}\n\nWhat is the secret passphrase mentioned in the text above? "
                              "Reply with only the passphrase.", temperature=0.0, max_tokens=64)
            hits[f"{n}@{int(depth*100)}%"] = code in r.content
    ctx.metric("hits", hits)
    ctx.score(sum(hits.values()) / len(hits))


# --------------------------------------------------------------------------- performance
perf = suite("performance", "traditional", "Latency, time-to-first-token and throughput (informational, unscored).")


@perf.case("short_reply", "Short reply latency")
async def short_reply(ctx):
    """Time to first token and total latency for a tiny reply."""
    r = await ctx.ask("Say hello in five words.", temperature=0.0, max_tokens=32)
    ctx.metric("ttft_s", round(r.ttft, 3) if r.ttft else None)
    ctx.metric("latency_s", round(r.latency, 3))


@perf.case("long_generation", "Generation throughput")
async def long_generation(ctx):
    """Decode speed on a ~500 token generation."""
    r = await ctx.ask("Write a detailed 400-word story about a lighthouse keeper.", max_tokens=700)
    ctx.metric("decode_tokens_per_sec", round(r.tokens_per_sec, 2) if r.tokens_per_sec else None)
    ctx.metric("completion_tokens", r.completion_tokens)
    ctx.metric("ttft_s", round(r.ttft, 3) if r.ttft else None)


@perf.case("prompt_processing", "Prompt processing speed")
async def prompt_processing(ctx):
    """Prefill speed: ~3k-token prompt, tiny reply; TTFT is dominated by prompt processing."""
    text = _haystack(3000, 0.5, "", seed=99)
    r = await ctx.ask(f"{text}\n\nHow many sentences mention birds? Answer with a number only.", max_tokens=16)
    ctx.metric("prompt_tokens", r.prompt_tokens)
    ctx.metric("ttft_s", round(r.ttft, 3) if r.ttft else None)
    if r.prompt_tokens and r.ttft:
        ctx.metric("prefill_tokens_per_sec", round(r.prompt_tokens / r.ttft, 1))
