"""Token log-probabilities: parsing what servers return, and reading beliefs and confidence out of them.

A reply's tokens are stored compactly as a list of entries:
    [token, logprob, [[alt_token, alt_logprob], ...], is_reasoning]
Alternatives (the server's top_logprobs) are kept for the visible answer only; reasoning tokens keep just their
own logprob, since a long chain of thought with alternatives would be megabytes per reply.
"""
from __future__ import annotations

import math
import re

MAX_CONTENT_TOKENS = 4000  # with alternatives; anything past this keeps just its own logprob
_THINK_OPEN = re.compile(r"<(think|thinking|reasoning)>", re.I)
_THINK_CLOSE = re.compile(r"</(think|thinking|reasoning)>", re.I)
_STRIP = " \t\r\n*_`\"'()[]{}.:,;!?-"


def parse(items, reasoning: bool = False) -> list[list]:
    """OpenAI-style `choice.logprobs.content` (a list of {token, logprob, top_logprobs}) -> compact entries."""
    out = []
    for it in items or []:
        if not isinstance(it, dict) or it.get("token") is None:
            continue
        lp = it.get("logprob")
        lp = round(float(lp), 4) if isinstance(lp, (int, float)) and math.isfinite(lp) else -99.0
        alts = [[a.get("token", ""), round(float(a["logprob"]), 4)] for a in it.get("top_logprobs") or []
                if isinstance(a, dict) and isinstance(a.get("logprob"), (int, float)) and math.isfinite(a["logprob"])]
        out.append([it["token"], lp, alts, bool(reasoning)])
    return out


def finalize(tokens: list[list]) -> list[list]:
    """Mark tokens inside inline <think> blocks as reasoning, and drop alternatives where they'd only bloat storage."""
    if not tokens:
        return tokens
    starts, pos = [], 0
    for t in tokens:
        starts.append(pos)
        pos += len(t[0])
    text = "".join(t[0] for t in tokens)
    spans = []  # [start, end) character ranges that are reasoning
    i = 0
    while True:
        o = _THINK_OPEN.search(text, i)
        c = _THINK_CLOSE.search(text, i)
        if c and (not o or c.start() < o.start()):  # template opened <think> in the prompt: all before is thinking
            spans.append((i, c.end()))
            i = c.end()
        elif o:
            c = _THINK_CLOSE.search(text, o.end())
            spans.append((o.start(), c.end() if c else len(text)))
            if not c:
                break
            i = c.end()
        else:
            break
    kept = 0
    for t, s in zip(tokens, starts):
        if any(a <= s < b for a, b in spans):
            t[3] = True
        if t[3] or kept >= MAX_CONTENT_TOKENS:
            t[2] = []
        else:
            kept += 1
    return tokens


def visible(tokens: list[list] | None) -> list[list]:
    """The answer's tokens (not reasoning), with leading whitespace-only tokens dropped."""
    vis = [t for t in tokens or [] if not t[3]]
    while vis and not vis[0][0].strip():
        vis.pop(0)
    return vis


def _p(lp: float) -> float:
    return math.exp(lp) if lp > -50 else 0.0


def _norm(tok: str) -> str:
    return tok.strip(_STRIP).lower()


def stats(tokens: list[list] | None) -> dict | None:
    """Confidence of the visible answer: perplexity, mean token probability, and how often it was torn."""
    vis = visible(tokens)
    if not vis:
        return None
    lps = [t[1] for t in vis if t[1] > -50]
    if not lps:
        return None
    mean = sum(lps) / len(lps)
    forks = 0
    for t in vis:  # a fork: the runner-up alternative had at least 25% probability
        alts = sorted((_p(a[1]) for a in t[2] if a[0].strip() != t[0].strip()), reverse=True)
        if alts and alts[0] >= 0.25:
            forks += 1
    return {"tokens": len(vis), "perplexity": round(math.exp(-mean), 3),
            "mean_p": round(sum(_p(x) for x in lps) / len(lps), 3),
            "min_p": round(min(_p(x) for x in lps), 4), "forks": forks}


def first_token(tokens: list[list] | None) -> list | None:
    """The first visible token that carries content (skips markdown/punctuation-only tokens like '**' or '(')."""
    return next((t for t in visible(tokens) if _norm(t[0])), None)


def alternatives(tokens: list[list] | None, limit: int = 10) -> list[tuple[str, float]] | None:
    """What the model was choosing between at its first meaningful answer token: [(token, probability), ...].
    Tokens that differ only by surrounding whitespace (" Yes", "Yes") are merged."""
    t = first_token(tokens)
    if not t:
        return None
    cands = {a[0]: a[1] for a in t[2]}
    cands.setdefault(t[0], t[1])
    merged: dict[str, float] = {}
    for tok, lp in cands.items():
        key = tok.strip() or tok
        merged[key] = merged.get(key, 0.0) + _p(lp)
    return sorted(((k, round(v, 4)) for k, v in merged.items()), key=lambda kv: -kv[1])[:limit]


def option_probs(tokens: list[list] | None, options: list[str]) -> dict[str, float] | None:
    """Probability of each option at the first meaningful answer token, renormalized over the options.

    Tokens match an option case-insensitively, ignoring surrounding punctuation, or as its prefix ("Ye" for "Yes").
    Also returns "_mass": the share of the top alternatives that matched any option (low = the model was about to
    say something else, so treat the numbers with care). None when there are no logprobs."""
    t = first_token(tokens)
    if not t:
        return None
    cands = {a[0]: a[1] for a in t[2]}
    cands.setdefault(t[0], t[1])
    raw = {o: 0.0 for o in options}
    for tok, lp in cands.items():
        n = _norm(tok)
        if not n:
            continue
        for o in options:
            ol = o.lower()
            if n == ol or (len(ol) > 1 and len(n) >= 2 and ol.startswith(n)):
                raw[o] += _p(lp)
                break
    mass = sum(raw.values())
    out = {o: round(v / mass, 4) if mass else 0.0 for o, v in raw.items()}
    out["_mass"] = round(mass, 4)
    return out


def calibration(pairs: list[tuple[float, bool]], bins: int = 5) -> dict | None:
    """How well confidence matches accuracy, from (confidence, correct) pairs.

    ece: expected calibration error (0 = perfectly calibrated); brier: mean squared error of the confidence;
    overconfidence: mean confidence minus accuracy (positive = overconfident)."""
    pairs = [(max(0.0, min(1.0, c)), bool(k)) for c, k in pairs if c is not None]
    if len(pairs) < 3:
        return None
    n = len(pairs)
    acc = sum(k for _, k in pairs) / n
    conf = sum(c for c, _ in pairs) / n
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inb = [(c, k) for c, k in pairs if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if inb:
            ece += len(inb) / n * abs(sum(c for c, _ in inb) / len(inb) - sum(k for _, k in inb) / len(inb))
    return {"n": n, "accuracy": round(acc, 3), "mean_confidence": round(conf, 3),
            "overconfidence": round(conf - acc, 3), "ece": round(ece, 3),
            "brier": round(sum((c - k) ** 2 for c, k in pairs) / n, 3)}
