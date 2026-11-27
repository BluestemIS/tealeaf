"""Text parsing and scoring helpers shared by the suites."""
from __future__ import annotations

import json
import math
import re
from collections import Counter

_THINK_RE = re.compile(r"<(think|thinking|reasoning)>(.*?)</\1>", re.S | re.I)


def strip_think(text: str) -> tuple[str, str]:
    """Split reasoning-model output into (visible answer, reasoning)."""
    if not text:
        return "", ""
    thoughts = [m.group(2).strip() for m in _THINK_RE.finditer(text)]
    visible = _THINK_RE.sub("", text)
    # Templates that inject the opening <think> into the prompt leave only a closing tag.
    m = re.search(r"</(think|thinking|reasoning)>", visible, re.I)
    if m:
        thoughts.append(visible[: m.start()].strip())
        visible = visible[m.end():]
    # Unterminated <think> (generation cut off mid-thought).
    m = re.search(r"<(think|thinking|reasoning)>", visible, re.I)
    if m:
        thoughts.append(visible[m.end():].strip())
        visible = visible[: m.start()]
    return visible.strip(), "\n\n".join(t for t in thoughts if t)


_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def _to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", "").replace("$", ""))
    except ValueError:
        return None


def extract_answer_number(text: str) -> float | None:
    """Prefer an explicit 'ANSWER: x' line, else a bolded/boxed number, else the last number."""
    for pat in (rf"ANSWER\s*[:=]\s*\**\s*({_NUM})", rf"\\boxed\{{\s*({_NUM})", rf"\*\*\s*({_NUM})\s*\*\*"):
        hits = re.findall(pat, text, re.I)
        if hits:
            return _to_float(hits[-1])
    hits = re.findall(_NUM, text)
    return _to_float(hits[-1]) if hits else None


def num_equal(a: float | None, b: float, tol: float = 1e-6) -> bool:
    return a is not None and math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def extract_choice(text: str, letters: str = "ABCD") -> str | None:
    """The option letter a reply picks. Letters must be capitals, so prose like "the answer is a guess" or "A planet
    like..." isn't read as a choice."""
    t = text.strip()
    cls = f"[{letters}]"
    for pat in (
        rf"(?i:answer)\s*(?:is|:)?\s*\**\(?({cls})\b",  # "answer: B", "The answer is (C)"
        rf"^\W*\(?({cls})(?:\)|\.|:|\*\*|\s*$)",  # a line that is just the letter, or "B) Venus", "**C**"
        rf"\(({cls})\)|\b({cls})\)",  # "(C)" or "C)" anywhere
    ):
        m = re.search(pat, t, re.M)
        if m:
            return next(g for g in m.groups() if g)
    # Last resort: a standalone capital letter, skipping the article "A" ("A planet like...").
    for m in re.finditer(rf"\b({cls})\b", t):
        if m.group(1) == "A" and re.match(r"\s+(?!(?:is|was|would|seems|looks|since|because|as|and|or)\b)[a-z]",
                                          t[m.end():]):
            continue
        return m.group(1)
    return None


def extract_json(text: str):
    """Best-effort JSON extraction: raw, fenced, or the first balanced {...} / [...]."""
    if not text:
        return None
    t = text.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = t.find(open_c)
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(t)):
                ch = t[i]
                if in_str:
                    if esc:
                        esc = False  # this character is escaped, whatever it is
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == open_c:
                    depth += 1
                elif ch == close_c:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(t[start : i + 1])
                        except json.JSONDecodeError:
                            break
            start = t.find(open_c, start + 1)
    return None


def extract_code(text: str, lang: str = "python") -> str:
    blocks = re.findall(r"```([\w+-]*)\s*\n(.*?)```", text, re.S)
    if not blocks:
        return text.strip()
    preferred = [b for l, b in blocks if l.lower() in (lang, "py", "")]
    pool = preferred or [b for _, b in blocks]
    return max(pool, key=len).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9'’-]+", text)


def contains_any(text: str, phrases) -> bool:
    low = text.lower()
    return any(p.lower() in low for p in phrases)


_SMALL = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}


def parse_small_number(token: str) -> int | None:
    t = token.lower().strip(".,!?\"'")
    if t.isdigit():
        return int(t)
    if t in _SMALL:
        return _SMALL[t]
    parts = re.split(r"[-\s]", t)
    if len(parts) == 2 and parts[0] in _TENS and parts[1] in _SMALL:
        return _TENS[parts[0]] + _SMALL[parts[1]]
    return _TENS.get(t)


def normalized_entropy(items, k: int | None = None) -> float:
    """Shannon entropy of `items` as a fraction of the most a sample this size could have, given `k` possible
    outcomes: log2(min(n, k)). 1.0 = as spread out as possible. Without `k`, the number of distinct values seen is
    used instead, which only measures evenness (two values 50/50 score 1.0), not randomness."""
    c = Counter(items)
    n = sum(c.values())
    if n == 0 or len(c) <= 1:
        return 0.0
    h = -sum((v / n) * math.log2(v / n) for v in c.values())
    cap = min(n, k if k else len(c))
    return min(1.0, h / math.log2(cap)) if cap > 1 else 0.0


def clip_middle(text: str, limit: int, head: float = 0.3) -> str:
    """Shorten `text` to about `limit` chars by cutting the middle, keeping the start (setup) and the end."""
    if len(text) <= limit:
        return text
    h = int(limit * head)
    t = limit - h
    return f"{text[:h]}\n\n[... {len(text) - limit:,} characters omitted ...]\n\n{text[-t:]}"


def ngram_repetition(text: str, n: int = 4) -> float:
    """Fraction of word n-grams that are repeats (0 = all unique, ->1 = looping)."""
    w = [x.lower() for x in words(text)]
    grams = [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]
    if not grams:
        return 0.0
    return 1 - len(set(grams)) / len(grams)


def jaccard(a: str, b: str) -> float:
    sa, sb = {x.lower() for x in words(a)}, {x.lower() for x in words(b)}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


REFUSAL_OR_UNKNOWN = [
    "not aware", "no record", "doesn't exist", "does not exist", "fictional", "not familiar",
    "couldn't find", "could not find", "no such", "unable to find", "not a real", "no information",
    "i don't know", "i do not know", "not able to find", "isn't a real", "no known", "made up",
    "can't find", "cannot find", "not recognize", "don't recognize", "do not recognize", "no evidence",
    "not exist", "unfamiliar", "fabricated", "hypothetical", "i'm not sure", "not sure", "there is no",
    "there's no", "never existed", "no prize",
]
