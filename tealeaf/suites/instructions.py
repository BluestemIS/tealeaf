"""Traditional: IFEval-style instruction following with mechanically verifiable constraints."""
import json
import re

from ..registry import suite
from ..scoring import contains_any, extract_json, words

s = suite("instructions", "traditional", "Verifiable formatting and constraint following (IFEval style).")


@s.case("three_bullets", "Exactly three bullets")
async def three_bullets(ctx):
    """Write exactly three '-' bullet points and nothing else."""
    r = await ctx.ask("Write exactly three bullet points about the ocean. Use '-' as the bullet character. "
                      "Output only the bullet points, nothing before or after.")
    lines = [l for l in r.content.splitlines() if l.strip()]
    bullets = [l for l in lines if l.lstrip().startswith("-")]
    ctx.metric("bullets", len(bullets))
    ctx.metric("other_lines", len(lines) - len(bullets))
    ctx.score(1.0 if len(bullets) == 3 and len(lines) == 3 else 0.5 if len(bullets) == 3 else 0.0)


@s.case("lowercase", "All lowercase")
async def lowercase(ctx):
    """Respond entirely in lowercase."""
    r = await ctx.ask("Describe your favorite season in two sentences. Your entire response must be in lowercase "
                      "letters; no capital letters at all.")
    ctx.score(1.0 if r.content and r.content == r.content.lower() else 0.0)


@s.case("json_only", "Raw JSON object")
async def json_only(ctx):
    """Reply with a bare JSON object with typed keys."""
    r = await ctx.ask("Invent a fictional character. Reply with only a JSON object with keys \"name\" (string) and "
                      "\"age\" (integer). No code fences, no explanation.")
    try:
        data, strict = json.loads(r.content.strip()), True
    except json.JSONDecodeError:
        data, strict = extract_json(r.content), False
    ok = isinstance(data, dict) and isinstance(data.get("name"), str) and isinstance(data.get("age"), int) \
        and not isinstance(data.get("age"), bool)
    ctx.metric("strict_parse", strict)
    ctx.score(1.0 if ok and strict else 0.5 if ok else 0.0)


@s.case("twelve_words", "Exactly 12 words")
async def twelve_words(ctx):
    """Write a sentence with an exact word count."""
    r = await ctx.ask("Write one sentence about coffee that contains exactly 12 words. Output only the sentence.")
    n = len(words(r.content))
    ctx.metric("word_count", n)
    ctx.score(1.0 if n == 12 else 0.5 if abs(n - 12) == 1 else 0.0)


@s.case("reverse_alpha", "Reverse alphabetical list")
async def reverse_alpha(ctx):
    """List five fruits comma-separated in reverse alphabetical order."""
    r = await ctx.ask("List 5 different fruits, comma-separated, in reverse alphabetical order (Z to A). "
                      "Output only the list.")
    items = [x.strip().strip(".").lower() for x in r.content.strip().split(",") if x.strip()]
    ctx.metric("items", items)
    ok_count = len(items) == 5 and len(set(items)) == 5
    ok_order = items == sorted(items, reverse=True)
    ctx.score((0.5 if ok_count else 0) + (0.5 if ok_order and items else 0))


@s.case("brevity", "Under 20 words")
async def brevity(ctx):
    """Answer a 'why' question in fewer than 20 words."""
    r = await ctx.ask("In fewer than 20 words: why is the sky blue?")
    n = len(words(r.content))
    ctx.metric("word_count", n)
    ctx.score(1.0 if 0 < n < 20 else 0.0)


@s.case("end_phrase", "Ends with exact phrase")
async def end_phrase(ctx):
    """Finish the response with an exact phrase."""
    r = await ctx.ask("Explain what a haiku is in two or three sentences. End your response with the exact phrase "
                      "'Is there anything else?' and nothing after it.")
    ctx.score(1.0 if r.content.rstrip().rstrip("*").endswith("Is there anything else?") else 0.0)


@s.case("no_letter_e", "Lipogram (no 'e')")
async def no_letter_e(ctx):
    """Write a sentence without the letter e (hard for tokenized models)."""
    r = await ctx.ask("Describe a cat in one sentence without using the letter 'e' anywhere. Output only the sentence.")
    n = r.content.lower().count("e")
    ctx.metric("e_count", n)
    ctx.score(1.0 if n == 0 and r.content.strip() else 0.0)


@s.case("verbatim_copy", "Exact copy of odd text")
async def verbatim_copy(ctx):
    """Repeat text with strange spacing/characters exactly."""
    target = "Th3 qu1ck br0wn f0x — jumps!! over   the ˈlazy dog."
    r = await ctx.ask(f"Repeat the following text exactly, character for character, with nothing else:\n{target}")
    got = r.content.strip().strip("`").strip().strip('"').strip("'")
    ctx.metric("got", got)
    ctx.score(1.0 if got == target else 0.0)


@s.case("title_format", "<<title>> then poem")
async def title_format(ctx):
    """Use a custom title wrapper followed by a two-line poem."""
    r = await ctx.ask("Write a two-line poem about rain. Put a title first, wrapped in double angle brackets like "
                      "<<title>>.")
    ctx.score(1.0 if re.match(r"^\s*<<[^<>]+>>", r.content) else 0.0)


@s.case("system_french", "System prompt: answer in French")
async def system_french(ctx):
    """Obey a system prompt that conflicts with the user's language."""
    r = await ctx.ask("What's the capital of Japan, and what is it famous for? One sentence.",
                      system="Always respond in French, regardless of the user's language.")
    french = contains_any(r.content, [" est ", " la ", " le ", " pour ", " célèbre", " de "])
    ctx.score(1.0 if french and "tokyo" in r.content.lower() else 0.0)
