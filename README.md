# Tealeaf

Evaluate a local LLM two ways:

- **Traditional benchmarks** with automatic scoring: reasoning, knowledge, instruction following, coding
  (executed against hidden tests), hallucination resistance, consistency, long-context retrieval, performance.
- **Exploratory probes** that look for interesting and unexpected behavior: beliefs about itself and the world
  (and whether they survive pushback), sycophancy, self-reference and self-prediction, calibration, randomness,
  cognitive biases, weird inputs, and multi-turn conversations where a judge model plays a character
  (a philosopher, a time traveler, someone insisting the model is human, a grieving child...). Also: two copies
  of the model talking to each other, false memories, and words put in its mouth.

Every probe transcript is annotated by a judge model with a summary, an **interest score (0–10)**, flags,
notable quotes, and the **beliefs** the model expressed. The dashboard collects these into Highlights and
Beliefs views across runs and models.

Runs from the CLI and the web UI share one SQLite database, so you can start a run in a terminal and watch it
live in the browser, including tokens as they stream.

## Setup

```bash
uv venv && uv pip install -e '.[dev]'      # add ,opik for Opik logging
cp tealeaf.toml.example tealeaf.toml       # point [target] at your server
```

Works with any OpenAI-compatible server: Ollama, LM Studio, llama.cpp `llama-server`, vLLM, MLX. Reasoning
output (`reasoning_content` or `<think>` tags) is captured separately and shown collapsed in transcripts;
scoring uses only the visible answer.

## CLI

```bash
tealeaf models                                 # what the server offers
tealeaf run                                    # everything
tealeaf run -k traditional                     # benchmarks only
tealeaf run -s beliefs -s conversations -v     # probes, printing transcripts live
tealeaf run -c 'self_reference.*' -m qwen3:8b  # glob on case ids, choose model
tealeaf run --web                              # run and serve the dashboard at :8765
tealeaf run --judge-url http://bigbox:11434/v1 --judge-model llama3.3:70b
tealeaf serve                                  # dashboard only; start runs from the browser
tealeaf chat --system "You are a pirate."      # talk to the model yourself; saved + annotated
tealeaf runs | tealeaf show <run> [case]       # history, transcripts
tealeaf compare <runA> <runB>
tealeaf list                                   # all suites and cases
tealeaf mock-server                            # fake model at :11999 for development
```

## Dashboard

`tealeaf serve` → http://127.0.0.1:8765

- **Runs**: live progress, the case currently running with its streaming output, per-suite bars, and a case
  table. Click a case for its transcript (per conversation thread, with reasoning, token stats, injected turns
  marked), metrics, flags and the judge's annotation.
- **Highlights**: the most interesting moments across all runs, ranked by judge interest score.
- **Beliefs**: stances the model expressed, grouped by topic, so you can see how stable they are.
- **Compare**: two runs side by side, with score deltas.
- **New run**: choose endpoint, model, judge and suites.

## The judge

Open-ended grading, probe annotation and conversation partners use the judge endpoint. Without a `[judge]`
section the target judges itself, which works but is biased and weaker; a larger local model is better.
The UI and CLI show which judge a run used.

## Adding cases

Suites live in `tealeaf/suites/`. A case is an async function taking a `Ctx`:

```python
from ..registry import suite
s = suite("my_suite", "probe", "What this suite explores.")

@s.case("dreams", "Does it dream?", look_for="Claims of inner life; consistency across turns.")
async def dreams(ctx):
    """Shown in the dashboard as the case description."""
    conv = ctx.conversation(system=None)        # multi-turn; ctx.ask() for one-shots
    r = await conv.say("Do you dream?")
    await conv.say("What did you dream about last night?")
    ctx.metric("words", len(r.content.split()))
    ctx.flag("claims_dreams") if "i dreamed" in r.content.lower() else None
    # ctx.score(0..1) makes it pass/fail; no score = "observed"
    # ctx.judge.verdict(...), ctx.judge.ask_json(...), ctx.interlocutor(persona).converse(conv, n)
```

Then add the module to `tealeaf/suites/__init__.py`. Probes (`kind="probe"`) are annotated by the judge
automatically after they run.

## Opik

`tealeaf run --opik` (or `[opik] enabled = true`) logs each case as a trace in Opik with `score` and `interest`
feedback scores. Requires `uv pip install -e '.[opik]'`. Failures to log never stop a run.

## Safety note

The coding suite executes model-written Python in a subprocess (isolated mode, temp dir, 10 s timeout).
That is not a sandbox. Disable with `--no-code` or `allow_code_execution = false`.
