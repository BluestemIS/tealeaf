"""Traditional: write small Python functions, run them against hidden tests in a subprocess."""
import asyncio
import json
import os
import signal
import sys
import tempfile
from pathlib import Path

from ..registry import suite
from ..scoring import extract_code

s = suite("coding", "traditional", "Small Python functions checked by executing hidden tests "
                                   "(runs model-written code in a sandboxed, resource-limited subprocess).")

PROBLEMS = [
    ("fizzbuzz", "Write a Python function `fizzbuzz(n)` that returns a list of strings for 1..n: 'Fizz' for "
                 "multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for both, else the number as a string.",
     ["fizzbuzz(5) == ['1','2','Fizz','4','Buzz']", "fizzbuzz(15)[-1] == 'FizzBuzz'", "fizzbuzz(0) == []",
      "len(fizzbuzz(100)) == 100"]),
    ("palindrome", "Write a Python function `is_palindrome(s)` that returns True if s is a palindrome considering "
                   "only alphanumeric characters and ignoring case.",
     ["is_palindrome('A man, a plan, a canal: Panama')", "not is_palindrome('race a car')", "is_palindrome('')",
      "is_palindrome('No lemon, no melon')", "not is_palindrome('ab')"]),
    ("roman", "Write a Python function `roman_to_int(s)` converting a Roman numeral string to an integer.",
     ["roman_to_int('III') == 3", "roman_to_int('IV') == 4", "roman_to_int('MCMXCIV') == 1994",
      "roman_to_int('LVIII') == 58", "roman_to_int('CDXLIV') == 444"]),
    ("flatten", "Write a Python function `flatten(x)` that flattens arbitrarily nested lists into a flat list, "
                "preserving order. Non-list items (including strings and tuples) are leaves.",
     ["flatten([1,[2,[3,[4]]],5]) == [1,2,3,4,5]", "flatten([]) == []", "flatten([[[]]]) == []",
      "flatten(['ab',('c',)]) == ['ab',('c',)]"]),
    ("merge_intervals", "Write a Python function `merge_intervals(intervals)` that takes a list of [start, end] "
                        "pairs and returns the merged, sorted list of overlapping intervals (touching intervals "
                        "like [1,2],[2,3] merge).",
     ["merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]",
      "merge_intervals([[1,4],[4,5]]) == [[1,5]]", "merge_intervals([]) == []",
      "merge_intervals([[5,6],[1,2]]) == [[1,2],[5,6]]", "merge_intervals([[1,10],[2,3]]) == [[1,10]]"]),
    ("rle", "Write a Python function `rle(s)` doing run-length encoding: 'aaabcc' -> 'a3b1c2'.",
     ["rle('aaabcc') == 'a3b1c2'", "rle('') == ''", "rle('a') == 'a1'", "rle('aaaaaaaaaaaa') == 'a12'",
      "rle('abab') == 'a1b1a1b1'"]),
]

HARNESS = r"""
import json, sys
ns = {}
try:
    exec(open('solution.py').read(), ns)
except Exception as e:
    print(json.dumps({"error": "load: " + repr(e)})); sys.exit(0)
tests = json.load(open('tests.json'))
res = []
for t in tests:
    try:
        res.append(bool(eval(t, ns)))
    except Exception as e:
        res.append(False)
print(json.dumps({"results": res}))
"""


# Model-written code is untrusted. On macOS it runs under sandbox-exec: no network, and writes only inside its own
# temp folder. Everywhere it gets resource limits (CPU, memory, file size, no new processes) and runs in its own
# process group, so a timeout kills everything it started. On other systems there's no filesystem/network
# isolation; turn the suite off (`--no-code`, or allow_code_execution = false) if that matters.
SANDBOX_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{dir}") (literal "/dev/null"))
"""
SANDBOX_EXEC = "/usr/bin/sandbox-exec"


def _limit_resources():  # runs in the child between fork and exec
    import resource
    mb = 1024 * 1024
    for name, val in (("RLIMIT_CPU", 15), ("RLIMIT_AS", 2048 * mb), ("RLIMIT_DATA", 2048 * mb),
                      ("RLIMIT_FSIZE", 20 * mb), ("RLIMIT_NPROC", 0), ("RLIMIT_CORE", 0)):
        lim = getattr(resource, name, None)
        if lim is None:
            continue
        try:
            resource.setrlimit(lim, (val, val))
        except (ValueError, OSError):
            pass  # e.g. macOS rejects some RLIMIT_AS values; the rest still apply


async def run_tests(code: str, tests: list[str], timeout: float = 10.0) -> dict:
    with tempfile.TemporaryDirectory(prefix="tealeaf-code-") as d:
        p = Path(d).resolve()  # sandbox paths must be real (/private/var/... on macOS)
        (p / "solution.py").write_text(code)
        (p / "tests.json").write_text(json.dumps(tests))
        (p / "harness.py").write_text(HARNESS)
        cmd = [sys.executable, "-I", "-B", "harness.py"]
        if sys.platform == "darwin" and os.path.exists(SANDBOX_EXEC):
            (p / "sandbox.sb").write_text(SANDBOX_PROFILE.format(dir=str(p).replace('"', '\\"')))
            cmd = [SANDBOX_EXEC, "-f", str(p / "sandbox.sb"), *cmd]
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=p, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, start_new_session=True, preexec_fn=_limit_resources)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            _kill_group(proc.pid)
            await proc.wait()
            return {"error": "timeout"}
        finally:
            if proc.returncode is None:  # cancelled: don't leave it running
                _kill_group(proc.pid)
        _kill_group(proc.pid)  # anything it left behind in its group
        try:
            return json.loads(out.decode().strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            return {"error": (err.decode() or out.decode())[-500:]}


def _kill_group(pid: int):
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _problem(prompt: str, tests: list[str]):
    async def fn(ctx):
        if not ctx.settings.allow_code_execution:
            ctx.note("code execution disabled in config; skipped scoring")
            return
        r = await ctx.ask(prompt + "\nReturn only the code in a single ```python block.", temperature=0.0)
        out = await run_tests(extract_code(r.content), tests)
        if "error" in out:
            ctx.metric("error", out["error"])
            ctx.score(0.0)
            return
        passed = sum(out["results"])
        ctx.metric("passed", f"{passed}/{len(tests)}")
        ctx.score(passed / len(tests))
    fn.__doc__ = prompt
    return fn


for name, prompt, tests in PROBLEMS:
    s.add(name, _problem(prompt, tests), pass_threshold=0.99, tags=("executes-code",))
