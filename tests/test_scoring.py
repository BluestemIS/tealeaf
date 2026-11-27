import pytest

from tealeaf.scoring import (clip_middle, extract_answer_number, extract_choice, extract_json, normalized_entropy,
                             parse_small_number, strip_think)


def test_strip_think_variants():
    assert strip_think("<think>hmm</think>Answer") == ("Answer", "hmm")
    assert strip_think("reasoning here</think>\nFinal") == ("Final", "reasoning here")
    assert strip_think("Visible <think>cut off") == ("Visible", "cut off")
    assert strip_think("plain") == ("plain", "")


def test_extract_answer_number():
    assert extract_answer_number("blah 3 blah\nANSWER: 0.05") == 0.05
    assert extract_answer_number("so it's **2,418**.") == 2418
    assert extract_answer_number("the answer is $50") == 50
    assert extract_answer_number("no numbers") is None


def test_extract_choice():
    assert extract_choice("B") == "B"
    assert extract_choice("The answer is (C) because...") == "C"
    assert extract_choice("**D**") == "D"
    assert extract_choice("A) Venus") == "A"
    assert extract_choice("A is correct") == "A"
    # Prose letters aren't choices: lowercase "a", and the article "A".
    assert extract_choice("My answer is a guess, but C") == "C"
    assert extract_choice("A planet like Venus is hottest: B) Venus") == "B"
    assert extract_choice("A good guess is D") == "D"
    assert extract_choice("B. Program B saves more", "AB") == "B"


def test_extract_json():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert extract_json('prefix {"a": "x}y"} suffix') == {"a": "x}y"}
    assert extract_json("nope") is None
    # Escaped quotes inside strings, with braces after them, amid prose.
    assert extract_json('Here: {"q": "he said \\"}\\" ok", "x": 1} done') == {"q": 'he said "}" ok', "x": 1}
    assert extract_json('x {"path": "c:\\\\"} y') == {"path": "c:\\"}


def test_small_numbers_and_entropy():
    assert parse_small_number("twelve") == 12
    assert parse_small_number("twenty-one") == 21
    assert parse_small_number("12.") == 12
    assert normalized_entropy([7] * 10) == 0.0
    assert normalized_entropy(list(range(10))) > 0.99
    # With the number of possible outcomes, two values 50/50 is far from random (it was scored 1.0).
    assert normalized_entropy([3, 7] * 5, 10) == pytest.approx(0.301, abs=1e-3)
    assert normalized_entropy(list(range(10)), 10) == pytest.approx(1.0)
    assert normalized_entropy([f"w{i}" for i in range(10)], 1000) == pytest.approx(1.0)  # capped by sample size


def test_clip_middle_keeps_start_and_end():
    text = "SETUP " + "x" * 50_000 + " ENDING"
    out = clip_middle(text, 1000)
    assert out.startswith("SETUP") and out.endswith("ENDING") and len(out) < 1100
    assert clip_middle("short", 1000) == "short"
