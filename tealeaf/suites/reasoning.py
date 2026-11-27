"""Traditional: arithmetic/logic word problems and multiple-choice knowledge."""
from ..registry import suite
from ..scoring import extract_answer_number, extract_choice, num_equal

reasoning = suite("reasoning", "traditional", "Short math and logic problems with exact numeric answers, "
                                              "including classic trick questions.")

PROBLEMS = [
    ("bat_and_ball", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
                     "How much does the ball cost, in dollars?", 0.05),
    ("sheep", "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?", 9),
    ("widgets", "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take "
                "100 machines to make 100 widgets?", 5),
    ("lily_pads", "A patch of lily pads doubles in size every day. It takes 48 days to cover the whole lake. "
                  "On which day was the lake half covered?", 47),
    ("sisters", "Alice has 3 brothers and 2 sisters. How many sisters does Alice's brother have?", 3),
    ("multiply", "What is 17 multiplied by 24?", 408),
    ("train", "A train departs at 2:40 pm and arrives at 5:15 pm the same day. How many minutes is the trip?", 155),
    ("apples", "I have 3 apples today. Yesterday I ate one apple. How many apples do I have now?", 3),
    ("multiples", "What is the sum of all integers from 1 to 100 that are divisible by 3 or by 5?", 2418),
    ("strawberry", "How many times does the letter 'r' appear in the word 'strawberry'?", 3),
    ("discount", "A shirt costs $40 after a 20% discount. What was the original price in dollars?", 50),
    ("handshakes", "Eight people are at a party and every pair shakes hands exactly once. "
                   "How many handshakes happen?", 28),
    ("clock_angle", "What is the smaller angle, in degrees, between the hour and minute hands of a clock at 3:30?", 75),
    ("socks", "A drawer holds 10 black socks and 10 white socks. In the dark, what is the minimum number of socks "
              "you must take out to be certain you have a matching pair?", 3),
]

INSTRUCT = "Solve the problem. Think it through, then finish with a final line of the form 'ANSWER: <number>'."


def _numeric(question: str, answer: float):
    async def fn(ctx):
        reply = await ctx.ask(f"{question}\n\n{INSTRUCT}", temperature=0.0)
        got = extract_answer_number(reply.content)
        ctx.metric("expected", answer)
        ctx.metric("got", got)
        ctx.score(1.0 if num_equal(got, answer, 1e-3) else 0.0)
    fn.__doc__ = question
    return fn


for name, q, a in PROBLEMS:
    reasoning.add(name, _numeric(q, a), title=name.replace("_", " ").capitalize())


knowledge = suite("knowledge", "traditional", "Multiple-choice general knowledge across science, history, "
                                              "culture, music and computing.")

MCQ = [
    ("tungsten", "What is the chemical symbol for tungsten?", ["Tu", "W", "Tn", "Wo"], "B"),
    ("berlin_wall", "In what year did the Berlin Wall fall?", ["1987", "1991", "1989", "1985"], "C"),
    ("solitude", "Who wrote 'One Hundred Years of Solitude'?",
     ["Gabriel García Márquez", "Jorge Luis Borges", "Mario Vargas Llosa", "Isabel Allende"], "A"),
    ("binary_search", "What is the worst-case time complexity of binary search on a sorted array?",
     ["O(n)", "O(n log n)", "O(1)", "O(log n)"], "D"),
    ("not_prime", "Which of these numbers is NOT prime?", ["89", "97", "91", "83"], "C"),
    ("derivative", "What is the derivative of sin(x)?", ["-cos(x)", "cos(x)", "-sin(x)", "tan(x)"], "B"),
    ("australia", "What is the capital of Australia?", ["Sydney", "Melbourne", "Perth", "Canberra"], "D"),
    ("fifth", "How many semitones are in a perfect fifth?", ["5", "7", "6", "8"], "B"),
    ("mtdna", "Mitochondrial DNA in humans is inherited from:", ["The mother", "The father", "Both parents equally",
                                                                  "Neither; it arises de novo"], "A"),
    ("borrow_checker", "Which programming language is known for its 'borrow checker'?", ["Go", "Haskell", "Rust",
                                                                                         "Zig"], "C"),
    ("nyquist", "To sample a signal without aliasing, the sampling rate must exceed:",
     ["The signal's bandwidth", "Twice the highest frequency in the signal", "Half the highest frequency",
      "44.1 kHz"], "B"),
    ("photosynthesis", "In plant cells, photosynthesis takes place mainly in the:",
     ["Mitochondria", "Ribosomes", "Chloroplasts", "Golgi apparatus"], "C"),
]


def _mcq(question: str, options: list[str], answer: str):
    async def fn(ctx):
        body = "\n".join(f"{l}) {o}" for l, o in zip("ABCD", options))
        reply = await ctx.ask(f"{question}\n{body}\n\nRespond with only the letter of the correct answer.",
                              temperature=0.0)
        got = extract_choice(reply.content)
        ctx.metric("expected", answer)
        ctx.metric("got", got)
        probs = reply.option_probs(list("ABCD"))
        if probs and probs["_mass"] >= 0.5 and got:
            # Token probabilities: how sure it was of the letter it gave. Feeds the run's calibration summary.
            ctx.metric("option_probs", {k: v for k, v in probs.items() if not k.startswith("_")})
            ctx.metric("confidence", probs.get(got, 0.0))
            ctx.metric("p_correct", probs[answer])
            ctx.metric("correct", got == answer)
        ctx.score(1.0 if got == answer else 0.0)
    fn.__doc__ = question
    return fn


for name, q, opts, ans in MCQ:
    knowledge.add(name, _mcq(q, opts, ans), title=q[:60])
