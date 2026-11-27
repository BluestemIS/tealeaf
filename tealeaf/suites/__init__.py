"""Importing this package registers every suite. Add a module here to add a suite."""
from . import (  # noqa: F401
    reasoning,       # traditional: reasoning + knowledge
    instructions,    # traditional
    coding,          # traditional
    reliability,     # traditional: hallucination, consistency, long_context, performance
    beliefs,         # probe
    sycophancy,      # probe
    self_reference,  # probe
    randomness,      # probe
    biases,          # probe
    conversations,   # probe
    weird_inputs,    # probe
    latent,          # probe: token probabilities vs what the model says
    faithfulness,    # probe: does the chain of thought drive the answer (llama.cpp for 2 of 3 tests)
    mood,            # probe: emotional tone (a steering sanity check)
)
