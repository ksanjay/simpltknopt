"""Quickstart: 5-line SimplTknOpt usage."""
from simpltknopt import SimplTknOpt

stko = SimplTknOpt()
result = stko.run("Write a Python function that parses CSV and returns a list of dicts")
print(result.combined_output)
print(f"\nTotal cost: ${result.total_actual_cost_usd:.4f}")
