"""CI/CD usage — non-interactive, JSON output, threshold override."""
import json
from simpltknopt import SimplTknOpt

stko = SimplTknOpt()

result = stko.run(
    "Summarize the key risks in a SOC 2 audit report",
    threshold=0.80,   # stricter quality requirement
    verify=True,      # verify every sub-task output
    interactive=False,
)

# Write machine-readable result
with open("stko_result.json", "w") as f:
    json.dump(json.loads(result.model_dump_json()), f, indent=2)

print(f"Cost: ${result.total_actual_cost_usd:.4f}")
print(f"Saved vs flagship: ${result.total_saved_vs_flagship:.4f}")
print("Output saved to stko_result.json")
