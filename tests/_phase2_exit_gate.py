"""Phase 2 exit gate tests — run manually to verify live API calls."""

import importlib
import sys

import src.config
importlib.reload(src.config)

from src.llm.client import call_llm, _get_today_spend
from src.llm.models import DeepTriggerResponse
from src.llm.prompts import build_deep_trigger_prompt

question = "Will the US Federal Reserve cut interest rates before June 2026?"
mid_price = 0.62
news = "Fed Chair Powell signals data-dependent approach; CPI came in at 3.2%."
x_data = "Traders on X betting heavily on no cut before summer. Bond market pricing in 1 cut max."

spend_before = _get_today_spend()
print(f"Spend before deep calls: ${spend_before:.6f}")

successes = 0
for i in range(2):
    prompt = build_deep_trigger_prompt(question, mid_price, news, x_data)
    raw = call_llm(prompt, tier="deep")
    if raw is None:
        print(f"  [{i+1}] FAIL: returned None")
        continue
    try:
        r = DeepTriggerResponse(**raw)
        print(f"  [{i+1}] OK  delta={r.sentiment_delta:+.3f}  uncertainty={r.uncertainty_penalty:.3f}  factors={r.key_factors}")
        successes += 1
    except Exception as e:
        print(f"  [{i+1}] PARSE FAIL: {e} | raw={raw}")

spend_after = _get_today_spend()
print(f"Spend after deep calls:  ${spend_after:.6f}")
print(f"Cost of 2 deep calls:   ${spend_after - spend_before:.6f}")
print(f"Result: {successes}/2 deep trigger calls succeeded")

if successes < 2:
    sys.exit(1)
