"""
utils/cost_tracker.py
-----------------------
Tracks token usage, cost, and latency for every LLM call in a graph run.

Implemented as a LangChain callback handler (CostTrackingCallback) rather
than manual tracker.track() calls around individual llm.invoke()s, because
this is a multi-agent graph: the supervisor makes its own LLM call, but so
does every ReAct step inside each specialist agent, and those are opaque
internals of create_react_agent. A callback hooked into on_llm_end catches
every one of them automatically, wherever in the graph they happen.

Pricing is USD per 1M tokens, last verified 2026-08.
"""

import time
from dataclasses import dataclass, field
from typing import Any
from langchain_core.callbacks import BaseCallbackHandler

MODEL_PRICING = {
    "gpt-4":        {"input": 30.00, "output": 60.00},
    "gpt-4o":       {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":  {"input": 0.15,  "output": 0.60},
    "gpt-4.1":      {"input": 2.00,  "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40,  "output": 1.60},
}

DEFAULT_MODEL = "gpt-4o-mini"  # agent orchestration makes many small calls; mini keeps a demo run cheap


def cost_for_tokens(input_tokens: int, output_tokens: int, model: str = DEFAULT_MODEL) -> float:
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o"])
    return input_tokens / 1_000_000 * pricing["input"] + output_tokens / 1_000_000 * pricing["output"]


@dataclass
class CallRecord:
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    cost_usd: float = field(init=False)

    def __post_init__(self):
        self.cost_usd = cost_for_tokens(self.input_tokens, self.output_tokens, self.model)


class CostTracker:
    def __init__(self):
        self.records: list[CallRecord] = []

    def add(self, stage: str, model: str, input_tokens: int, output_tokens: int, latency_seconds: float):
        self.records.append(CallRecord(
            stage=stage, model=model, input_tokens=input_tokens,
            output_tokens=output_tokens, latency_seconds=latency_seconds,
        ))

    def summary(self) -> dict:
        return {
            "total_cost_usd": round(sum(r.cost_usd for r in self.records), 5),
            "total_latency_seconds": round(sum(r.latency_seconds for r in self.records), 2),
            "total_input_tokens": sum(r.input_tokens for r in self.records),
            "total_output_tokens": sum(r.output_tokens for r in self.records),
            "n_llm_calls": len(self.records),
            "by_stage": [
                {"stage": r.stage, "model": r.model, "input_tokens": r.input_tokens,
                 "output_tokens": r.output_tokens, "cost_usd": round(r.cost_usd, 5),
                 "latency_seconds": round(r.latency_seconds, 2)}
                for r in self.records
            ],
        }

    def reset(self):
        self.records = []


class CostTrackingCallback(BaseCallbackHandler):
    """LangChain callback that records every LLM call's real token usage
    into a CostTracker. Attach via config={"callbacks": [callback]} on any
    graph.invoke() / agent.invoke() call -- it will catch every LLM call
    that happens anywhere inside that invocation, including ones buried
    inside prebuilt agents we don't otherwise have a hook into."""

    def __init__(self, tracker: CostTracker, stage_name: str = "llm_call"):
        self.tracker = tracker
        self.stage_name = stage_name
        self._start_times: dict[str, float] = {}

    def on_llm_start(self, serialized: dict, prompts: list[str], *, run_id, **kwargs: Any) -> None:
        self._start_times[str(run_id)] = time.perf_counter()

    def on_chat_model_start(self, serialized: dict, messages, *, run_id, **kwargs: Any) -> None:
        self._start_times[str(run_id)] = time.perf_counter()

    def on_llm_end(self, response, *, run_id, **kwargs: Any) -> None:
        start = self._start_times.pop(str(run_id), None)
        latency = time.perf_counter() - start if start is not None else 0.0

        model = DEFAULT_MODEL
        input_tokens = output_tokens = 0

        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage")
        model = llm_output.get("model_name", model)

        if not usage and response.generations:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            if msg is not None:
                usage_meta = getattr(msg, "usage_metadata", None)
                if usage_meta:
                    input_tokens = usage_meta.get("input_tokens", 0)
                    output_tokens = usage_meta.get("output_tokens", 0)
                response_meta = getattr(msg, "response_metadata", {}) or {}
                model = response_meta.get("model_name", model)

        if usage:
            input_tokens = usage.get("prompt_tokens", input_tokens)
            output_tokens = usage.get("completion_tokens", output_tokens)

        self.tracker.add(self.stage_name, model, input_tokens, output_tokens, latency)
