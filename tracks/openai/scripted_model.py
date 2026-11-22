"""A scripted stand-in for a real model, so the agent runs with no API calls.

Each step is what the "model" says on one turn:
    {"tool": "ratio_calc", "args": {...}}       call a tool (or a handoff, e.g.
                                                 "transfer_to_risk_agent")
    {"text": "The current ratio is 1.07."}      answer and stop

Everything else is real: the Agents SDK loop, the tools in core/, the
guardrail, the handoff, and the trace adapter. That makes this useful for
offline tests, and it's the basis of the mock track (M2), which will replay
recorded runs and planted failures.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

Step = dict[str, Any]


class ScriptedModel(Model):
    # Shared across instances: a run with a handoff uses several scripted
    # models, and the SDK requires every tool call ID in a run to be unique.
    _calls = 0

    def __init__(self, steps: list[Step], tokens_per_turn: tuple[int, int] = (500, 50)) -> None:
        self.steps = list(steps)
        self.turn = 0
        self.tokens_in, self.tokens_out = tokens_per_turn

    async def get_response(self, system_instructions, input, model_settings, tools, output_schema,
                           handoffs, tracing, *, previous_response_id=None, conversation_id=None,
                           prompt=None) -> ModelResponse:
        if self.turn >= len(self.steps):
            raise RuntimeError(f"script ran out after {self.turn} turns; the agent wanted another turn")
        step = self.steps[self.turn]
        self.turn += 1

        if "tool" in step:
            ScriptedModel._calls += 1
            n = ScriptedModel._calls
            item = ResponseFunctionToolCall(
                type="function_call", id=f"fc_{n}", call_id=f"call_{n}",
                name=step["tool"], arguments=json.dumps(step.get("args", {})), status="completed",
            )
        else:
            item = ResponseOutputMessage(
                type="message", id=f"msg_{self.turn}", role="assistant", status="completed",
                content=[ResponseOutputText(type="output_text", text=step["text"], annotations=[])],
            )
        usage = Usage(requests=1, input_tokens=self.tokens_in, output_tokens=self.tokens_out,
                      total_tokens=self.tokens_in + self.tokens_out)
        return ModelResponse(output=[item], usage=usage, response_id=None)

    def stream_response(self, *args, **kwargs) -> AsyncIterator:
        raise NotImplementedError("ScriptedModel is non-streaming, like the lab's Workers AI setup")
