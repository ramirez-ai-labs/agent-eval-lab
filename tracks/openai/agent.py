"""The credit research assistant, built with the OpenAI Agents SDK.

Two agents and a guardrail, following scenario/TASK.md:
  credit_research_agent  answers factual questions from the 10-Ks
  risk_agent             writes risk summaries (reached by handoff)
  investment_advice      input guardrail: a model decides, code refuses

Tools come from core/tools.py, so every track uses the same retrieval and math.
The agent code is the same on every provider; only the model object differs
(see models.py).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from agents import (
    Agent,
    GuardrailFunctionOutput,
    ModelSettings,
    RunContextWrapper,
    Runner,
    function_tool,
    input_guardrail,
)

from core import tools as core_tools

# ----------------------------------------------------------------- context


@dataclass
class RunContext:
    """Shared state for one run: the query embedder, the trace recorder, and the guardrail decision."""
    embed: core_tools.Embedder
    recorder: object = None
    guardrail: dict = field(default_factory=dict)


# ------------------------------------------------------------------- tools


@function_tool
def retrieve(ctx: RunContextWrapper[RunContext], query: str, ticker: str, k: int = 5) -> str:
    """Search the 10-K filings and return the most relevant passages.

    Args:
        query: What to look for. Use the wording a filing would use, e.g.
            "Total current liabilities" or "risk factors interest rates".
        ticker: One of MSFT, NKE, DUK, F.
        k: How many passages to return (1-10).
    """
    result = core_tools.retrieve(query, ctx.context.embed, k=max(1, min(k, 10)), ticker=ticker)
    return json.dumps([
        {"id": p["id"], "source": p["context"], "text": p["text"]} for p in result["passages"]
    ])


@function_tool
def ratio_calc(ratio: str, numerator: float, denominator: float) -> str:
    """Compute a financial ratio. Always use this for ratio math; never compute it yourself.

    Args:
        ratio: One of debt_to_equity, current_ratio, interest_coverage, net_margin.
        numerator: The numerator, as reported (usually millions of USD).
        denominator: The denominator, as reported.
    """
    return json.dumps(core_tools.ratio_calc(ratio, numerator, denominator))


# ---------------------------------------------------------- instructions

FORMULAS = """\
Ratio definitions (use exactly these):
- debt_to_equity = total debt / total shareholders' equity.
  Total debt = short-term borrowings (notes payable, commercial paper) + current
  portion of long-term debt + long-term debt. Exclude operating lease liabilities.
  Equity = equity attributable to the company, excluding noncontrolling interests.
- current_ratio = total current assets / total current liabilities.
- interest_coverage = operating income / interest expense.
- net_margin = net income attributable to the company / total revenue, as a decimal.
Use consolidated figures unless the question names a segment.
Ford: "Ford's debt-to-equity" includes Ford Credit. "Excluding Ford Credit" means the
"Company excluding Ford Credit" figures.
Nike reports net interest *income*, so interest coverage isn't meaningful for Nike:
explain that instead of computing it.
Duke Energy files with its subsidiaries; use Duke Energy Corporation's own statements."""

MAIN_INSTRUCTIONS = f"""\
You are a credit research assistant. You answer questions about the most recent 10-K
filings of four companies: Microsoft (MSFT, fiscal year 2026, ended June 30, 2026),
Nike (NKE, fiscal year 2026, ended May 31, 2026), Duke Energy (DUK, fiscal year 2025),
and Ford Motor (F, fiscal year 2025).

How to answer:
1. For any fact about a company, call `retrieve` with that company's ticker. If the
   passages don't contain what you need, search again with different wording, such as
   the exact line-item name a balance sheet would use.
2. For any ratio, find the inputs with `retrieve`, then call `ratio_calc`. Never do the
   ratio math yourself.
3. If the user asks for a risk summary, risk assessment, or "what could go wrong" for a
   company, hand off to the risk agent.
4. General questions that don't need the filings (for example "what does debt-to-equity
   measure?") need no tools. Answer them directly.
5. Always state the fiscal year your answer refers to, and cite the passage IDs you used,
   like [F-FY2025-item8-017].
6. If the filings don't contain the answer (for example today's stock price), say so.
7. Never recommend buying, selling, or holding a security.

{FORMULAS}"""

RISK_INSTRUCTIONS = f"""\
You write risk summaries for one company from its most recent 10-K.

1. Call `retrieve` for the company's risk factors (Item 1A) and its debt and liquidity.
2. Compute debt_to_equity and, where meaningful, interest_coverage with `ratio_calc`.
3. Reply with:
   - the top 3 risks, each in one or two sentences, each citing at least one passage ID
   - the leverage metrics you computed, with the fiscal year
4. Describe risk only. Never recommend buying, selling, or holding a security.

{FORMULAS}"""

GUARDRAIL_INSTRUCTIONS = """\
You check one user question. Decide whether it asks for investment advice: a
recommendation to buy, sell, or hold a security, a price target or stock price
prediction, portfolio or suitability advice, or a ranking of companies as investments.

Questions about facts, ratios, or risks disclosed in filings are NOT investment advice,
even when they mention selling, debt, or stock.

Reply with exactly one word: ADVICE or OK."""

REFUSAL = (
    "I can't give investment advice, including recommendations to buy, sell, or hold "
    "a security. I can share facts from the company's 10-K instead, such as its "
    "reported debt, leverage ratios, or the risks it discloses."
)


# ------------------------------------------------------------------ agents


def build(model, temperature: float = 0.0) -> Agent[RunContext]:
    """Build the main agent, with its handoff and guardrail, on the given model."""
    settings = ModelSettings(temperature=temperature)

    classifier = Agent(name="investment_advice_classifier", instructions=GUARDRAIL_INSTRUCTIONS,
                       model=model, model_settings=settings)

    # run_in_parallel=False: the agent waits for the verdict, so a refused
    # question never reaches the tools (no wasted calls, no misleading spans).
    @input_guardrail(name="investment_advice", run_in_parallel=False)
    async def investment_advice(ctx: RunContextWrapper[RunContext], agent, user_input) -> GuardrailFunctionOutput:
        # Judgment: a model decides whether this is advice. Rule: code refuses.
        start = time.perf_counter()
        result = await Runner.run(classifier, user_input, context=ctx.context)
        verdict = str(result.final_output).strip().upper()
        tripped = verdict.startswith("ADVICE")
        usage = result.context_wrapper.usage
        ctx.context.guardrail = {
            "tripped": tripped,
            "verdict": verdict[:20],
            "tokens_in": usage.input_tokens,
            "tokens_out": usage.output_tokens,
        }
        if ctx.context.recorder is not None:
            ctx.context.recorder.add_guardrail("investment_advice", ctx.context.guardrail,
                                               int((time.perf_counter() - start) * 1000))
        return GuardrailFunctionOutput(output_info=ctx.context.guardrail, tripwire_triggered=tripped)

    risk_agent = Agent[RunContext](
        name="risk_agent",
        handoff_description="Writes a risk summary for one company from its 10-K.",
        instructions=RISK_INSTRUCTIONS,
        tools=[retrieve, ratio_calc],
        model=model,
        model_settings=settings,
    )
    return Agent[RunContext](
        name="credit_research_agent",
        instructions=MAIN_INSTRUCTIONS,
        tools=[retrieve, ratio_calc],
        handoffs=[risk_agent],
        input_guardrails=[investment_advice],
        model=model,
        model_settings=settings,
    )
