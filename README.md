# agent-eval-lab

**Ship agents you can trust.** Build a small agent, then learn to test what it *does*: the tools it picks, the handoffs it makes, and the guardrails it keeps. Then apply the same tests to your own agent.

> **Status: in progress.** Built so far: 10-K extraction, hybrid retrieval with its own eval, the agent (OpenAI Agents SDK), a 36-case dataset, trace checks, an LLM judge, the mock track with planted failures, and a CI eval gate. Next: the full 36-case reference run, then the Codex lab. See [PLAN.md](PLAN.md) for the design and milestones.

## Why this exists

Most evals grade what an agent *says*. But an agent can give the right answer after calling the wrong tool, skipping a guardrail, or using 10x the tokens. Only the trace shows that.

```text
User: "What's Company X's debt-to-equity ratio?"

Agent A (good path)                     Agent B (bad path, same answer)
 1. retrieval → balance sheet passage    1. retrieval → balance sheet passage
 2. ratio_calc(debt, equity) → 1.42      2. (does the math in its head) → 1.42
 3. answer: 1.42                         3. answer: 1.42
```

An answer-only eval passes both agents. A trace check catches Agent B.

## What you'll learn

1. **Build** an agent with tools, a handoff, a guardrail, and tracing using the OpenAI Agents SDK.
2. **Explain** why answer-only evals miss agent failures, and point to the trace evidence.
3. **Write** trace-based checks for tool choice, handoffs, and guardrails, plus retrieval and groundedness checks.
4. **Gate** changes in CI, including changes made by a coding agent like Codex.
5. **Apply** the same process to your own agent.

## The scenario

A **credit research assistant** that answers questions about public companies' SEC 10-K filings. It has:

- a retrieval tool
- a ratio calculator
- a risk-summary agent it hands off to
- a guardrail that declines to give investment advice

Each part exists so one eval check has something to catch.

## Runs for free

| Tier | You need | Cost |
|---|---|---|
| **Mock** | Nothing (fully offline) | Free |
| **Workers AI** (default) | A free Cloudflare account | Free within the daily allowance |
| **OpenAI** | Your own API key | About $1–2 for a full run on a small model |

The same agent code runs on every tier. You switch with one flag: `--provider openai`.

Everything runs on your machine except the model calls, and nothing gets deployed. That keeps the lab free and easy to follow.

> **Running this in production?** The evals don't change. Swap local search for a vector store, add an AI gateway, and run the same CI gate against your deployed agent.

## Start here

Five minutes, no accounts or keys:

```bash
git clone https://github.com/ramirez-ai-labs/agent-eval-lab.git
cd agent-eval-lab
uv sync
uv run pytest                             # every test, offline
uv run python evals/retrieval_eval.py     # how well search finds the evidence
```

Then add a free Cloudflare token and run the agent itself. [SETUP.md](SETUP.md) walks through it, including what each error means.

**Coming with the workshop:**

| You are | Go to |
|---|---|
| A learner | The self-paced path |
| A facilitator | `workshop/FACILITATOR.md` |
| An engineer | `evals/` and `coding-agents/codex/` |
| A technical leader | `workshop/exec-briefing/` |
| A team adopting this | `workshop/ADAPT.md` |

## Provider status

| Provider | Status |
|---|---|
| Workers AI | Working: agent on `gpt-oss-120b` (Chat Completions), judge on `gemma-4-26b` |
| OpenAI | Agent code ready (Responses API); reference run planned |
| Mock | Working: replays the committed Workers AI reference run, offline, with planted failures. Grades every PR in CI |
| Anthropic | Planned (v1.1) |
| LangGraph | Planned (v1.2) |

## Related labs

- **Retrieval evaluation:** [rag-eval-lab](https://github.com/ramirez-ai-labs/rag-eval-lab) → [gcp-rag-eval-lab](https://github.com/ramirez-ai-labs/gcp-rag-eval-lab) → [snowflake-rag-eval-lab](https://github.com/ramirez-ai-labs/snowflake-rag-eval-lab)
- **Answer evaluation:** [ai-vic-eval](https://github.com/ramirez-ai-labs/ai-vic-eval), [chatbot-evaluation-system](https://github.com/ramirez-ai-labs/chatbot-evaluation-system)
- **Coding-agent evaluation:** [codex-evaluation-benchmark](https://github.com/ramirez-ai-labs/codex-evaluation-benchmark)
