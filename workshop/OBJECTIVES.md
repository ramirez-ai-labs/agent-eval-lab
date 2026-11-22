# Learning objectives

What learners should be able to *do* after each part of the workshop. Every objective has a matching knowledge check in `ASSESSMENT.md`, so we can tell whether it was met.

## Course outcomes

By the end, learners can:

1. **Build** an agent with tools, a handoff, a guardrail, and tracing using the OpenAI Agents SDK.
2. **Explain** why answer-only evals miss agent failures, and point to the trace evidence.
3. **Write** trace-based checks for tool choice, handoffs, and guardrails, plus retrieval and groundedness checks.
4. **Gate** changes in CI, including changes made by a coding agent like Codex.
5. **Apply** the same process to their own agent.

## Lab 1 — Build the agent (20 min)

| # | Objective | Outcome |
|---|---|---|
| 1.1 | Add a tool to an agent and describe when the model should call it | 1 |
| 1.2 | Configure a handoff and a guardrail, and say what each one is for | 1 |
| 1.3 | Read a trace and name the step where each tool, handoff, and guardrail ran | 1, 2 |

## Lab 2 — Evaluate it and find the failure (25 min)

| # | Objective | Outcome |
|---|---|---|
| 2.1 | Run the eval suite and read the per-check report | 3 |
| 2.2 | Find a failure that the answer check passed but a trace check caught, and explain why | 2, 3 |
| 2.3 | Write one new trace check from a rule in `scenario/TASK.md` | 3 |

## Lab 3 — A coding agent changes it; the evals measure it (20 min)

| # | Objective | Outcome |
|---|---|---|
| 3.1 | Compare before/after eval reports and decide whether a change helped | 4 |
| 3.2 | Explain why a guardrail failure blocks a merge even when the overall pass rate went up | 4 |

## Take-home — Apply it to your own agent

| # | Objective | Outcome |
|---|---|---|
| T.1 | Write a short task spec for your own agent: tools, handoffs, refusal rules | 5 |
| T.2 | Write three checks for it, including at least one trace check | 3, 5 |
| T.3 | Run them in CI with a guardrail hard-fail | 4, 5 |
