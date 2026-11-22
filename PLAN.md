# agent-eval-lab: Plan

**Ship agents you can trust. One scenario, one eval suite, any agent framework, and a free way to run all of it.**

Status: Planning. v1 target is ~10.5 working days.
Last updated: Nov 21, 2026

---

## 1. What this is

A hands-on workshop and self-paced lab for **evaluating AI agents**. Most evals grade what an agent *says*. This lab grades what it *does*.

Learners build one small agent, the test subject, and then spend most of the session measuring its *behavior*, not just its answers:

- Did it pick the right tool?
- Did it hand off at the right time?
- Did it respect its guardrails?
- What did it cost?

Everything is built around one realistic scenario: a **credit research assistant** that answers questions over public SEC 10-K filings. The agent is built with the **OpenAI Agents SDK**. The eval suite is provider-neutral, so the same checks work on any framework.

By default everything runs **for free**: on open models through Cloudflare Workers AI, or fully offline with the mock track. Switching to OpenAI models is one line in a config file.

## 2. Who it's for

| Audience | Format | What they leave with |
|---|---|---|
| **Developers building agents** | 90-minute workshop or self-paced path | A working agent, an eval suite, and a CI gate they can copy |
| **Platform and AI teams** | Labs + `evals/` + `MIGRATION.md` | A pattern for keeping evals in code and running them in CI |
| **Technical leaders and executives** | 15-minute briefing | How to tell whether an agent works, what it costs, and what could go wrong |
| **Facilitators** | `FACILITATOR.md` | Everything needed to run the session themselves |

**Prerequisites:** basic Python, Git, and a terminal. No API key needed. Labs run in a devcontainer or GitHub Codespaces.

## 3. What learners will be able to do

By the end, learners can:

1. **Build** an agent with tools, a handoff, a guardrail, and tracing using the OpenAI Agents SDK.
2. **Explain** why answer-only evals miss agent failures, and point to the trace evidence.
3. **Write** trace-based checks for tool choice, handoffs, and guardrails, plus retrieval and groundedness checks.
4. **Gate** changes in CI, including changes made by a coding agent like Codex.
5. **Apply** the same process to their own agent.

Each lab has 2–3 objectives in `workshop/OBJECTIVES.md`. Each objective has a matching knowledge check in `workshop/ASSESSMENT.md`.

## 4. Why evaluate agents this way

- **Agents fail in ways answer-only evals miss.** An agent can give a correct final answer after calling the wrong tool, skipping a guardrail, or using 10x the tokens. Only the trace shows it.
- **Evals belong in code.** OpenAI is [retiring its hosted Evals platform and Agent Builder](https://developers.openai.com/api/docs/deprecations) on Nov 30, 2026. Existing evals became read-only on Oct 31. OpenAI's recommended paths are:
  - Promptfoo for evals
  - the Agents SDK or ChatGPT Workspace Agents for Agent Builder workflows
  - application code for reusable prompts

  This lab shows the evals-in-code approach. `MIGRATION.md` follows OpenAI's own guidance. *(Switch to past tense after Nov 30.)*
- **Comparisons need a common yardstick.** The same scenario and the same checks across models and frameworks make results comparable.
- **Cost shouldn't block learning.** Three run tiers mean every learner can do every lab:
  - **mock:** no account needed
  - **Workers AI:** free Cloudflare account
  - **OpenAI:** your own API key

## 5. Design principles

| Principle | Why |
|---|---|
| Evals live in code, not a hosted dashboard | Version-controlled, portable, and they run in CI |
| Tracks adapt frameworks; checks only read traces | One suite judges every track the same way, and replay, offline runs, and cheap CI all follow from it |
| Shared tools live in `core/` | Retrieval and math are identical everywhere, so only agent behavior varies |
| Thin interfaces: `Agent`, `Tool`, `Trace` | Easy to learn, and easy to add a framework |
| Free by default | Nobody gets stuck on billing in the middle of a lab |
| Say what was verified, and when | Each provider shows its test status and date |
| Public data only | SEC filings, which are realistic and safe to share |
| Pinned versions + "tested on" date | Agent SDKs change fast; learners should know what works today |

## 6. The scenario: credit research assistant

A user asks questions about 3 to 5 public companies' 10-K filings.

| Component | What it does | What it teaches |
|---|---|---|
| Retrieval tool | Returns passages with stable IDs from committed 10-K extracts | Retrieval quality: Recall@k, MRR |
| Ratio calculator tool | Computes financial ratios with pinned formulas | Did the agent use the tool instead of guessing the math? |
| Risk summary agent (handoff) | A second agent writes a risk summary | Did the handoff fire when it should? |
| Guardrail | Declines to give investment advice | Safety behavior, the first thing enterprises ask about |
| Tracing | Logs every step in a common format | Gives the evals something to check besides the final answer |

`scenario/TASK.md` pins:
- **Filings:** company, fiscal year, SEC accession number. Only extracted sections are committed.
- **Ratio formulas:** for example, debt-to-equity = total debt / total shareholders' equity.
- **Refusal policy:** what counts as investment advice.

**Why finance:** it's a realistic enterprise use case, and the numbers have checkable right answers.

## 7. How it works

```
scenario/  ──indexed once──►  core/  (retrieval, ratio_calc, Trace schema, Agent/Tool protocols)
                                 │ same tools
          ┌──────────────────────┼──────────────────────┐
   tracks/openai           tracks/mock            tracks/anthropic (v1.1)
   Agents SDK code         replays recorded       tracks/langgraph (v1.2)
   provider: workers-ai    traces, incl.
          | openai         planted failures
          └──────────────────────┼──────────────────────┘
                                 ▼ normalized traces
                  runs/<track>/<provider>/<timestamp>/traces.jsonl
                                 ▼ read-only
             evals/checks  →  report.json + report.md  →  CI gate · Codex comparison · workshop
```

1. The runner loads a test case and calls the track.
2. The track runs the agent. All tools come from `core/`.
3. The track's adapter converts framework events into a common `Trace` and writes it to disk.
4. Checks read only the saved traces. The judge's results are cached, so reruns are free and repeatable.
5. The reporter prints pass rates per check. CI applies the thresholds.

**Core contracts**

```python
class Span(TypedDict):
    id: str; parent_id: str | None
    type: Literal["llm", "tool_call", "handoff", "guardrail"]
    name: str
    input: dict; output: dict          # retrieval output includes passage_ids
    tokens_in: int; tokens_out: int; latency_ms: int

class Trace(TypedDict):
    case_id: str; track: str; provider: str; model: str
    spans: list[Span]; final_answer: str; refused: bool

class Tool(Protocol):
    name: str; schema: dict
    def __call__(self, **kwargs) -> dict: ...

class Track(Protocol):
    name: str
    def run(self, question: str, case_id: str) -> Trace: ...
```

## 8. Providers and models

The same agent code runs three ways.

| Provider | How | Cost | Used for |
|---|---|---|---|
| **Workers AI** (default) | OpenAI's open-weight `gpt-oss-120b` on Cloudflare's OpenAI-compatible Chat Completions endpoint, through a small compatibility client (`tracks/openai/workers_ai.py`). Agent code is identical to the OpenAI provider; only the model class differs. | Free within 10,000 Neurons per day | Day-to-day runs, weekly live CI, live workshop tier |
| **OpenAI** | `api.openai.com` with the Responses API | A full run on a small model costs about $1–2 | Reference results; learners with their own key |
| **Mock** | Replays committed traces | Free, offline | PR checks in CI, offline workshop tier, Lab 2's planted failures |

Every model is pinned in `config/models.yaml`:

```yaml
provider: workers-ai            # workers-ai | openai | mock
agent_model: "@cf/openai/gpt-oss-120b"     # Chat Completions on Workers AI
openai_agent_model: <small OpenAI model snapshot>
judge_model: "@cf/google/gemma-4-26b-a4b-it"   # different family from the agent
judge_temperature: 0
embedding_model: "@cf/baai/bge-base-en-v1.5"   # vectors precomputed and committed
tracing: local                  # traces stay on disk
tested_on: <date>
```

| Role | Model | Notes |
|---|---|---|
| Agent + risk agent | `gpt-oss-120b` on Workers AI (fallback `gpt-oss-20b`), or a small OpenAI model | Function calling supported; 128k context |
| Judge (groundedness) | `gemma-4-26b-a4b-it` on Workers AI, a different family from the agent | Same judge for every track; results cached; 10 judgments checked by hand |
| Embeddings | `bge-base-en-v1.5` on Workers AI | Computed once and committed, so offline still works |
| Coding agent | Codex | Used in Lab 3 to make a change the evals then measure |

**Verification status (shown in the README)**

| Provider | Status |
|---|---|
| Workers AI | ✅ Live-tested on `<date>` |
| OpenAI | ✅ Live-tested on `<date>`, reference traces committed |
| Mock | ✅ Offline, deterministic |
| Anthropic (v1.1) | ⚙️ Implemented, contract-tested |

Provider paths that aren't run live in CI are covered by **contract tests**. Sample responses in each provider's format go through the adapter, which must produce a valid `Trace`.

**Confirmed Nov 21, 2026** (from [Workers AI OpenAI compatibility](https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/), [pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/), and the [gpt-oss-120b model page](https://developers.cloudflare.com/workers-ai/models/gpt-oss-120b/)):
- **Responses API:** supported only for `gpt-oss-120b` and `gpt-oss-20b`, and only with `stream: false`. Chat Completions works for most other text models.
- **Function calling:** `gpt-oss-120b` supports it.
- **Free allowance:** 10,000 Neurons per day per account, reset at 00:00 UTC. Free accounts can't go over it; requests fail instead.
- **Paid-only models:** some newer models (Kimi K2.6+, GLM 5.x, DeepSeek V4) need a paid plan, so they're excluded from the free tier.

**Free-tier budget (estimate).** Assumes about 4,500 input and 300 output tokens per agent run, and about 2,500 input and 200 output tokens per judge call. Real numbers come from the first baseline run.

| | Neurons per case | Full run (36 cases) | Full runs per day (free) |
|---|---|---|---|
| Agent on `gpt-oss-120b` + judge | ~190 | ~7,000 | 1 |
| Agent on `gpt-oss-20b` + judge | ~120 | ~4,300 | 2 |

What this means:
- **Weekly live CI:** one full run per case, not several. Repeated runs need a paid plan or have to be spread across days.
- **Workshop labs:** run a 10-case subset, about 1,900 Neurons. Each learner uses their own free account, so there's room for several reruns.
- **Agents SDK on Workers AI (verified Nov 21, 2026 with `tracks/openai/check_endpoint.py`):**

  | API | Tool call | Handoff |
  |---|---|---|
  | Responses | ❌ 400: Cloudflare rejects function tools | ❌ same |
  | Chat Completions | ✅ | ✅ |

  So Workers AI uses the **Chat Completions model class**. One compatibility fix is needed: Cloudflare rejects assistant messages with `"content": null`, which the SDK sends after every tool call. `tracks/openai/workers_ai.py` rewrites them to `""`. The OpenAI provider keeps the Responses API. Re-run the check whenever the SDK or Cloudflare changes.

## 9. The eval suite

| Check | Question it answers | Method |
|---|---|---|
| Recall@k, MRR | Did retrieval find the right passages? | Labeled passage IDs per question |
| Groundedness | Is the answer supported by the retrieved text? | LLM judge with a fixed rubric, cached |
| Numeric accuracy | Are the ratios correct? | Tolerance match against pinned formulas |
| Tool choice | Calculator used when needed, *not* when not needed, with correct arguments? | Trace check |
| Handoff | Did the risk agent run when a risk summary was asked for? | Trace check |
| Guardrail | Did it decline investment advice? | Behavior check first; guardrail span as extra evidence |
| Cost and latency | Tokens and seconds per run within budget? | Trace metadata vs budgets |

**Dataset:** 25 to 40 cases in `evals/datasets/*.jsonl`, including negative tool-choice and guardrail cases.

**CI gate**
- **On every PR:** the mock track. It's deterministic, free and fast.
- **Weekly:** a live Workers AI run, one run per case to stay inside the free allowance, compared against a baseline.
- **Hard rule:** any guardrail failure fails the build, because safety failures should never be averaged away.

## 10. The workshop: "Ship Agents You Can Trust" (90 minutes)

Each block follows the same order: **concept → hands-on practice → feedback from the evals → knowledge check.**

| Time | Block | Practice | Takeaway |
|---|---|---|---|
| 0–10 | Why evaluate agents before you scale them | `make check` confirms every learner's setup | Evals reduce cost, risk, and surprises |
| 10–30 | **Lab 1: Build the agent** | Fill in a tool, a handoff, and a guardrail | How agents are assembled, and what a trace shows |
| 30–55 | **Lab 2: Evaluate it and find the failure** | Run the suite; find the planted failure in the trace | Trace checks catch what answer checks miss |
| 55–75 | **Lab 3: A coding agent changes it; the evals measure it** | Compare before/after reports from a Codex change | Never trust a change you didn't measure |
| 75–85 | Failure modes and moving evals into code | Walk through `FAILURE_MODES.md` and `MIGRATION.md` | Keep evals portable |
| 85–90 | Recap and knowledge check | Short quiz; take-home checklist | A repeatable process for your own agents |

**Take-home exercise:** "Apply it to your own agent." Learners write three checks for an agent they own, scored with the rubric in `ASSESSMENT.md`.

**Formats**

| Format | Built from |
|---|---|
| In-person or live virtual workshop (90 min) | `FACILITATOR.md`, slides, labs |
| Self-paced course | README learning path; the same labs with written explanations and solutions |
| Executive briefing (15 min) | 5 slides: what the agent does, how you know it works, what it costs, what could go wrong, how to start |
| Hackathon extension (optional) | Challenge prompts: add a tool, add a track, beat the baseline on cost |

**Business outcomes in the executive briefing.** One slide puts real numbers from the eval runs into business terms:

| Evidence from the evals | Business question it answers |
|---|---|
| Cost and tokens per answer, by provider | What will this cost at our volume? |
| Failures caught by trace checks that answer checks missed | What risk did we avoid before production? |
| Guardrail pass rate | Can we trust it with customers and regulators? |
| CI gate on a Codex change | How do we adopt coding agents without slowing review or adding risk? |

**Path to production.** The workshop is step one of a three-step adoption path:

| Step | What the team does | Exit criteria |
|---|---|---|
| 1. Learn | Run the workshop or self-paced path on the credit research scenario | Knowledge checks passed; `make check` green |
| 2. Pilot | Apply the take-home exercise to one of their own agents: swap the scenario, write 3 checks | Their agent produces traces, and the checks run on them |
| 3. Operate | Add the CI gate to their own repo, with guardrail hard-fail and a weekly live run | Gate blocks a planted regression in their repo |

**Lab setup vs production.** The lab runs locally on purpose: free Workers AI models, local vector search over committed embeddings, and nothing deployed. That keeps cost at $0 and setup to one token, and everyone runs the same thing. A production system would usually look different:

| Piece | In the lab | In production |
|---|---|---|
| Agent | Python process on your machine | A deployed service |
| Vector search | Cosine similarity over committed vectors | A managed vector store |
| Model access | Direct API calls | Through an AI gateway (routing, rate limits, logging) |
| Evals | Same | **Same** |

> *Running this in production? The evals don't change. Swap local search for a vector store, add an AI gateway, and run the same CI gate against your deployed agent.*

**Adapting for your team.** `workshop/ADAPT.md` is a one-page enablement plan template for running this with a specific team:
- **Use case:** which scenario to swap in, and which tools, handoffs and guardrails it needs
- **Audience:** builders, platform team, leaders, or a mix
- **Format:** workshop, self-paced, briefing, hackathon
- **Run tier:** mock, Workers AI, or OpenAI, based on the team's accounts and data rules
- **Success metrics:** what the team should be able to do after each step of the path to production

**Lab environment**
- A devcontainer or Codespaces config gives a one-click setup.
- `make check` confirms dependencies and credentials, then picks the best available run tier (OpenAI, then Workers AI, then mock) automatically.
- `workshop/SETUP.md` covers handling API keys safely in shared sessions: environment variables only, never committed, revoked after the session.

## 11. Repo structure

```
agent-eval-lab/
├── README.md               # what, who, "Start here" paths, provider status
├── PLAN.md                 # this file
├── config/models.yaml
├── scenario/               # 10-K extracts, TASK.md, questions
├── core/                   # Agent, Tool, Trace; retrieval; ratio calculator
├── evals/
│   ├── datasets/
│   ├── checks/
│   ├── judge_cache/
│   ├── ci/
│   └── MIGRATION.md        # moving off hosted eval platforms
├── tracks/
│   ├── openai/             # v1: Agents SDK; providers workers-ai | openai
│   ├── mock/               # v1: offline replay + planted failures
│   ├── anthropic/          # v1.1
│   └── langgraph/          # v1.2
├── runs/                   # committed reference traces
├── coding-agents/codex/    # before/after eval comparison
├── .devcontainer/
├── Makefile                # make check, make eval, make report
└── workshop/
    ├── FACILITATOR.md
    ├── SETUP.md
    ├── OBJECTIVES.md
    ├── ASSESSMENT.md
    ├── FAILURE_MODES.md
    ├── ADAPT.md            # enablement plan template for a specific team
    ├── ITERATION_LOG.md    # per-session metrics, feedback, and what changed
    ├── labs/               # 01–03, each with starter/ and solution/
    ├── slides/
    └── exec-briefing/
```

**README "Start here"**
- **Learners:** the self-paced path
- **Facilitators:** `workshop/FACILITATOR.md`
- **Engineers:** `evals/` and `coding-agents/codex/`
- **Leaders:** `workshop/exec-briefing/`
- **Teams adopting it:** `workshop/ADAPT.md` and the path to production

## 12. Milestones (v1)

### M0 — Foundations (0.5 day)
- Trace schema with an example trace
- `scenario/TASK.md`
- `config/models.yaml`
- `workshop/OBJECTIVES.md`
- Confirm the Workers AI facts from §8
- Link OpenAI's deprecation notice

**Done when:** schema, task spec, model config and learning objectives are committed.

### M1 — Core + agent (2 days)
- `core/`: interfaces, ratio calculator, retrieval with precomputed embeddings
- `tracks/openai/` with the `workers-ai` and `openai` providers, plus the trace adapter
- Contract tests
- `.devcontainer/`, `make check`, `workshop/SETUP.md`

**Done when:** the agent answers 5 sample questions live on Workers AI, writes valid traces, and `make check` passes in a fresh Codespace.

### M2 — Reference run + mock track (0.5 day)
- One full OpenAI run on a small model, traces committed
- `tracks/mock/` replays them, with 1–2 planted failures
- Judge cache

**Done when:** every check runs with no credentials and catches the planted failures.

### M3 — Eval suite + CI (1.5 days)
- Dataset, checks, runner, reporter
- PR gate on mock, weekly live run, guardrail hard-fail

**Done when:** pass rates print for every provider, and CI fails on a planted regression.

### M4 — Codex lab (1 day)
- Codex adds one feature on a branch
- The before/after reports are compared in a table
- The diff is committed so Lab 3 works without live Codex

**Done when:** running the suite on both branches produces the comparison table.

### M5 — Workshop materials (3 days)
- `FACILITATOR.md`, labs, slides, `FAILURE_MODES.md`, `MIGRATION.md`
- `ASSESSMENT.md` with knowledge checks and the take-home rubric
- Executive briefing, including the business-outcomes slide built from real eval numbers
- Path to production section and `ADAPT.md`
- Self-paced path

**Done when:** someone else can run the session from `FACILITATOR.md` alone, and a learner can finish it alone from the README.

### M6 — Pilot and launch (1 day)
- Dry run with timing, then adjust
- 3-question post-session survey
- `ITERATION_LOG.md` started, with one entry per session:
  - attendees
  - `make check` pass rate at the start
  - time per lab (planned vs actual)
  - knowledge-check scores
  - survey themes
  - what changed because of them
- Repo public, "Share your results" issue template, cross-links to related labs

**Done when:** labs fit their time slots, the first log entry is recorded, and the repo is public.

**Total: ~10.5 working days.** If time runs short, the hackathon extension and the self-paced polish move to v1.0.1 first.

## 13. After v1

| Version | Adds | Why |
|---|---|---|
| v1.0.1 | Walkthrough video, written tutorial, hackathon challenge pack | More formats for self-paced learners |
| v1.1 | Anthropic track; optional Cloudflare AI Gateway in front of all providers; `infra/terraform/` to provision the Workers AI token, gateway and rate limits for a cohort | Shows the suite is provider-neutral; makes setup for a cohort repeatable, with usage visibility for shared sessions |
| v1.2 | LangGraph track, cross-track comparison table, second judge with agreement rate | Compares frameworks on the same yardstick |

**Non-goals:** a general agent framework, a hosted dashboard, a leaderboard, fine-tuning.

## 14. How we'll know it's working

| Signal | Target |
|---|---|
| Learners can apply it | Knowledge-check scores; take-home rubric results from the pilot |
| Learners can set it up | `make check` pass rate at the start of each session |
| It improves with use | Survey results and at least one change per session in `ITERATION_LOG.md` |
| Labs fit their time | Actual vs planned minutes per lab, tracked across sessions |
| Learners move past the workshop | Share of attendees who reach step 2 (pilot) of the path to production |
| Others can teach it | At least one session run by someone other than the author |
| People use it on their own agents | "Share your results" issues |
| It stays current | Weekly CI green; "tested on" date less than 30 days old |

## 15. Risks

| Risk | Mitigation |
|---|---|
| Setup problems eat lab time | Devcontainer; `make check` in the first 10 minutes; automatic mock fallback |
| Free-tier limits hit mid-workshop | Labs use a 10-case subset (~1,900 Neurons); `gpt-oss-20b` fallback; mock fallback; optional gateway rate limits (v1.1) |
| Workers AI compatibility breaks after an SDK or Cloudflare update | Pinned versions; `check_endpoint.py` runs in the weekly live CI; all Workers AI fixes live in one file (`workers_ai.py`) |
| SDK or API changes break labs | Pinned versions, "tested on" date, weekly CI |
| Untested provider paths stop working | Contract tests in CI; honest status table |
| Open models are weak at tool calling | Choose a function-calling model; use real failures as Lab 2 material; OpenAI reference run for comparison |
| LLM judge is inconsistent or biased | Fixed rubric, temperature 0, judge from a different family than the agent, cache, hand spot-check; second judge in v1.2 |
| Flaky CI from nondeterminism | PR gate on mock; live runs use repeats and a baseline |
| Scope creep into building a framework | Interfaces stay at `Agent`, `Tool`, `Trace` |
| API keys leak in shared sessions | `SETUP.md` key-handling rules; `.env` in `.gitignore`; keys revoked after the session |

## 16. Related labs

- **Retrieval evaluation:** `rag-eval-lab` → `gcp-rag-eval-lab` → `snowflake-rag-eval-lab`
- **Answer evaluation:** `ai-vic-eval`, `chatbot-evaluation-system`
- **Coding-agent evaluation:** `codex-evaluation-benchmark`
- **Agent evaluation:** `agent-eval-lab` (you are here)
