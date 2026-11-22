# Setup

Three levels. Each one adds a little setup, and you can stop at any of them.

| Level | What you can do | You need | Time |
| --- | --- | --- | --- |
| **1. Offline** | Run every test, the retrieval eval, and the grader | Python and `uv` | 5 minutes |
| **2. Workers AI** (default) | Run the agent and the judge on open models, for free | A free Cloudflare account | +10 minutes |
| **3. OpenAI** | Run the same agent on OpenAI models | An OpenAI API key | +5 minutes |

## 1. Offline: no accounts, no keys

You need Python 3.12 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/ramirez-ai-labs/agent-eval-lab.git
cd agent-eval-lab
uv sync                                   # installs the exact versions in uv.lock
uv run pytest                             # every test runs offline
uv run python evals/retrieval_eval.py     # Recall@k, Hit@k and MRR for three search modes
```

This works with no credentials because the filings, the search index and the query embeddings are all committed. It's also exactly what CI runs on every pull request.

## 2. Workers AI: free open models

The agent runs on `@cf/openai/gpt-oss-120b` and the judge on `@cf/google/gemma-4-26b-a4b-it`, through Cloudflare Workers AI. The free tier covers this lab.

### Get your account ID and a token

1. Create a free account at [dash.cloudflare.com](https://dash.cloudflare.com/sign-up).
2. **Account ID:** it's the 32-character code in the dashboard URL after you log in (`dash.cloudflare.com/<account-id>/...`). The account home page also shows it. The ID isn't a secret, but it's yours, so keep it in `.env` too.
3. **API token:** go to **My Profile → API Tokens → Create Token → Create Custom Token**, and set:

   | Field | Value |
   | --- | --- |
   | Token name | `agent-eval-lab-workers-ai` |
   | Permissions | **Account · Workers AI · Read**, and **Account · Workers AI · Edit** |
   | Account resources | Include · your account |

   Give the token **only** the Workers AI permission. The *Edit Cloudflare Workers* template looks close, but it can deploy and change Workers on your account, and this lab never needs that.
4. Copy the token when Cloudflare shows it. It's shown once.

### Add them to `.env`

```bash
cp .env.example .env
```

Fill in `CF_ACCOUNT_ID` and `CF_API_TOKEN`. `.env` is gitignored: never commit it, and don't paste the token into chats, issues or screenshots.

### Check the connection

```bash
uv run python tracks/openai/check_endpoint.py --api chat
```

Two checks should pass: the agent calls the calculator tool, and it hands off to a second agent. This uses a few hundred Neurons.

### Run the agent, then grade it

```bash
uv run python -m tracks.openai.track       # 5 sample cases from the dataset
uv run python -m evals.judge --calibrate   # test the judge before trusting it
uv run python -m evals.run_checks runs/openai/workers-ai/<stamp>/traces.jsonl --judge
```

Traces are written to `runs/` on your machine. Hosted trace export is off, so nothing is sent to a tracing dashboard.

### The free allowance

Workers AI's free tier is **10,000 Neurons per day**, and it resets at **00:00 UTC** (4 PM Pacific in winter, 5 PM in summer). Approximate costs:

| Step | Neurons |
| --- | --- |
| Connection check | a few hundred |
| Agent, 5 sample cases | about 1,000 (estimate) |
| Judge calibration, 12 calls | a few hundred (estimate) |
| Agent, all 36 cases | about 7,000 (estimate): its own day |
| Rebuild the search index (`scenario/embed.py`) | about 3,300 (measured). You don't need to: the index is committed |

Judge verdicts are cached in `evals/cache/judge.json`, so re-grading a run costs nothing. If you use up the allowance, the lab tells you, and everything in level 1 keeps working.

## 3. OpenAI: the reference track

1. Add `OPENAI_API_KEY` to `.env`.
2. Set `agent.openai_model` in `config/models.yaml`. It ships as `TODO` until the reference run pins a model.
3. Run with `--provider openai`:

```bash
uv run python -m tracks.openai.track --provider openai
```

The agent code is the same. On OpenAI it uses the Responses API, and on Workers AI it uses Chat Completions ([PLAN.md §8](PLAN.md) explains why). A full run on a small model costs about $1–2.

## Rebuilding the filings (optional)

You only need this to change which filings or sections the lab uses. `scenario/extract.py` downloads the 10-Ks from SEC EDGAR, and EDGAR requires every client to identify itself:

```bash
SEC_USER_AGENT="Your Name you@example.com"   # in .env
uv run python scenario/extract.py
uv run python scenario/embed.py              # ~3,300 Neurons: at most once a day
```

## When something goes wrong

| You see | What it means | Fix |
| --- | --- | --- |
| `Missing CF_ACCOUNT_ID or CF_API_TOKEN` | `.env` is missing or empty | `cp .env.example .env` and fill it in |
| `Cloudflare rejected the API token` (401) | The token is wrong, or valid but without the Workers AI permission | Create a token with **Workers AI · Read** and **Edit** |
| `The token is valid but can't access this account` (403) | `CF_ACCOUNT_ID` belongs to a different account | Copy the account ID from the same account as the token |
| `The Workers AI free allowance … is used up` (429, code 4006) | You've used today's 10,000 Neurons | Wait for 00:00 UTC. Level 1 still works |
| `passages.jsonl changed since the vectors were built` | The filings were rebuilt, but the index wasn't | `uv run python scenario/embed.py` |
| `uv sync --locked` fails in CI | `pyproject.toml` changed without updating the lockfile | Run `uv lock` and commit `uv.lock` |
| `evidence not found in any passage` | A dataset label no longer matches the filings | Fix the evidence text in `evals/datasets/cases.jsonl`. This failure is on purpose: labels shouldn't drift silently |
