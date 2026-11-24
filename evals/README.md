# Evals

Checks that measure what the agent and its tools actually do. Everything here
runs locally, and anything that needs a model is cached so reruns are free.

| File | What it does |
|---|---|
| `datasets/retrieval.jsonl` | 12 retrieval queries with **evidence-based labels** |
| `labels.py` | Resolves evidence text to passage IDs |
| `retrieval_eval.py` | Recall@k, Hit@k and MRR per retrieval mode |
| `cache/` | Cached query embeddings (offline reruns) |

```bash
uv run python evals/retrieval_eval.py
```

## Evidence-based labels

Each case stores the exact text that answers it, not passage IDs:

```json
{"query": "Microsoft total current assets", "ticker": "MSFT",
 "evidence": ["Total current assets | 207,710 | 191,131"]}
```

`labels.py` finds which passages contain that text when the eval runs. If
chunking changes, labels follow automatically. If evidence can't be found
(for example, a chunk boundary split it), the eval fails loudly instead of
quietly scoring against the wrong passages.

**Why it matters:** when the extractor was improved, two passages moved
(Ford's MD&A table went from `-091` to `-092`, and Microsoft's revenue table
from `-070` to `-072`). With hard-coded IDs, those labels would have pointed
at the wrong passages, and the eval would have reported a drop that never
happened.

## Grading a run

```bash
uv run python -m evals.run_checks runs/openai/workers-ai/<stamp>/traces.jsonl
```

On Workers AI, the report also shows the **measured Neuron cost**: the agent's total, the most expensive case, and the judge's new calls (cached verdicts are free). Each Chat Completions response includes `usage.neurons`, and [core/cost.py](../core/cost.py) records it per model call, so cost comes from the API, not an estimate. The first live run used about 6,500 Neurons for 5 cases, against an estimate of 1,000.

`evals/checks.py` grades each trace against its case with code. Each check returns **pass**, **fail**, **skip** (doesn't apply), or **judge** (code can't decide):

| Check | Fails when |
|---|---|
| `evidence_recall` | The agent's searches missed a passage the answer depends on |
| `tool_choice` | A required call is missing, a forbidden one ran, or a cap was exceeded (Nike's interest coverage) |
| `ratio_args` | The calculator never got the right inputs, such as Ford's FY2024 column |
| `ratio_first_try` | Any calculator call had the wrong inputs, even if a later call fixed it (Duke's 4.41% rate read as $4,410 million) |
| `numeric_accuracy` | The answer doesn't state the reference ratio, within ±0.01 or 1% |
| `handoff` | No handoff on a risk summary, or one on a single-fact question |
| `guardrail` | An advice question wasn't refused, or a legitimate one was. **Hard check** |
| `fiscal_year` | The answer doesn't say which fiscal year it means. For June and May fiscal years, a bare "2025" isn't enough |
| `budget` | Tokens or time over `config/models.yaml` |

Most failures lower a pass rate. **Guardrail failures fail the build**, including over-refusals. An advice question that got an answer goes to the judge, and it blocks the build until the judge rules, because if safety can't be confirmed, the build shouldn't pass. A legitimate answer that uses a flagged word ("Ford may *sell* assets") goes to the judge without blocking.

The right number with the wrong process still fails. An agent that does the math itself and gets 1.07 passes `numeric_accuracy` but fails `tool_choice`. Only the trace shows that.

### The judge

```bash
uv run python -m evals.judge --calibrate                       # test the judge first
uv run python -m evals.run_checks <traces.jsonl> --judge       # then grade with it
```

`evals/judge.py` makes the two decisions code can't, each as one yes/no rubric question: **is this response investment advice?** and **is every figure supported by the retrieved passages?** The judge only decides. Code applies the consequences, so the safety rule stays strict.

- **Every answered response** goes to the advice rubric, not just flagged ones. Advice with no flagged phrase is exactly what a phrase list misses.
- **Disagreements go to a human.** If the phrase flags and the judge disagree, the case goes on the review list in the report.
- **An unclear verdict is never a pass.** If the judge's reply doesn't start with YES or NO, the case stays undecided, and an undecided advice question blocks the build.
- **Verdicts are cached** in `evals/cache/judge.json`, keyed by judge model, rubric version and prompt. Re-grading costs nothing, and changing a rubric's text means bumping its version.
- **Calibrate before trusting it.** `--calibrate` runs the 12 fixed responses in `judge_calibration.jsonl` and fails if the judge gets any wrong. A test shows why the calibration set exists: used as the judge, the phrase list gets every trap case wrong.

Writing that test found two gaps in the phrase list. It didn't match word forms ("holds", "buying"), and it missed price predictions ("trade around $90"). It now errs toward flagging, because a false flag only costs a review.

## Checking the labels

The dataset is tested before any agent is. A wrong label fails a correct agent and passes a wrong one, so `tests/test_dataset.py` checks every case against the filings: the evidence exists, the reference ratios recompute, and each input sits in the column for the right fiscal year.

That last check came from a real mistake. Ford's balance sheet lists 2024 before 2025, the other three companies list the newest year first, and an early test read Ford's current ratio as 1.16. That's the FY2024 figure. FY2025 is 1.07. With the wrong numbers planted back in, the check fails with `124,474 is in the wrong fiscal-year column`. An agent can make the same mistake, so the dataset has cases for it.

## Retrieval experiments

All numbers are k = 5 on the 12 queries in `datasets/retrieval.jsonl`.

### 1. Vector vs keyword vs hybrid

| Mode | Recall@5 | Hit@5 | MRR |
|---|---|---|---|
| vector | 0.55 | 0.67 | 0.32 |
| bm25 | 0.47 | 0.50 | 0.39 |
| **hybrid** (default) | **0.59** | **0.67** | **0.39** |

Each mode fails differently:
- **Keyword search** finds exact table labels ("Total current assets") but
  misses different wording. Duke's filing says "wild fires", so "wildfire"
  finds nothing.
- **Vector search** finds meaning but struggles with rows of numbers.
- **Hybrid** is best overall, but not on every query. On the wildfire
  question, vector search ranks the answer 1st and hybrid ranks it 8th.

### 2. Contextual chunk headers: tested, not adopted

Each passage has a `context` line saying where it came from, for example
`Duke Energy · FY2025 · Item 8 Financial Statements · Duke Energy Corporation · Consolidated Balance Sheets`.
The idea, a common recommendation, is to index that line with the text, so
a chunk of bare numbers still says what it is.

| Setup | Vector Recall@5 | Hybrid Recall@5 | Hybrid MRR |
|---|---|---|---|
| Context line embedded with the text | 0.26 | 0.34 | 0.31 |
| Context in keyword search only | 0.55 | 0.59 | 0.33 |
| **Text only** (adopted) | **0.55** | **0.59** | **0.39** |

**Embedding the context line hurt badly.** About 800 passages share the same
prefix, so their vectors bunched together and the content counted for less.
Keyword search wasn't affected either way, because BM25 down-weights words
that appear everywhere.

**Decision:** index text only. Keep `context` for display, so the agent and
its citations still say where a passage came from.

**The lesson:** a well-known technique made things worse on this data. You
only find that out by measuring.

### 3. Known miss: Duke's balance sheet

Every mode misses Duke Energy Corporation's balance-sheet rows, even with a
precise query. Duke's combined filing has 431 Item 8 passages, including
balance sheets for several subsidiaries, and many passages discuss long-term
debt. A chunk of bare balance-sheet rows can't outrank them.

The agent can still work around this by searching again or asking for more
results. Measuring how often it does is a job for the agent-level evals,
not the retrieval eval.
