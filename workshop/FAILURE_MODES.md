# Failure modes: what actually went wrong building this lab

These are real failures from building this lab, not invented examples. For each one: what it looked like, what caught it, and the fix. Most were caught by a check, not by reading code. That's the argument for evals in one page.

Use them in the workshop as "find the bug" exercises. Show the symptom, ask what check would catch it, then reveal what did.

## The patterns

1. **Check the labels before the agent.** A wrong label fails a correct agent and passes a wrong one. Two of the worst bugs here were in the reference data, not the agent.
2. **A real call catches what tests don't.** Three times, every test passed and the first real request failed. Tests only reach the code paths you thought of.
3. **The right number can come from the wrong process.** Only a trace shows how an answer was produced.
4. **Measure; don't trust the recommendation or the estimate.** A popular retrieval technique made things worse, and a cost estimate was wrong by a full run.
5. **When safety is undecided, it isn't a pass.** The gate has to say what happens while a judgment is pending.

---

## Data and labels

### Ford lists the oldest year first

| | |
| --- | --- |
| **Symptom** | Ford's FY2025 current ratio came out as **1.16**. The correct value is **1.07** |
| **Cause** | Microsoft, Nike and Duke list the newest year first. Ford's balance sheet runs `December 31, 2024 \| December 31, 2025`, so taking the first column gives FY2024 |
| **Caught by** | A dataset check that each ratio input sits in the column for the case's fiscal year (`tests/test_dataset.py`). With the wrong figures planted back in, it fails with `124,474 is in the wrong fiscal-year column` |
| **Fix** | Corrected the test that used 1.16, documented the column order in `scenario/TASK.md` §3, and kept two dataset cases that depend on it |
| **Lesson** | The first column of a financial table isn't always the latest year. An agent can make the same mistake, so the dataset now tests for it |

### Risk cases with half their evidence

| | |
| --- | --- |
| **Symptom** | The four risk-summary cases expected leverage ratios, but their evidence listed only risk-factor passages |
| **Caught by** | The same column check: `9,227 isn't in any evidence row`. The inputs for the expected ratios had nowhere to come from |
| **Fix** | Split evidence into `evidence` (all required: the balance-sheet rows) and `evidence_any` (alternatives: any one of several risk factors) |
| **Lesson** | A check written for one bug found a second one. Checks that cross-reference fields are cheap and catch a lot |

### Ford's Item 8 is a pointer

| | |
| --- | --- |
| **Symptom** | Ford's extracted financial statements were a few lines long |
| **Cause** | Ford's Item 8 just says the statements follow the signature pages |
| **Fix** | The extractor starts Ford's Item 8 at the last auditor's report heading |
| **Lesson** | Filings follow the rules loosely. Check extraction output before building on it |

### Words split by HTML tags

| | |
| --- | --- |
| **Symptom** | Text like `RIS K FACTORS`, which keyword search can't match |
| **Cause** | The filing styles letters in separate tags, and the extractor put a line break between every tag |
| **Fix** | Line breaks only at block-level tags; inline text joins with no separator |

### Passages too long to embed

| | |
| --- | --- |
| **Symptom** | 23 passages exceeded what the embedding model reads (about 2,000 characters), so their ends were never indexed |
| **Fix** | Paragraphs over the passage size are split at sentence boundaries. Table rows still stay whole |

---

## Retrieval

### A well-known technique made retrieval worse

| | |
| --- | --- |
| **Symptom** | Embedding each passage's context line ("Duke Energy · FY2025 · Item 8 · …") with its text dropped vector Recall@5 from **0.55 to 0.26** |
| **Cause** | About 800 passages share the same prefix, so their vectors bunched together and the content counted for less |
| **Caught by** | The retrieval eval (`evals/retrieval_eval.py`), run before and after |
| **Fix** | Index text only, and keep the context line for display and citations |
| **Lesson** | Commonly recommended doesn't mean right for your data |

### Duke's balance sheet can't be found (open)

| | |
| --- | --- |
| **Symptom** | No search mode retrieves Duke Energy Corporation's own balance-sheet rows |
| **Cause** | 431 Item 8 passages, including subsidiaries' balance sheets, and many passages about long-term debt. A chunk of bare numbers can't outrank them |
| **Status** | Open. Whether the agent recovers by searching again is a question for the agent-level evals, so it stays in the dataset |

---

## Provider and SDK

### Workers AI rejects function tools on the Responses API

| | |
| --- | --- |
| **Symptom** | `400` on the first tool call through the Responses API, even though Workers AI documents Responses for `gpt-oss` |
| **Caught by** | `tracks/openai/check_endpoint.py`: one tool-call check and one handoff check, against each API |
| **Fix** | On Workers AI the agent uses the Chat Completions model class. The agent code doesn't change |
| **Lesson** | "Supported" in the docs and "works with your SDK" aren't the same. Check the exact path you depend on |

### `content: null` rejected

| | |
| --- | --- |
| **Symptom** | `400` on the turn after every tool call |
| **Cause** | The SDK sends assistant messages that contain only tool calls with `"content": null`, the standard OpenAI format. Workers AI's Chat Completions rejects it |
| **Fix** | A small HTTP transport in `tracks/openai/workers_ai.py` rewrites `null` to `""` before the request is sent. It's the only Workers AI-specific code in the agent |

### The error explainer passed its test and failed for real

| | |
| --- | --- |
| **Symptom** | The used-up-allowance message showed a bare `Error code: 429` instead of the explanation |
| **Cause** | The unit test put Cloudflare's message in the exception text. The real client puts it in the response body |
| **Caught by** | Running against the real limit |
| **Fix** | The explainer reads the error text, the parsed body and the raw response |
| **Lesson** | A fake that's built from what you *expect* the error to look like tests your expectation, not the API |

### A dependency only one branch had

| | |
| --- | --- |
| **Symptom** | The judge crashed with `ImportError` instead of explaining the used-up allowance |
| **Cause** | It imported the error explainer, which existed only on an unmerged branch. No test reached the error path |
| **Caught by** | One real call, rejected for free because the allowance was used up |
| **Fix** | Use the explainer when it's there, and fall back to status plus message otherwise. The fallback leaves out the request URL, because it contains the Cloudflare account ID |

---

## Eval design

### The phrase list missed word forms and price predictions

| | |
| --- | --- |
| **Symptom** | "Nike **holds** short-term investments" and "I'd expect Nike to **trade around $90**" weren't flagged |
| **Cause** | `\bhold\b` doesn't match "holds", and price predictions weren't in the list |
| **Caught by** | A test that uses the phrase list *as* the judge on the calibration set. It should get exactly the trap cases wrong. It also got these wrong |
| **Fix** | Match word forms and price predictions, and err toward flagging. A flag never fails a case on its own; it only adds a review |
| **Lesson** | This is why the guardrail check uses a judge: you can't list every phrasing in advance |

### An advice answer that didn't block the build

| | |
| --- | --- |
| **Symptom** | "Honestly? Sell Nike." was marked *awaiting the judge*, and the build passed |
| **Caught by** | Running the grader end to end on a few planted traces and reading the output. The unit tests passed |
| **Fix** | An advice question that got an answer blocks the build until the judge rules. A legitimate answer that only uses a flagged word ("Ford may *sell* assets") goes on the review list without blocking |
| **Lesson** | Decide what "pending" means for a hard rule. For safety, undecided isn't a pass |

### The right number, the wrong process

| | |
| --- | --- |
| **Symptom** | An agent that does the arithmetic itself and states 1.07 |
| **Caught by** | `tool_choice` fails: no calculator call. `numeric_accuracy` passes, and an answer-only eval would too |
| **Lesson** | This is the lab's core example. Only the trace shows how an answer was produced |

---

## Test harness and operations

### Two fake models, one tool call ID

| | |
| --- | --- |
| **Symptom** | The SDK rejected a handoff test with a duplicate tool call ID |
| **Cause** | Each scripted model numbered its calls from 1, so the main agent and the risk agent both produced `call_1` |
| **Fix** | One counter shared by every scripted model |
| **Lesson** | Test doubles have bugs too. Here, the SDK's own validation caught it |

### A cost estimate that was off by a full run

| | |
| --- | --- |
| **Symptom** | The free allowance ran out mid-afternoon, with the agent's first live run still to do |
| **Cause** | Three full embedding runs at about 3,300 Neurons each (the plan estimated 3,000) used nearly all of the 10,000. A PR description had counted two runs |
| **Fix** | Record measured costs in `PLAN.md` and `SETUP.md`, rebuild the index at most once a day, and post a correction on the PR |
| **Lesson** | Measure costs on the real provider. On a free tier, being wrong means waiting until tomorrow |
