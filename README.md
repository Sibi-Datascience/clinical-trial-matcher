---
title: Multi-Agent Clinical Trial Matcher
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# 🧬 Multi-Agent Clinical Trial Matcher

A LangGraph-based **agentic AI system** — not a simple RAG app — that takes a
free-text patient case note and autonomously coordinates three specialized
agents to find and score matching clinical trials from the live
[ClinicalTrials.gov](https://clinicaltrials.gov/data-api/api) API.

**Live demo:** _add your Hugging Face Space link here after deploying_

## Why this project is different from a typical "RAG chatbot"

Most GenAI portfolio projects retrieve a document and summarize it. This
project demonstrates **agentic orchestration**: multiple agents with
different responsibilities, structured (typed) hand-offs between them, a
real external tool call to a live government API, and a final reasoning
step that cross-references two independent data sources (patient data vs.
trial eligibility text) to produce a justified, scored output.

## Architecture

```
                 ┌────────────────────┐
  Patient        │  Agent 1            │   Structured Pydantic
  case note ───► │  The Extractor      │──► PatientProfile
                 │  (LLM structured    │    (age, diagnoses,
                 │   output)           │     biomarkers, meds...)
                 └────────────────────┘
                            │
                            ▼
                 ┌────────────────────┐
                 │  Agent 2            │   List[TrialSummary]
                 │  The Researcher     │──► fetched live from
                 │  (ClinicalTrials.gov│    ClinicalTrials.gov
                 │   API v2 tool call) │    API v2
                 └────────────────────┘
                            │
                            ▼
                 ┌────────────────────┐
                 │  Agent 3            │   List[TrialMatch]
                 │  The Evaluator      │──► match_score 0-100
                 │  (cross-references  │    + rationale +
                 │   criteria vs.      │    matched/unmatched
                 │   patient profile)  │    criteria
                 └────────────────────┘
                            │
                            ▼
                    Streamlit results UI
```

The pipeline is implemented as a `langgraph.graph.StateGraph` (`src/graph.py`)
with a shared, strongly-typed `AgentState` (`src/schemas.py`) — so every
hand-off between agents is validated data, not free-form text.

## Tech stack

| Layer            | Choice                                              |
|-------------------|------------------------------------------------------|
| Orchestration     | LangGraph (`StateGraph`)                             |
| LLM               | Groq (`llama-3.3-70b-versatile`) via `langchain-groq` |
| Structured output | Pydantic v2 + `with_structured_output`               |
| External tool     | ClinicalTrials.gov API v2 (no key required)           |
| UI                | Streamlit                                             |
| Dev/experiment    | Kaggle Notebook (`notebooks/kaggle_demo.ipynb`)       |
| Deployment        | Hugging Face Spaces                                   |

## Project structure

```
clinical-trial-matcher/
├── app.py                     # Streamlit entry point (HF Spaces runs this)
├── requirements.txt
├── .env.example
├── src/
│   ├── schemas.py             # Pydantic models: PatientProfile, TrialSummary, TrialMatch, AgentState
│   ├── tools.py                # ClinicalTrials.gov API v2 wrapper
│   ├── agents.py               # extractor_agent, researcher_agent, evaluator_agent
│   └── graph.py                 # LangGraph wiring (StateGraph)
└── notebooks/
    └── kaggle_demo.ipynb        # Dev/test the pipeline in Kaggle, no Streamlit needed
```

## Run it locally

```bash
git clone <your-repo-url>
cd clinical-trial-matcher
pip install -r requirements.txt
cp .env.example .env   # add your free Groq API key: https://console.groq.com/keys
streamlit run app.py
```

## Develop and test in Kaggle first

1. Upload `notebooks/kaggle_demo.ipynb` to a new Kaggle Notebook.
2. Add your `GROQ_API_KEY` as a Kaggle **Secret** (Add-ons → Secrets).
3. Run all cells — this installs dependencies, defines the same
   `schemas.py` / `tools.py` / `agents.py` / `graph.py` logic inline, and
   runs the full pipeline against a sample case note, printing the
   extracted profile, fetched trials, and scored matches.
4. Once you're happy with the prompts/logic, copy any tweaks back into the
   `src/` files before deploying.

## Deploy to Hugging Face Spaces

1. Create a new Space → SDK: **Streamlit**.
2. Push this entire folder's contents to the Space's repo (the YAML block
   at the top of this README configures the Space automatically).
3. In the Space's **Settings → Repository secrets**, add `GROQ_API_KEY`.
4. The Space will build and launch `app.py` automatically.

```bash
# from inside this project folder
git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
git push space main
```

## Example input

```
62-year-old female with Stage IIIB non-small cell lung cancer (adenocarcinoma),
EGFR exon 19 deletion positive, ECOG performance status 1. Previously treated
with carboplatin/pemetrexed, progressed after 4 cycles. No prior immunotherapy.
Currently residing in Boston, Massachusetts.
```

The Extractor turns this into a typed `PatientProfile`, the Researcher
queries ClinicalTrials.gov for recruiting NSCLC trials near Boston, and the
Evaluator scores each trial's eligibility text against the profile with a
short rationale.

## Design notes / talking points for interviews

- **Structured hand-offs, not string concatenation.** Each agent returns a
  validated Pydantic object, so a downstream agent can never receive
  malformed data silently — the graph fails loudly if a contract is broken.
- **Real external tool use.** The Researcher agent calls a live, unauthenticated
  government API — not a static or mocked dataset — so the demo is
  genuinely dynamic.
- **Explainability by construction.** The Evaluator is prompted to cite
  matched/unmatched criteria drawn only from the trial's own text, which
  keeps scores auditable and reduces hallucinated eligibility rules.
- **Extensible graph.** Because orchestration is a `StateGraph` rather than
  a linear script, adding a conditional retry edge (e.g. broaden the search
  if zero trials are found) or a human-in-the-loop approval node is a small
  , additive change.

## Limitations & disclaimer

This is a portfolio/demo project. It is **not** validated for clinical use,
does not constitute medical advice, and should not be used to make real
patient-care or trial-enrollment decisions. Eligibility text parsing is
LLM-based and can miss nuance that a clinical research coordinator would
catch.
