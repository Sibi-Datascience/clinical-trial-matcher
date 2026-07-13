"""
The three specialized agents. Each is a plain function that takes an
AgentState and returns an updated AgentState — this is the LangGraph node
contract. Keeping agents as pure functions (not classes) makes the graph
easy to test node-by-node.
"""
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from .schemas import AgentState, PatientProfile, TrialMatch
from .tools import search_clinical_trials


def get_llm(temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Agent 1 — The Extractor
# ---------------------------------------------------------------------------
EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a clinical NLP specialist. Extract structured patient data from a "
            "free-text clinical case note. Be conservative: only include what is explicitly "
            "stated or very strongly implied. Normalize diagnosis and biomarker names to "
            "standard oncology / medical terminology (e.g. 'lung cancer' -> 'non-small cell "
            "lung cancer' only if the note supports that specificity).",
        ),
        ("human", "Case note:\n\n{case_note}"),
    ]
)


def extractor_agent(state: AgentState) -> AgentState:
    llm = get_llm()
    structured_llm = llm.with_structured_output(PatientProfile)
    chain = EXTRACT_PROMPT | structured_llm
    patient: PatientProfile = chain.invoke({"case_note": state.case_note})
    state.patient = patient
    state.log.append(
        f"✅ Extractor: found {len(patient.diagnoses)} diagnosis(es), "
        f"{len(patient.biomarkers)} biomarker(s), {len(patient.medications)} medication(s)"
    )
    return state


# ---------------------------------------------------------------------------
# Agent 2 — The Researcher
# ---------------------------------------------------------------------------
def researcher_agent(state: AgentState) -> AgentState:
    trials = search_clinical_trials(state.patient, max_results=10)
    state.trials = trials
    if trials:
        state.log.append(f"🔎 Researcher: fetched {len(trials)} recruiting trial(s) from ClinicalTrials.gov")
    else:
        state.log.append("🔎 Researcher: no matching recruiting trials found on ClinicalTrials.gov")
    return state


# ---------------------------------------------------------------------------
# Agent 3 — The Evaluator
# ---------------------------------------------------------------------------
EVAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a clinical trial eligibility evaluator. Compare the patient profile "
            "against a single trial's eligibility criteria text. Score match_score from 0-100. "
            "List matched_criteria and unmatched_criteria as short bullet strings drawn only "
            "from the eligibility text provided. Give a one-line recommendation: 'Strong match', "
            "'Possible match', or 'Not eligible'. Be conservative and evidence-based — never "
            "invent eligibility rules that are not present in the text.",
        ),
        (
            "human",
            "PATIENT PROFILE:\n{patient}\n\n"
            "TRIAL {nct_id} — {title}\n"
            "ELIGIBILITY CRITERIA:\n{eligibility}",
        ),
    ]
)


def evaluator_agent(state: AgentState) -> AgentState:
    if not state.trials:
        state.log.append("🧠 Evaluator: skipped (no trials to evaluate)")
        return state

    llm = get_llm()
    structured_llm = llm.with_structured_output(TrialMatch)
    chain = EVAL_PROMPT | structured_llm

    matches = []
    for trial in state.trials:
        match: TrialMatch = chain.invoke(
            {
                "patient": state.patient.model_dump_json(),
                "nct_id": trial.nct_id,
                "title": trial.title,
                "eligibility": trial.eligibility_text or "No eligibility text available.",
            }
        )
        match.nct_id = trial.nct_id
        match.title = trial.title
        matches.append(match)

    matches.sort(key=lambda m: m.match_score, reverse=True)
    state.matches = matches
    state.log.append(f"🧠 Evaluator: scored {len(matches)} trial(s)")
    return state
