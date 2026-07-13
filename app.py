"""
Multi-Agent Clinical Trial Matcher — Streamlit front-end.

This is the entry point Hugging Face Spaces will run. It calls the
LangGraph pipeline (src/graph.py) which chains three agents:
Extractor -> Researcher -> Evaluator.
"""
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.graph import run_pipeline  # noqa: E402

st.set_page_config(page_title="Clinical Trial Matcher", page_icon="🧬", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🧬 About")
    st.write(
        "A multi-agent system that reads a free-text patient case note, "
        "extracts structured clinical entities, searches live ClinicalTrials.gov "
        "data, and scores each trial's eligibility match."
    )
    st.markdown("**Pipeline**")
    st.markdown("1. 🧾 **Extractor** — LLM structured output\n"
                "2. 🔎 **Researcher** — ClinicalTrials.gov API v2\n"
                "3. 🧠 **Evaluator** — eligibility scoring + rationale")
    st.divider()
    st.markdown("Built with **LangGraph** + **Groq** + **Streamlit**")
    st.markdown("[ClinicalTrials.gov API docs](https://clinicaltrials.gov/data-api/api)")
    st.divider()
    st.caption(
        "⚠️ For demonstration purposes only. Not medical advice and not "
        "validated for clinical decision-making."
    )
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is not set. Add it in your Space's Settings → Secrets.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("Multi-Agent Clinical Trial Matcher")
st.caption("Patient case note → structured extraction → live trial search → eligibility scoring")

SAMPLE_NOTE = (
    "62-year-old female with Stage IIIB non-small cell lung cancer (adenocarcinoma), "
    "EGFR exon 19 deletion positive, ECOG performance status 1. Previously treated with "
    "carboplatin/pemetrexed, progressed after 4 cycles. No prior immunotherapy. "
    "Currently residing in Boston, Massachusetts. No significant cardiac or renal "
    "comorbidities."
)

col1, col2 = st.columns([4, 1])
with col1:
    case_note = st.text_area(
        "Patient case note",
        height=200,
        placeholder="Paste a de-identified patient case note here...",
    )
with col2:
    st.write("")
    st.write("")
    if st.button("Use sample note"):
        case_note = SAMPLE_NOTE
        st.session_state["case_note"] = SAMPLE_NOTE

if "case_note" in st.session_state and not case_note:
    case_note = st.session_state["case_note"]

run = st.button("🚀 Run Agent Team", type="primary", use_container_width=False)

if run:
    if not case_note or not case_note.strip():
        st.warning("Please paste a case note first.")
        st.stop()
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is not set — cannot call the LLM.")
        st.stop()

    with st.status("Running multi-agent pipeline...", expanded=True) as status:
        try:
            result = run_pipeline(case_note)
        except Exception as e:
            status.update(label="Pipeline failed", state="error")
            st.exception(e)
            st.stop()

        for line in result.log:
            st.write(line)
        status.update(label="Pipeline complete ✅", state="complete")

    st.subheader("🧾 Extracted Patient Profile")
    if result.patient:
        st.json(result.patient.model_dump())
    else:
        st.info("No structured profile was extracted.")

    st.subheader("🏆 Trial Matches")
    if not result.matches:
        st.info("No trials were found or scored for this patient profile.")
    else:
        for m in result.matches:
            badge = "🟢" if m.match_score >= 70 else "🟡" if m.match_score >= 40 else "🔴"
            with st.expander(f"{badge} {m.match_score}/100 — {m.title}  ({m.nct_id})"):
                st.markdown(f"**Recommendation:** {m.recommendation}")
                st.markdown(f"**Rationale:** {m.rationale}")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**✅ Matched criteria**")
                    for c in m.matched_criteria:
                        st.markdown(f"- {c}")
                with c2:
                    st.markdown("**❌ Unmatched criteria**")
                    for c in m.unmatched_criteria:
                        st.markdown(f"- {c}")
                st.markdown(f"[View full trial on ClinicalTrials.gov ↗](https://clinicaltrials.gov/study/{m.nct_id})")
