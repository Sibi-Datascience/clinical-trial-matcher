"""
Multi-Agent Clinical Trial Matcher — Streamlit front-end.

This is the entry point Streamlit Cloud (or Hugging Face Spaces) runs.
It calls the LangGraph pipeline (src/graph.py) which chains three agents:
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
        st.error("GROQ_API_KEY is not set. Add it in your app's Settings → Secrets.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("Multi-Agent Clinical Trial Matcher")
st.caption("Patient case note → structured extraction → live trial search → eligibility scoring")

SAMPLE_NOTES = {
    "Lung cancer (NSCLC, EGFR+)": (
        "62-year-old female with Stage IIIB non-small cell lung cancer (adenocarcinoma), "
        "EGFR exon 19 deletion positive, ECOG performance status 1. Previously treated with "
        "carboplatin/pemetrexed, progressed after 4 cycles. No prior immunotherapy. "
        "Currently residing in Boston, Massachusetts. No significant cardiac or renal "
        "comorbidities."
    ),
    "Breast cancer (HER2+)": (
        "45-year-old female with Stage II invasive ductal carcinoma of the breast, "
        "HER2-positive, ER-negative, PR-negative. ECOG performance status 0. Completed "
        "neoadjuvant chemotherapy (docetaxel, carboplatin, trastuzumab, pertuzumab) with "
        "partial response. Underwent lumpectomy. No prior history of cardiac disease. "
        "Currently residing in Chicago, Illinois."
    ),
    "Type 2 diabetes": (
        "58-year-old male with a 10-year history of type 2 diabetes mellitus, HbA1c 8.9%, "
        "on metformin and insulin glargine, poorly controlled. BMI 33. History of mild "
        "diabetic peripheral neuropathy. No history of diabetic ketoacidosis. Currently "
        "residing in Austin, Texas. No known cardiovascular events."
    ),
    "Colorectal cancer (KRAS mutant)": (
        "70-year-old male with Stage IV metastatic colorectal cancer, KRAS G12C mutation "
        "positive, microsatellite stable (MSS). ECOG performance status 1. Liver metastases "
        "present. Previously treated with FOLFOX and bevacizumab, progressed after 6 months. "
        "Currently residing in Seattle, Washington."
    ),
    "Rheumatoid arthritis": (
        "39-year-old female with a 6-year history of seropositive rheumatoid arthritis "
        "(anti-CCP positive, RF positive), currently on methotrexate with inadequate "
        "response, moderate-to-severe disease activity (DAS28 5.4). No prior biologic "
        "therapy. Currently residing in Denver, Colorado."
    ),
}

if "case_note_input" not in st.session_state:
    st.session_state["case_note_input"] = ""

col1, col2 = st.columns([4, 1])
with col2:
    st.write("")
    chosen_sample = st.selectbox("Sample note", list(SAMPLE_NOTES.keys()), label_visibility="collapsed")
    if st.button("Use sample note"):
        st.session_state["case_note_input"] = SAMPLE_NOTES[chosen_sample]
        st.rerun()
with col1:
    case_note = st.text_area(
        "Patient case note",
        height=200,
        placeholder="Paste a de-identified patient case note here...",
        key="case_note_input",
    )

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
