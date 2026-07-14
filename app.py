"""
Multi-Agent Clinical Trial Matcher — Streamlit front-end.

Entry point for Streamlit Cloud / Hugging Face Spaces. Streams the
LangGraph pipeline (src/graph.py) node-by-node so the UI can show each
agent — Extractor, Researcher, Evaluator — working in real time, with a
full explainability panel after every step.
"""
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.graph import stream_pipeline  # noqa: E402

st.set_page_config(page_title="Clinical Trial Matcher", page_icon="🧬", layout="wide")

# ---------------------------------------------------------------------------
# Theme — clinical / lab-instrument palette, deliberately not the default
# dark-mode-with-one-neon-accent look. Deep navy surfaces, a signature
# three-node "agent rail" that mirrors the real execution graph, teal for
# confirmed/complete, amber for in-progress, coral reserved for negative
# results only (never used decoratively).
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #0B1220;
    --surface: #121A2B;
    --surface-2: #1A2338;
    --border: #263049;
    --text: #E8ECF4;
    --text-muted: #8C97AC;
    --teal: #2DD4BF;
    --amber: #F2B155;
    --coral: #F2707A;
}

.stApp { background-color: var(--bg); font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }
code, .mono { font-family: 'JetBrains Mono', monospace !important; }

.hero-sub { color: var(--text-muted); font-size: 1.02rem; margin-top: -0.6rem; }

/* ---- Agent rail (signature element) ---- */
.rail { display: flex; align-items: center; justify-content: center; margin: 1.6rem 0 0.4rem 0; }
.rail-node-wrap { display: flex; flex-direction: column; align-items: center; width: 190px; }
.rail-node {
    width: 56px; height: 56px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; border: 2px solid var(--border);
    background: var(--surface); color: var(--text-muted);
    transition: all 0.3s ease;
}
.rail-node.active {
    border-color: var(--amber); color: var(--amber);
    box-shadow: 0 0 0 0 rgba(242,177,85,0.5);
    animation: pulse 1.4s infinite;
}
.rail-node.done { border-color: var(--teal); background: rgba(45,212,191,0.12); color: var(--teal); }
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(242,177,85,0.45); }
    70% { box-shadow: 0 0 0 12px rgba(242,177,85,0); }
    100% { box-shadow: 0 0 0 0 rgba(242,177,85,0); }
}
.rail-line { height: 2px; width: 70px; background: var(--border); margin-top: -32px; }
.rail-line.done { background: var(--teal); }
.rail-label { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 0.85rem; margin-top: 0.5rem; color: var(--text); }
.rail-status { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.1rem; text-align: center; min-height: 1.2rem; }

/* ---- Cards ---- */
.card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.7rem;
}
.chip {
    display: inline-block; background: var(--surface-2); border: 1px solid var(--border);
    color: var(--text); border-radius: 6px; padding: 0.15rem 0.55rem; margin: 0.15rem 0.3rem 0.15rem 0;
    font-size: 0.82rem; font-family: 'JetBrains Mono', monospace;
}
.field-label { color: var(--text-muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.2rem; }

/* ---- Score bar ---- */
.score-track { background: var(--surface-2); border-radius: 6px; height: 10px; width: 100%; overflow: hidden; }
.score-fill { height: 100%; border-radius: 6px; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🧬 About")
    st.write(
        "A multi-agent system — not a single LLM call — that reads a patient "
        "case note, extracts structured clinical data, searches live "
        "ClinicalTrials.gov data, and scores eligibility with a written rationale."
    )
    st.markdown("**Built with**")
    st.markdown("`LangGraph` · `Groq` · `Streamlit` · `ClinicalTrials.gov API v2`")
    st.divider()
    st.caption(
        "⚠️ For demonstration purposes only. Not medical advice and not "
        "validated for clinical decision-making."
    )
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is not set. Add it in your app's Settings → Secrets.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Multi-Agent Clinical Trial Matcher")
st.markdown(
    '<p class="hero-sub">Three specialized agents collaborate — extract, research, evaluate — '
    "to turn a free-text case note into scored, justified trial matches.</p>",
    unsafe_allow_html=True,
)

with st.expander("ℹ️ How this works (plain-English explainer)"):
    st.markdown(
        """
This isn't a single AI call — it's a **pipeline of three agents**, each with one job,
handing structured data to the next:

1. **🧾 Extractor** — reads your free-text note and pulls out structured facts
   (age, diagnoses, biomarkers, medications...) using the LLM's *structured output*
   mode, so the result is a validated data object, not just prose.
2. **🔎 Researcher** — takes that structured profile and calls the real, live
   **ClinicalTrials.gov API** to fetch currently recruiting trials matching the
   patient's condition and location. This is a genuine tool call to an external
   system, not a static or made-up dataset.
3. **🧠 Evaluator** — reads each trial's actual eligibility criteria text and
   compares it line-by-line against the patient profile, producing a 0–100 match
   score, a plain-English rationale, and which specific criteria matched or didn't.

Each agent's output is *typed and validated* before the next agent ever sees it —
so a bad hand-off between agents fails loudly instead of silently confusing the
next step.
"""
    )

# ---------------------------------------------------------------------------
# Sample notes + input
# ---------------------------------------------------------------------------
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
    if st.button("Use sample note", use_container_width=True):
        st.session_state["case_note_input"] = SAMPLE_NOTES[chosen_sample]
        st.rerun()
with col1:
    case_note = st.text_area(
        "Patient case note",
        height=200,
        placeholder="Paste a de-identified patient case note here...",
        key="case_note_input",
    )

run = st.button("🚀 Run Agent Team", type="primary")

# ---------------------------------------------------------------------------
# Pipeline rail renderer
# ---------------------------------------------------------------------------
AGENTS = [("🧾", "Extractor", "Structured extraction"), ("🔎", "Researcher", "ClinicalTrials.gov search"), ("🧠", "Evaluator", "Eligibility scoring")]


def node_status(steps_done: int, idx: int) -> str:
    threshold = idx + 1
    if steps_done >= threshold:
        return "done"
    if steps_done >= idx:
        return "active"
    return "pending"


def render_rail(steps_done: int, live_notes: list) -> str:
    parts = ['<div class="rail">']
    for i, (icon, name, desc) in enumerate(AGENTS):
        status = node_status(steps_done, i)
        note = live_notes[i] if i < len(live_notes) else desc
        parts.append('<div class="rail-node-wrap">')
        parts.append(f'<div class="rail-node {status}">{icon}</div>')
        parts.append(f'<div class="rail-label">{name}</div>')
        parts.append(f'<div class="rail-status">{note}</div>')
        parts.append("</div>")
        if i < len(AGENTS) - 1:
            line_status = "done" if steps_done >= i + 1 else ""
            parts.append(f'<div class="rail-line {line_status}"></div>')
    parts.append("</div>")
    return "".join(parts)


def score_color(score: int) -> str:
    if score >= 70:
        return "var(--teal)"
    if score >= 40:
        return "var(--amber)"
    return "var(--coral)"


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
if run:
    if not case_note or not case_note.strip():
        st.warning("Please paste a case note first.")
        st.stop()
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is not set — cannot call the LLM.")
        st.stop()

    rail_ph = st.empty()
    rail_ph.markdown(render_rail(0, ["Reading case note..."]), unsafe_allow_html=True)

    extractor_ph = st.container()
    researcher_ph = st.container()
    evaluator_ph = st.container()

    final_state = None
    try:
        for state in stream_pipeline(case_note):
            steps_done = len(state.log)
            live_notes = []
            if steps_done >= 1 and state.patient:
                live_notes.append(f"{len(state.patient.diagnoses)} diagnosis(es) found")
            if steps_done >= 2:
                live_notes.append(f"{len(state.trials)} trial(s) fetched")
            if steps_done >= 3:
                live_notes.append(f"{len(state.matches)} trial(s) scored")
            rail_ph.markdown(render_rail(steps_done, live_notes), unsafe_allow_html=True)
            final_state = state
    except Exception as e:
        rail_ph.markdown(render_rail(0, ["Pipeline failed"]), unsafe_allow_html=True)
        st.exception(e)
        st.stop()

    result = final_state

    # ---- Agent 1 explainability: extracted profile ----
    with extractor_ph:
        st.subheader("🧾 Agent 1 · Extractor — structured patient profile")
        if result.patient:
            p = result.patient
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="card"><div class="field-label">Age</div>{p.age or "—"}</div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="card"><div class="field-label">Sex</div>{p.sex or "—"}</div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="card"><div class="field-label">ECOG status</div>{p.ecog_status if p.ecog_status is not None else "—"}</div>', unsafe_allow_html=True)

            def chip_row(label, items):
                chips = "".join(f'<span class="chip">{x}</span>' for x in items) or '<span class="field-label">none stated</span>'
                st.markdown(f'<div class="card"><div class="field-label">{label}</div>{chips}</div>', unsafe_allow_html=True)

            chip_row("Diagnoses", p.diagnoses)
            chip_row("Biomarkers", p.biomarkers)
            chip_row("Medications", p.medications)
            if p.stage or p.location:
                st.markdown(
                    f'<div class="card"><div class="field-label">Stage</div>{p.stage or "—"} '
                    f'&nbsp;&nbsp;·&nbsp;&nbsp; <span class="field-label">Location</span> {p.location or "—"}</div>',
                    unsafe_allow_html=True,
                )
            with st.expander("Raw structured JSON"):
                st.json(p.model_dump())
        else:
            st.info("No structured profile was extracted.")

    # ---- Agent 2 explainability: raw trials fetched ----
    with researcher_ph:
        st.subheader("🔎 Agent 2 · Researcher — trials fetched from ClinicalTrials.gov")
        if result.trials:
            st.caption(f"Query: recruiting trials for **{', '.join(result.patient.diagnoses[:3])}**" + (f" near **{result.patient.location}**" if result.patient.location else ""))
            with st.expander(f"View all {len(result.trials)} fetched trials (before scoring)"):
                for t in result.trials:
                    st.markdown(
                        f'<div class="card"><b>{t.title}</b> &nbsp;<span class="chip">{t.nct_id}</span> '
                        f'<span class="chip">{t.status}</span><br>'
                        f'<span class="field-label">Locations</span> {", ".join(t.locations) or "—"}<br>'
                        f'<a href="{t.url}" target="_blank">View on ClinicalTrials.gov ↗</a></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No recruiting trials were found for this patient's condition/location.")

    # ---- Agent 3 explainability: scored matches ----
    with evaluator_ph:
        st.subheader("🧠 Agent 3 · Evaluator — eligibility scoring")
        if result.matches:
            scores = [m.match_score for m in result.matches]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Trials found", len(result.trials))
            m2.metric("Trials scored", len(result.matches))
            m3.metric("Average score", f"{sum(scores)/len(scores):.0f}")
            m4.metric("Top score", max(scores))

            df = pd.DataFrame(
                [{"Trial": (m.title[:40] + "…") if len(m.title) > 40 else m.title, "Score": m.match_score} for m in result.matches]
            ).set_index("Trial")
            st.bar_chart(df, color="#2DD4BF", horizontal=True)

            st.markdown("#### Trial-by-trial rationale")
            for m in result.matches:
                with st.expander(f"{m.match_score}/100 — {m.title}  ({m.nct_id})"):
                    st.markdown(
                        f'<div class="score-track"><div class="score-fill" style="width:{m.match_score}%; background:{score_color(m.match_score)};"></div></div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Recommendation:** {m.recommendation}")
                    st.markdown(f"**Rationale:** {m.rationale}")
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        st.markdown("**✅ Matched criteria**")
                        for c in m.matched_criteria:
                            st.markdown(f"- {c}")
                    with cc2:
                        st.markdown("**❌ Unmatched criteria**")
                        for c in m.unmatched_criteria:
                            st.markdown(f"- {c}")
                    st.markdown(f"[View full trial on ClinicalTrials.gov ↗](https://clinicaltrials.gov/study/{m.nct_id})")

            # ---- Downloadable report ----
            report_lines = [
                "# Clinical Trial Match Report",
                "",
                f"**Diagnoses:** {', '.join(result.patient.diagnoses) or '—'}",
                f"**Biomarkers:** {', '.join(result.patient.biomarkers) or '—'}",
                f"**Location:** {result.patient.location or '—'}",
                "",
                "## Scored trials",
            ]
            for m in result.matches:
                report_lines += [
                    f"### {m.match_score}/100 — {m.title} ({m.nct_id})",
                    f"- Recommendation: {m.recommendation}",
                    f"- Rationale: {m.rationale}",
                    f"- Matched: {', '.join(m.matched_criteria) or '—'}",
                    f"- Unmatched: {', '.join(m.unmatched_criteria) or '—'}",
                    f"- Link: https://clinicaltrials.gov/study/{m.nct_id}",
                    "",
                ]
            st.download_button(
                "⬇️ Download full report (Markdown)",
                data="\n".join(report_lines),
                file_name="trial_match_report.md",
                mime="text/markdown",
            )
        else:
            st.info("No trials to score.")
