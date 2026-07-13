"""
Pydantic data models shared across all three agents.

Using Pydantic (rather than free-text) is what lets Agent 1's output be
*structured* and directly consumable by Agent 2 and Agent 3 — this is the
core of "structured output" that recruiters look for.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class PatientProfile(BaseModel):
    """Structured extraction target for Agent 1 (The Extractor)."""
    age: Optional[int] = Field(None, description="Patient age in years")
    sex: Optional[str] = Field(None, description="Biological sex: male/female")
    diagnoses: List[str] = Field(
        default_factory=list,
        description="Primary and secondary diagnoses/conditions, normalized to standard terminology",
    )
    biomarkers: List[str] = Field(
        default_factory=list,
        description="Genetic mutations / biomarker status, e.g. 'EGFR exon 19 deletion', 'HER2-negative'",
    )
    medications: List[str] = Field(
        default_factory=list, description="Current or prior medications / treatment lines"
    )
    stage: Optional[str] = Field(None, description="Disease stage if applicable, e.g. 'Stage IIIB'")
    ecog_status: Optional[int] = Field(None, description="ECOG performance status 0-5, if mentioned")
    location: Optional[str] = Field(None, description="Patient's city/state/country, if mentioned")
    other_notes: Optional[str] = Field(None, description="Any other clinically relevant detail")


class TrialSummary(BaseModel):
    """Raw trial data fetched by Agent 2 (The Researcher)."""
    nct_id: str
    title: str
    status: str
    conditions: List[str] = Field(default_factory=list)
    eligibility_text: str = ""
    locations: List[str] = Field(default_factory=list)
    url: str = ""


class TrialMatch(BaseModel):
    """Scoring output of Agent 3 (The Evaluator)."""
    nct_id: str = ""
    title: str = ""
    match_score: int = Field(..., ge=0, le=100, description="0-100 eligibility match score")
    rationale: str = Field(..., description="Short clinical justification for the score")
    matched_criteria: List[str] = Field(default_factory=list)
    unmatched_criteria: List[str] = Field(default_factory=list)
    recommendation: str = Field(
        ..., description="One of: 'Strong match', 'Possible match', 'Not eligible'"
    )


class AgentState(BaseModel):
    """The shared state object that flows through the LangGraph pipeline."""
    case_note: str
    patient: Optional[PatientProfile] = None
    trials: List[TrialSummary] = Field(default_factory=list)
    matches: List[TrialMatch] = Field(default_factory=list)
    log: List[str] = Field(default_factory=list)
