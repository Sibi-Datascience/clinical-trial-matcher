"""
Tool used by Agent 2 (The Researcher): a thin, robust wrapper around the
official ClinicalTrials.gov API v2 (no API key required).

Docs: https://clinicaltrials.gov/data-api/api
"""
from typing import List
import requests

from .schemas import PatientProfile, TrialSummary

CTGOV_BASE = "https://clinicaltrials.gov/api/v2/studies"


def search_clinical_trials(patient: PatientProfile, max_results: int = 10) -> List[TrialSummary]:
    """
    Query ClinicalTrials.gov for currently recruiting trials matching the
    patient's condition(s). Falls back gracefully if fields are missing.
    """
    if not patient or not patient.diagnoses:
        return []

    # ClinicalTrials.gov's query.cond field accepts free-text boolean-ish queries.
    condition_query = " OR ".join(patient.diagnoses[:3])

    params = {
        "query.cond": condition_query,
        "pageSize": max_results,
        "format": "json",
        "filter.overallStatus": "RECRUITING",
    }
    if patient.location:
        params["query.locn"] = patient.location

    try:
        resp = requests.get(CTGOV_BASE, params=params, timeout=25)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        # Return empty list rather than crashing the pipeline; caller logs this.
        return []

    trials: List[TrialSummary] = []
    for study in data.get("studies", []):
        protocol = study.get("protocolSection", {})
        ident = protocol.get("identificationModule", {})
        status_mod = protocol.get("statusModule", {})
        cond_mod = protocol.get("conditionsModule", {})
        elig_mod = protocol.get("eligibilityModule", {})
        loc_mod = protocol.get("contactsLocationsModule", {})

        nct_id = ident.get("nctId", "")
        locations = [
            f"{loc.get('city', '')}, {loc.get('country', '')}"
            for loc in loc_mod.get("locations", [])[:5]
        ]

        trials.append(
            TrialSummary(
                nct_id=nct_id,
                title=ident.get("briefTitle", ""),
                status=status_mod.get("overallStatus", ""),
                conditions=cond_mod.get("conditions", []),
                eligibility_text=(elig_mod.get("eligibilityCriteria", "") or "")[:4000],
                locations=locations,
                url=f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "",
            )
        )
    return trials
