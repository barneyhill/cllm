"""
Data models for CLLM (Claim LLM) tool.

These models represent the V3 workflow for claim verification:
- Claims: Atomic factual statements extracted from manuscripts
- Results: Grouped claims with evaluation status
- Concordance: Comparison between LLM and peer review results
"""

from typing import List, Optional

from pydantic import BaseModel


# ============================================================================
# CLAIM MODELS
# ============================================================================


class LLMClaimV3(BaseModel):
    """Atomic factual claim extracted from a manuscript.

    Attributes:
        claim_id: Unique identifier (e.g., "C1", "C2")
        claim: The atomic factual claim text
        claim_type: EXPLICIT or IMPLICIT
        source_text: Exact excerpt from manuscript
        evidence_type: List of evidence types (DATA, CITATION, KNOWLEDGE, INFERENCE, SPECULATION)
        evidence_reasoning: Brief explanation of evidence type classification
    """
    claim_id: str
    claim: str
    claim_type: str
    source_text: str
    evidence_type: List[str]
    evidence_reasoning: str


class LLMClaimsResponseV3(BaseModel):
    """Response from claim extraction stage.

    Attributes:
        claims: List of extracted claims
    """
    claims: List[LLMClaimV3]


# ============================================================================
# RESULT MODELS
# ============================================================================


class LLMResultV3(BaseModel):
    """Grouped claims with evaluation status.

    A result represents a logical grouping of related claims that together
    support a coherent scientific finding.

    Attributes:
        result_id: Unique identifier for this result (e.g., "R1", "R2")
        claim_ids: List of claim IDs in this result (e.g., ["C1", "C2"])
        reviewer_id: ORCID ID of reviewer, or "LLM" for LLM evaluations
        reviewer_name: Name of reviewer, or "LLM" for LLM evaluations
        status: SUPPORTED, UNSUPPORTED, or UNCERTAIN
        status_reasoning: Brief explanation of status evaluation
    """
    result_id: str
    claim_ids: List[str]
    reviewer_id: str
    reviewer_name: str
    status: str
    status_reasoning: str


class LLMResultsResponseV3(BaseModel):
    """Response from result grouping stage.

    Attributes:
        results: List of results
    """
    results: List[LLMResultV3]


# ============================================================================
# CONCORDANCE MODELS
# ============================================================================


class LLMResultsConcordanceRow(BaseModel):
    """Comparison between LLM and peer review results.

    Represents a single comparison of how LLM and peer reviewers
    evaluated overlapping claims.

    Attributes:
        llm_result_id: Result ID from LLM evaluation (e.g., "R2"), or None if no LLM result
        peer_result_id: Result ID from peer review evaluation (e.g., "R4"), or None if no peer result
        llm_status: LLM's status evaluation, or None if no LLM result
        peer_status: Peer reviewer's status evaluation, or None if no peer result
        agreement_status: "agree", "disagree", or "partial"
        notes: Optional explanation of comparison
    """
    llm_result_id: Optional[str] = None
    peer_result_id: Optional[str] = None
    llm_status: Optional[str] = None
    peer_status: Optional[str] = None
    agreement_status: str
    notes: Optional[str] = None


class LLMResultsConcordanceResponse(BaseModel):
    """Response from concordance analysis stage.

    Attributes:
        concordance: List of concordance comparisons
    """
    concordance: List[LLMResultsConcordanceRow]
