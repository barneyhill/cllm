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
        agreement_status: "agree", "disagree", or "disjoint"
        notes: Optional explanation of comparison
        n_llm: Number of claims in LLM result, or None if no LLM result
        n_peer: Number of claims in peer result, or None if no peer result
        n_itx: Number of claims in the intersection (shared between both)
    """
    llm_result_id: Optional[str] = None
    peer_result_id: Optional[str] = None
    llm_status: Optional[str] = None
    peer_status: Optional[str] = None
    agreement_status: str
    notes: Optional[str] = None
    n_llm: Optional[int] = None
    n_peer: Optional[int] = None
    n_itx: Optional[int] = None


class LLMResultsConcordanceResponse(BaseModel):
    """Response from concordance analysis stage.

    Attributes:
        concordance: List of concordance comparisons
    """
    concordance: List[LLMResultsConcordanceRow]


# ============================================================================
# DATABASE EXPORT MODELS - New format for database import
# ============================================================================


class DBSubmission(BaseModel):
    """Submission record for database."""
    id: str  # UUID
    user_id: Optional[int] = None  # Null for CLI usage, set by backend
    manuscript_title: Optional[str] = None
    manuscript_doi: Optional[str] = None
    status: str = "completed"
    created_at: str
    updated_at: str


class DBContent(BaseModel):
    """Content record for database."""
    id: str  # UUID
    submission_id: str
    content_type: str  # 'manuscript' or 'peer_review'
    content_text: str
    created_at: str


class DBPrompt(BaseModel):
    """Prompt record for database."""
    id: str  # Hash of prompt_text + model
    prompt_text: str
    prompt_type: str  # 'extract', 'eval_llm', 'eval_peer', 'compare'
    model: str
    created_at: str


class DBClaim(BaseModel):
    """Claim record for database."""
    id: str  # UUID
    content_id: str
    claim_id: str  # "C1", "C2" from LLM
    claim: str
    claim_type: str
    source_text: str
    evidence_type: str  # JSON string
    evidence_reasoning: str
    prompt_id: str
    created_at: str


class DBResult(BaseModel):
    """Result record for database."""
    id: str  # UUID
    content_id: str
    result_id: str  # "R1", "R2" from LLM
    result_type: str  # 'llm' or 'peer'
    reviewer_id: str
    reviewer_name: str
    result_status: str
    result_reasoning: str
    prompt_id: str
    created_at: str


class DBClaimResult(BaseModel):
    """Junction table record linking claims to results."""
    claim_id: str  # UUID
    result_id: str  # UUID


class DBComparison(BaseModel):
    """Comparison record for database."""
    id: str  # UUID
    submission_id: str
    llm_result_id: Optional[str] = None  # UUID, nullable
    peer_result_id: Optional[str] = None  # UUID, nullable
    llm_status: Optional[str] = None
    peer_status: Optional[str] = None
    agreement_status: str
    notes: Optional[str] = None
    prompt_id: str
    created_at: str


class DBExport(BaseModel):
    """Complete database export structure."""
    submission: DBSubmission
    content: List[DBContent]
    prompts: List[DBPrompt]
    claims: List[DBClaim]
    results: List[DBResult]
    claim_results: List[DBClaimResult]
    comparisons: List[DBComparison]
