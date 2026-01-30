"""
Data models for CLLM (Claim LLM) tool.

These models represent the V3 workflow for claim verification:
- Claims: Atomic factual statements extracted from manuscripts
- Results: Grouped claims with evaluation status
- Concordance: Comparison between LLM and peer review results
"""

from typing import List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# CLAIM MODELS
# ============================================================================


class LLMClaimResponseV3(BaseModel):
    """Claim as returned by LLM (without claim_id).

    This is the format the LLM returns. The claim_id is added post-hoc.

    Attributes:
        claim: The atomic factual claim text
        claim_type: EXPLICIT or IMPLICIT
        source: Exact excerpt from manuscript or reference to figure
        source_type: List indicating source types (TEXT, IMAGE)
        evidence: Brief explanation of evidence type classification
        evidence_type: List of evidence types (DATA, CITATION, KNOWLEDGE, INFERENCE, SPECULATION)
    """
    claim: str
    claim_type: str
    source: str
    source_type: List[str]
    evidence: str
    evidence_type: List[str]


class LLMClaimV3(BaseModel):
    """Atomic factual claim extracted from a manuscript (with claim_id added post-hoc).

    Attributes:
        claim_id: Unique identifier (e.g., "C1", "C2") - added after LLM response
        claim: The atomic factual claim text
        claim_type: EXPLICIT or IMPLICIT
        source: Exact excerpt from manuscript or reference to figure
        source_type: List indicating source types (TEXT, IMAGE)
        evidence: Brief explanation of evidence type classification
        evidence_type: List of evidence types (DATA, CITATION, KNOWLEDGE, INFERENCE, SPECULATION)
    """
    claim_id: str
    claim: str
    claim_type: str
    source: str
    source_type: List[str]
    evidence: str
    evidence_type: List[str]


class LLMClaimsResponseV3(BaseModel):
    """Response from claim extraction stage (from LLM).

    Attributes:
        claims: List of extracted claims (without claim_ids)
    """
    claims: List[LLMClaimResponseV3]


# ============================================================================
# RESULT MODELS
# ============================================================================


class LLMResultResponseV3(BaseModel):
    """Result as returned by LLM (without result_id, reviewer_id, reviewer_name).

    This is the format the LLM returns. The result_id and reviewer fields are added post-hoc.

    Attributes:
        claim_ids: List of claim IDs in this result (e.g., ["C1", "C2"])
        result: Description of the scientific finding (e.g., "The authors show that protein X phosphorylates protein Y.")
        evaluation_type: SUPPORTED, UNSUPPORTED, or UNCERTAIN
        evaluation: Brief explanation of evaluation_type assessment
        result_type: MAJOR or MINOR - significance of the result
    """
    claim_ids: List[str]
    result: str
    evaluation_type: str
    evaluation: str
    result_type: str


class LLMResultV3(BaseModel):
    """Grouped claims with evaluation status (with identifiers added post-hoc).

    A result represents a logical grouping of related claims that together
    support a coherent scientific finding.

    Attributes:
        result_id: Unique identifier for this result (e.g., "R1", "R2") - added after LLM response
        claim_ids: List of claim IDs in this result (e.g., ["C1", "C2"])
        result: Description of the scientific finding
        reviewer_id: ORCID ID of reviewer, or "OpenEval" for LLM evaluations - added after LLM response
        reviewer_name: Name of reviewer, or "OpenEval" for LLM evaluations - added after LLM response
        evaluation_type: SUPPORTED, UNSUPPORTED, or UNCERTAIN
        evaluation: Brief explanation of evaluation_type assessment
        result_type: MAJOR or MINOR - significance of the result
    """
    result_id: str
    claim_ids: List[str]
    result: str
    reviewer_id: str
    reviewer_name: str
    evaluation_type: str
    evaluation: str
    result_type: str


class LLMResultsResponseV3(BaseModel):
    """Response from result grouping stage (from LLM).

    Attributes:
        results: List of results (without result_ids or reviewer fields)
    """
    results: List[LLMResultResponseV3]


# ============================================================================
# CONCORDANCE MODELS
# ============================================================================


class LLMResultsConcordanceRowResponse(BaseModel):
    """Concordance row as returned by LLM (without comparison_id, comparison_type, and claim counts).

    This is the format the LLM returns. The comparison_id, comparison_type, and claim counts are added post-hoc.

    Attributes:
        openeval_result_id: Result ID from OpenEval evaluation (e.g., "R2"), or None if no OpenEval result
        peer_result_id: Result ID from peer review evaluation (e.g., "R4"), or None if no peer result
        comparison: Explanation of the comparison between evaluations
    """
    openeval_result_id: Optional[str] = None
    peer_result_id: Optional[str] = None
    comparison: Optional[str] = None


class LLMResultsConcordanceRow(BaseModel):
    """Comparison between OpenEval and peer review results.

    Represents a single comparison of how OpenEval and peer reviewers
    evaluated overlapping claims.

    Attributes:
        comparison_id: Unique identifier (e.g., "CMP1", "CMP2") - added after LLM response
        openeval_result_id: Result ID from OpenEval evaluation (e.g., "R2"), or None if no OpenEval result
        peer_result_id: Result ID from peer review evaluation (e.g., "R4"), or None if no peer result
        openeval_evaluation_type: Evaluation type from OpenEval (SUPPORTED/UNSUPPORTED/UNCERTAIN), or None - looked up post-hoc
        peer_evaluation_type: Evaluation type from peer review (SUPPORTED/UNSUPPORTED/UNCERTAIN), or None - looked up post-hoc
        openeval_result_type: Result type from OpenEval (MAJOR/MINOR), or None - looked up post-hoc
        peer_result_type: Result type from peer review (MAJOR/MINOR), or None - looked up post-hoc
        comparison_type: "agree", "disagree", "partial", or "disjoint" - calculated post-hoc
        comparison: Explanation of the comparison between evaluations
        n_openeval: Number of claims in OpenEval result, or None if no OpenEval result
        n_peer: Number of claims in peer result, or None if no peer result
        n_itx: Number of claims in the intersection (shared between both)
    """
    comparison_id: str
    openeval_result_id: Optional[str] = None
    peer_result_id: Optional[str] = None
    openeval_evaluation_type: Optional[str] = None  # Looked up post-hoc
    peer_evaluation_type: Optional[str] = None  # Looked up post-hoc
    openeval_result_type: Optional[str] = None  # Looked up post-hoc
    peer_result_type: Optional[str] = None  # Looked up post-hoc
    comparison_type: Optional[str] = None  # Calculated post-hoc
    comparison: Optional[str] = None
    n_openeval: Optional[int] = None
    n_peer: Optional[int] = None
    n_itx: Optional[int] = None


class LLMResultsConcordanceResponse(BaseModel):
    """Response from concordance analysis stage (from LLM).

    Attributes:
        concordance: List of concordance comparisons (without comparison_id or comparison_type)
    """
    concordance: List[LLMResultsConcordanceRowResponse]


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
    source: str
    source_type: str  # JSON string
    evidence: str
    evidence_type: str  # JSON string
    prompt_id: str
    created_at: str


class DBResult(BaseModel):
    """Result record for database."""
    id: str  # UUID
    content_id: str
    result_id: str  # "R1", "R2" from LLM
    result_category: str  # 'openeval' or 'peer'
    result: str  # Description of the scientific finding
    reviewer_id: str
    reviewer_name: str
    evaluation_type: str  # 'SUPPORTED', 'UNSUPPORTED', or 'UNCERTAIN'
    evaluation: str
    result_type: str  # 'MAJOR' or 'MINOR'
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
    comparison_id: str  # "CMP1", "CMP2" from post-processing
    openeval_result_id: Optional[str] = None  # UUID, nullable
    peer_result_id: Optional[str] = None  # UUID, nullable
    openeval_evaluation_type: Optional[str] = None  # 'SUPPORTED', 'UNSUPPORTED', or 'UNCERTAIN', nullable
    peer_evaluation_type: Optional[str] = None  # 'SUPPORTED', 'UNSUPPORTED', or 'UNCERTAIN', nullable
    openeval_result_type: Optional[str] = None  # 'MAJOR' or 'MINOR', nullable
    peer_result_type: Optional[str] = None  # 'MAJOR' or 'MINOR', nullable
    comparison_type: str  # 'agree', 'disagree', 'partial', or 'disjoint'
    comparison: Optional[str] = None
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


# =============================================================================
# Score Models (eLife Assessment)
# =============================================================================

class ScoreResponse(BaseModel):
    """
    Response from LLM for paper assessment (before enrichment).

    This model represents the raw structured output from the LLM when
    generating an eLife-style assessment.
    """
    assessment: str = Field(
        ...,
        description="Comprehensive 3-5 paragraph evaluation of the paper"
    )
    findings_significance: str = Field(
        ...,
        description="Significance rating: Landmark, Fundamental, Important, Valuable, or Useful"
    )
    evidence_strength: str = Field(
        ...,
        description="Evidence rating: Exceptional, Compelling, Convincing, Solid, Incomplete, or Inadequate"
    )


class Score(BaseModel):
    """
    Complete paper assessment with metadata (enriched post-LLM).

    This model includes the LLM assessment plus file paths and taxonomy
    information for traceability.
    """
    assessment: str = Field(
        ...,
        description="Comprehensive evaluation text"
    )
    findings_significance: str = Field(
        ...,
        description="Significance category from taxonomy"
    )
    evidence_strength: str = Field(
        ...,
        description="Evidence strength category from taxonomy"
    )
    taxonomy_type: str = Field(
        ...,
        description="Assessment taxonomy used (e.g., 'elife', 'icml', 'nature')"
    )
    manuscript_path: str = Field(
        ...,
        description="Path to source manuscript file"
    )
    claims_path: str = Field(
        ...,
        description="Path to source claims file"
    )
    results_path: str = Field(
        ...,
        description="Path to source results file"
    )
    created_at: str = Field(
        ...,
        description="ISO 8601 timestamp of assessment creation"
    )
