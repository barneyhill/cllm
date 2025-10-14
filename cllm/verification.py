"""
Verification service for CLLM - 4-stage claim verification workflow.

This module implements:
1. Extract atomic factual claims from manuscript
2. LLM groups claims into results (with evaluation)
3. Peer review groups claims into results (with evaluation)
4. Compare results between LLM and peer reviewer
"""

import json
import time
from pathlib import Path
from typing import List, Tuple

from anthropic import Anthropic

from .config import config
from .models import (
    LLMClaimV3,
    LLMClaimsResponseV3,
    LLMResultV3,
    LLMResultsResponseV3,
    LLMResultsConcordanceRow,
    LLMResultsConcordanceResponse,
)
from .prompts.prompt_fallback import (
    STAGE1_FALLBACK,
    STAGE2_FALLBACK,
    STAGE3_FALLBACK,
    STAGE4_FALLBACK,
)


# ============================================================================
# PROMPT LOADING
# ============================================================================


def load_prompt(filename: str, fallback: str) -> str:
    """Load prompt from file, falling back to hardcoded prompt if not found.

    Args:
        filename: Name of prompt file in prompts/ directory
        fallback: Hardcoded prompt to use if file doesn't exist

    Returns:
        Prompt text
    """
    prompts_dir = Path(__file__).parent / "prompts"
    prompt_file = prompts_dir / filename

    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8")
    return fallback


def get_anthropic_client():
    """Get configured Anthropic client with extended timeout."""
    return Anthropic(
        api_key=config.anthropic_api_key,
        timeout=config.timeout,
    )


def extract_json_from_response(response_text: str) -> str:
    """Extract JSON from LLM response, handling markdown code fences.

    Args:
        response_text: Raw response text from LLM

    Returns:
        Cleaned JSON string
    """
    text = response_text.strip()

    # Check for markdown code fences
    if "```" in text:
        lines = text.split("\n")
        start_idx = None
        end_idx = None

        # Find the first opening fence
        for i, line in enumerate(lines):
            if line.strip().startswith("```"):
                start_idx = i + 1
                break

        # Find the closing fence
        if start_idx is not None:
            for i in range(start_idx, len(lines)):
                if lines[i].strip() == "```":
                    end_idx = i
                    break

            if end_idx is not None and start_idx < end_idx:
                text = "\n".join(lines[start_idx:end_idx])
            elif start_idx is not None:
                text = "\n".join(lines[start_idx:])

    # If text is empty, try to find JSON starting with { or [
    if not text.strip():
        for start_char in ["{", "["]:
            idx = response_text.find(start_char)
            if idx >= 0:
                text = response_text[idx:]
                break

    return text.strip()


# ============================================================================
# STAGE 1: EXTRACT CLAIMS FROM MANUSCRIPT
# ============================================================================

# Load prompt from file (with fallback)
STAGE1_PROMPT_TEMPLATE = load_prompt("extract.txt", STAGE1_FALLBACK)


def extract_claims(manuscript_text: str) -> Tuple[List[LLMClaimV3], float]:
    """Stage 1: Extract atomic factual claims from manuscript.

    Args:
        manuscript_text: Full text of the manuscript

    Returns:
        Tuple of (list of extracted claims, processing time in seconds)

    Raises:
        ValueError: If LLM response is invalid or cannot be parsed
    """
    client = get_anthropic_client()
    start_time = time.time()

    prompt = STAGE1_PROMPT_TEMPLATE.replace("$MANUSCRIPT_TEXT", manuscript_text)

    message = client.messages.create(
        model=config.anthropic_model,
        max_tokens=30000,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text
    json_text = extract_json_from_response(response_text)

    if not json_text or not json_text.strip():
        raise ValueError(
            f"Failed to extract JSON from response. Response text: {response_text[:1000]}"
        )

    try:
        response_data = json.loads(json_text)
        llm_response = LLMClaimsResponseV3(**response_data)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse LLM response as JSON: {e}\n"
            f"Extracted JSON: {json_text[:500]}\n"
            f"Full response: {response_text[:1000]}"
        )
    except Exception as e:
        raise ValueError(
            f"Failed to validate LLM response: {e}\n"
            f"Extracted JSON: {json_text[:500]}\n"
            f"Full response: {response_text[:1000]}"
        )

    processing_time = time.time() - start_time
    return llm_response.claims, processing_time


# ============================================================================
# STAGE 2: LLM GROUPS CLAIMS INTO RESULTS
# ============================================================================

# Load prompt from file (with fallback)
STAGE2_PROMPT_TEMPLATE = load_prompt("llm_eval.txt", STAGE2_FALLBACK)


def llm_group_claims_into_results(
    manuscript_text: str, claims: List[LLMClaimV3]
) -> Tuple[List[LLMResultV3], float]:
    """Stage 2: LLM groups claims into results and evaluates each result.

    Args:
        manuscript_text: Full text of the manuscript (for context)
        claims: List of extracted claims from Stage 1

    Returns:
        Tuple of (list of results, processing time in seconds)

    Raises:
        ValueError: If LLM response is invalid or cannot be parsed
    """
    client = get_anthropic_client()
    start_time = time.time()

    claims_json = json.dumps(
        [
            {
                "claim_id": c.claim_id,
                "claim": c.claim,
                "claim_type": c.claim_type,
                "source_text": c.source_text,
                "evidence_type": c.evidence_type,
                "evidence_reasoning": c.evidence_reasoning,
            }
            for c in claims
        ],
        indent=2,
    )

    prompt = (
        STAGE2_PROMPT_TEMPLATE.replace("$MANUSCRIPT_TEXT", manuscript_text).replace(
            "$CLAIMS_JSON", claims_json
        )
    )

    # Note: reviewer_id and reviewer_name will be set to "LLM" by the prompt

    message = client.messages.create(
        model=config.anthropic_model,
        max_tokens=30000,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text
    json_text = extract_json_from_response(response_text)

    if not json_text or not json_text.strip():
        raise ValueError(
            f"Failed to extract JSON from response. Response text: {response_text[:1000]}"
        )

    try:
        response_data = json.loads(json_text)
        llm_response = LLMResultsResponseV3(**response_data)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse LLM response as JSON: {e}\n"
            f"Extracted JSON: {json_text[:500]}\n"
            f"Full response: {response_text[:1000]}"
        )
    except Exception as e:
        raise ValueError(
            f"Failed to validate LLM response: {e}\n"
            f"Extracted JSON: {json_text[:500]}\n"
            f"Full response: {response_text[:1000]}"
        )

    processing_time = time.time() - start_time
    return llm_response.results, processing_time


# ============================================================================
# STAGE 3: PEER REVIEW GROUPS CLAIMS INTO RESULTS
# ============================================================================

# Load prompt from file (with fallback)
STAGE3_PROMPT_TEMPLATE = load_prompt("peer_eval.txt", STAGE3_FALLBACK)


def peer_review_group_claims_into_results(
    claims: List[LLMClaimV3], review_text: str
) -> Tuple[List[LLMResultV3], float]:
    """Stage 3: Extract results from peer review based on manuscript claims.

    Args:
        claims: List of extracted claims from Stage 1
        review_text: Full text of peer review

    Returns:
        Tuple of (list of results from reviewer perspective, processing time in seconds)

    Raises:
        ValueError: If LLM response is invalid or cannot be parsed
    """
    client = get_anthropic_client()
    start_time = time.time()

    claims_json = json.dumps(
        [
            {
                "claim_id": c.claim_id,
                "claim": c.claim,
                "claim_type": c.claim_type,
                "source_text": c.source_text,
                "evidence_type": c.evidence_type,
                "evidence_reasoning": c.evidence_reasoning,
            }
            for c in claims
        ],
        indent=2,
    )

    prompt = STAGE3_PROMPT_TEMPLATE.replace("$CLAIMS_JSON", claims_json).replace(
        "$REVIEW_TEXT", review_text
    )

    message = client.messages.create(
        model=config.anthropic_model,
        max_tokens=30000,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text
    json_text = extract_json_from_response(response_text)

    if not json_text or not json_text.strip():
        raise ValueError(
            f"Failed to extract JSON from response. Response text: {response_text[:1000]}"
        )

    try:
        response_data = json.loads(json_text)
        llm_response = LLMResultsResponseV3(**response_data)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse LLM response as JSON: {e}\n"
            f"Extracted JSON: {json_text[:500]}\n"
            f"Full response: {response_text[:1000]}"
        )
    except Exception as e:
        raise ValueError(
            f"Failed to validate LLM response: {e}\n"
            f"Extracted JSON: {json_text[:500]}\n"
            f"Full response: {response_text[:1000]}"
        )

    processing_time = time.time() - start_time
    return llm_response.results, processing_time


# ============================================================================
# STAGE 4: COMPARE RESULTS BETWEEN LLM AND PEER REVIEW
# ============================================================================

# Load prompt from file (with fallback)
STAGE4_PROMPT_TEMPLATE = load_prompt("compare.txt", STAGE4_FALLBACK)


def compare_results(
    llm_results: List[LLMResultV3], peer_results: List[LLMResultV3]
) -> Tuple[List[LLMResultsConcordanceRow], float]:
    """Stage 4: Compare results between LLM and peer review.

    Args:
        llm_results: Results from LLM evaluation (Stage 2)
        peer_results: Results from peer review (Stage 3)

    Returns:
        Tuple of (list of concordance rows, processing time in seconds)

    Raises:
        ValueError: If LLM response is invalid or cannot be parsed
    """
    client = get_anthropic_client()
    start_time = time.time()

    llm_results_json = json.dumps(
        [
            {
                "result_id": r.result_id,
                "claim_ids": r.claim_ids,
                "reviewer_id": r.reviewer_id,
                "reviewer_name": r.reviewer_name,
                "status": r.status,
                "status_reasoning": r.status_reasoning,
            }
            for r in llm_results
        ],
        indent=2,
    )

    peer_results_json = json.dumps(
        [
            {
                "result_id": r.result_id,
                "claim_ids": r.claim_ids,
                "reviewer_id": r.reviewer_id,
                "reviewer_name": r.reviewer_name,
                "status": r.status,
                "status_reasoning": r.status_reasoning,
            }
            for r in peer_results
        ],
        indent=2,
    )

    prompt = STAGE4_PROMPT_TEMPLATE.replace(
        "$LLM_RESULTS_JSON", llm_results_json
    ).replace("$PEER_RESULTS_JSON", peer_results_json)

    message = client.messages.create(
        model=config.anthropic_model,
        max_tokens=30000,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text
    json_text = extract_json_from_response(response_text)

    if not json_text or not json_text.strip():
        raise ValueError(
            f"Failed to extract JSON from response. Response text: {response_text[:1000]}"
        )

    try:
        response_data = json.loads(json_text)
        llm_response = LLMResultsConcordanceResponse(**response_data)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse LLM response as JSON: {e}\n"
            f"Extracted JSON: {json_text[:500]}\n"
            f"Full response: {response_text[:1000]}"
        )
    except Exception as e:
        raise ValueError(
            f"Failed to validate LLM response: {e}\n"
            f"Extracted JSON: {json_text[:500]}\n"
            f"Full response: {response_text[:1000]}"
        )

    processing_time = time.time() - start_time
    return llm_response.concordance, processing_time


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def calculate_results_metrics(
    llm_results: List[LLMResultV3],
    peer_results: List[LLMResultV3],
    concordance: List[LLMResultsConcordanceRow],
) -> dict:
    """Calculate summary metrics for V3 workflow.

    Args:
        llm_results: LLM evaluation results
        peer_results: Peer review results
        concordance: Concordance analysis

    Returns:
        Dictionary with metrics including agreement rate
    """
    total_comparisons = len(concordance)
    agreements = sum(1 for row in concordance if row.agreement_status == "agree")
    disagreements = sum(1 for row in concordance if row.agreement_status == "disagree")
    partial = sum(1 for row in concordance if row.agreement_status == "partial")

    agreement_rate = (agreements / total_comparisons * 100) if total_comparisons > 0 else 0.0

    return {
        "total_comparisons": total_comparisons,
        "agreements": agreements,
        "disagreements": disagreements,
        "partial": partial,
        "agreement_rate": agreement_rate,
    }
