"""
Verification service for CLLM - 4-stage claim verification workflow.

This module implements:
1. Extract atomic factual claims from manuscript
2. LLM groups claims into results (with evaluation)
3. Peer review groups claims into results (with evaluation)
4. Compare results between LLM and peer reviewer
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Type, TypeVar, Union
import warnings

from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel

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
from .utils import generate_uuid, generate_prompt_id, get_current_timestamp

T = TypeVar("T", bound=BaseModel)

# ========================================================================
# PROMPT-TOKEN LIMIT (for API calls; only used to roughly check prompt length)
# ========================================================================
# Common token limits for LLM API calls (as of 2024):
#   - OpenAI GPT-4 (Turbo): 128k tokens, but safest for prompts: 100k-120k tokens
#   - OpenAI GPT-3.5 Turbo (1106): 16k tokens (older: 4k)
#   - Anthropic Claude 2/3: up to 200k tokens for context window (Claude 3 Opus),
#     but practical prompt limits often 100k-120k tokens before failures/timeouts.
# Adjust this value to your chosen model's context window (prompt+completion).
MAX_PROMPT_TOKENS = (
    100000  # Set to 100k for Claude models (conservative limit for 200k context)
)

# ============================================================================
# PROMPT LOADING
# ============================================================================


def warn_if_prompt_too_long(
    prompt: str, max_prompt_tokens: int = MAX_PROMPT_TOKENS, stage_name: str = "Prompt"
):
    """
    Warn if the given prompt likely exceeds max_prompt_tokens.
    Uses a rough estimate: 1 token ≈ 4 characters.
    """
    approx_token_count = len(prompt) // 4
    if approx_token_count > max_prompt_tokens:
        warnings.warn(
            f"[{stage_name}] Prompt may exceed {max_prompt_tokens} tokens! "
            f"Prompt length estimate: {approx_token_count} tokens.",
            UserWarning,
        )


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


def get_llm_client() -> Union[Anthropic, OpenAI]:
    """Get configured LLM client based on provider setting."""
    if config.llm_provider == "anthropic":
        return Anthropic(
            api_key=config.anthropic_api_key,
            timeout=config.timeout,
        )
    elif config.llm_provider == "openai":
        return OpenAI(
            api_key=config.openai_api_key,
            timeout=config.timeout,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")


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


def call_llm_structured(
    client: Union[Anthropic, OpenAI],
    prompt: str,
    response_model: Type[T],
    max_tokens: int = 64000,
) -> Tuple[T, Dict[str, int], Any]:
    """Call LLM with structured output support.

    Args:
        client: LLM client (Anthropic or OpenAI)
        prompt: The prompt to send
        response_model: Pydantic model for structured response
        max_tokens: Maximum tokens in response

    Returns:
        Tuple of (parsed_response, usage_dict, raw_response)
        where usage_dict contains 'input_tokens' and 'output_tokens'
        and raw_response is the full API response object

    Raises:
        ValueError: If response cannot be parsed
    """
    if config.llm_provider == "anthropic":
        # Anthropic: text response -> JSON parse -> Pydantic validation
        message = client.messages.create(
            model=config.anthropic_model,
            max_tokens=max_tokens,
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
            parsed_response = response_model(**response_data)
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

        usage = {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        }

        return parsed_response, usage, message

    elif config.llm_provider == "openai":
        # OpenAI: structured outputs with direct Pydantic model
        # Note: GPT-5 only supports temperature=1, so we don't set it
        completion = client.beta.chat.completions.parse(
            model=config.openai_model,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_model,
            max_completion_tokens=max_tokens,
        )

        parsed_response = completion.choices[0].message.parsed

        if parsed_response is None:
            raise ValueError(
                f"OpenAI returned None for parsed response. "
                f"Refusal: {completion.choices[0].message.refusal}"
            )

        usage = {
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
        }

        return parsed_response, usage, completion

    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")


# ============================================================================
# STAGE 1: EXTRACT CLAIMS FROM MANUSCRIPT
# ============================================================================

# Load prompt from file (with fallback)
STAGE1_PROMPT_TEMPLATE = load_prompt("extract.txt", STAGE1_FALLBACK)


def extract_claims(
    manuscript_text: str, verbose: bool = False, return_metrics: bool = False
) -> Tuple[List[LLMClaimV3], float, Optional[Dict[str, Any]], Optional[Any]]:
    """Stage 1: Extract atomic factual claims from manuscript.

    Args:
        manuscript_text: Full text of the manuscript
        verbose: If True, print detailed logging information
        return_metrics: If True, return metrics dictionary as third element

    Returns:
        Tuple of (list of extracted claims, processing time in seconds, optional metrics dict, optional raw_response)

    Raises:
        ValueError: If LLM response is invalid or cannot be parsed
    """
    client = get_llm_client()
    start_time = time.time()

    prompt = STAGE1_PROMPT_TEMPLATE.replace("$MANUSCRIPT_TEXT", manuscript_text)
    warn_if_prompt_too_long(
        prompt, MAX_PROMPT_TOKENS, "STAGE 1: Extract Claims From Manuscript"
    )

    llm_response, usage, raw_response = call_llm_structured(
        client=client,
        prompt=prompt,
        response_model=LLMClaimsResponseV3,
        max_tokens=64000,
    )

    # Post-process: Add sequential claim_id to each claim (C1, C2, C3, ...)
    claims_with_ids = []
    for idx, claim_response in enumerate(llm_response.claims, start=1):
        claim = LLMClaimV3(
            claim_id=f"C{idx}",
            claim=claim_response.claim,
            claim_type=claim_response.claim_type,
            source_text=claim_response.source_text,
            evidence_type=claim_response.evidence_type,
            evidence_reasoning=claim_response.evidence_reasoning,
        )
        claims_with_ids.append(claim)

    processing_time = time.time() - start_time

    # Build metrics dict if verbose or return_metrics requested
    metrics = None
    if verbose or return_metrics:
        model = (
            config.anthropic_model
            if config.llm_provider == "anthropic"
            else config.openai_model
        )
        metrics = {
            "model": model,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "num_claims": len(claims_with_ids),
            "processing_time_seconds": processing_time,
        }

    # Verbose: print metrics
    if verbose and metrics:
        print(f"[EXTRACT] Input tokens: {metrics['input_tokens']:,}", file=sys.stderr)
        print(f"[EXTRACT] Output tokens: {metrics['output_tokens']:,}", file=sys.stderr)
        print(f"[EXTRACT] Claims extracted: {metrics['num_claims']}", file=sys.stderr)
        print(
            f"[EXTRACT] Total time: {metrics['processing_time_seconds']:.2f}s",
            file=sys.stderr,
        )

    return claims_with_ids, processing_time, metrics, raw_response if return_metrics else None


# ============================================================================
# STAGE 2: LLM GROUPS CLAIMS INTO RESULTS
# ============================================================================

# Load prompt from file (with fallback)
STAGE2_PROMPT_TEMPLATE = load_prompt("llm_eval.txt", STAGE2_FALLBACK)


def llm_group_claims_into_results(
    manuscript_text: str,
    claims: List[LLMClaimV3],
    verbose: bool = False,
    return_metrics: bool = False,
) -> Tuple[List[LLMResultV3], float, Optional[Dict[str, Any]]]:
    """Stage 2: LLM groups claims into results and evaluates each result.

    Args:
        manuscript_text: Full text of the manuscript (for context)
        claims: List of extracted claims from Stage 1
        verbose: If True, print detailed logging information
        return_metrics: If True, return metrics dictionary as third element

    Returns:
        Tuple of (list of results, processing time in seconds, optional metrics dict)

    Raises:
        ValueError: If LLM response is invalid or cannot be parsed
    """
    client = get_llm_client()
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

    prompt = STAGE2_PROMPT_TEMPLATE.replace(
        "$MANUSCRIPT_TEXT", manuscript_text
    ).replace("$CLAIMS_JSON", claims_json)

    warn_if_prompt_too_long(
        prompt, MAX_PROMPT_TOKENS, "STAGE 2: LLM Group Claims Into Results"
    )

    # Note: reviewer_id and reviewer_name will be set to "LLM" by the prompt

    llm_response, usage, raw_response = call_llm_structured(
        client=client,
        prompt=prompt,
        response_model=LLMResultsResponseV3,
        max_tokens=64000,
    )

    # Post-process: Add sequential result_id and reviewer fields (R1, R2, R3, ...)
    results_with_ids = []
    for idx, result_response in enumerate(llm_response.results, start=1):
        result = LLMResultV3(
            result_id=f"R{idx}",
            claim_ids=result_response.claim_ids,
            result=result_response.result,
            reviewer_id="OpenEval",
            reviewer_name="OpenEval",
            status=result_response.status,
            status_reasoning=result_response.status_reasoning,
        )
        results_with_ids.append(result)

    processing_time = time.time() - start_time

    # Build metrics dict if verbose or return_metrics requested
    metrics = None
    if verbose or return_metrics:
        model = (
            config.anthropic_model
            if config.llm_provider == "anthropic"
            else config.openai_model
        )
        metrics = {
            "model": model,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "num_results": len(results_with_ids),
            "processing_time_seconds": processing_time,
        }

    # Verbose: print metrics
    if verbose and metrics:
        print(f"[EVAL-LLM] Input tokens: {metrics['input_tokens']:,}", file=sys.stderr)
        print(
            f"[EVAL-LLM] Output tokens: {metrics['output_tokens']:,}", file=sys.stderr
        )
        print(f"[EVAL-LLM] Results created: {metrics['num_results']}", file=sys.stderr)
        print(
            f"[EVAL-LLM] Total time: {metrics['processing_time_seconds']:.2f}s",
            file=sys.stderr,
        )

    return results_with_ids, processing_time, metrics


# ============================================================================
# STAGE 3: PEER REVIEW GROUPS CLAIMS INTO RESULTS
# ============================================================================

# Load prompt from file (with fallback)
STAGE3_PROMPT_TEMPLATE = load_prompt("peer_eval.txt", STAGE3_FALLBACK)


def peer_review_group_claims_into_results(
    claims: List[LLMClaimV3],
    review_text: str,
    verbose: bool = False,
    return_metrics: bool = False,
) -> Tuple[List[LLMResultV3], float, Optional[Dict[str, Any]]]:
    """Stage 3: Extract results from peer review based on manuscript claims.

    Args:
        claims: List of extracted claims from Stage 1
        review_text: Full text of peer review
        verbose: If True, print detailed logging information
        return_metrics: If True, return metrics dictionary as third element

    Returns:
        Tuple of (list of results from reviewer perspective, processing time in seconds, optional metrics dict)

    Raises:
        ValueError: If LLM response is invalid or cannot be parsed
    """
    client = get_llm_client()
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

    warn_if_prompt_too_long(
        prompt, MAX_PROMPT_TOKENS, "STAGE 3: Peer Review Groups Claims Into Results"
    )

    llm_response, usage, raw_response = call_llm_structured(
        client=client,
        prompt=prompt,
        response_model=LLMResultsResponseV3,
        max_tokens=64000,
    )

    # Post-process: Add sequential result_id and reviewer fields (R1, R2, R3, ...)
    results_with_ids = []
    for idx, result_response in enumerate(llm_response.results, start=1):
        result = LLMResultV3(
            result_id=f"R{idx}",
            claim_ids=result_response.claim_ids,
            result=result_response.result,
            reviewer_id="PEER_REVIEW",
            reviewer_name="Peer Reviewer",
            status=result_response.status,
            status_reasoning=result_response.status_reasoning,
        )
        results_with_ids.append(result)

    processing_time = time.time() - start_time

    # Build metrics dict if verbose or return_metrics requested
    metrics = None
    if verbose or return_metrics:
        model = (
            config.anthropic_model
            if config.llm_provider == "anthropic"
            else config.openai_model
        )
        metrics = {
            "model": model,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "num_results": len(results_with_ids),
            "processing_time_seconds": processing_time,
        }

    # Verbose: print metrics
    if verbose and metrics:
        print(f"[EVAL-PEER] Input tokens: {metrics['input_tokens']:,}", file=sys.stderr)
        print(
            f"[EVAL-PEER] Output tokens: {metrics['output_tokens']:,}", file=sys.stderr
        )
        print(f"[EVAL-PEER] Results created: {metrics['num_results']}", file=sys.stderr)
        print(
            f"[EVAL-PEER] Total time: {metrics['processing_time_seconds']:.2f}s",
            file=sys.stderr,
        )

    return results_with_ids, processing_time, metrics


# ============================================================================
# STAGE 4: COMPARE RESULTS BETWEEN LLM AND PEER REVIEW
# ============================================================================

# Load prompt from file (with fallback)
STAGE4_PROMPT_TEMPLATE = load_prompt("compare.txt", STAGE4_FALLBACK)


def compute_jaccard_pairings(
    llm_results: List[LLMResultV3], peer_results: List[LLMResultV3]
) -> List[dict]:
    """Compute pairwise Jaccard indices between OpenEval and peer results based on claim overlap.

    The Jaccard index measures similarity between two sets: |A ∩ B| / |A ∪ B|

    Args:
        llm_results: Results from OpenEval evaluation
        peer_results: Results from peer review evaluation

    Returns:
        List of pairings sorted by Jaccard index (descending), each containing:
        - openeval_result_id: ID of OpenEval result
        - peer_result_id: ID of peer result
        - jaccard_index: Similarity score (0.0 to 1.0)
        - shared_claims: List of claim IDs that appear in both results
    """
    pairings = []

    for llm_result in llm_results:
        llm_claims = set(llm_result.claim_ids)

        for peer_result in peer_results:
            peer_claims = set(peer_result.claim_ids)

            # Compute Jaccard index
            intersection = llm_claims & peer_claims
            union = llm_claims | peer_claims

            jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0

            # Only include non-zero similarities
            if jaccard > 0:
                pairings.append(
                    {
                        "openeval_result_id": llm_result.result_id,
                        "peer_result_id": peer_result.result_id,
                        "jaccard_index": round(jaccard, 3),
                        "shared_claims": sorted(list(intersection)),
                    }
                )

    # Sort by Jaccard index descending
    pairings.sort(key=lambda x: x["jaccard_index"], reverse=True)

    return pairings


def compare_results(
    llm_results: List[LLMResultV3],
    peer_results: List[LLMResultV3],
    verbose: bool = False,
    return_metrics: bool = False,
) -> Tuple[List[LLMResultsConcordanceRow], float, Optional[Dict[str, Any]]]:
    """Stage 4: Compare results between LLM and peer review.

    Args:
        llm_results: Results from LLM evaluation (Stage 2)
        peer_results: Results from peer review (Stage 3)
        verbose: If True, print detailed logging information
        return_metrics: If True, return metrics dictionary as third element

    Returns:
        Tuple of (list of concordance rows, processing time in seconds, optional metrics dict)

    Raises:
        ValueError: If LLM response is invalid or cannot be parsed
    """
    from .utils import calculate_comparison_metrics, format_metrics_report

    client = get_llm_client()
    start_time = time.time()

    # Compute Jaccard pairings based on claim overlap
    jaccard_pairings = compute_jaccard_pairings(llm_results, peer_results)

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

    jaccard_pairings_json = json.dumps(jaccard_pairings, indent=2)

    prompt = (
        STAGE4_PROMPT_TEMPLATE.replace("$LLM_RESULTS_JSON", llm_results_json)
        .replace("$PEER_RESULTS_JSON", peer_results_json)
        .replace("$JACCARD_PAIRINGS_JSON", jaccard_pairings_json)
    )

    warn_if_prompt_too_long(
        prompt,
        MAX_PROMPT_TOKENS,
        "STAGE 4: Compare Results Between LLM and Peer Review",
    )

    llm_response, usage, raw_response = call_llm_structured(
        client=client,
        prompt=prompt,
        response_model=LLMResultsConcordanceResponse,
        max_tokens=64000,
    )

    # Post-process: Calculate agreement_status based on openeval_status and peer_status
    for row in llm_response.concordance:
        # Calculate agreement_status based on presence and values of statuses
        if row.openeval_status is not None and row.peer_status is not None:
            # Both statuses present - check if they agree
            if row.openeval_status == row.peer_status:
                # If both agree on the same status (including both UNCERTAIN), mark as "agree"
                row.agreement_status = "agree"
            # Both present but different - check if either is UNCERTAIN
            elif row.openeval_status == "UNCERTAIN" or row.peer_status == "UNCERTAIN":
                # One is UNCERTAIN and the other is certain (but different), mark as "partial"
                row.agreement_status = "partial"
            else:
                # Both are certain (SUPPORTED or UNSUPPORTED) but disagree
                row.agreement_status = "disagree"
        else:
            # Only one status present (or both None) - mark as disjoint
            row.agreement_status = "disjoint"

    # Enhance concordance rows with claim count metrics
    # Create lookup dictionaries for faster access
    llm_results_dict = {r.result_id: r for r in llm_results}
    peer_results_dict = {r.result_id: r for r in peer_results}

    for row in llm_response.concordance:
        # Get the OpenEval result if present
        if row.openeval_result_id and row.openeval_result_id in llm_results_dict:
            llm_result = llm_results_dict[row.openeval_result_id]
            llm_claims = set(llm_result.claim_ids)
            row.n_openeval = len(llm_claims)
        else:
            llm_claims = set()
            row.n_openeval = None

        # Get the peer result if present
        if row.peer_result_id and row.peer_result_id in peer_results_dict:
            peer_result = peer_results_dict[row.peer_result_id]
            peer_claims = set(peer_result.claim_ids)
            row.n_peer = len(peer_claims)
        else:
            peer_claims = set()
            row.n_peer = None

        # Calculate intersection
        row.n_itx = len(llm_claims & peer_claims)

    processing_time = time.time() - start_time

    # Build metrics dict if verbose or return_metrics requested
    metrics_dict = None
    if verbose or return_metrics:
        model = (
            config.anthropic_model
            if config.llm_provider == "anthropic"
            else config.openai_model
        )

        # Count status categories for breakdown
        status_counts = {"SUPPORTED": 0, "UNSUPPORTED": 0, "UNCERTAIN": 0}
        for row in llm_response.concordance:
            if row.openeval_status in status_counts:
                status_counts[row.openeval_status] += 1
            if row.peer_status in status_counts:
                status_counts[row.peer_status] += 1

        # Calculate comprehensive comparison metrics
        comparison_metrics = calculate_comparison_metrics(
            llm_results, peer_results, llm_response.concordance
        )

        metrics_dict = {
            "model": model,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "num_comparisons": len(llm_response.concordance),
            "status_breakdown": status_counts,
            "processing_time_seconds": processing_time,
            "jaccard_pairings": jaccard_pairings,
            "comparison_metrics": comparison_metrics,
        }

    # Verbose: print metrics
    if verbose and metrics_dict:
        print(
            f"[COMPARE] Input tokens: {metrics_dict['input_tokens']:,}", file=sys.stderr
        )
        print(
            f"[COMPARE] Output tokens: {metrics_dict['output_tokens']:,}",
            file=sys.stderr,
        )
        print(
            f"[COMPARE] Comparisons created: {metrics_dict['num_comparisons']}",
            file=sys.stderr,
        )
        print(
            f"[COMPARE] Jaccard pairings computed: {len(metrics_dict['jaccard_pairings'])}",
            file=sys.stderr,
        )
        print(
            f"[COMPARE] Status breakdown - Supported: {metrics_dict['status_breakdown']['SUPPORTED']}, "
            f"Unsupported: {metrics_dict['status_breakdown']['UNSUPPORTED']}, "
            f"Uncertain: {metrics_dict['status_breakdown']['UNCERTAIN']}",
            file=sys.stderr,
        )
        print(
            f"[COMPARE] Total time: {metrics_dict['processing_time_seconds']:.2f}s",
            file=sys.stderr,
        )
        print("", file=sys.stderr)

        # Print comprehensive metrics report
        metrics_report = format_metrics_report(metrics_dict["comparison_metrics"])
        print(metrics_report, file=sys.stderr)

    return llm_response.concordance, processing_time, metrics_dict


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
    disjoint = sum(1 for row in concordance if row.agreement_status == "disjoint")

    agreement_rate = (
        (agreements / total_comparisons * 100) if total_comparisons > 0 else 0.0
    )

    return {
        "total_comparisons": total_comparisons,
        "agreements": agreements,
        "disagreements": disagreements,
        "partial": partial,
        "disjoint": disjoint,
        "agreement_rate": agreement_rate,
    }
