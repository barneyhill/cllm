"""Utility functions for CLLM - metrics calculation."""

from typing import List, Dict, Any
from .models import LLMResultV3, LLMResultsConcordanceRow


def calculate_comparison_metrics(
    llm_results: List[LLMResultV3],
    peer_results: List[LLMResultV3],
    concordance: List[LLMResultsConcordanceRow],
) -> Dict[str, Any]:
    """Calculate comprehensive metrics for result comparison.

    This includes:
    - Basic counts (N1, N2, overlap, only in each)
    - Classification agreement percentages
    - Category-wise precision, recall, F1
    - Coverage metrics

    Args:
        llm_results: Results from LLM evaluation
        peer_results: Results from peer review
        concordance: Concordance analysis

    Returns:
        Dictionary with comprehensive metrics
    """
    # Basic counts
    n1 = len(llm_results)
    n2 = len(peer_results)

    # Count unique result IDs that have matches (for coverage calculation)
    llm_ids_with_match = set(
        row.llm_result_id for row in concordance
        if row.llm_result_id is not None and row.peer_result_id is not None
    )
    peer_ids_with_match = set(
        row.peer_result_id for row in concordance
        if row.peer_result_id is not None and row.llm_result_id is not None
    )

    n_overlap_llm = len(llm_ids_with_match)
    n_overlap_peer = len(peer_ids_with_match)

    # Count unique result IDs that appear only in one set
    llm_ids_only = set(
        row.llm_result_id for row in concordance
        if row.llm_result_id is not None and row.peer_result_id is None
    )
    peer_ids_only = set(
        row.peer_result_id for row in concordance
        if row.peer_result_id is not None and row.llm_result_id is None
    )

    n_only1 = len(llm_ids_only)
    n_only2 = len(peer_ids_only)

    # Classification agreement (for overlapping results only)
    overlapping_rows = [
        row for row in concordance
        if row.llm_result_id is not None and row.peer_result_id is not None
    ]

    if overlapping_rows:
        agrees = sum(1 for row in overlapping_rows if row.agreement_status == "agree")
        disagrees = sum(1 for row in overlapping_rows if row.agreement_status == "disagree")

        agreement_pct = (agrees / len(overlapping_rows)) * 100 if overlapping_rows else 0
        disagreement_pct = (disagrees / len(overlapping_rows)) * 100 if overlapping_rows else 0
    else:
        agreement_pct = 0
        disagreement_pct = 0
        agrees = 0
        disagrees = 0

    # Category-wise comparison (SUPPORTED, UNSUPPORTED, UNCERTAIN)
    categories = ["SUPPORTED", "UNSUPPORTED", "UNCERTAIN"]
    category_metrics = {}

    for category in categories:
        # Count matches for this category
        matches = sum(
            1 for row in overlapping_rows
            if row.llm_status == category
            and row.peer_status == category
            and row.agreement_status == "agree"
        )

        # Count all LLM results with this category
        llm_with_category = sum(
            1 for row in overlapping_rows
            if row.llm_status == category
        )

        # Count all peer results with this category
        peer_with_category = sum(
            1 for row in overlapping_rows
            if row.peer_status == category
        )

        # Calculate precision, recall, F1
        precision = (matches / llm_with_category * 100) if llm_with_category > 0 else 0
        recall = (matches / peer_with_category * 100) if peer_with_category > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        category_metrics[category] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "llm_count": llm_with_category,
            "peer_count": peer_with_category,
            "matches": matches,
        }

    # Coverage metrics (what percentage of each set has at least one match)
    coverage_1_by_2 = (n_overlap_llm / n1 * 100) if n1 > 0 else 0
    coverage_2_by_1 = (n_overlap_peer / n2 * 100) if n2 > 0 else 0

    # Build summary
    return {
        # Basic counts
        "n1": n1,
        "n2": n2,
        "n_overlap_llm": n_overlap_llm,
        "n_overlap_peer": n_overlap_peer,
        "n_only1": n_only1,
        "n_only2": n_only2,

        # Overall agreement
        "agreement_count": agrees,
        "disagreement_count": disagrees,
        "agreement_pct": agreement_pct,
        "disagreement_pct": disagreement_pct,

        # Category-wise
        "category_metrics": category_metrics,

        # Coverage
        "coverage_1_by_2": coverage_1_by_2,
        "coverage_2_by_1": coverage_2_by_1,
    }


def format_metrics_report(metrics: Dict[str, Any]) -> str:
    """Format metrics dictionary into a human-readable report.

    Args:
        metrics: Metrics dictionary from calculate_comparison_metrics

    Returns:
        Formatted multi-line string report
    """
    lines = []

    lines.append("=" * 60)
    lines.append("COMPARISON METRICS")
    lines.append("=" * 60)

    # Basic counts
    lines.append("")
    lines.append("Basic Counts:")
    lines.append(f"  N₁ (LLM results):          {metrics['n1']}")
    lines.append(f"  N₂ (Peer results):         {metrics['n2']}")
    lines.append(f"  N_overlap_LLM (LLM w/ match): {metrics['n_overlap_llm']}")
    lines.append(f"  N_overlap_Peer (Peer w/ match): {metrics['n_overlap_peer']}")
    lines.append(f"  N_only₁ (only LLM):        {metrics['n_only1']}")
    lines.append(f"  N_only₂ (only Peer):       {metrics['n_only2']}")

    # Classification agreement
    lines.append("")
    lines.append("Classification Agreement (for overlapping results):")
    lines.append(f"  Agreement:                 {metrics['agreement_count']} ({metrics['agreement_pct']:.1f}%)")
    lines.append(f"  Disagreement:              {metrics['disagreement_count']} ({metrics['disagreement_pct']:.1f}%)")

    # Category-wise
    lines.append("")
    lines.append("Category-wise Metrics:")
    for category, cat_metrics in metrics['category_metrics'].items():
        lines.append(f"  {category}:")
        lines.append(f"    Precision:               {cat_metrics['precision']:.1f}%")
        lines.append(f"    Recall:                  {cat_metrics['recall']:.1f}%")
        lines.append(f"    F1:                      {cat_metrics['f1']:.1f}")
        lines.append(f"    LLM count:               {cat_metrics['llm_count']}")
        lines.append(f"    Peer count:              {cat_metrics['peer_count']}")
        lines.append(f"    Matches:                 {cat_metrics['matches']}")

    # Coverage
    lines.append("")
    lines.append("Coverage Metrics:")
    lines.append(f"  Set 1 covered by Set 2:    {metrics['coverage_1_by_2']:.1f}%")
    lines.append(f"  Set 2 covered by Set 1:    {metrics['coverage_2_by_1']:.1f}%")

    # Summary line
    lines.append("")
    lines.append("Summary:")
    # Total unique results = all from set 1 + all from set 2 (no double counting needed in concordance)
    total = metrics['n1'] + metrics['n2']
    num_only = metrics['n_only1'] + metrics['n_only2']

    # Calculate percentages based on concordance rows with matches
    total_matched_rows = metrics['agreement_count'] + metrics['disagreement_count']
    agree_pct = metrics['agreement_pct']
    disagree_pct = metrics['disagreement_pct']

    summary = (
        f"Of {total_matched_rows} concordance row pairs, {metrics['agreement_count']} ({agree_pct:.1f}%) agree, "
        f"{metrics['disagreement_count']} ({disagree_pct:.1f}%) disagree. "
        f"{num_only} results from {metrics['n1']} LLM + {metrics['n2']} peer = {total} total appear in only one set."
    )
    lines.append(f"  {summary}")

    # Per-category agreement
    cat_agreement = []
    for category in ["SUPPORTED", "UNSUPPORTED", "UNCERTAIN"]:
        cat_metrics = metrics['category_metrics'][category]
        if cat_metrics['llm_count'] > 0 or cat_metrics['peer_count'] > 0:
            cat_agreement.append(f"{category.lower()} {cat_metrics['f1']:.0f}%")

    if cat_agreement:
        lines.append(f"  Agreement by class: {', '.join(cat_agreement)}")

    lines.append("=" * 60)

    return "\n".join(lines)
