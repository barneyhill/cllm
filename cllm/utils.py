"""Utility functions for CLLM - metrics calculation, UUID generation, and prompt hashing."""

import base64
import hashlib
import io
import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import urllib.request

from PIL import Image

from .models import LLMResultV3, LLMResultsConcordanceRow


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


def generate_prompt_id(prompt_text: str, model: str) -> str:
    """Generate a deterministic ID for a prompt based on its text and model.

    Uses SHA256 hash of (prompt_text + model) and takes first 16 characters.

    Args:
        prompt_text: The full prompt text
        model: The model name

    Returns:
        16-character hex string
    """
    combined = f"{prompt_text}|{model}"
    hash_obj = hashlib.sha256(combined.encode('utf-8'))
    return hash_obj.hexdigest()[:16]


def get_current_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.utcnow().isoformat() + 'Z'


def is_video_url(url: str) -> bool:
    """Check if URL represents video file or video thumbnail.

    Detects video extensions anywhere in the URL path, including thumbnails
    like video.mp4.jpg which are JPEG images of video files that cannot be
    processed by vision models.

    Args:
        url: URL to check

    Returns:
        True if URL contains video extension, False otherwise

    Example:
        >>> is_video_url("https://example.com/video.mp4")
        True
        >>> is_video_url("https://example.com/video.mp4.jpg")
        True
        >>> is_video_url("https://example.com/image.jpg")
        False
    """
    # Pattern matches video extension anywhere in URL:
    # .mp4, .mov, .avi, .webm, .mkv, .flv, .wmv, .m4v
    # Followed by: dot, slash, query param, or end of string
    video_pattern = r'\.(mp4|mov|avi|webm|mkv|flv|wmv|m4v)(?:\.|/|\?|$)'
    return bool(re.search(video_pattern, url, re.IGNORECASE))


def extract_figure_urls_from_markdown(markdown_text: str) -> List[str]:
    """Extract figure URLs from markdown text.

    Matches markdown image syntax: ![description](url)
    Filters for common image extensions: jpg, jpeg, png, gif, webp
    Excludes video files and video thumbnails

    Args:
        markdown_text: Full markdown text to search

    Returns:
        List of figure URLs found in the markdown (excluding videos)

    Example:
        >>> text = "![Figure 1](https://example.com/fig1.jpg)"
        >>> extract_figure_urls_from_markdown(text)
        ['https://example.com/fig1.jpg']
    """
    # Pattern matches: ![any text](http(s)://url.extension)
    # where extension is one of the supported image formats
    pattern = r"!\[([^\]]*)\]\((https?://[^\)]+\.(?:jpg|jpeg|png|gif|webp))\)"

    matches = re.findall(pattern, markdown_text, re.IGNORECASE)

    # Extract just the URLs (second element of each match tuple)
    # and filter out video URLs
    urls = [match[1] for match in matches if not is_video_url(match[1])]

    return urls


def extract_figures_with_metadata(markdown_text: str) -> List[Tuple[str, str]]:
    """Extract figures with their descriptions and URLs from markdown.

    Excludes video files and video thumbnails.

    Args:
        markdown_text: Full markdown text to search

    Returns:
        List of (description, url) tuples (excluding videos)

    Example:
        >>> text = "![Figure 1: Cell diagram](https://example.com/fig1.jpg)"
        >>> extract_figures_with_metadata(text)
        [('Figure 1: Cell diagram', 'https://example.com/fig1.jpg')]
    """
    pattern = r"!\[([^\]]*)\]\((https?://[^\)]+\.(?:jpg|jpeg|png|gif|webp))\)"

    matches = re.findall(pattern, markdown_text, re.IGNORECASE)

    # Filter out video URLs
    filtered_matches = [match for match in matches if not is_video_url(match[1])]

    return filtered_matches


def fetch_and_resize_image(url: str, max_size: int = 2000) -> Optional[Tuple[str, str]]:
    """Fetch an image from URL, resize it to fit within max_size, and return base64.

    Anthropic API requires images to be max 2000x2000px when sending multiple images.
    This function downloads the image, resizes it maintaining aspect ratio, and
    converts to base64 JPEG format.

    Args:
        url: URL of the image to fetch
        max_size: Maximum dimension (width or height) in pixels

    Returns:
        Tuple of (base64_data, media_type) or None if fetch/resize fails

    Example:
        >>> data, media_type = fetch_and_resize_image("https://example.com/image.jpg")
        >>> print(media_type)
        'image/jpeg'
    """
    try:
        # Fetch image from URL
        with urllib.request.urlopen(url, timeout=10) as response:
            image_data = response.read()

        # Open with Pillow
        image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if necessary (handles RGBA, P, etc.)
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')

        # Calculate new size maintaining aspect ratio
        width, height = image.size
        if width > max_size or height > max_size:
            # Resize to fit within max_size x max_size
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))

            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert to JPEG and encode as base64
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85, optimize=True)
        buffer.seek(0)

        base64_data = base64.b64encode(buffer.read()).decode('utf-8')

        return base64_data, 'image/jpeg'

    except Exception as e:
        # Log error but don't crash - just skip this image
        print(f"Warning: Failed to fetch/resize image {url}: {e}", file=__import__('sys').stderr)
        return None


def process_figure_urls_for_api(figure_urls: List[str], max_size: int = 2000) -> List[Tuple[str, str]]:
    """Process a list of figure URLs for API consumption.

    Fetches and resizes images, returning base64-encoded data.
    Skips any images that fail to download or process.

    Args:
        figure_urls: List of image URLs
        max_size: Maximum dimension (width or height) in pixels

    Returns:
        List of (base64_data, media_type) tuples for successfully processed images

    Example:
        >>> urls = ["https://example.com/fig1.jpg", "https://example.com/fig2.jpg"]
        >>> processed = process_figure_urls_for_api(urls)
        >>> len(processed)
        2
    """
    processed_images = []

    for url in figure_urls:
        result = fetch_and_resize_image(url, max_size)
        if result:
            processed_images.append(result)

    return processed_images


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
        row.openeval_result_id for row in concordance
        if row.openeval_result_id is not None and row.peer_result_id is not None
    )
    peer_ids_with_match = set(
        row.peer_result_id for row in concordance
        if row.peer_result_id is not None and row.openeval_result_id is not None
    )

    n_overlap_llm = len(llm_ids_with_match)
    n_overlap_peer = len(peer_ids_with_match)

    # Count unique result IDs that appear only in one set
    llm_ids_only = set(
        row.openeval_result_id for row in concordance
        if row.openeval_result_id is not None and row.peer_result_id is None
    )
    peer_ids_only = set(
        row.peer_result_id for row in concordance
        if row.peer_result_id is not None and row.openeval_result_id is None
    )

    n_only1 = len(llm_ids_only)
    n_only2 = len(peer_ids_only)

    # Create lookup dictionaries for evaluation_types
    llm_results_dict = {r.result_id: r for r in llm_results}
    peer_results_dict = {r.result_id: r for r in peer_results}

    # Classification agreement (for overlapping results only)
    overlapping_rows = [
        row for row in concordance
        if row.openeval_result_id is not None and row.peer_result_id is not None
    ]

    if overlapping_rows:
        agrees = sum(1 for row in overlapping_rows if row.comparison_type == "agree")
        disagrees = sum(1 for row in overlapping_rows if row.comparison_type == "disagree")

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
            if row.openeval_result_id in llm_results_dict
            and row.peer_result_id in peer_results_dict
            and llm_results_dict[row.openeval_result_id].evaluation_type == category
            and peer_results_dict[row.peer_result_id].evaluation_type == category
            and row.comparison_type == "agree"
        )

        # Count all LLM results with this category
        llm_with_category = sum(
            1 for row in overlapping_rows
            if row.openeval_result_id in llm_results_dict
            and llm_results_dict[row.openeval_result_id].evaluation_type == category
        )

        # Count all peer results with this category
        peer_with_category = sum(
            1 for row in overlapping_rows
            if row.peer_result_id in peer_results_dict
            and peer_results_dict[row.peer_result_id].evaluation_type == category
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
