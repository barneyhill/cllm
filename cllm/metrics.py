"""
Metrics module for tracking LLM usage and costs.

Provides unified metrics tracking across all workflow steps:
- Token usage (input, cached, output)
- Cost breakdown by token type
- Processing time
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Union
from datetime import datetime

from anthropic.types import Message
from litellm import model_cost


# ============================================================================
# Helper Functions
# ============================================================================

def _get(obj, key, default=0):
    """Get attribute from object or dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ============================================================================
# Cost Breakdown Functions
# ============================================================================

def anthropic_cost_breakdown(message: Message, model_key: str = "claude-sonnet-4-5") -> Dict[str, Any]:
    """
    Calculate cost breakdown for Anthropic API call.

    Args:
        message: Anthropic Message object with usage information
        model_key: Model identifier for litellm pricing

    Returns:
        Dictionary with token counts and costs
    """
    prices = model_cost[model_key]
    limit = prices.get("max_input_tokens", 200_000)

    u = message.usage
    input_tokens = int(_get(u, "input_tokens"))
    output_tokens = int(_get(u, "output_tokens"))
    cache_read_input_tokens = int(_get(u, "cache_read_input_tokens"))
    cache_creation_input_tokens = int(_get(u, "cache_creation_input_tokens"))
    cached_tokens = cache_read_input_tokens + cache_creation_input_tokens
    total_tokens = input_tokens + output_tokens

    # Split base vs above 200k
    def split(n):
        return (min(n, limit), max(n - limit, 0))

    in_b, in_a = split(input_tokens)
    out_b, out_a = split(output_tokens)
    cr_b, cr_a = split(cache_read_input_tokens)
    cc_b, cc_a = split(cache_creation_input_tokens)

    # Prices
    in_p = prices["input_cost_per_token"]
    in_p_hi = prices.get("input_cost_per_token_above_200k_tokens", in_p)
    out_p = prices["output_cost_per_token"]
    out_p_hi = prices.get("output_cost_per_token_above_200k_tokens", out_p)
    cr_p = prices.get("cache_read_input_token_cost", 0.0)
    cr_p_hi = prices.get("cache_read_input_token_cost_above_200k_tokens", cr_p)
    cc_p = prices.get("cache_creation_input_token_cost", 0.0)
    cc_p_hi = prices.get("cache_creation_input_token_cost_above_200k_tokens", cc_p)

    # Costs
    input_cost = in_b * in_p + in_a * in_p_hi
    cached_cost = cr_b * cr_p + cr_a * cr_p_hi + cc_b * cc_p + cc_a * cc_p_hi
    output_cost = out_b * out_p + out_a * out_p_hi
    total_cost = input_cost + cached_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_cost": input_cost,
        "cached_cost": cached_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def openai_cost_breakdown(response: Dict[str, Any], model_key: str = "gpt-5", service_tier: str = "standard") -> Dict[str, Any]:
    """
    Calculate cost breakdown for OpenAI API call.

    Args:
        response: OpenAI completion response dict with usage information
        model_key: Model identifier for litellm pricing
        service_tier: Service tier (standard, flex, priority)

    Returns:
        Dictionary with token counts and costs
    """
    prices = model_cost[model_key]

    def tier_price(base):
        if service_tier == "flex":
            return prices.get(f"{base}_flex", prices[base])
        if service_tier == "priority":
            return prices.get(f"{base}_priority", prices[base])
        return prices[base]

    in_p = tier_price("input_cost_per_token")
    out_p = tier_price("output_cost_per_token")
    cache_p = tier_price("cache_read_input_token_cost") if "cache_read_input_token_cost" in prices else in_p

    u = response["usage"]
    input_tokens = int(u.get("input_tokens", 0) or u.get("prompt_tokens", 0))
    output_tokens = int(u.get("output_tokens", 0) or u.get("completion_tokens", 0))
    cached_tokens = int((u.get("input_tokens_details", {}) or {}).get("cached_tokens", 0))
    total_tokens = input_tokens + output_tokens
    uncached_input_tokens = max(input_tokens - cached_tokens, 0)

    input_cost = uncached_input_tokens * in_p
    cached_cost = cached_tokens * cache_p
    output_cost = output_tokens * out_p
    total_cost = input_cost + cached_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_cost": input_cost,
        "cached_cost": cached_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


# ============================================================================
# Metrics Saving
# ============================================================================

def save_metrics(
    output_path: Path,
    command: str,
    model: str,
    provider: str,
    response: Union[Message, Dict[str, Any]],
    processing_time_sec: float,
    service_tier: str = "standard"
) -> None:
    """
    Save metrics to JSON file.

    Args:
        output_path: Path to save metrics JSON file
        command: The cllm command that was executed
        model: Model identifier string
        provider: Provider name (anthropic or openai)
        response: Raw API response (Message for Anthropic, dict for OpenAI)
        processing_time_sec: Time taken for the LLM call in seconds
        service_tier: OpenAI service tier (only used for openai provider)
    """
    # Get cost breakdown based on provider
    if provider == "anthropic":
        # Map model name to litellm key
        model_key = model.replace("claude-", "claude-")  # litellm uses same format
        cost_breakdown = anthropic_cost_breakdown(response, model_key=model_key)
    elif provider == "openai":
        # Convert response to dict if needed
        response_dict = response if isinstance(response, dict) else {
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "input_tokens_details": getattr(response.usage, "prompt_tokens_details", {}),
            }
        }
        model_key = model  # OpenAI uses model name directly
        cost_breakdown = openai_cost_breakdown(response_dict, model_key=model_key, service_tier=service_tier)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Build full metrics dictionary
    metrics = {
        "command": command,
        "model": model,
        "created_at": int(datetime.now().timestamp()),
        **cost_breakdown,
        "processing_time_sec": processing_time_sec,
    }

    # Write to file
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


# ============================================================================
# Timing Context Manager
# ============================================================================

class Timer:
    """Context manager for timing operations."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        return False
