#!/usr/bin/env python3
"""
CLLM - Claim LLM CLI Tool

A command-line tool for scientific claim extraction, evaluation, and comparison.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import click

from .config import config
from .verification import (
    extract_claims,
    llm_group_claims_into_results,
    peer_review_group_claims_into_results,
    compare_results,
)
from .models import (
    LLMClaimV3,
    LLMResultV3,
)


@click.group()
@click.version_option(version="1.0.0", prog_name="cllm")
def cli():
    """
    CLLM - Claim LLM CLI Tool

    A tool for extracting, evaluating, and comparing scientific claims.
    """
    pass


@cli.command()
@click.argument("manuscript", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), required=True, help="Output JSON file for claims")
@click.option("-m", "--metadata", type=click.Path(path_type=Path), help="Optional JSON file for metadata")
def extract(manuscript: Path, output: Path, metadata: Optional[Path]):
    """
    Extract atomic factual claims from a manuscript.

    Example:
        cllm extract -o claims.json manuscript.txt
    """
    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
        sys.exit(1)

    click.echo(f"📄 Reading manuscript from: {manuscript}")

    # Read manuscript text
    try:
        manuscript_text = manuscript.read_text(encoding="utf-8")
    except Exception as e:
        click.echo(f"❌ Error reading manuscript: {e}", err=True)
        sys.exit(1)

    click.echo(f"🔍 Extracting claims from manuscript ({len(manuscript_text)} characters)...")

    # Extract claims
    try:
        claims, processing_time = extract_claims(manuscript_text)
        click.echo(f"✅ Extracted {len(claims)} claims in {processing_time:.2f}s")
    except Exception as e:
        click.echo(f"❌ Error extracting claims: {e}", err=True)
        sys.exit(1)

    # Convert to JSON (just array of claims)
    claims_array = [
        {
            "claim_id": c.claim_id,
            "claim": c.claim,
            "claim_type": c.claim_type,
            "source_text": c.source_text,
            "evidence_type": c.evidence_type,
            "evidence_reasoning": c.evidence_reasoning,
        }
        for c in claims
    ]

    # Write claims to output file
    try:
        output.write_text(json.dumps(claims_array, indent=2), encoding="utf-8")
        click.echo(f"💾 Saved {len(claims)} claims to: {output}")
    except Exception as e:
        click.echo(f"❌ Error writing output: {e}", err=True)
        sys.exit(1)

    # Write metadata if requested
    if metadata:
        metadata_data = {
            "command": "extract",
            "manuscript_file": str(manuscript),
            "manuscript_length": len(manuscript_text),
            "num_claims": len(claims),
            "processing_time_seconds": processing_time,
        }
        try:
            metadata.write_text(json.dumps(metadata_data, indent=2), encoding="utf-8")
            click.echo(f"📊 Saved metadata to: {metadata}")
        except Exception as e:
            click.echo(f"⚠️  Warning: Could not write metadata: {e}", err=True)


@cli.command()
@click.argument("manuscript", type=click.Path(exists=True, path_type=Path))
@click.option("-c", "--claims", type=click.Path(exists=True, path_type=Path), required=True, help="Input claims JSON file")
@click.option("-p", "--peer-reviews", type=click.Path(exists=True, path_type=Path), help="Peer review file (if provided, evaluates from reviewer perspective)")
@click.option("-o", "--output", type=click.Path(path_type=Path), required=True, help="Output JSON file for evaluations")
@click.option("-m", "--metadata", type=click.Path(path_type=Path), help="Optional JSON file for metadata")
def eval(manuscript: Path, claims: Path, peer_reviews: Optional[Path], output: Path, metadata: Optional[Path]):
    """
    Evaluate claims and group them into results.

    Without -p: LLM evaluates the claims based on manuscript.
    With -p: Groups claims based on peer review commentary.

    Examples:
        cllm eval -c claims.json -o eval_llm.json manuscript.txt
        cllm eval -c claims.json -p reviews.txt -o eval_peers.json manuscript.txt
    """
    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
        sys.exit(1)

    # Read manuscript
    click.echo(f"📄 Reading manuscript from: {manuscript}")
    try:
        manuscript_text = manuscript.read_text(encoding="utf-8")
    except Exception as e:
        click.echo(f"❌ Error reading manuscript: {e}", err=True)
        sys.exit(1)

    # Read claims
    click.echo(f"📋 Reading claims from: {claims}")
    try:
        claims_data = json.loads(claims.read_text(encoding="utf-8"))
        if not isinstance(claims_data, list):
            raise ValueError("Invalid claims format: expected JSON array")

        claims_list = [
            LLMClaimV3(
                claim_id=c["claim_id"],
                claim=c["claim"],
                claim_type=c["claim_type"],
                source_text=c["source_text"],
                evidence_type=c["evidence_type"],
                evidence_reasoning=c["evidence_reasoning"],
            )
            for c in claims_data
        ]
        click.echo(f"   Loaded {len(claims_list)} claims")
    except Exception as e:
        click.echo(f"❌ Error reading claims: {e}", err=True)
        sys.exit(1)

    # Evaluate based on mode
    if peer_reviews:
        # Peer review mode
        click.echo(f"📝 Reading peer reviews from: {peer_reviews}")
        try:
            review_text = peer_reviews.read_text(encoding="utf-8")
        except Exception as e:
            click.echo(f"❌ Error reading peer reviews: {e}", err=True)
            sys.exit(1)

        click.echo(f"🔍 Grouping claims based on peer review commentary...")
        try:
            results, processing_time = peer_review_group_claims_into_results(claims_list, review_text)
            click.echo(f"✅ Created {len(results)} peer review results in {processing_time:.2f}s")
            eval_source = "PEER_REVIEW"
        except Exception as e:
            click.echo(f"❌ Error evaluating with peer reviews: {e}", err=True)
            sys.exit(1)
    else:
        # LLM evaluation mode
        click.echo(f"🤖 Evaluating claims with LLM...")
        try:
            results, processing_time = llm_group_claims_into_results(manuscript_text, claims_list)
            click.echo(f"✅ Created {len(results)} LLM results in {processing_time:.2f}s")
            eval_source = "LLM"
        except Exception as e:
            click.echo(f"❌ Error evaluating with LLM: {e}", err=True)
            sys.exit(1)

    # Convert to JSON (just array of results)
    results_array = [
        {
            "result_id": r.result_id,
            "claim_ids": r.claim_ids,
            "reviewer_id": r.reviewer_id,
            "reviewer_name": r.reviewer_name,
            "status": r.status,
            "status_reasoning": r.status_reasoning,
        }
        for r in results
    ]

    # Write results to output file
    try:
        output.write_text(json.dumps(results_array, indent=2), encoding="utf-8")
        click.echo(f"💾 Saved {len(results)} results to: {output}")
    except Exception as e:
        click.echo(f"❌ Error writing output: {e}", err=True)
        sys.exit(1)

    # Write metadata if requested
    if metadata:
        metadata_data = {
            "command": "eval",
            "manuscript_file": str(manuscript),
            "claims_file": str(claims),
            "peer_reviews_file": str(peer_reviews) if peer_reviews else None,
            "source": eval_source,
            "num_results": len(results),
            "processing_time_seconds": processing_time,
        }
        try:
            metadata.write_text(json.dumps(metadata_data, indent=2), encoding="utf-8")
            click.echo(f"📊 Saved metadata to: {metadata}")
        except Exception as e:
            click.echo(f"⚠️  Warning: Could not write metadata: {e}", err=True)


@cli.command()
@click.argument("eval_peers", type=click.Path(exists=True, path_type=Path))
@click.argument("eval_llm", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), required=True, help="Output JSON file for comparison")
@click.option("-m", "--metadata", type=click.Path(path_type=Path), help="Optional JSON file for metadata")
def cmp(eval_peers: Path, eval_llm: Path, output: Path, metadata: Optional[Path]):
    """
    Compare peer review and LLM evaluations.

    Example:
        cllm cmp eval_peers.json eval_llm.json -o compare.json
    """
    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
        sys.exit(1)

    # Read peer review results
    click.echo(f"📝 Reading peer review results from: {eval_peers}")
    try:
        peers_data = json.loads(eval_peers.read_text(encoding="utf-8"))
        if not isinstance(peers_data, list):
            raise ValueError("Invalid results format: expected JSON array")

        peer_results = [
            LLMResultV3(
                result_id=r["result_id"],
                claim_ids=r["claim_ids"],
                reviewer_id=r["reviewer_id"],
                reviewer_name=r["reviewer_name"],
                status=r["status"],
                status_reasoning=r["status_reasoning"],
            )
            for r in peers_data
        ]
        click.echo(f"   Loaded {len(peer_results)} peer results")
    except Exception as e:
        click.echo(f"❌ Error reading peer review results: {e}", err=True)
        sys.exit(1)

    # Read LLM results
    click.echo(f"🤖 Reading LLM results from: {eval_llm}")
    try:
        llm_data = json.loads(eval_llm.read_text(encoding="utf-8"))
        if not isinstance(llm_data, list):
            raise ValueError("Invalid results format: expected JSON array")

        llm_results = [
            LLMResultV3(
                result_id=r["result_id"],
                claim_ids=r["claim_ids"],
                reviewer_id=r["reviewer_id"],
                reviewer_name=r["reviewer_name"],
                status=r["status"],
                status_reasoning=r["status_reasoning"],
            )
            for r in llm_data
        ]
        click.echo(f"   Loaded {len(llm_results)} LLM results")
    except Exception as e:
        click.echo(f"❌ Error reading LLM results: {e}", err=True)
        sys.exit(1)

    # Compare results
    click.echo(f"⚖️  Comparing results...")
    try:
        concordance, processing_time = compare_results(llm_results, peer_results)
        click.echo(f"✅ Generated {len(concordance)} concordance rows in {processing_time:.2f}s")
    except Exception as e:
        click.echo(f"❌ Error comparing results: {e}", err=True)
        sys.exit(1)

    # Calculate metrics
    agreements = sum(1 for c in concordance if c.agreement_status == "agree")
    disagreements = sum(1 for c in concordance if c.agreement_status == "disagree")
    partial = sum(1 for c in concordance if c.agreement_status == "partial")
    agreement_rate = (agreements / len(concordance) * 100) if concordance else 0.0

    click.echo(f"📊 Agreement rate: {agreement_rate:.1f}%")
    click.echo(f"   Agreements: {agreements}, Disagreements: {disagreements}, Partial: {partial}")

    # Convert to JSON (just array of concordance rows)
    concordance_array = [
        {
            "llm_result_id": c.llm_result_id,
            "peer_result_id": c.peer_result_id,
            "llm_status": c.llm_status,
            "peer_status": c.peer_status,
            "agreement_status": c.agreement_status,
            "notes": c.notes,
        }
        for c in concordance
    ]

    # Write concordance to output file
    try:
        output.write_text(json.dumps(concordance_array, indent=2), encoding="utf-8")
        click.echo(f"💾 Saved {len(concordance)} concordance rows to: {output}")
    except Exception as e:
        click.echo(f"❌ Error writing output: {e}", err=True)
        sys.exit(1)

    # Write metadata if requested
    if metadata:
        metadata_data = {
            "command": "cmp",
            "eval_peers_file": str(eval_peers),
            "eval_llm_file": str(eval_llm),
            "total_comparisons": len(concordance),
            "agreements": agreements,
            "disagreements": disagreements,
            "partial": partial,
            "agreement_rate": agreement_rate,
            "processing_time_seconds": processing_time,
        }
        try:
            metadata.write_text(json.dumps(metadata_data, indent=2), encoding="utf-8")
            click.echo(f"📊 Saved metadata to: {metadata}")
        except Exception as e:
            click.echo(f"⚠️  Warning: Could not write metadata: {e}", err=True)


if __name__ == "__main__":
    cli()
