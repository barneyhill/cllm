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
    STAGE1_PROMPT_TEMPLATE,
    STAGE2_PROMPT_TEMPLATE,
    STAGE3_PROMPT_TEMPLATE,
    STAGE4_PROMPT_TEMPLATE,
)
from .models import (
    LLMClaimV3,
    LLMResultV3,
)
from .report import json_to_pdf_table
from .db_export import export_to_database_format, save_db_export


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
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging (token counts, timing)")
def extract(manuscript: Path, output: Path, metadata: Optional[Path], verbose: bool):
    """
    Extract atomic factual claims from a manuscript.

    Example:
        cllm extract -o claims.json manuscript.txt
        cllm extract -o claims.json -v manuscript.txt  # with verbose logging
    """
    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
        sys.exit(1)

    if not verbose:
        click.echo(f"📄 Reading manuscript from: {manuscript}")

    # Read manuscript text
    try:
        manuscript_text = manuscript.read_text(encoding="utf-8")
    except Exception as e:
        click.echo(f"❌ Error reading manuscript: {e}", err=True)
        sys.exit(1)

    if not verbose:
        click.echo(f"🔍 Extracting claims from manuscript ({len(manuscript_text)} characters)...")

    # Extract claims (request metrics if metadata file specified)
    try:
        claims, processing_time, metrics, raw_response = extract_claims(manuscript_text, verbose=verbose, return_metrics=(metadata is not None))
        if not verbose:
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
        # Build full command string
        cmd_parts = ["cllm", "extract", str(manuscript), "-o", str(output)]
        if metadata:
            cmd_parts.extend(["-m", str(metadata)])
        if verbose:
            cmd_parts.append("-v")

        metadata_data = {
            "command": " ".join(cmd_parts),
        }

        # Add metrics if available
        if metrics:
            metadata_data.update(metrics)

        try:
            metadata.write_text(json.dumps(metadata_data, indent=2), encoding="utf-8")
            if not verbose:
                click.echo(f"📊 Saved metadata to: {metadata}")
        except Exception as e:
            click.echo(f"⚠️  Warning: Could not write metadata: {e}", err=True)


@cli.command()
@click.argument("manuscript", type=click.Path(exists=True, path_type=Path))
@click.option("-c", "--claims", type=click.Path(exists=True, path_type=Path), required=True, help="Input claims JSON file")
@click.option("-p", "--peer-reviews", type=click.Path(exists=True, path_type=Path), help="Peer review file (if provided, evaluates from reviewer perspective)")
@click.option("-o", "--output", type=click.Path(path_type=Path), required=True, help="Output JSON file for evaluations")
@click.option("-m", "--metadata", type=click.Path(path_type=Path), help="Optional JSON file for metadata")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging (token counts, timing)")
def eval(manuscript: Path, claims: Path, peer_reviews: Optional[Path], output: Path, metadata: Optional[Path], verbose: bool):
    """
    Evaluate claims and group them into results.

    Without -p: LLM evaluates the claims based on manuscript.
    With -p: Groups claims based on peer review commentary.

    Examples:
        cllm eval -c claims.json -o eval_llm.json manuscript.txt
        cllm eval -c claims.json -p reviews.txt -o eval_peers.json manuscript.txt
        cllm eval -c claims.json -o eval_llm.json -v manuscript.txt  # with verbose logging
    """
    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
        sys.exit(1)

    # Read manuscript
    if not verbose:
        click.echo(f"📄 Reading manuscript from: {manuscript}")
    try:
        manuscript_text = manuscript.read_text(encoding="utf-8")
    except Exception as e:
        click.echo(f"❌ Error reading manuscript: {e}", err=True)
        sys.exit(1)

    # Read claims
    if not verbose:
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
        if not verbose:
            click.echo(f"   Loaded {len(claims_list)} claims")
    except Exception as e:
        click.echo(f"❌ Error reading claims: {e}", err=True)
        sys.exit(1)

    # Evaluate based on mode
    if peer_reviews:
        # Peer review mode
        if not verbose:
            click.echo(f"📝 Reading peer reviews from: {peer_reviews}")
        try:
            review_text = peer_reviews.read_text(encoding="utf-8")
        except Exception as e:
            click.echo(f"❌ Error reading peer reviews: {e}", err=True)
            sys.exit(1)

        if not verbose:
            click.echo(f"🔍 Grouping claims based on peer review commentary...")
        try:
            results, processing_time, metrics_eval = peer_review_group_claims_into_results(
                claims_list, review_text, verbose=verbose, return_metrics=(metadata is not None)
            )
            if not verbose:
                click.echo(f"✅ Created {len(results)} peer review results in {processing_time:.2f}s")
            eval_source = "PEER_REVIEW"
        except Exception as e:
            click.echo(f"❌ Error evaluating with peer reviews: {e}", err=True)
            sys.exit(1)
    else:
        # LLM evaluation mode
        if not verbose:
            click.echo(f"🤖 Evaluating claims with LLM...")
        try:
            results, processing_time, metrics_eval = llm_group_claims_into_results(
                manuscript_text, claims_list, verbose=verbose, return_metrics=(metadata is not None)
            )
            if not verbose:
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
            "result": r.result,
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
        # Build full command string
        cmd_parts = ["cllm", "eval", str(manuscript), "-c", str(claims)]
        if peer_reviews:
            cmd_parts.extend(["-p", str(peer_reviews)])
        cmd_parts.extend(["-o", str(output)])
        if metadata:
            cmd_parts.extend(["-m", str(metadata)])
        if verbose:
            cmd_parts.append("-v")

        metadata_data = {
            "command": " ".join(cmd_parts),
        }

        # Add metrics if available
        if metrics_eval:
            metadata_data.update(metrics_eval)

        try:
            metadata.write_text(json.dumps(metadata_data, indent=2), encoding="utf-8")
            if not verbose:
                click.echo(f"📊 Saved metadata to: {metadata}")
        except Exception as e:
            click.echo(f"⚠️  Warning: Could not write metadata: {e}", err=True)


@cli.command()
@click.argument("eval_peers", type=click.Path(exists=True, path_type=Path))
@click.argument("eval_llm", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), required=True, help="Output JSON file for comparison")
@click.option("-m", "--metadata", type=click.Path(path_type=Path), help="Optional JSON file for metadata")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging (token counts, timing, metrics)")
def cmp(eval_peers: Path, eval_llm: Path, output: Path, metadata: Optional[Path], verbose: bool):
    """
    Compare peer review and LLM evaluations.

    Example:
        cllm cmp eval_peers.json eval_llm.json -o compare.json
        cllm cmp eval_peers.json eval_llm.json -o compare.json -v  # with verbose logging and metrics
    """
    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
        sys.exit(1)

    # Read peer review results
    if not verbose:
        click.echo(f"📝 Reading peer review results from: {eval_peers}")
    try:
        peers_data = json.loads(eval_peers.read_text(encoding="utf-8"))
        if not isinstance(peers_data, list):
            raise ValueError("Invalid results format: expected JSON array")

        peer_results = [
            LLMResultV3(
                result_id=r["result_id"],
                claim_ids=r["claim_ids"],
                result=r["result"],
                reviewer_id=r["reviewer_id"],
                reviewer_name=r["reviewer_name"],
                status=r["status"],
                status_reasoning=r["status_reasoning"],
            )
            for r in peers_data
        ]
        if not verbose:
            click.echo(f"   Loaded {len(peer_results)} peer results")
    except Exception as e:
        click.echo(f"❌ Error reading peer review results: {e}", err=True)
        sys.exit(1)

    # Read LLM results
    if not verbose:
        click.echo(f"🤖 Reading LLM results from: {eval_llm}")
    try:
        llm_data = json.loads(eval_llm.read_text(encoding="utf-8"))
        if not isinstance(llm_data, list):
            raise ValueError("Invalid results format: expected JSON array")

        llm_results = [
            LLMResultV3(
                result_id=r["result_id"],
                claim_ids=r["claim_ids"],
                result=r["result"],
                reviewer_id=r["reviewer_id"],
                reviewer_name=r["reviewer_name"],
                status=r["status"],
                status_reasoning=r["status_reasoning"],
            )
            for r in llm_data
        ]
        if not verbose:
            click.echo(f"   Loaded {len(llm_results)} LLM results")
    except Exception as e:
        click.echo(f"❌ Error reading LLM results: {e}", err=True)
        sys.exit(1)

    # Compare results (request metrics if metadata file specified)
    if not verbose:
        click.echo(f"⚖️  Comparing results...")
    try:
        concordance, processing_time, metrics_cmp = compare_results(
            llm_results, peer_results, verbose=verbose, return_metrics=(metadata is not None)
        )
        if not verbose:
            click.echo(f"✅ Generated {len(concordance)} concordance rows in {processing_time:.2f}s")
    except Exception as e:
        click.echo(f"❌ Error comparing results: {e}", err=True)
        sys.exit(1)

    # Calculate basic metrics (only shown in non-verbose mode; verbose mode shows comprehensive metrics)
    agreements = sum(1 for c in concordance if c.agreement_status == "agree")
    disagreements = sum(1 for c in concordance if c.agreement_status == "disagree")
    disjoint = sum(1 for c in concordance if c.agreement_status == "disjoint")
    agreement_rate = (agreements / len(concordance) * 100) if concordance else 0.0

    if not verbose:
        click.echo(f"📊 Agreement rate: {agreement_rate:.1f}%")
        click.echo(f"   Agreements: {agreements}, Disagreements: {disagreements}, Disjoint: {disjoint}")

    # Convert to JSON (just array of concordance rows)
    concordance_array = [
        {
            "openeval_result_id": c.openeval_result_id,
            "peer_result_id": c.peer_result_id,
            "openeval_status": c.openeval_status,
            "peer_status": c.peer_status,
            "agreement_status": c.agreement_status,
            "comparison": c.comparison,
            "n_openeval": c.n_openeval,
            "n_peer": c.n_peer,
            "n_itx": c.n_itx,
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
        # Build full command string
        cmd_parts = ["cllm", "cmp", str(eval_peers), str(eval_llm), "-o", str(output)]
        if metadata:
            cmd_parts.extend(["-m", str(metadata)])
        if verbose:
            cmd_parts.append("-v")

        metadata_data = {
            "command": " ".join(cmd_parts),
        }

        # Add metrics if available
        if metrics_cmp:
            metadata_data.update(metrics_cmp)

        try:
            metadata.write_text(json.dumps(metadata_data, indent=2), encoding="utf-8")
            if not verbose:
                click.echo(f"📊 Saved metadata to: {metadata}")
        except Exception as e:
            click.echo(f"⚠️  Warning: Could not write metadata: {e}", err=True)


@cli.command()
@click.argument("input_json", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), required=True, help="Output PDF file")
@click.option("-f", "--format", type=click.Choice(["pdf"]), default="pdf", help="Output format (currently only PDF)")
@click.option("-t", "--type", "data_type", type=click.Choice(["comparison", "evaluation", "claim"]), required=True, help="Type of data being rendered")
def generate(input_json: Path, output: Path, format: str, data_type: str):
    """
    Generate a PDF table report from JSON data.

    Examples:
        cllm generate -o table.pdf -f pdf -t comparison compare.json
        cllm generate -o table.pdf -f pdf -t evaluation eval_llm.json
        cllm generate -o table.pdf -f pdf -t claim claims.json
    """
    click.echo(f"📄 Reading {data_type} data from: {input_json}")

    # Map data type to title
    titles = {
        "comparison": "Results Comparison Table",
        "evaluation": "Evaluation Results Table",
        "claim": "Claims Table"
    }

    try:
        json_to_pdf_table(
            json_data=input_json,
            output_filename=output,
            title=titles.get(data_type, "Data Table")
        )
        click.echo(f"✅ Generated PDF report: {output}")
    except Exception as e:
        click.echo(f"❌ Error generating PDF: {e}", err=True)
        sys.exit(1)


@cli.command(hidden=True)
@click.argument("manuscript", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_dir", type=click.Path(path_type=Path), required=True, help="Output directory")
@click.option("-p", "--peer-reviews", type=click.Path(exists=True, path_type=Path), help="Peer review file (optional, runs full workflow if provided)")
@click.option("-m", "--metrics", is_flag=True, help="Save metrics for each stage")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def workflow(manuscript: Path, output_dir: Path, peer_reviews: Optional[Path], metrics: bool, verbose: bool):
    """
    Run the CLLM workflow from manuscript to evaluation/comparison.

    Without -p: Runs stages 1-2 (extract claims, LLM evaluation)
    With -p: Runs all 4 stages (extract, LLM eval, peer eval, comparison)

    Examples:
        cllm workflow manuscript.txt -o output/                    # extract + LLM eval only
        cllm workflow manuscript.txt -o output/ -p reviews.txt     # full workflow
        cllm workflow manuscript.txt -o output/ -p reviews.txt -m  # with metrics
    """
    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    claims_file = output_dir / "claims.json"
    eval_llm_file = output_dir / "eval_llm.json"
    eval_peer_file = output_dir / "eval_peer.json"
    cmp_file = output_dir / "cmp.json"

    # Define metrics paths if requested
    metrics_extract_file = output_dir / "metrics_extract.json" if metrics else None
    metrics_eval_llm_file = output_dir / "metrics_eval_llm.json" if metrics else None
    metrics_eval_peer_file = output_dir / "metrics_eval_peer.json" if metrics else None
    metrics_cmp_file = output_dir / "metrics_cmp.json" if metrics else None

    click.echo("=" * 60)
    click.echo("CLLM WORKFLOW")
    click.echo("=" * 60)
    click.echo(f"Manuscript: {manuscript}")
    if peer_reviews:
        click.echo(f"Reviews: {peer_reviews}")
        click.echo(f"Mode: Full workflow (4 stages)")
    else:
        click.echo(f"Mode: Extract + LLM evaluation (2 stages)")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Metrics: {'enabled' if metrics else 'disabled'}")
    click.echo(f"Verbose: {'enabled' if verbose else 'disabled'}")
    click.echo("=" * 60)

    # ========================================================================
    # STAGE 1: EXTRACT CLAIMS
    # ========================================================================
    click.echo("\n📝 STAGE 1: Claim Extraction")
    click.echo("-" * 60)

    if not verbose:
        click.echo(f"📄 Reading manuscript from: {manuscript}")

    try:
        manuscript_text = manuscript.read_text(encoding="utf-8")
    except Exception as e:
        click.echo(f"❌ Error reading manuscript: {e}", err=True)
        sys.exit(1)

    if not verbose:
        click.echo(f"🔍 Extracting claims from manuscript ({len(manuscript_text)} characters)...")

    try:
        claims, processing_time, metrics_data = extract_claims(
            manuscript_text,
            verbose=verbose,
            return_metrics=metrics
        )
        if not verbose:
            click.echo(f"✅ Extracted {len(claims)} claims in {processing_time:.2f}s")
    except Exception as e:
        click.echo(f"❌ Error extracting claims: {e}", err=True)
        sys.exit(1)

    # Write claims
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

    try:
        claims_file.write_text(json.dumps(claims_array, indent=2), encoding="utf-8")
        click.echo(f"💾 Saved claims to: {claims_file}")
    except Exception as e:
        click.echo(f"❌ Error writing claims: {e}", err=True)
        sys.exit(1)

    # Write metrics if requested
    if metrics and metrics_data:
        try:
            cmd_parts = ["cllm", "extract", str(manuscript), "-o", str(claims_file)]
            if metrics_extract_file:
                cmd_parts.extend(["-m", str(metrics_extract_file)])
            if verbose:
                cmd_parts.append("-v")

            metadata_data = {"command": " ".join(cmd_parts)}
            metadata_data.update(metrics_data)

            metrics_extract_file.write_text(json.dumps(metadata_data, indent=2), encoding="utf-8")
            click.echo(f"📊 Saved metrics to: {metrics_extract_file}")
        except Exception as e:
            click.echo(f"⚠️  Warning: Could not write metrics: {e}", err=True)

    # ========================================================================
    # STAGE 2: LLM EVALUATION
    # ========================================================================
    click.echo("\n🤖 STAGE 2: LLM Evaluation")
    click.echo("-" * 60)

    if not verbose:
        click.echo(f"🤖 Evaluating claims with LLM...")

    try:
        llm_results, processing_time, metrics_data = llm_group_claims_into_results(
            manuscript_text, claims, verbose=verbose, return_metrics=metrics
        )
        if not verbose:
            click.echo(f"✅ Created {len(llm_results)} LLM results in {processing_time:.2f}s")
    except Exception as e:
        click.echo(f"❌ Error evaluating with LLM: {e}", err=True)
        sys.exit(1)

    # Write LLM results
    results_array = [
        {
            "result_id": r.result_id,
            "claim_ids": r.claim_ids,
            "result": r.result,
            "reviewer_id": r.reviewer_id,
            "reviewer_name": r.reviewer_name,
            "status": r.status,
            "status_reasoning": r.status_reasoning,
        }
        for r in llm_results
    ]

    try:
        eval_llm_file.write_text(json.dumps(results_array, indent=2), encoding="utf-8")
        click.echo(f"💾 Saved LLM results to: {eval_llm_file}")
    except Exception as e:
        click.echo(f"❌ Error writing LLM results: {e}", err=True)
        sys.exit(1)

    # Write metrics if requested
    if metrics and metrics_data:
        try:
            cmd_parts = ["cllm", "eval", str(manuscript), "-c", str(claims_file), "-o", str(eval_llm_file)]
            if metrics_eval_llm_file:
                cmd_parts.extend(["-m", str(metrics_eval_llm_file)])
            if verbose:
                cmd_parts.append("-v")

            metadata_data = {"command": " ".join(cmd_parts)}
            metadata_data.update(metrics_data)

            metrics_eval_llm_file.write_text(json.dumps(metadata_data, indent=2), encoding="utf-8")
            click.echo(f"📊 Saved metrics to: {metrics_eval_llm_file}")
        except Exception as e:
            click.echo(f"⚠️  Warning: Could not write metrics: {e}", err=True)

    # ========================================================================
    # STAGE 3 & 4: PEER REVIEW EVALUATION AND COMPARISON (only if peer reviews provided)
    # ========================================================================
    if not peer_reviews:
        # No peer reviews - skip stages 3 and 4, but generate database export
        click.echo("\n💾 Generating database export...")
        click.echo("-" * 60)

        try:
            # Collect prompts used in workflow
            model = config.anthropic_model if config.llm_provider == "anthropic" else config.openai_model
            prompts = {
                "extract": {"text": STAGE1_PROMPT_TEMPLATE, "model": model},
                "eval_llm": {"text": STAGE2_PROMPT_TEMPLATE, "model": model},
            }

            # Generate database export (no peer review or comparison data)
            db_export = export_to_database_format(
                manuscript_text=manuscript_text,
                peer_review_text=None,
                claims=claims,
                llm_results=llm_results,
                peer_results=None,
                concordance=None,
                prompts=prompts,
            )

            # Save to file
            db_export_file = output_dir / "db_export.json"
            save_db_export(db_export, db_export_file)
            click.echo(f"✅ Generated database export: {db_export_file.name}")

        except Exception as e:
            click.echo(f"⚠️  Warning: Could not generate database export: {e}", err=True)

        click.echo("\n" + "=" * 60)
        click.echo("✅ WORKFLOW COMPLETE (2 stages)")
        click.echo("=" * 60)
        click.echo(f"Output directory: {output_dir}")
        click.echo(f"  - Claims: {claims_file.name}")
        click.echo(f"  - LLM evaluation: {eval_llm_file.name}")
        click.echo(f"  - Database export: db_export.json")
        if metrics:
            click.echo(f"  - Metrics files: metrics_extract.json, metrics_eval_llm.json")
        click.echo("=" * 60)
        return

    # ========================================================================
    # STAGE 3: PEER REVIEW EVALUATION
    # ========================================================================
    click.echo("\n📝 STAGE 3: Peer Review Evaluation")
    click.echo("-" * 60)

    if not verbose:
        click.echo(f"📝 Reading peer reviews from: {peer_reviews}")

    try:
        review_text = peer_reviews.read_text(encoding="utf-8")
    except Exception as e:
        click.echo(f"❌ Error reading peer reviews: {e}", err=True)
        sys.exit(1)

    if not verbose:
        click.echo(f"🔍 Grouping claims based on peer review commentary...")

    try:
        peer_results, processing_time, metrics_data = peer_review_group_claims_into_results(
            claims, review_text, verbose=verbose, return_metrics=metrics
        )
        if not verbose:
            click.echo(f"✅ Created {len(peer_results)} peer review results in {processing_time:.2f}s")
    except Exception as e:
        click.echo(f"❌ Error evaluating with peer reviews: {e}", err=True)
        sys.exit(1)

    # Write peer results
    results_array = [
        {
            "result_id": r.result_id,
            "claim_ids": r.claim_ids,
            "result": r.result,
            "reviewer_id": r.reviewer_id,
            "reviewer_name": r.reviewer_name,
            "status": r.status,
            "status_reasoning": r.status_reasoning,
        }
        for r in peer_results
    ]

    try:
        eval_peer_file.write_text(json.dumps(results_array, indent=2), encoding="utf-8")
        click.echo(f"💾 Saved peer results to: {eval_peer_file}")
    except Exception as e:
        click.echo(f"❌ Error writing peer results: {e}", err=True)
        sys.exit(1)

    # Write metrics if requested
    if metrics and metrics_data:
        try:
            cmd_parts = ["cllm", "eval", str(manuscript), "-c", str(claims_file), "-p", str(peer_reviews), "-o", str(eval_peer_file)]
            if metrics_eval_peer_file:
                cmd_parts.extend(["-m", str(metrics_eval_peer_file)])
            if verbose:
                cmd_parts.append("-v")

            metadata_data = {"command": " ".join(cmd_parts)}
            metadata_data.update(metrics_data)

            metrics_eval_peer_file.write_text(json.dumps(metadata_data, indent=2), encoding="utf-8")
            click.echo(f"📊 Saved metrics to: {metrics_eval_peer_file}")
        except Exception as e:
            click.echo(f"⚠️  Warning: Could not write metrics: {e}", err=True)

    # ========================================================================
    # STAGE 4: COMPARE RESULTS
    # ========================================================================
    click.echo("\n⚖️  STAGE 4: Compare Results")
    click.echo("-" * 60)

    if not verbose:
        click.echo(f"⚖️  Comparing results...")

    try:
        concordance, processing_time, metrics_data = compare_results(
            llm_results, peer_results, verbose=verbose, return_metrics=metrics
        )
        if not verbose:
            click.echo(f"✅ Generated {len(concordance)} concordance rows in {processing_time:.2f}s")
    except Exception as e:
        click.echo(f"❌ Error comparing results: {e}", err=True)
        sys.exit(1)

    # Calculate basic metrics
    agreements = sum(1 for c in concordance if c.agreement_status == "agree")
    disagreements = sum(1 for c in concordance if c.agreement_status == "disagree")
    disjoint = sum(1 for c in concordance if c.agreement_status == "disjoint")
    agreement_rate = (agreements / len(concordance) * 100) if concordance else 0.0

    if not verbose:
        click.echo(f"📊 Agreement rate: {agreement_rate:.1f}%")
        click.echo(f"   Agreements: {agreements}, Disagreements: {disagreements}, Disjoint: {disjoint}")

    # Write concordance
    concordance_array = [
        {
            "openeval_result_id": c.openeval_result_id,
            "peer_result_id": c.peer_result_id,
            "openeval_status": c.openeval_status,
            "peer_status": c.peer_status,
            "agreement_status": c.agreement_status,
            "comparison": c.comparison,
            "n_openeval": c.n_openeval,
            "n_peer": c.n_peer,
            "n_itx": c.n_itx,
        }
        for c in concordance
    ]

    try:
        cmp_file.write_text(json.dumps(concordance_array, indent=2), encoding="utf-8")
        click.echo(f"💾 Saved concordance to: {cmp_file}")
    except Exception as e:
        click.echo(f"❌ Error writing concordance: {e}", err=True)
        sys.exit(1)

    # Write metrics if requested
    if metrics and metrics_data:
        try:
            cmd_parts = ["cllm", "cmp", str(eval_peer_file), str(eval_llm_file), "-o", str(cmp_file)]
            if metrics_cmp_file:
                cmd_parts.extend(["-m", str(metrics_cmp_file)])
            if verbose:
                cmd_parts.append("-v")

            metadata_data = {"command": " ".join(cmd_parts)}
            metadata_data.update(metrics_data)

            metrics_cmp_file.write_text(json.dumps(metadata_data, indent=2), encoding="utf-8")
            click.echo(f"📊 Saved metrics to: {metrics_cmp_file}")
        except Exception as e:
            click.echo(f"⚠️  Warning: Could not write metrics: {e}", err=True)

    # ========================================================================
    # DATABASE EXPORT
    # ========================================================================
    click.echo("\n💾 Generating database export...")
    click.echo("-" * 60)

    try:
        # Collect prompts used in workflow
        model = config.anthropic_model if config.llm_provider == "anthropic" else config.openai_model
        prompts = {
            "extract": {"text": STAGE1_PROMPT_TEMPLATE, "model": model},
            "eval_llm": {"text": STAGE2_PROMPT_TEMPLATE, "model": model},
            "eval_peer": {"text": STAGE3_PROMPT_TEMPLATE, "model": model},
            "compare": {"text": STAGE4_PROMPT_TEMPLATE, "model": model},
        }

        # Generate database export
        db_export = export_to_database_format(
            manuscript_text=manuscript_text,
            peer_review_text=review_text,
            claims=claims,
            llm_results=llm_results,
            peer_results=peer_results,
            concordance=concordance,
            prompts=prompts,
        )

        # Save to file
        db_export_file = output_dir / "db_export.json"
        save_db_export(db_export, db_export_file)
        click.echo(f"✅ Generated database export: {db_export_file.name}")

    except Exception as e:
        click.echo(f"⚠️  Warning: Could not generate database export: {e}", err=True)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    click.echo("\n" + "=" * 60)
    click.echo("✅ WORKFLOW COMPLETE (4 stages)")
    click.echo("=" * 60)
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"  - Claims: {claims_file.name}")
    click.echo(f"  - LLM evaluation: {eval_llm_file.name}")
    click.echo(f"  - Peer evaluation: {eval_peer_file.name}")
    click.echo(f"  - Comparison: {cmp_file.name}")
    click.echo(f"  - Database export: db_export.json")
    if metrics:
        click.echo(f"  - Metrics files: metrics_*.json")
    click.echo("=" * 60)


if __name__ == "__main__":
    cli()
