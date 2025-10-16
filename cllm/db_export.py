"""
Database export utilities for CLLM.

Converts CLLM workflow outputs (claims, results, concordance) into
database-ready format with UUIDs and proper foreign key relationships.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

from .models import (
    LLMClaimV3,
    LLMResultV3,
    LLMResultsConcordanceRow,
    DBSubmission,
    DBContent,
    DBPrompt,
    DBClaim,
    DBResult,
    DBClaimResult,
    DBComparison,
    DBExport,
)
from .utils import generate_uuid, generate_prompt_id, get_current_timestamp


def export_to_database_format(
    manuscript_text: str,
    peer_review_text: Optional[str],
    claims: List[LLMClaimV3],
    llm_results: List[LLMResultV3],
    peer_results: Optional[List[LLMResultV3]],
    concordance: Optional[List[LLMResultsConcordanceRow]],
    prompts: Dict[str, Dict[str, str]],  # {stage: {text: str, model: str}}
) -> DBExport:
    """Convert CLLM workflow outputs to database export format.

    Args:
        manuscript_text: Full text of manuscript
        peer_review_text: Full text of peer reviews (optional)
        claims: List of extracted claims
        llm_results: List of LLM evaluation results
        peer_results: List of peer review results (optional)
        concordance: List of concordance rows (optional)
        prompts: Dictionary of prompts used {stage: {text: str, model: str}}

    Returns:
        DBExport object ready for database import
    """
    timestamp = get_current_timestamp()

    # Generate IDs
    submission_id = generate_uuid()
    manuscript_content_id = generate_uuid()
    peer_content_id = generate_uuid() if peer_review_text else None

    # Create submission
    submission = DBSubmission(
        id=submission_id,
        user_id=None,  # Will be set by backend
        manuscript_title=None,
        manuscript_doi=None,
        status="completed",
        created_at=timestamp,
        updated_at=timestamp,
    )

    # Create content records
    content_list = [
        DBContent(
            id=manuscript_content_id,
            submission_id=submission_id,
            content_type="manuscript",
            content_text=manuscript_text,
            created_at=timestamp,
        )
    ]

    if peer_review_text and peer_content_id:
        content_list.append(
            DBContent(
                id=peer_content_id,
                submission_id=submission_id,
                content_type="peer_review",
                content_text=peer_review_text,
                created_at=timestamp,
            )
        )

    # Create prompt records
    prompt_list = []
    prompt_id_map = {}  # stage -> prompt_id

    for stage, prompt_info in prompts.items():
        prompt_id = generate_prompt_id(prompt_info['text'], prompt_info['model'])
        prompt_id_map[stage] = prompt_id

        prompt_list.append(
            DBPrompt(
                id=prompt_id,
                prompt_text=prompt_info['text'],
                prompt_type=stage,
                model=prompt_info['model'],
                created_at=timestamp,
            )
        )

    # Create claim records with UUID mapping
    claim_list = []
    claim_uuid_map = {}  # claim_id (e.g., "C1") -> UUID

    for claim in claims:
        claim_uuid = generate_uuid()
        claim_uuid_map[claim.claim_id] = claim_uuid

        claim_list.append(
            DBClaim(
                id=claim_uuid,
                content_id=manuscript_content_id,
                claim_id=claim.claim_id,
                claim=claim.claim,
                claim_type=claim.claim_type,
                source_text=claim.source_text,
                evidence_type=json.dumps(claim.evidence_type),
                evidence_reasoning=claim.evidence_reasoning,
                prompt_id=prompt_id_map.get('extract', ''),
                created_at=timestamp,
            )
        )

    # Create LLM result records with UUID mapping
    result_list = []
    llm_result_uuid_map = {}  # result_id (e.g., "R1") -> UUID
    claim_result_list = []

    for result in llm_results:
        result_uuid = generate_uuid()
        llm_result_uuid_map[result.result_id] = result_uuid

        result_list.append(
            DBResult(
                id=result_uuid,
                content_id=manuscript_content_id,
                result_id=result.result_id,
                result_type='llm',
                reviewer_id=result.reviewer_id,
                reviewer_name=result.reviewer_name,
                result_status=result.status,
                result_reasoning=result.status_reasoning,
                prompt_id=prompt_id_map.get('eval_llm', ''),
                created_at=timestamp,
            )
        )

        # Create junction records for this result
        for claim_id in result.claim_ids:
            if claim_id in claim_uuid_map:
                claim_result_list.append(
                    DBClaimResult(
                        claim_id=claim_uuid_map[claim_id],
                        result_id=result_uuid,
                    )
                )

    # Create peer result records with UUID mapping
    peer_result_uuid_map = {}  # result_id (e.g., "R1") -> UUID

    if peer_results and peer_content_id:
        for result in peer_results:
            result_uuid = generate_uuid()
            peer_result_uuid_map[result.result_id] = result_uuid

            result_list.append(
                DBResult(
                    id=result_uuid,
                    content_id=peer_content_id,
                    result_id=result.result_id,
                    result_type='peer',
                    reviewer_id=result.reviewer_id,
                    reviewer_name=result.reviewer_name,
                    result_status=result.status,
                    result_reasoning=result.status_reasoning,
                    prompt_id=prompt_id_map.get('eval_peer', ''),
                    created_at=timestamp,
                )
            )

            # Create junction records for this result
            for claim_id in result.claim_ids:
                if claim_id in claim_uuid_map:
                    claim_result_list.append(
                        DBClaimResult(
                            claim_id=claim_uuid_map[claim_id],
                            result_id=result_uuid,
                        )
                    )

    # Create comparison records
    comparison_list = []

    if concordance:
        for row in concordance:
            comparison_uuid = generate_uuid()

            # Map LLM/peer result IDs to UUIDs
            llm_result_uuid = None
            if row.llm_result_id and row.llm_result_id in llm_result_uuid_map:
                llm_result_uuid = llm_result_uuid_map[row.llm_result_id]

            peer_result_uuid = None
            if row.peer_result_id and row.peer_result_id in peer_result_uuid_map:
                peer_result_uuid = peer_result_uuid_map[row.peer_result_id]

            comparison_list.append(
                DBComparison(
                    id=comparison_uuid,
                    submission_id=submission_id,
                    llm_result_id=llm_result_uuid,
                    peer_result_id=peer_result_uuid,
                    llm_status=row.llm_status,
                    peer_status=row.peer_status,
                    agreement_status=row.agreement_status,
                    notes=row.notes,
                    prompt_id=prompt_id_map.get('compare', ''),
                    created_at=timestamp,
                )
            )

    # Build final export object
    return DBExport(
        submission=submission,
        content=content_list,
        prompts=prompt_list,
        claims=claim_list,
        results=result_list,
        claim_results=claim_result_list,
        comparisons=comparison_list,
    )


def save_db_export(export: DBExport, output_path: Path):
    """Save database export to JSON file.

    Args:
        export: DBExport object
        output_path: Path to output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(export.model_dump(), f, indent=2)
