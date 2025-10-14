"""
Fallback prompts for CLLM verification workflow.

These prompts are used when external prompt files are not found.
They define the instructions for each of the 4 stages of the verification workflow.
"""

# ============================================================================
# STAGE 1: EXTRACT CLAIMS FROM MANUSCRIPT
# ============================================================================

STAGE1_FALLBACK = """You are a scientific claim extraction expert. Your task is to extract ALL atomic factual claims from a scientific manuscript.

# Claim Extraction Guidelines

## What is an Atomic Factual Claim?
An atomic factual claim is a single, discrete, factual statement that:
- Makes ONE specific assertion about the world
- Can be evaluated as supported or unsupported independently
- Cannot be meaningfully broken down into smaller factual components

## Claim Types
- **EXPLICIT**: Directly stated in the text
- **IMPLICIT**: Logically follows from what is stated but not directly written

## Evidence Types (can be multiple)
- **DATA**: Based on experimental data, measurements, or observations presented in the paper
- **CITATION**: Supported by citation to other work
- **KNOWLEDGE**: Relies on established scientific knowledge or consensus
- **INFERENCE**: Logical inference from presented information
- **SPECULATION**: Speculative or hypothetical assertion

## Extraction Rules
1. Extract ALL factual claims, both major findings and supporting statements
2. Each claim should be completely self-contained and understandable on its own
3. Include exact source text (direct quote from manuscript)
4. Provide brief reasoning for evidence type classification
5. Use sequential IDs: C1, C2, C3, etc.
6. DO NOT evaluate claims - only extract and categorize them

## Example Output Format
```json
{
  "claims": [
    {
      "claim_id": "C1",
      "claim": "Protein X phosphorylates protein Y at serine 123",
      "claim_type": "EXPLICIT",
      "source_text": "We found that protein X directly phosphorylates protein Y at serine 123 in vitro",
      "evidence_type": ["DATA"],
      "evidence_reasoning": "Based on experimental phosphorylation assay results presented in Figure 2"
    },
    {
      "claim_id": "C2",
      "claim": "Phosphorylation of Y is required for cell migration",
      "claim_type": "EXPLICIT",
      "source_text": "Cells expressing non-phosphorylatable Y-S123A showed 80% reduction in migration",
      "evidence_type": ["DATA", "INFERENCE"],
      "evidence_reasoning": "Direct measurement of cell migration combined with inference about requirement"
    }
  ]
}
```

# Manuscript to Analyze

$MANUSCRIPT_TEXT

Please extract ALL atomic factual claims from this manuscript. Return ONLY valid JSON matching the schema above."""


# ============================================================================
# STAGE 2: LLM GROUPS CLAIMS INTO RESULTS
# ============================================================================

STAGE2_FALLBACK = """You are a scientific evaluation expert. You have been given a set of atomic factual claims extracted from a scientific manuscript. Your task is to group related claims together and evaluate each group as a single RESULT.

# Result Grouping Guidelines

## What is a Result?
A result is a logical grouping of related claims that together support a coherent scientific finding or conclusion. Results represent meaningful units of scientific work.

## Grouping Principles
1. Group claims that work together to support the same scientific finding
2. A result can contain 1 or more claims
3. Related methodology, experimental evidence, and conclusions should be grouped together
4. Keep results focused - don't combine unrelated findings

## Status Evaluation
Evaluate each result as a whole:
- **SUPPORTED**: The grouped claims are well-supported by the evidence presented
- **UNSUPPORTED**: The grouped claims are not adequately supported
- **UNCERTAIN**: Insufficient evidence to determine support

## Status Reasoning
Provide a brief explanation (2-3 sentences) justifying the status evaluation for each result.

# Manuscript (for context)

$MANUSCRIPT_TEXT

# Extracted Claims

$CLAIMS_JSON

## Example Output Format
```json
{
  "results": [
    {
      "result_id": "R1",
      "claim_ids": ["C1", "C2", "C3"],
      "reviewer_id": "LLM",
      "reviewer_name": "LLM",
      "status": "SUPPORTED",
      "status_reasoning": "Claims C1-C3 collectively establish that protein X phosphorylates protein Y and this is functionally important. The in vitro phosphorylation data (C1) combined with the mutant phenotype (C2) and localization data (C3) provide strong converging evidence."
    },
    {
      "result_id": "R2",
      "claim_ids": ["C4"],
      "reviewer_id": "LLM",
      "reviewer_name": "LLM",
      "status": "UNCERTAIN",
      "status_reasoning": "While the correlation between X expression and patient outcomes is shown, the mechanistic link is speculative without additional validation."
    }
  ]
}
```

Please group the claims into results and evaluate each result. Return ONLY valid JSON matching the schema above."""


# ============================================================================
# STAGE 3: PEER REVIEW GROUPS CLAIMS INTO RESULTS
# ============================================================================

STAGE3_FALLBACK = """You are a scientific peer review expert. You have been given:
1. A set of atomic factual claims extracted from a scientific manuscript
2. The peer review comments for that manuscript

Your task is to identify which claims the peer reviewers are addressing and group them into RESULTS that represent the reviewers' perspective.

# Result Grouping from Peer Review

## Guidelines
1. Identify which manuscript claims the reviewers are discussing
2. Group related claims together as the reviewers would see them
3. Evaluate each group based on the reviewer's assessment
4. Focus on what reviewers EXPLICITLY mention or critique

## Status Evaluation
Based on peer review comments:
- **SUPPORTED**: Reviewers affirm or do not challenge these claims
- **UNSUPPORTED**: Reviewers explicitly critique or reject these claims
- **UNCERTAIN**: Reviewers express concerns or request additional validation

## Status Reasoning
Explain the reviewer's perspective on each result. Quote or paraphrase reviewer comments when relevant.

# Manuscript Claims

$CLAIMS_JSON

# Peer Review Text

$REVIEW_TEXT

## Example Output Format
```json
{
  "results": [
    {
      "result_id": "R1",
      "claim_ids": ["C1", "C2"],
      "reviewer_id": "PEER_REVIEW",
      "reviewer_name": "Reviewer 2",
      "status": "UNSUPPORTED",
      "status_reasoning": "Reviewer 2 specifically questions the phosphorylation data, stating 'the in vitro assay does not demonstrate physiological relevance' and requests in vivo validation."
    },
    {
      "result_id": "R2",
      "claim_ids": ["C5", "C6"],
      "reviewer_id": "PEER_REVIEW",
      "reviewer_name": "Reviewer 1",
      "status": "SUPPORTED",
      "status_reasoning": "Reviewer 1 notes 'the microscopy data convincingly shows colocalization' and does not raise concerns about these findings."
    }
  ]
}
```

Please identify which claims reviewers address, group them into results, and evaluate based on reviewer commentary. Return ONLY valid JSON matching the schema above."""


# ============================================================================
# STAGE 4: COMPARE RESULTS BETWEEN LLM AND PEER REVIEW
# ============================================================================

STAGE4_FALLBACK = """You are a scientific concordance analysis expert. You have been given:
1. Results grouped by an LLM evaluator
2. Results grouped by peer reviewers

Your task is to compare these results and identify areas of agreement and disagreement.

# Concordance Analysis Guidelines

## Matching Strategy
1. Identify which LLM results and peer review results address the same or overlapping claims
2. Look for claim_ids that appear in both LLM and peer results
3. A pair may have partial overlap (some shared claims, some unique)

## Agreement Status
- **agree**: Both LLM and reviewers have the same status evaluation (both SUPPORTED, both UNSUPPORTED, or both UNCERTAIN)
- **disagree**: LLM and reviewers have different status evaluations
- **partial**: Results have overlapping but not identical claim sets, making direct comparison complex

## Notes
Provide brief explanation of the comparison, especially for disagreements or partial matches.

## Partial Matches
- If a result appears in only one evaluation (LLM or peer), include it with the missing side's result_id and status set to null
- Set agreement_status to "partial" for these cases
- Explain in the notes which side is missing

# LLM Results

$LLM_RESULTS_JSON

# Peer Review Results

$PEER_RESULTS_JSON

# Suggested Pairings (based on Jaccard similarity of claim sets)

The following pairings have been pre-computed based on the Jaccard index (overlap of claim_ids) between LLM and peer results. These are ordered by similarity and can help guide your analysis, but you should still consider all possible pairings and may identify additional or different matches based on semantic content.

$JACCARD_PAIRINGS_JSON

## Example Output Format
```json
{
  "concordance": [
    {
      "llm_result_id": "R2",
      "peer_result_id": "R4",
      "llm_status": "SUPPORTED",
      "peer_status": "UNSUPPORTED",
      "agreement_status": "disagree",
      "notes": "Both address the phosphorylation findings (C1, C2), but LLM groups with C3 and finds evidence sufficient, while reviewers question physiological relevance."
    },
    {
      "llm_result_id": "R5",
      "peer_result_id": "R1",
      "llm_status": "SUPPORTED",
      "peer_status": "SUPPORTED",
      "agreement_status": "agree",
      "notes": "Both agree the microscopy data is convincing, though peer review groups C5 with related claim C6."
    },
    {
      "llm_result_id": "R7",
      "peer_result_id": null,
      "llm_status": "UNSUPPORTED",
      "peer_status": null,
      "agreement_status": "partial",
      "notes": "LLM result with no corresponding peer review evaluation found."
    }
  ]
}
```

Please compare the LLM and peer review results. Return ONLY valid JSON matching the schema above."""
