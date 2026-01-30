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
1. Results grouped by an OpenEval evaluator (LLM peer reviewer)
2. Results grouped by peer reviewers

Your task is to compare these results and identify areas of agreement and disagreement.

# Concordance Analysis Guidelines

## Matching Strategy
1. Identify which OpenEval results and peer review results address the same or overlapping claims
2. Look for claim_ids that appear in both OpenEval and peer results
3. Results may address overlapping or completely different claims

## Comparison
Provide brief explanation of the comparison between OpenEval and peer evaluations.

## Disjoint Cases
- If a result appears in only one evaluation (OpenEval or peer), include it with the missing side's result_id and status set to null
- Explain in the comparison which side is missing

# OpenEval Results

$LLM_RESULTS_JSON

# Peer Review Results

$PEER_RESULTS_JSON

# Suggested Pairings (based on Jaccard similarity of claim sets)

The following pairings have been pre-computed based on the Jaccard index (overlap of claim_ids) between OpenEval and peer results. These are ordered by similarity and can help guide your analysis, but you should still consider all possible pairings and may identify additional or different matches based on semantic content.

$JACCARD_PAIRINGS_JSON

## Example Output Format
```json
{
  "concordance": [
    {
      "openeval_result_id": "R2",
      "peer_result_id": "R4",
      "comparison": "Both address the phosphorylation findings (C1, C2), but OpenEval groups with C3 and finds evidence sufficient, while reviewers question physiological relevance."
    },
    {
      "openeval_result_id": "R5",
      "peer_result_id": "R1",
      "comparison": "Both agree the microscopy data is convincing, though peer review groups C5 with related claim C6."
    },
    {
      "openeval_result_id": "R7",
      "peer_result_id": null,
      "comparison": "OpenEval result with no corresponding peer review evaluation found."
    }
  ]
}
```

Please compare the OpenEval and peer review results. Return ONLY valid JSON matching the schema above."""

SCORE_ELIFE_FALLBACK = """You are a senior scientific editor tasked with providing a holistic assessment of a scientific paper.

## Inputs

You will receive:
1. The manuscript text
2. Claims extracted from the manuscript
3. Results from a review process

## Context

The results have been generated as part of a review process. You may use them to inform your assessment or produce an independent evaluation. 

## eLife Assessment Framework

### Findings Significance:
- Landmark: findings with profound implications that are expected to have widespread influence
- Fundamental: findings that substantially advance our understanding of major research questions
- Important: findings that have theoretical or practical implications beyond a single subfield
- Valuable: findings that have theoretical or practical implications for a subfield
- Useful: findings that have focused importance and scope

### Evidence Strength:
- Exceptional: exemplary use of existing approaches that establish new standards for a field
- Compelling: evidence that features methods, data and analyses more rigorous than the current state-of-the-art
- Convincing: appropriate and validated methodology in line with current state-of-the-art
- Solid: methods, data and analyses broadly support the claims with only minor weaknesses
- Incomplete: main claims are only partially supported
- Inadequate: methods, data and analyses do not support the primary claims

## Task

Provide a comprehensive 3-5 paragraph assessment that:
1. Summarizes the research objectives and approach
2. Describes key findings and significance
3. Evaluates evidence quality
4. Places work in broader context
5. Notes limitations

Then assign significance and evidence ratings.

## Output Format

{
  "assessment": "Your 3-5 paragraph evaluation...",
  "findings_significance": "One of: Landmark, Fundamental, Important, Valuable, Useful",
  "evidence_strength": "One of: Exceptional, Compelling, Convincing, Solid, Incomplete, Inadequate"
}

$MANUSCRIPT_TEXT

$CLAIMS_JSON

$RESULTS_JSON
"""
