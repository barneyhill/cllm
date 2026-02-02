# CLLM - Claim LLM

A command-line tool for extracting, evaluating, and comparing scientific claims using Large Language Models.

## Overview

CLLM implements a 4-stage workflow for scientific claim verification:

1. **Extract Claims**: Extract atomic factual claims from a manuscript
2. **LLM Evaluation**: Group claims into results and evaluate them
3. **Peer Review Evaluation**: Group claims based on peer review commentary
4. **Compare Results**: Compare LLM and peer review evaluations

## Installation

### Prerequisites

- Python >=3.10
- Anthropic API key

### Install with uv (recommended)

```bash
# From the cllm directory
cd cllm
uv pip install -e .
```

### Install with pip

```bash
# From the cllm directory
cd cllm
pip install -e .
```

## Configuration

Set your Anthropic API key as an environment variable:

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

Optional configuration:

```bash
# Use a different model (default: claude-sonnet-4-5-20250929)
export ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Set timeout in seconds (default: 600.0)
export ANTHROPIC_TIMEOUT=600.0
```

## Usage

### Complete Workflow Example

```bash
# 1. Extract claims from manuscript
cllm extract manuscript.txt -o claims.json

# 2. Get LLM evaluation
cllm eval manuscript.txt -c claims.json -o eval_llm.json

# 3. Get peer review evaluation
cllm eval manuscript.txt -c claims.json -p peer_reviews.txt -o eval_peers.json

# 4. Compare evaluations
cllm cmp eval_peers.json eval_llm.json -o comparison.json

# Optional: Save metadata for any command
cllm extract manuscript.txt -o claims.json -m metadata.json
```

## Commands

### 1. Extract Claims

Extract atomic factual claims from a scientific manuscript:

```bash
cllm extract manuscript.txt -o claims.json
```

**Options:**
- `-o, --output`: Output JSON file for claims (required)
- `-m, --metadata`: Optional JSON file for metadata

**Output Format:**

The output file contains a JSON array of claim objects:

```json
[
  {
    "claim_id": "C1",
    "claim": "The claim text",
    "claim_type": "EXPLICIT",
    "source_text": "Source quote from manuscript",
    "evidence_type": ["DATA", "CITATION"],
    "evidence_reasoning": "Explanation of evidence type"
  },
  {
    "claim_id": "C2",
    "claim": "Another claim text",
    "claim_type": "IMPLICIT",
    "source_text": "Source quote",
    "evidence_type": ["INFERENCE"],
    "evidence_reasoning": "Explanation"
  }
]
```

**Metadata Format** (if `-m` is used):

```json
{
  "command": "extract",
  "manuscript_file": "manuscript.txt",
  "manuscript_length": 50000,
  "num_claims": 42,
  "processing_time_seconds": 15.3
}
```

### 2. Evaluate Claims

Evaluate claims and group them into results.

**LLM Evaluation:**
```bash
cllm eval manuscript.txt -c claims.json -o eval_llm.json
```

**Peer Review Evaluation:**
```bash
cllm eval manuscript.txt -c claims.json -p reviews.txt -o eval_peers.json
```

**Options:**
- `-c, --claims`: Input claims JSON file (required)
- `-p, --peer-reviews`: Peer review file (optional, changes evaluation mode)
- `-o, --output`: Output JSON file for evaluations (required)
- `-m, --metadata`: Optional JSON file for metadata

**Output Format:**

The output file contains a JSON array of result objects:

```json
[
  {
    "result_id": "R1",
    "claim_ids": ["C1", "C2", "C3"],
    "reviewer_id": "LLM",
    "reviewer_name": "LLM",
    "status": "SUPPORTED",
    "status_reasoning": "Explanation of why these claims are grouped and supported"
  },
  {
    "result_id": "R2",
    "claim_ids": ["C4"],
    "reviewer_id": "0000-0002-1234-5678",
    "reviewer_name": "Jane Reviewer",
    "status": "UNCERTAIN",
    "status_reasoning": "Explanation of uncertainty"
  }
]
```

**Metadata Format** (if `-m` is used):

```json
{
  "command": "eval",
  "manuscript_file": "manuscript.txt",
  "claims_file": "claims.json",
  "peer_reviews_file": "reviews.txt",
  "source": "LLM",
  "num_results": 15,
  "processing_time_seconds": 22.7
}
```

### 3. Compare Results

Compare peer review and LLM evaluations:

```bash
cllm cmp eval_peers.json eval_llm.json -o comparison.json
```

**Options:**
- `-o, --output`: Output JSON file for comparison (required)
- `-m, --metadata`: Optional JSON file for metadata

**Output Format:**

The output file contains a JSON array of concordance objects:

```json
[
  {
    "llm_result_id": "R2",
    "peer_result_id": "R4",
    "llm_status": "SUPPORTED",
    "peer_status": "UNSUPPORTED",
    "agreement_status": "disagree",
    "notes": "Explanation of disagreement"
  },
  {
    "llm_result_id": "R5",
    "peer_result_id": null,
    "llm_status": "SUPPORTED",
    "peer_status": null,
    "agreement_status": "disjoint",
    "notes": "LLM result with no corresponding peer review"
  }
]
```

**Note:** For disjoint cases (where a result exists in only one evaluation), the missing side's `result_id` and `status` will be `null`.

**Metadata Format** (if `-m` is used):

```json
{
  "command": "cmp",
  "eval_peers_file": "eval_peers.json",
  "eval_llm_file": "eval_llm.json",
  "total_comparisons": 20,
  "agreements": 15,
  "disagreements": 3,
  "disjoint": 2,
  "agreement_rate": 75.0,
  "processing_time_seconds": 8.5
}
```

## Input File Formats

### Manuscript Files

Plain text files (`.txt`) containing the scientific manuscript:

```
Title: My Scientific Study

Abstract:
This study investigates...

Introduction:
Previous research has shown...
```

### Peer Review Files

Plain text files (`.txt`) containing peer review comments:

```
Reviewer 1:
The manuscript presents interesting findings, but...

Reviewer 2:
I have concerns about the methodology...
```

## Features

- 🔍 **Atomic Claim Extraction**: Identifies discrete, factual statements
- 🤖 **LLM Evaluation**: Groups and evaluates claims using Claude
- 📝 **Peer Review Analysis**: Extracts reviewer perspectives on claims
- ⚖️ **Concordance Analysis**: Compares LLM and peer review assessments
- 💾 **JSON Output**: All results saved in structured JSON format
- 📊 **Metrics**: Calculates agreement rates and statistics

## Architecture

CLLM is a standalone tool with no external dependencies beyond the Anthropic API:

```
cllm/
├── cllm/
│   ├── __init__.py
│   ├── cli.py           # CLI interface
│   ├── models.py        # Pydantic data models
│   ├── verification.py  # 4-stage verification logic
│   └── config.py        # Configuration management
├── pyproject.toml       # Package configuration
└── README.md
```

## Claim Types

- **EXPLICIT**: Directly stated in the text
- **IMPLICIT**: Logically follows from what is stated

## Evidence Types

- **DATA**: Based on experimental data or observations
- **CITATION**: Supported by citation to other work
- **KNOWLEDGE**: Relies on established scientific knowledge
- **INFERENCE**: Logical inference from presented information
- **SPECULATION**: Speculative or hypothetical assertion

## Status Values

- **SUPPORTED**: Claims are well-supported by evidence
- **UNSUPPORTED**: Claims are not adequately supported
- **UNCERTAIN**: Insufficient evidence to determine support

## Agreement Status

- **agree**: Same status evaluation from LLM and reviewers
- **disagree**: Different status evaluations
- **disjoint**: One evaluator produced a result, but the other did not

## Troubleshooting

### API Key Not Set

```
❌ Configuration error: ANTHROPIC_API_KEY environment variable is required.
```

**Solution**: Set your API key:
```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

### Rate Limiting

If you encounter rate limiting from the Anthropic API, try:
- Waiting a few moments between commands
- Using shorter documents
- Checking your API usage limits

### Large Documents

For very large manuscripts (>100,000 characters), processing may take longer. The default timeout is 600 seconds (10 minutes).

## Example Workflow

See the example below for a complete end-to-end workflow:

```bash
# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Extract claims (with optional metadata)
cllm extract paper.txt -o claims.json -m metadata_extract.json
# Output: 💾 Saved 42 claims to: claims.json
#         📊 Saved metadata to: metadata_extract.json

# Get LLM evaluation
cllm eval paper.txt -c claims.json -o eval_llm.json -m metadata_eval_llm.json
# Output: 💾 Saved 15 results to: eval_llm.json
#         📊 Saved metadata to: metadata_eval_llm.json

# Get peer review evaluation
cllm eval paper.txt -c claims.json -p reviews.txt -o eval_peers.json
# Output: 💾 Saved 12 results to: eval_peers.json

# Compare evaluations
cllm cmp eval_peers.json eval_llm.json -o comparison.json -m metadata_cmp.json
# Output: 📊 Agreement rate: 75.0%
#         💾 Saved 20 concordance rows to: comparison.json
#         📊 Saved metadata to: metadata_cmp.json
```

**Note:** The `-m` flag is optional for all commands. If omitted, only the main output file is created.

## Development

### Running Tests

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests (when available)
pytest
```

### Project Structure

- `cli.py`: Command-line interface using Click
- `models.py`: Pydantic models for data validation
- `verification.py`: Core 4-stage verification workflow
- `config.py`: Configuration management

## License

MIT

## Support

For issues or questions, please open an issue on GitHub.
