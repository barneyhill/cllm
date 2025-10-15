#!/usr/bin/env python3
"""
Simple example of how to use the json_to_pdf_table function.
"""

import json
from json_to_pdf_table import json_to_pdf_table


# Example 1: Use with Python list of dictionaries
print("Example 1: Creating PDF from Python data structure...")

data = [
    {
        "llm_result_id": "R7",
        "peer_result_id": "R7",
        "llm_status": "UNCERTAIN",
        "peer_status": "UNCERTAIN",
        "agreement_status": "agree",
        "notes": "Both assess forward-looking adaptability/ontology/community claims as UNCERTAIN (shared: C72, C73, C75, C76, C82, C85). LLM also includes C68, C69, C70, C87.",
        "n_llm": 10,
        "n_peer": 6,
        "n_itx": 6,
        "llm_reasoning": "Proposed benefits for reprocessing, artifact/batch-effect analyses, and transparency (C68–C71, C87), broader platform adaptability and extensions (C72–C76), and ontology/incentivization claims (C82, C85) are forward-looking. They are plausible but not supported by empirical case studies, quantitative evaluations, or user adoption evidence in the manuscript.",
        "peer_reasoning": "These forward-looking claims about universal applicability, platform adaptability, ontology development, and community deposition are not validated by data in the reviews. Reviewer 2 specifically requests details on versioning and backwards compatibility (especially for archival contexts), indicating that long-term adaptability and adoption remain to be demonstrated."
    }
]

json_to_pdf_table(
    data,
    "/mnt/user-data/outputs/example1.pdf",
    title="Example 1: Direct Python Data"
)


# Example 2: Load from JSON file and convert
print("\nExample 2: Creating PDF from JSON file...")

# First, save sample data to a JSON file
sample_file = "sample_data.json"
with open(sample_file, 'w') as f:
    json.dump(data, f, indent=2)

# Now convert the JSON file to PDF
json_to_pdf_table(
    sample_file,
    "/mnt/user-data/outputs/example2.pdf",
    title="Example 2: From JSON File"
)

print("\n✓ All examples completed successfully!")
print("Check the /mnt/user-data/outputs/ directory for the generated PDFs.")