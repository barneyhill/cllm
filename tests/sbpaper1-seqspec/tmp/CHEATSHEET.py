#!/usr/bin/env python3
"""
QUICK REFERENCE CHEATSHEET
Copy and paste these examples for common scenarios
"""

from json_to_pdf_table import json_to_pdf_table
from advanced_pdf_table import json_to_pdf_table_advanced
from reportlab.lib.pagesizes import landscape, letter, A4, A3

# =============================================================================
# BASIC USAGE - Copy this for 90% of cases
# =============================================================================

# Your data
data = [
    {"id": "1", "status": "OK", "notes": "Some long text here..."},
    {"id": "2", "status": "ERROR", "notes": "Another long text..."}
]

# Simple conversion
json_to_pdf_table(data, "output.pdf", title="My Table")


# =============================================================================
# LOAD FROM JSON FILE
# =============================================================================

json_to_pdf_table("data.json", "output.pdf", title="My Table")


# =============================================================================
# LARGER PAGE SIZE (for tables with many columns)
# =============================================================================

json_to_pdf_table_advanced(
    data,
    "output.pdf",
    title="Wide Table",
    pagesize=landscape(A3)  # or landscape(TABLOID)
)


# =============================================================================
# CUSTOM COLORS
# =============================================================================

json_to_pdf_table_advanced(
    data,
    "output.pdf",
    title="Blue Theme",
    header_bg_color='#1E88E5',  # Blue header
    alt_row_color='#E3F2FD'     # Light blue alternating rows
)


# =============================================================================
# EXCLUDE COLUMNS
# =============================================================================

json_to_pdf_table_advanced(
    data,
    "output.pdf",
    title="Filtered",
    exclude_columns={'internal_id', 'temp_field', 'debug_info'}
)


# =============================================================================
# CUSTOM COLUMN ORDER
# =============================================================================

json_to_pdf_table_advanced(
    data,
    "output.pdf",
    title="Ordered",
    column_order=['id', 'name', 'status', 'date', 'notes']
)


# =============================================================================
# SPECIFY LONG TEXT COLUMNS (for better formatting)
# =============================================================================

json_to_pdf_table_advanced(
    data,
    "output.pdf",
    title="Custom Text Columns",
    long_text_columns={'description', 'comments', 'analysis', 'feedback'}
)


# =============================================================================
# CUSTOM COLUMN WIDTHS
# =============================================================================

json_to_pdf_table_advanced(
    data,
    "output.pdf",
    title="Custom Widths",
    column_widths={
        'id': 0.5,           # Narrow
        'name': 1.0,         # Normal
        'description': 3.0,  # Wide
        'status': 0.8        # Medium
    }
)


# =============================================================================
# LARGER FONTS (for presentations)
# =============================================================================

json_to_pdf_table_advanced(
    data,
    "output.pdf",
    title="Large Fonts",
    base_font_size=10,
    small_font_size=9
)


# =============================================================================
# SMALLER FONTS (to fit more data)
# =============================================================================

json_to_pdf_table_advanced(
    data,
    "output.pdf",
    title="Compact",
    base_font_size=7,
    small_font_size=6,
    margin=0.3  # Narrower margins too
)


# =============================================================================
# FULL CUSTOMIZATION (all options)
# =============================================================================

json_to_pdf_table_advanced(
    data,
    "output.pdf",
    title="Fully Custom Table",
    pagesize=landscape(A4),
    header_bg_color='#2C3E50',
    alt_row_color='#F8F9FA',
    base_font_size=9,
    small_font_size=7,
    margin=0.5,
    long_text_columns={'notes', 'reasoning', 'description'},
    exclude_columns={'temp', 'debug'},
    column_order=['id', 'name', 'status'],
    column_widths={'notes': 2.5, 'id': 0.5}
)


# =============================================================================
# COMPARISON: LLM vs PEER (your specific use case)
# =============================================================================

comparison_data = [
    {
        "llm_result_id": "R7",
        "peer_result_id": "R7",
        "llm_status": "UNCERTAIN",
        "peer_status": "UNCERTAIN",
        "agreement_status": "agree",
        "notes": "Both assess forward-looking claims...",
        "n_llm": 10,
        "n_peer": 6,
        "n_itx": 6,
        "llm_reasoning": "Long reasoning text...",
        "peer_reasoning": "Long reasoning text..."
    }
]

json_to_pdf_table(
    comparison_data,
    "comparison.pdf",
    title="LLM vs Peer Review Comparison"
)

# Or with customization:
json_to_pdf_table_advanced(
    comparison_data,
    "comparison_custom.pdf",
    title="LLM vs Peer Review Comparison",
    pagesize=landscape(A3),  # More space for reasoning columns
    long_text_columns={'notes', 'llm_reasoning', 'peer_reasoning'},
    column_widths={
        'llm_result_id': 0.6,
        'peer_result_id': 0.6,
        'llm_status': 0.8,
        'peer_status': 0.8,
        'agreement_status': 0.8,
        'notes': 1.5,
        'n_llm': 0.4,
        'n_peer': 0.4,
        'n_itx': 0.4,
        'llm_reasoning': 2.5,
        'peer_reasoning': 2.5
    }
)

print("✓ Cheatsheet examples ready to copy!")