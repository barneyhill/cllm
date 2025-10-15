# JSON to Landscape PDF Table Converter

A robust Python solution for converting JSON data to professional landscape PDF tables with automatic text wrapping for long content.

## Features

✅ **Landscape orientation** for maximum table width  
✅ **Automatic text wrapping** in cells with long content  
✅ **Smart column width allocation** based on content type  
✅ **Professional styling** with customizable colors and fonts  
✅ **Two versions**: Simple (quick start) and Advanced (full customization)

## Installation

```bash
pip install reportlab
```

## Quick Start - Simple Version

Use `json_to_pdf_table.py` for straightforward conversion:

```python
from json_to_pdf_table import json_to_pdf_table

# Your JSON data
data = [
    {
        "id": "R7",
        "status": "UNCERTAIN",
        "notes": "This is a long text column that will automatically wrap...",
        "reasoning": "Another long text field with detailed explanation..."
    }
]

# Create PDF
json_to_pdf_table(
    data,
    "output.pdf",
    title="My Results Table"
)
```

### Loading from JSON file:

```python
json_to_pdf_table(
    "data.json",  # Path to your JSON file
    "output.pdf",
    title="My Results Table"
)
```

## Advanced Version - Full Customization

Use `advanced_pdf_table.py` for complete control:

```python
from advanced_pdf_table import json_to_pdf_table_advanced
from reportlab.lib.pagesizes import landscape, A4, A3

# Example 1: Custom colors
json_to_pdf_table_advanced(
    data,
    "output.pdf",
    title="Custom Styled Table",
    header_bg_color='#1E88E5',      # Blue header
    alt_row_color='#E3F2FD',        # Light blue rows
    base_font_size=9
)

# Example 2: Custom page size
json_to_pdf_table_advanced(
    data,
    "output.pdf",
    title="A3 Landscape",
    pagesize=landscape(A3),
    margin=0.75
)

# Example 3: Exclude columns
json_to_pdf_table_advanced(
    data,
    "output.pdf",
    title="Filtered View",
    exclude_columns={'internal_id', 'temp_field'}
)

# Example 4: Custom column order
json_to_pdf_table_advanced(
    data,
    "output.pdf",
    title="Ordered Columns",
    column_order=['name', 'status', 'notes', 'date']
)

# Example 5: Specify which columns have long text
json_to_pdf_table_advanced(
    data,
    "output.pdf",
    title="Custom Text Columns",
    long_text_columns={'description', 'comments', 'analysis'}
)

# Example 6: Custom column widths
json_to_pdf_table_advanced(
    data,
    "output.pdf",
    title="Custom Widths",
    column_widths={
        'id': 0.5,
        'name': 1.0,
        'description': 3.0,
        'status': 0.8
    }
)
```

## Advanced Customization Options

### AdvancedPDFTableGenerator Parameters

```python
from advanced_pdf_table import AdvancedPDFTableGenerator
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib import colors

generator = AdvancedPDFTableGenerator(
    pagesize=landscape(letter),      # Page size
    margin=0.5,                      # Margin in inches
    header_bg_color='#2C3E50',      # Header background
    header_text_color=colors.white,  # Header text color
    alt_row_color='#F8F9FA',        # Alternating row color
    grid_color=colors.grey,          # Grid line color
    base_font_size=8,                # Normal text size
    small_font_size=7                # Long text columns size
)

generator.generate_pdf(
    data,
    "output.pdf",
    title="Custom Generator",
    column_widths={'notes': 3.0},
    long_text_columns={'notes', 'reasoning'},
    exclude_columns={'temp'},
    column_order=['id', 'status', 'notes']
)
```

## How It Works

### Text Wrapping
The key to handling long text is using `Paragraph` objects from ReportLab instead of plain strings. This enables automatic text wrapping within cells:

```python
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

# Create a paragraph style
style = ParagraphStyle(
    'CellStyle',
    fontSize=8,
    wordWrap='CJK'  # Enables word wrapping
)

# Use Paragraph objects in table cells
cell_content = Paragraph("Long text that will wrap automatically", style)
```

### Column Width Allocation
The scripts intelligently allocate column widths:

- **Long text columns** (notes, reasoning, descriptions): 2.5x width
- **ID/Status columns**: 0.8x width  
- **Numeric columns** (n_*): 0.5x width
- **Default columns**: 1.0x width

You can override these with custom widths.

### Styling
- Professional header with dark background
- Alternating row colors for readability
- Vertical alignment: TOP (important for cells with varying heights)
- Grid lines for clear separation
- Proper padding for comfortable reading

## Common Use Cases

### 1. Analysis Results with Long Reasoning
```python
data = [
    {
        "id": "R1",
        "status": "SUPPORTED",
        "reasoning": "Very long explanation with multiple sentences..."
    }
]

json_to_pdf_table(data, "results.pdf", title="Analysis Results")
```

### 2. Comparison Tables
```python
json_to_pdf_table_advanced(
    data,
    "comparison.pdf",
    title="LLM vs Peer Comparison",
    column_order=['id', 'llm_status', 'peer_status', 'agreement'],
    exclude_columns=['internal_temp']
)
```

### 3. Large Datasets on A3
```python
json_to_pdf_table_advanced(
    data,
    "large_table.pdf",
    pagesize=landscape(A3),  # More space
    base_font_size=7,        # Smaller font
    margin=0.3               # Narrow margins
)
```

## Page Size Options

```python
from reportlab.lib.pagesizes import letter, A4, A3, A2, TABLOID

# US Letter (8.5 x 11 inches)
pagesize=landscape(letter)

# A4 (210 x 297 mm)
pagesize=landscape(A4)

# A3 (297 x 420 mm) - for large tables
pagesize=landscape(A3)

# Tabloid (11 x 17 inches)
pagesize=landscape(TABLOID)
```

## Troubleshooting

### Text Not Wrapping
Make sure you're passing data as a list of dictionaries, not markdown strings.

### Columns Too Narrow
Use custom column widths:
```python
column_widths={'description': 3.0, 'notes': 2.5}
```

### Text Too Small
Increase font sizes:
```python
base_font_size=10,
small_font_size=9
```

### Table Doesn't Fit on Page
- Use larger page size: `landscape(A3)`
- Reduce font size
- Exclude unnecessary columns
- Reduce margins

## Files Included

- `json_to_pdf_table.py` - Simple version for quick conversion
- `advanced_pdf_table.py` - Advanced version with full customization
- `simple_example.py` - Basic usage examples
- `README.md` - This file

## Tips for Best Results

1. **Identify long text columns** - Explicitly specify which columns contain lengthy text
2. **Use landscape orientation** - Gives you 11" width vs 8.5"
3. **Start with defaults** - The auto-calculated widths work well for most cases
4. **Test with sample data** - Generate a PDF with a few rows first
5. **Adjust incrementally** - Tweak one parameter at a time

## Example Output

The generated PDFs include:
- Professional header with title
- Formatted column headers (spaces instead of underscores, title case)
- Wrapped text in long columns
- Alternating row colors
- Grid lines
- Proper alignment and padding

## License

This code uses ReportLab, which is available under BSD license.