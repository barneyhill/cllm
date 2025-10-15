"""
Report generation for CLLM.

Generate PDF tables from JSON data files.
"""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.enums import TA_LEFT, TA_CENTER


def json_to_pdf_table(json_data, output_filename, title=None):
    """
    Convert JSON data to a landscape PDF table with text wrapping.

    Args:
        json_data: List of dictionaries or path to JSON file
        output_filename: Path for the output PDF file
        title: Optional title for the document
    """

    # Load JSON data if it's a file path
    if isinstance(json_data, str) or isinstance(json_data, Path):
        with open(json_data, 'r') as f:
            data = json.load(f)
    else:
        data = json_data

    if not data:
        raise ValueError("JSON data is empty")

    # Set up the PDF document with landscape orientation
    doc = SimpleDocTemplate(
        str(output_filename),
        pagesize=landscape(letter),
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.75*inch,
        bottomMargin=0.5*inch
    )

    # Container for the 'Flowable' objects
    elements = []

    # Set up styles
    styles = getSampleStyleSheet()

    # Custom style for table headers
    header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.whitesmoke,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        leading=11
    )

    # Custom style for table cells with normal text
    cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontSize=8,
        alignment=TA_LEFT,
        fontName='Helvetica',
        leading=10,
        wordWrap='CJK'
    )

    # Custom style for table cells with small text (for long content)
    cell_style_small = ParagraphStyle(
        'TableCellSmall',
        parent=styles['Normal'],
        fontSize=7,
        alignment=TA_LEFT,
        fontName='Helvetica',
        leading=9,
        wordWrap='CJK'
    )

    # Add title if provided
    if title:
        title_style = styles['Title']
        elements.append(Paragraph(title, title_style))
        elements.append(Paragraph("<br/>", styles['Normal']))

    # Extract headers from the first dictionary
    headers = list(data[0].keys())

    # Create header row with Paragraph objects
    table_data = [[Paragraph(str(h).replace('_', ' ').title(), header_style) for h in headers]]

    # Define which columns typically have long text
    # These will use smaller font
    long_text_columns = {
        'notes', 'llm_reasoning', 'peer_reasoning', 'reasoning',
        'status_reasoning', 'claim', 'source_text', 'evidence_reasoning'
    }

    # Add data rows
    for row in data:
        row_data = []
        for header in headers:
            value = row.get(header, '')

            # Convert value to string
            if value is None:
                value = ''
            elif isinstance(value, list):
                # Handle list values (like evidence_type or claim_ids)
                value = ', '.join(str(v) for v in value)
            else:
                value = str(value)

            # Use smaller font for columns with typically long text
            if header.lower() in long_text_columns:
                row_data.append(Paragraph(value, cell_style_small))
            else:
                row_data.append(Paragraph(value, cell_style))

        table_data.append(row_data)

    # Calculate column widths based on landscape letter size
    # Landscape letter is 11 x 8.5 inches
    available_width = 10 * inch  # 11 inches minus 1 inch for margins

    # Custom width allocation for specific columns
    col_widths = []
    for header in headers:
        header_lower = header.lower()
        if header_lower in ['llm_reasoning', 'peer_reasoning', 'reasoning', 'status_reasoning']:
            # Long text columns get more space
            col_widths.append(2.5)
        elif header_lower in ['claim', 'source_text']:
            # Claim text gets more space
            col_widths.append(2.0)
        elif header_lower in ['notes', 'evidence_reasoning']:
            # Notes column gets medium space
            col_widths.append(1.8)
        elif header_lower in ['n_llm', 'n_peer', 'n_itx']:
            # Number columns get less space
            col_widths.append(0.5)
        elif header_lower in ['llm_status', 'peer_status', 'agreement_status', 'status', 'claim_type']:
            # Status columns get less space
            col_widths.append(0.8)
        elif header_lower in ['claim_id', 'result_id', 'llm_result_id', 'peer_result_id']:
            # ID columns get minimal space
            col_widths.append(0.6)
        elif header_lower in ['evidence_type', 'claim_ids']:
            # List columns get medium space
            col_widths.append(1.2)
        else:
            # Default width
            col_widths.append(1.0)

    # Normalize widths to fit available space
    total_width = sum(col_widths)
    col_widths = [w * available_width / total_width for w in col_widths]

    # Create the table
    table = Table(table_data, colWidths=col_widths, repeatRows=1)

    # Apply table styling
    table_style = TableStyle([
        # Header styling
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 0), (-1, 0), 12),

        # Data rows styling
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),

        # Grid
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('LINEBELOW', (0, 0), (-1, 0), 2, colors.HexColor('#2C3E50')),

        # Alternating row colors
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')]),

        # Vertical alignment
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ])

    table.setStyle(table_style)

    # Add the table to elements
    elements.append(table)

    # Build the PDF
    doc.build(elements)
