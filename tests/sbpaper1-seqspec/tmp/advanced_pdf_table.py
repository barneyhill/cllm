#!/usr/bin/env python3
"""
Advanced JSON to PDF table converter with extensive customization options.
"""

import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape, A4, A3
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT


class AdvancedPDFTableGenerator:
    """
    Advanced PDF table generator with customization options.
    """
    
    def __init__(self, 
                 pagesize=landscape(letter),
                 margin=0.5*inch,
                 header_bg_color='#2C3E50',
                 header_text_color=colors.whitesmoke,
                 alt_row_color='#F8F9FA',
                 grid_color=colors.grey,
                 base_font_size=8,
                 small_font_size=7):
        """
        Initialize with custom styling options.
        
        Args:
            pagesize: Page size (landscape(letter), landscape(A4), etc.)
            margin: Page margin
            header_bg_color: Header background color (hex or color object)
            header_text_color: Header text color
            alt_row_color: Alternating row color (hex or color object)
            grid_color: Grid line color
            base_font_size: Base font size for cells
            small_font_size: Font size for long text columns
        """
        self.pagesize = pagesize
        self.margin = margin
        self.header_bg_color = self._parse_color(header_bg_color)
        self.header_text_color = header_text_color
        self.alt_row_color = self._parse_color(alt_row_color)
        self.grid_color = grid_color
        self.base_font_size = base_font_size
        self.small_font_size = small_font_size
        
        # Set up styles
        self._setup_styles()
    
    def _parse_color(self, color):
        """Convert hex string to color object if needed."""
        if isinstance(color, str) and color.startswith('#'):
            return colors.HexColor(color)
        return color
    
    def _setup_styles(self):
        """Set up paragraph styles."""
        styles = getSampleStyleSheet()
        
        self.header_style = ParagraphStyle(
            'TableHeader',
            parent=styles['Normal'],
            fontSize=self.base_font_size + 1,
            textColor=self.header_text_color,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            leading=self.base_font_size + 3
        )
        
        self.cell_style = ParagraphStyle(
            'TableCell',
            parent=styles['Normal'],
            fontSize=self.base_font_size,
            alignment=TA_LEFT,
            fontName='Helvetica',
            leading=self.base_font_size + 2,
            wordWrap='CJK'
        )
        
        self.cell_style_small = ParagraphStyle(
            'TableCellSmall',
            parent=styles['Normal'],
            fontSize=self.small_font_size,
            alignment=TA_LEFT,
            fontName='Helvetica',
            leading=self.small_font_size + 2,
            wordWrap='CJK'
        )
        
        self.title_style = styles['Title']
    
    def generate_pdf(self, 
                     json_data, 
                     output_filename, 
                     title=None,
                     column_widths=None,
                     long_text_columns=None,
                     exclude_columns=None,
                     column_order=None):
        """
        Generate PDF from JSON data with advanced options.
        
        Args:
            json_data: List of dictionaries or path to JSON file
            output_filename: Output PDF file path
            title: Optional document title
            column_widths: Dict mapping column names to relative widths
            long_text_columns: Set/list of column names that should use small font
            exclude_columns: Set/list of column names to exclude from output
            column_order: List specifying the order of columns
        """
        
        # Load JSON data
        if isinstance(json_data, str):
            with open(json_data, 'r') as f:
                data = json.load(f)
        else:
            data = json_data
        
        if not data:
            raise ValueError("JSON data is empty")
        
        # Set defaults for customization options
        if long_text_columns is None:
            long_text_columns = {'notes', 'llm_reasoning', 'peer_reasoning', 
                                'reasoning', 'description', 'comments'}
        if exclude_columns is None:
            exclude_columns = set()
        
        # Extract and order headers
        all_headers = list(data[0].keys())
        
        # Filter out excluded columns
        headers = [h for h in all_headers if h not in exclude_columns]
        
        # Apply custom column ordering if specified
        if column_order:
            # Ensure all headers are included
            remaining = [h for h in headers if h not in column_order]
            headers = [h for h in column_order if h in headers] + remaining
        
        # Set up the PDF document
        doc = SimpleDocTemplate(
            output_filename,
            pagesize=self.pagesize,
            rightMargin=self.margin,
            leftMargin=self.margin,
            topMargin=self.margin + 0.25*inch,
            bottomMargin=self.margin
        )
        
        elements = []
        
        # Add title
        if title:
            elements.append(Paragraph(title, self.title_style))
            elements.append(Paragraph("<br/>", self.cell_style))
        
        # Create header row
        table_data = [[Paragraph(str(h).replace('_', ' ').title(), self.header_style) 
                      for h in headers]]
        
        # Add data rows
        for row in data:
            row_data = []
            for header in headers:
                value = row.get(header, '')
                
                if value is None:
                    value = ''
                else:
                    value = str(value)
                
                # Use appropriate style based on column type
                if header.lower() in long_text_columns:
                    row_data.append(Paragraph(value, self.cell_style_small))
                else:
                    row_data.append(Paragraph(value, self.cell_style))
            
            table_data.append(row_data)
        
        # Calculate column widths
        page_width, page_height = self.pagesize
        available_width = page_width - (2 * self.margin)
        
        # Use custom widths if provided, otherwise auto-calculate
        if column_widths:
            col_widths_list = [column_widths.get(h, 1.0) for h in headers]
        else:
            col_widths_list = self._auto_calculate_widths(headers, long_text_columns)
        
        # Normalize widths
        total_width = sum(col_widths_list)
        col_widths_list = [w * available_width / total_width for w in col_widths_list]
        
        # Create table
        table = Table(table_data, colWidths=col_widths_list, repeatRows=1)
        
        # Apply styling
        table_style = self._create_table_style()
        table.setStyle(table_style)
        
        elements.append(table)
        
        # Build PDF
        doc.build(elements)
        print(f"✓ PDF created successfully: {output_filename}")
    
    def _auto_calculate_widths(self, headers, long_text_columns):
        """Auto-calculate column widths based on header names."""
        col_widths = []
        for header in headers:
            header_lower = header.lower()
            
            if header_lower in long_text_columns:
                col_widths.append(2.5)
            elif any(x in header_lower for x in ['id', 'status', 'agreement']):
                col_widths.append(0.8)
            elif header_lower.startswith('n_'):
                col_widths.append(0.5)
            else:
                col_widths.append(1.0)
        
        return col_widths
    
    def _create_table_style(self):
        """Create the table style."""
        return TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), self.header_bg_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), self.header_text_color),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), self.base_font_size + 1),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            
            # Data rows styling
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), self.base_font_size),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, self.grid_color),
            ('LINEBELOW', (0, 0), (-1, 0), 2, self.header_bg_color),
            
            # Alternating row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), 
             [colors.white, self.alt_row_color]),
            
            # Vertical alignment
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ])


# Convenience function for backward compatibility
def json_to_pdf_table_advanced(json_data, output_filename, **kwargs):
    """
    Convenience wrapper for the advanced generator.
    
    Example usage:
        json_to_pdf_table_advanced(
            data,
            "output.pdf",
            title="My Table",
            pagesize=landscape(A4),
            header_bg_color='#1E88E5',
            long_text_columns={'description', 'notes'},
            exclude_columns={'internal_id'},
            column_order=['name', 'status', 'notes']
        )
    """
    generator = AdvancedPDFTableGenerator(
        pagesize=kwargs.pop('pagesize', landscape(letter)),
        margin=kwargs.pop('margin', 0.5*inch),
        header_bg_color=kwargs.pop('header_bg_color', '#2C3E50'),
        header_text_color=kwargs.pop('header_text_color', colors.whitesmoke),
        alt_row_color=kwargs.pop('alt_row_color', '#F8F9FA'),
        grid_color=kwargs.pop('grid_color', colors.grey),
        base_font_size=kwargs.pop('base_font_size', 8),
        small_font_size=kwargs.pop('small_font_size', 7)
    )
    
    generator.generate_pdf(json_data, output_filename, **kwargs)


# Example usage
if __name__ == "__main__":
    sample_data = [
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
    
    # Example 1: Custom colors and styling
    json_to_pdf_table_advanced(
        sample_data,
        "/mnt/user-data/outputs/advanced_example1.pdf",
        title="Custom Styled Table",
        header_bg_color='#1E88E5',  # Blue header
        alt_row_color='#E3F2FD',     # Light blue alternating rows
        base_font_size=9
    )
    
    # Example 2: A4 size with custom column order
    json_to_pdf_table_advanced(
        sample_data,
        "/mnt/user-data/outputs/advanced_example2.pdf",
        title="A4 Landscape with Custom Order",
        pagesize=landscape(A4),
        column_order=['llm_result_id', 'llm_status', 'peer_status', 
                     'agreement_status', 'llm_reasoning']
    )
    
    # Example 3: Exclude certain columns
    json_to_pdf_table_advanced(
        sample_data,
        "/mnt/user-data/outputs/advanced_example3.pdf",
        title="Filtered Columns",
        exclude_columns={'peer_result_id', 'n_itx'}
    )
    
    print("\n✓ All advanced examples completed!")