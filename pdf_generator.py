import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.units import cm
import textwrap

def register_korean_font():
    """Registers a Korean font for ReportLab."""
    font_path = "C:\\Windows\\Fonts\\malgun.ttf"
    if os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont('Malgun', font_path))
            return 'Malgun'
        except Exception as e:
            print(f"Font registration error: {e}")
            return 'Helvetica' # Fallback
    else:
        # Check standard paths or fallback
        return 'Helvetica'

def draw_header(c, title, page_num):
    c.saveState()
    c.setFont('Malgun', 10)
    c.drawString(2*cm, 28.5*cm, title)
    c.drawRightString(19*cm, 28.5*cm, f"Page {page_num}")
    c.line(2*cm, 28.3*cm, 19*cm, 28.3*cm)
    c.restoreState()

def generate_workbook(problems, title="Math Workbook", author="GenAI Teacher", output_path="workbook.pdf"):
    """
    Generates a PDF workbook from a list of problem dictionaries.
    Each problem dict should have: 'number', 'content' (or 'text'), 'options', 'solution'.
    """
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    font_name = register_korean_font()
    
    # --- Cover Page ---
    c.setFont(font_name, 24)
    c.drawCentredString(width/2, height/2 + 2*cm, title)
    c.setFont(font_name, 14)
    c.drawCentredString(width/2, height/2 - 1*cm, f"Created by: {author}")
    c.drawCentredString(width/2, height/2 - 2*cm, "Generated with Gemini Math Extractor")
    c.showPage()
    
    # --- Problem Pages ---
    y_position = 25 * cm
    page_num = 1
    
    # Filter valid problems
    valid_probs = [p for p in problems if p]
    
    for i, prob in enumerate(valid_probs):
        draw_header(c, title, page_num)
        
        # Problem Number
        p_num = prob.get('number', i+1)
        c.setFont(font_name, 12)
        c.drawString(2*cm, y_position, f"Problem {p_num}.")
        
        # Problem Content
        content = prob.get('content') or prob.get('text', '')
        # Basic text wrapping (ReportLab doesn't handle complex LaTeX natively easily without platypus, 
        # but for now we stick to basic text drawing for simplicity or use textobject)
        
        text_obj = c.beginText(2.5*cm, y_position - 1*cm)
        text_obj.setFont(font_name, 11)
        
        # Split by newlines first
        lines = content.split('\n')
        wrapped_lines = []
        for line in lines:
            wrapped_lines.extend(textwrap.wrap(line, width=80)) # adjust width heuristic
            
        for line in wrapped_lines:
            text_obj.textLine(line)
        
        c.drawText(text_obj)
        
        # Calculate space used (approximate)
        text_height = len(wrapped_lines) * 14 # 14 points per line
        current_y = y_position - 1*cm - text_height - 1*cm
        
        # Options
        options = prob.get('options', [])
        if options:
            opt_text_obj = c.beginText(2.5*cm, current_y)
            opt_text_obj.setFont(font_name, 10)
            for j, opt in enumerate(options):
                opt_text_obj.textLine(f"{j+1}. {opt}")
            c.drawText(opt_text_obj)
            current_y -= (len(options) * 14 + 10)
        
        # Workspace line
        c.setDash(3, 3)
        c.line(2*cm, current_y - 1*cm, 19*cm, current_y - 1*cm)
        c.drawString(17*cm, current_y - 0.8*cm, "Workspace")
        c.setDash(1, 0) # Reset dash
        
        # Move Y for next problem
        # Give some space for working
        space_per_prob = 8 * cm
        y_position = current_y - space_per_prob
        
        # Check if new page needed
        if y_position < 4 * cm:
            c.showPage()
            page_num += 1
            y_position = 25 * cm

    c.showPage()
    
    # --- Answer Key Page ---
    page_num += 1
    draw_header(c, title + " - Answer Key", page_num)
    
    c.setFont(font_name, 16)
    c.drawString(2*cm, 25*cm, "Answer Key & Hints")
    
    y = 23 * cm
    c.setFont(font_name, 10)
    
    for i, prob in enumerate(valid_probs):
        p_num = prob.get('number', i+1)
        
        # Extract answer (simple heuristic or use 'solution' field)
        # We'll just show the full solution in a concise way or just the answer if parsed
        # For this version, let's show the first 2 lines of solution or "See detailed solution"
        
        sol_full = prob.get('solution', 'No solution provided.')
        # Clean up XML tags if any
        import re
        sol_clean = re.sub(r'<[^>]+>', '', sol_full).strip()
        sol_summary = sol_clean.split('\n')[0][:100] + "..." if len(sol_clean) > 100 else sol_clean
        
        line = f"{p_num}. {sol_summary}"
        
        if y < 3*cm:
            c.showPage()
            page_num += 1
            draw_header(c, title + " - Answer Key", page_num)
            y = 25 * cm
            
        c.drawString(2.5*cm, y, line)
        y -= 0.8 * cm

    c.save()
    return output_path

if __name__ == "__main__":
    # Test
    dummy_probs = [
        {
            "number": 1,
            "content": "Solve for x: 2x + 5 = 15",
            "options": ["x=1", "x=5", "x=10"],
            "solution": "2x = 10 -> x = 5"
        },
        {
            "number": 2,
            "content": "What is the area of a circle with radius 3?",
            "options": ["9pi", "3pi", "6pi"],
            "solution": "Area = pi * r^2 = 9pi"
        }
    ]
    generate_workbook(dummy_probs, output_path="test_workbook.pdf")
    print("Test PDF generated: test_workbook.pdf")
