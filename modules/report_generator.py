from fpdf import FPDF
from datetime import datetime

def parse_insights(insights):

    sections = {
        "KEY DRIVERS": "",
        "RISK FACTORS": "",
        "BUSINESS INTERPRETATION": "",
        "RECOMMENDATIONS": ""
    }

    current_section = None

    for line in insights.split("\n"):

        line_clean = line.strip().upper().replace(":", "")

        for section in sections:
            if section in line_clean:
                current_section = section
                break
        else:
            if current_section:
                sections[current_section] += line.strip() + " "

    return sections

def generate_report(insights, results):

    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, "AI Autonomous Data Scientist Report", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)

    pdf.ln(10)

    # EDA
    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,"EDA Analysis",ln=True)

    pdf.ln(5)

    pdf.image("saved_charts/target_distribution.png", w=170)

    pdf.image("saved_charts/correlation_heatmap.png", w=170)

    pdf.ln(5)

    # Section: Model Performance
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "1. Model Performance", ln=True)

    pdf.ln(5)

    # Table Header
    pdf.set_font("Arial", "B", 12)
    pdf.cell(90, 10, "Model", border=1)
    pdf.cell(90, 10, "Accuracy", border=1, ln=True)

    # Table Rows
    pdf.set_font("Arial", "", 12)

    best_model = max(results, key=results.get)

    for model, score in results.items():

        if model == best_model:
            pdf.set_font("Arial", "B", 12)
        else:
            pdf.set_font("Arial", "", 12)

        pdf.cell(90, 10, model, border=1)
        pdf.cell(90, 10, f"{score:.4f}", border=1, ln=True)

    pdf.ln(10)

    # Best Model Section
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "2. Best Model", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"The best performing model is: {best_model}", ln=True)

    pdf.ln(10)

    # AI Insights Section
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "3. AI Generated Insights", ln=True)

    pdf.ln(5)

    sections = parse_insights(insights)

    for title, content in sections.items():

        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, title.title(), ln=True)

        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, content)

        pdf.ln(3)

    pdf.ln(10)
    
    file_path = "ai_data_scientist_report.pdf"

    pdf.output(file_path)

    return file_path