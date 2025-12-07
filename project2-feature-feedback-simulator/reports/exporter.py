import os
import csv
from datetime import datetime
from fpdf import FPDF   # Lightweight PDF library

SAVE_DIR = os.path.join("reports", "saved_results")

# Ensure directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

class ReportExporter:

    # ------------------------------
    # Filename helper
    # ------------------------------
    @staticmethod
    def _make_filename(base, ext):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{base}_{timestamp}.{ext}"

    # ------------------------------
    # CSV Export
    # ------------------------------
    @staticmethod
    def export_csv(results_dict, filename="feedback"):
        """
        results_dict format:
        {
            "Persona Name": "Feedback text...",
            "Persona 2": "Feedback text..."
        }
        """
        file_path = os.path.join(SAVE_DIR, 
                                 ReportExporter._make_filename(filename, "csv"))

        with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Persona", "Feedback"])

            for persona, feedback in results_dict.items():
                writer.writerow([persona, feedback])

        return file_path

    # ------------------------------
    # PDF Export
    # ------------------------------
    @staticmethod
    def export_pdf(title, content, filename="report"):
        file_path = os.path.join(SAVE_DIR, 
                                 ReportExporter._make_filename(filename, "pdf"))

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, title, ln=True)

        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, content)

        pdf.output(file_path)

        return file_path

    # ------------------------------
    # Raw text export (optional)
    # ------------------------------
    @staticmethod
    def save_txt(content, filename="feedback"):
        file_path = os.path.join(SAVE_DIR, 
                                 ReportExporter._make_filename(filename, "txt"))

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return file_path
