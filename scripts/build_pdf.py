#!/usr/bin/env python3
"""Build bao-cao-khoi-phuc-dau-tieng-viet.pdf tu file .md.

Dung fpdf2 + font Noto Sans (co tieng Viet). Chay tu repo:
    ~/workspace/.venv-nlp/bin/python scripts/build_pdf.py
File .md tim o ./ (bo cuc phang) hoac reports/ (bo cuc repo);
PDF ghi canh file .md. Bieu do comparison_chart.png neu chua co
se duoc tao bang matplotlib tu so lieu Bang 4.1 (file PNG goc
khong nam trong repo).
"""
import re
import sys
from pathlib import Path

from fpdf import FPDF
from fpdf.fonts import FontFace

HERE = Path(__file__).parent
_MD_CANDIDATES = [
    HERE / "bao-cao-khoi-phuc-dau-tieng-viet.md",                 # bo cuc phang
    HERE.parent / "reports" / "bao-cao-khoi-phuc-dau-tieng-viet.md",  # bo cuc repo
]
MD_FILE = next((p for p in _MD_CANDIDATES if p.exists()), None)
if MD_FILE is None:
    raise SystemExit("Khong tim thay bao-cao-khoi-phuc-dau-tieng-viet.md "
                     "(da thu ./ va reports/)")
PDF_FILE = MD_FILE.with_suffix(".pdf")
FONT_DIR = Path("/usr/share/fonts/truetype/noto")


def ensure_chart(path):
    """Tao bieu do so sanh 4 he neu file chua co (du lieu tu Bang 4.1)."""
    if path.exists():
        return path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    systems = ["Từ điển", "Bigram", "PhoBERT", "Tự build"]
    word = [77.4, 91.9, 83.4, 92.5]
    sent = [2.6, 30.8, 6.7, 30.8]
    amb = [74.6, 91.3, 81.8, 92.4]
    x = range(len(systems))
    plt.figure(figsize=(8, 4.5))
    plt.bar([i - 0.25 for i in x], word, 0.25, label="Word")
    plt.bar([i for i in x], sent, 0.25, label="Sentence")
    plt.bar([i + 0.25 for i in x], amb, 0.25, label="Từ đa nghĩa")
    plt.xticks(list(x), systems)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title("So sánh 4 hệ khôi phục dấu (test 10k câu)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Da tao bieu do {path} (file goc khong co trong repo)")
    return path


class ReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Noto", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "Khôi phục dấu tiếng Việt bằng phân loại token với Transformer",
                 align="R")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Noto", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Trang {self.page_no()}/{{nb}}", align="C")


def parse_markdown(text):
    """Chia .md thanh cac block: ('h1'|'h2'|'h3'|'para'|'list'|'num'|
    'table'|'image'|'hr', noi dung)."""
    blocks = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if line.startswith("# "):
            blocks.append(("h1", line[2:].strip()))
            i += 1
        elif line.startswith("## "):
            blocks.append(("h2", line[3:].strip()))
            i += 1
        elif line.startswith("### "):
            blocks.append(("h3", line[4:].strip()))
            i += 1
        elif line.strip() == "---":
            blocks.append(("hr", None))
            i += 1
        elif line.startswith("!["):
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
            if m is None:
                print(f"CANH BAO: dong anh khong dung dinh dang, bo qua: {line}",
                      file=sys.stderr)
                i += 1
                continue
            blocks.append(("image", (m.group(1), m.group(2))))
            i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if all(set(c) <= set(":- ") for c in cells):
                    i += 1
                    continue
                rows.append(cells)
                i += 1
            blocks.append(("table", rows))
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                item = lines[i][2:].strip()
                i += 1
                # Dong tiep theo thut dau la phan noi cua item nay.
                while i < len(lines) and lines[i].startswith("  ") and lines[i].strip():
                    item += " " + lines[i].strip()
                    i += 1
                items.append(item)
            blocks.append(("list", items))
        elif re.match(r"\d+\. ", line):
            items = []
            while i < len(lines) and re.match(r"\d+\. ", lines[i]):
                item = re.sub(r"^\d+\. ", "", lines[i]).strip()
                i += 1
                while i < len(lines) and lines[i].startswith("  ") and lines[i].strip():
                    item += " " + lines[i].strip()
                    i += 1
                items.append(item)
            blocks.append(("num", items))
        else:
            # Doan van: gop cac dong lien nhau thanh 1 doan.
            para = [line.strip()]
            i += 1
            while (i < len(lines) and lines[i].strip()
                   and not lines[i].startswith(("#", "|", "- ", "!["))
                   and not re.match(r"\d+\. ", lines[i])
                   and lines[i].strip() != "---"):
                para.append(lines[i].strip())
                i += 1
            blocks.append(("para", " ".join(para)))
    return blocks


def clean_inline(text):
    """fpdf2 markdown=True ho tro **bold** va __italic__; doi *italic*
    thanh __italic__ va bo backtick. Noto Sans thieu 3 glyph ≥ → − nen
    thay bang ASCII tuong duong."""
    text = text.replace("`", "")
    text = text.replace("≥", ">=").replace("→", "->").replace("−", "-")
    text = re.sub(r"\*(?!\*)([^*]+)\*(?!\*)", r"__\1__", text)
    return text


def build():
    pdf = ReportPDF("P", "mm", "A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.alias_nb_pages()
    pdf.add_font("Noto", "", str(FONT_DIR / "NotoSans-Regular.ttf"))
    pdf.add_font("Noto", "B", str(FONT_DIR / "NotoSans-Bold.ttf"))
    pdf.add_font("Noto", "I", str(FONT_DIR / "NotoSans-Italic.ttf"))
    pdf.add_font("Noto", "BI", str(FONT_DIR / "NotoSans-BoldItalic.ttf"))
    pdf.set_margins(20, 18, 20)

    blocks = parse_markdown(MD_FILE.read_text(encoding="utf-8"))
    pdf.add_page()
    for kind, content in blocks:
        if kind == "h1":
            pdf.set_font("Noto", "B", 20)
            pdf.set_text_color(15, 60, 60)
            pdf.multi_cell(0, 9, clean_inline(content), markdown=True)
            pdf.ln(2)
        elif kind == "h2":
            pdf.ln(4)
            pdf.set_font("Noto", "B", 15)
            pdf.set_text_color(15, 60, 60)
            pdf.multi_cell(0, 8, clean_inline(content), markdown=True)
            pdf.ln(1)
        elif kind == "h3":
            pdf.ln(2)
            pdf.set_font("Noto", "B", 12)
            pdf.set_text_color(14, 124, 123)
            pdf.multi_cell(0, 7, clean_inline(content), markdown=True)
            pdf.ln(1)
        elif kind == "hr":
            pdf.ln(2)
            pdf.set_draw_color(14, 124, 123)
            pdf.line(pdf.l_margin, pdf.get_y(),
                     pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(4)
        elif kind == "para":
            pdf.set_font("Noto", "", 11)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(0, 6.5, clean_inline(content), markdown=True)
            pdf.ln(2)
        elif kind == "list":
            pdf.set_font("Noto", "", 11)
            pdf.set_text_color(30, 30, 30)
            for item in content:
                x = pdf.get_x()
                pdf.cell(6, 6.5, "•")
                pdf.multi_cell(0, 6.5, clean_inline(item), markdown=True)
                pdf.set_x(x)
            pdf.ln(2)
        elif kind == "num":
            pdf.set_font("Noto", "", 11)
            pdf.set_text_color(30, 30, 30)
            for n, item in enumerate(content, 1):
                x = pdf.get_x()
                pdf.cell(8, 6.5, f"{n}.")
                pdf.multi_cell(0, 6.5, clean_inline(item), markdown=True)
                pdf.set_x(x)
            pdf.ln(2)
        elif kind == "image":
            caption, path = content
            # Anh tim canh file .md truoc, roi canh script; bieu do so sanh
            # neu chua co thi tao ra (xem ensure_chart).
            img = MD_FILE.parent / path
            if not img.exists():
                img = HERE / path
            if path == "comparison_chart.png" and not img.exists():
                img = ensure_chart(MD_FILE.parent / path)
            if img.exists():
                pdf.ln(2)
                # image() tu sang trang neu khong du cho va tu day con
                # tro xuong duoi anh — khong cong them chieu cao.
                pdf.image(str(img), x=25, w=160)
                pdf.ln(2)
                pdf.set_font("Noto", "", 9)
                pdf.set_text_color(100, 100, 100)
                pdf.multi_cell(0, 5, caption, align="C")
                pdf.ln(2)
            else:
                print(f"CANH BAO: khong tim thay anh {path}, bo qua khoi PDF",
                      file=sys.stderr)
        elif kind == "table":
            pdf.set_font("Noto", "", 10)
            pdf.set_text_color(30, 30, 30)
            ncols = len(content[0])
            width = pdf.w - pdf.l_margin - pdf.r_margin
            col_w = width / ncols
            with pdf.table(col_widths=[col_w] * ncols,
                           text_align="LEFT",
                           line_height=6) as table:
                for r, row in enumerate(content):
                    tr = table.row()
                    for cell in row:
                        # O bang khong ho tro markdown: cell co **...**
                        # thi bo dau ** va in dam ca o (vd hang Tu build).
                        cell = clean_inline(cell)
                        bold = "**" in cell
                        cell = cell.replace("**", "")
                        style = (FontFace(emphasis="BOLD")
                                 if (r == 0 or bold) else None)
                        tr.cell(cell, style=style)
            pdf.ln(3)

    pdf.output(str(PDF_FILE))
    print(f"Da tao {PDF_FILE} ({PDF_FILE.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
