#!/usr/bin/env python3
"""Render slide 11 (ket qua) + slide 12 (phan tich loi & demo) dang PNG
2560x1440, cung style voi slide 10 (nen teal nhat, the trang bo tron).

Chay: ~/workspace/.venv-nlp/bin/python render_slides.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
W, H = 2560, 1440

BG = (231, 243, 242)
TITLE = (10, 61, 62)
TEAL = (14, 124, 123)
CARD_BORDER = (191, 224, 222)
BODY = (46, 58, 58)
GRAY = (90, 110, 109)
WHITE = (255, 255, 255)
ROW_ALT = (240, 248, 247)
ROW_HL = (205, 235, 233)
TABLE_HEAD_BG = (14, 124, 123)

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


def wrap(draw, text, fnt, max_w):
    """Ngat dong theo chieu rong (do bang font)."""
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def bullet_block(draw, x, y, max_w, items, fnt, gap=26, lh=58):
    """Ve list bullet (chu thuong, nhu slide 10); tra ve y sau block."""
    for text in items:
        lines = wrap(draw, text, fnt, max_w - 40)
        draw.ellipse([x, y + 20, x + 16, y + 36], fill=TEAL)
        for j, ln in enumerate(lines):
            draw.text((x + 40, y + j * lh), ln, font=fnt, fill=BODY)
        y += len(lines) * lh + gap
    return y


def card(draw, box, radius=36):
    draw.rounded_rectangle(box, radius=radius, fill=WHITE,
                           outline=CARD_BORDER, width=3)


def slide11():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((110, 85), "Kết quả thực nghiệm: Transformer tự build tốt nhất",
           font=font(72, True), fill=TITLE)

    # --- The bang ben trai ---
    cx, cy, cw, ch = 110, 250, 1330, 780
    card(d, [cx, cy, cx + cw, cy + ch])
    d.text((cx + 50, cy + 40), "Bảng so sánh (test 10.000 câu, 239.189 từ)",
           font=font(46, True), fill=TEAL)
    # Bang: 4 cot
    cols = [cx + 50, cx + 450, cx + 720, cx + 990]
    headers = ["Hệ thống", "Word", "Sentence", "Từ đa nghĩa"]
    rows = [
        ("Từ điển", "77,4%", "2,6%", "74,6%"),
        ("Bigram", "91,9%", "30,8%", "91,3%"),
        ("PhoBERT", "83,4%", "6,7%", "81,8%"),
        ("Tự build", "92,5%", "30,8%", "92,4%"),
    ]
    ty = cy + 140
    rh = 110
    # Header
    d.rectangle([cx + 30, ty, cx + cw - 30, ty + rh], fill=TABLE_HEAD_BG)
    for c, htxt in enumerate(headers):
        d.text((cols[c], ty + 28), htxt, font=font(40, True), fill=WHITE)
    for r, row in enumerate(rows):
        ry = ty + rh * (r + 1)
        if r == 3:
            d.rectangle([cx + 30, ry, cx + cw - 30, ry + rh], fill=ROW_HL)
        elif r % 2 == 1:
            d.rectangle([cx + 30, ry, cx + cw - 30, ry + rh], fill=ROW_ALT)
        for c, cell in enumerate(row):
            f = font(42, True) if (r == 3 or c == 0) else font(42)
            d.text((cols[c], ry + 28), cell, font=f, fill=TITLE if r == 3 else BODY)
    d.text((cx + 50, cy + ch - 70),
           "Tự build vượt bigram ở word (+0,6) và từ đa nghĩa (+1,1).",
           font=font(36), fill=GRAY)

    # --- The bieu do ben phai ---
    gx, gy, gw, gh = 1490, 250, 960, 780
    card(d, [gx, gy, gx + gw, gy + gh])
    d.text((gx + 50, gy + 40), "Biểu đồ so sánh", font=font(46, True), fill=TEAL)
    chart = Image.open(HERE / "comparison_chart.png")
    chart = chart.resize((860, int(860 * chart.size[1] / chart.size[0])))
    img.paste(chart, (gx + 50, gy + 140))

    # --- Dong ket luan duoi cung ---
    d.text((110, 1100),
           "Ngược với kỳ vọng ban đầu: PhoBERT (83,4%) thua cả bigram (91,9%) —",
           font=font(44, True), fill=TITLE)
    d.text((110, 1170),
           "chỉ train 3 epoch, loss còn 1,95, đầu phân loại 2.326 lớp chưa học đủ.",
           font=font(40), fill=BODY)
    d.text((110, 1270),
           "Mục tiêu 95% chưa đạt — tốt nhất 92,5%. SOTA công bố: ~97–98%.",
           font=font(40), fill=TEAL)
    img.save(HERE / "slide-11-ket-qua.png")
    print("saved slide-11-ket-qua.png")


def slide12():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((110, 85), "Phân tích lỗi & Demo", font=font(72, True), fill=TITLE)

    # --- The trai: phan tich loi ---
    cx, cy, cw, ch = 110, 250, 1170, 880
    card(d, [cx, cy, cx + cw, cy + ch])
    d.text((cx + 50, cy + 40), "Lỗi của hệ tự build (17.918 từ sai)",
           font=font(46, True), fill=TEAL)
    y = bullet_block(d, cx + 50, cy + 150, cw - 100, [
        "55,1% số lỗi chỉ sai thanh điệu (cùng nguyên âm, khác dấu).",
        "88,6% số lỗi rơi vào từ đa nghĩa.",
        "Chỉ 6,3% số lỗi ở từ chưa thấy trong train.",
    ], font(40))
    d.text((cx + 50, y + 10), "10 cặp nhầm lẫn hàng đầu (đều chỉ khác thanh điệu):",
           font=font(38, True), fill=TITLE)
    pairs = ["nhưng → những (73)", "từ → tự (45)", "hoá → hóa (45)",
             "khí → khi (42)", "còn → con (41)", "trọng → trong (39)",
             "tự → từ (38)", "cổ → có (38)", "nền → nên (35)", "loài → loại (32)"]
    py = y + 80
    for i in range(0, 10, 2):
        d.text((cx + 50, py), f"• {pairs[i]}", font=font(36), fill=BODY)
        d.text((cx + 620, py), f"• {pairs[i + 1]}", font=font(36), fill=BODY)
        py += 58

    # --- The phai: demo + huong phat trien ---
    gx, gy, gw, gh = 1330, 250, 1120, 880
    card(d, [gx, gy, gx + gw, gy + gh])
    d.text((gx + 50, gy + 40), "Demo Gradio (Cell 9)", font=font(46, True), fill=TEAL)
    y = bullet_block(d, gx + 50, gy + 150, gw - 100, [
        "Nhập câu không dấu, nhận câu có dấu trong ~1 giây.",
        "Test 3 câu mẫu: 20/27 từ đúng — các câu ví dụ dồn nhiều từ đa nghĩa.",
        "Ví dụ: “ha noi la thu do cua viet nam” → “hà nội là thủ đô của viết năm”.",
    ], font(38), gap=20, lh=54)
    d.text((gx + 50, y + 10), "Hướng phát triển", font=font(46, True), fill=TEAL)
    bullet_block(d, gx + 50, y + 100, gw - 100, [
        "Train PhoBERT thêm epoch + tăng lr; mở rộng corpus lên 500.000 câu.",
        "Thử seq2seq (mT5/T5) + constrained decoding / CRF.",
    ], font(38), gap=20, lh=54)

    # --- Dong duoi cung ---
    d.text((110, 1200),
           "Kết luận: tự build 92,5% là hệ tốt nhất; lỗi chủ đạo là nhầm thanh điệu ở từ đa nghĩa.",
           font=font(40, True), fill=TITLE)
    d.text((110, 1270),
           "Code + số liệu đầy đủ trong 3 notebook Kaggle (xem Phụ lục báo cáo).",
           font=font(38), fill=TEAL)
    img.save(HERE / "slide-12-loi-demo.png")
    print("saved slide-12-loi-demo.png")


if __name__ == "__main__":
    slide11()
    slide12()
