# Trang 10 (thêm vào slide NLP) — Tự build model thì sao?

Ảnh trang: `slide-10-tu-build-model.png` (2560×1440, cùng tông nền/teal với 9 trang gốc).
Chèn vào PowerPoint: thêm 1 slide trắng → Insert → Pictures → chọn file PNG → kéo full-bleed.
Chưa chèn sẵn vào file `.pptx` vì 9 trang gốc là ảnh render, không có text chỉnh được — chèn tay 1 phút là xong.

## Nội dung trang

**Tiêu đề:** Tự build model thì sao? Ba hệ thống để so sánh

| | 1. Baseline | 2. Tự build | 3. PhoBERT |
|---|---|---|---|
| Cách làm | Từ điển + n-gram, nhanh, dễ làm, không cần GPU | Transformer nhỏ / BiLSTM, tự thiết kế, train từ đầu | Đã học sẵn ~20 GB tiếng Việt, fine-tune token classification |
| Điểm mạnh | Mốc so sánh cho cả hai mô hình còn lại | Hiểu sâu kiến trúc — điểm cộng khi thuyết trình | Thường mạnh nhất, kỳ vọng word accuracy cao nhất |
| Điểm yếu | Yếu khi một từ có nhiều ứng viên đúng ngữ pháp | Cần corpus lớn + train lâu; thường thấp hơn PhoBERT vài % word accuracy | 2–3 epochs, lr ~2e-5, batch 16–32, chạy Kaggle GPU |

**Khuyến nghị:** làm cả ba, so sánh trên cùng tập test, cùng word accuracy + sentence accuracy — điền số vào bảng ở trang 7 sau khi đo, đừng đoán trước.

**Gợi ý khởi điểm cho model tự build:** encoder 4–6 lớp, hidden 256, dropout 0,1, Adam lr 1e-3 — chỉnh theo validation. (Chỉ là điểm khởi đầu, không phải số đo.)

## Nói khoảng 60 giây khi tới trang này

> "Tự build được, nhưng có đánh đổi. PhoBERT đã học sẵn 20 GB tiếng Việt nên ban đầu em kỳ vọng nó mạnh nhất. Vì vậy em làm cả ba: baseline để có mốc, model tự build để hiểu sâu kiến trúc, và PhoBERT để có số liệu mạnh nhất — cả ba đo trên cùng một tập test."

## Cập nhật sau khi có kết quả (16/11/2026)

Kỳ vọng ở trang này đã **sai**: kết quả thực tế ở trang 11–12 cho thấy
model tự build (92,5%) thắng PhoBERT (83,4%). Khi thuyết trình, sau khi
nói câu trên thì chuyển sang trang 11 và nói thêm:

> "Kết quả thực tế ngược với kỳ vọng: model tự build đạt 92,5% word
> accuracy, cao nhất trong 4 hệ, còn PhoBERT chỉ đạt 83,4% — thấp hơn cả
> bigram. Nguyên nhân chính là PhoBERT chỉ train 3 epoch, loss còn 1,95,
> đầu phân loại 2.326 lớp chưa học đủ, trong khi model tự build train
> 12 epoch với learning rate cao hơn."

**Trang 11 — Kết quả thực nghiệm** (`slide-11-ket-qua.png`): bảng so sánh
4 hệ + biểu đồ + kết luận ngược kỳ vọng.
**Trang 12 — Phân tích lỗi & Demo** (`slide-12-loi-demo.png`): 55,1% lỗi
chỉ sai thanh điệu, top 10 cặp nhầm lẫn, demo Gradio, hướng phát triển.
File PowerPoint đủ 12 trang:
`~/workspace/your_files/khoi-phuc-dau-tieng-viet/khoi-phuc-dau-tieng-viet-cover-12slides.pptx`
(bản 10 trang giữ nguyên, không sửa).

## Còn chờ user quyết (chưa sửa vào plan/slide)

- Deadline thật + làm cá nhân hay nhóm: PLAN.md đang giả định cá nhân, 4 tuần; slide trang 8 đang để 5 tuần (Tuần 1 dữ liệu, Tuần 2 baseline, Tuần 3–4 PhoBERT, Tuần 5 đánh giá/demo). 4 giai đoạn giống nhau, chỉ khác tổng số tuần — slide trang 8 đã ghi chú "giữ nguyên 4 giai đoạn và nén/giãn từng giai đoạn" nếu thời gian khác.
- ~~Có làm cả 3 hệ thống không, hay chỉ PhoBERT + baseline.~~ → Đã chốt ngày 16/11: làm cả 3 hệ (baseline + tự build + PhoBERT).
