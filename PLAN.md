# Kế hoạch đề án NLP: Khôi phục dấu tiếng Việt

**Đề tài:** Khôi phục dấu tiếng Việt (Vietnamese Diacritic Restoration)
**Tên tiếng Anh:** *Vietnamese Diacritic Restoration using Transformer-based Token Classification*
**Mô hình chính:** PhoBERT (token classification) so sánh với baseline từ điển + n-gram
**Môi trường:** Kaggle (GPU miễn phí), Python, Hugging Face Transformers

> Giả định: làm cá nhân, 4 tuần. Nếu nhóm hoặc thời gian khác, điều chỉnh số tuần tương ứng.

## Mục tiêu

- Xây hệ thống nhập câu tiếng Việt không dấu, trả về câu có dấu đúng.
- So sánh baseline truyền thống với mô hình PhoBERT fine-tune.
- Đạt word accuracy ≥ 95% trên tập test (mục tiêu tham khảo; báo cáo số thực tế đo được).
- Có demo chạy được: nhập câu không dấu → hiện câu đã khôi phục dấu.

## Tuần 1 — Dữ liệu

- [ ] Thu thập corpus tiếng Việt có dấu (~100k–500k câu): Wikipedia tiếng Việt / báo / truyện.
- [ ] Làm sạch: bỏ câu quá ngắn/dài, bỏ ký tự lạ, chuẩn hóa khoảng trắng.
- [ ] Tạo cặp song song: bỏ dấu bằng `unidecode`-style mapping để làm input, giữ bản gốc làm label.
- [ ] Chia train/val/test theo tỉ lệ 80/10/10 (chia theo câu, không trộn).
- [ ] Thống kê dữ liệu: số câu, số từ, tỉ lệ từ có dấu, từ đa nghĩa thường gặp (vd: "co" → có/cô/cỏ...).

## Tuần 2 — Baseline

- [ ] Baseline 1: tra từ điển tần suất — mỗi từ không dấu map sang dạng có dấu phổ biến nhất.
- [ ] Baseline 2: n-gram (bigram/trigram) chọn dạng có dấu theo ngữ cảnh.
- [ ] Đánh giá baseline: word accuracy, sentence accuracy trên tập test.
- [ ] Phân tích lỗi baseline: lỗi ở từ nào, trường hợp nào (từ đa nghĩa, từ hiếm).

## Tuần 3 — Mô hình PhoBERT

- [ ] Chuẩn bị dữ liệu dạng token classification: mỗi token không dấu → nhãn là dạng có dấu tương ứng.
- [ ] Fine-tune `vinai/phobert-base` trên Kaggle GPU (2–3 epochs, lr ~2e-5, batch 16–32).
- [ ] Đánh giá trên val/test, so sánh trực tiếp với baseline (cùng tập test, cùng metric).
- [ ] Thử nghiệm phụ (nếu còn thời gian): tăng dữ liệu, thử `phobert-large`, hoặc seq2seq (mT5).

## Tuần 4 — Đánh giá, demo, báo cáo

- [ ] Đánh giá chi tiết: word accuracy, sentence accuracy, confusion matrix cho các từ đa nghĩa phổ biến.
- [ ] Phân tích lỗi của PhoBERT so với baseline (cải thiện ở đâu, còn sai ở đâu).
- [ ] Demo: Gradio / Streamlit — nhập câu không dấu, hiện kết quả + tô màu từ đã sửa.
- [ ] Viết báo cáo: giới thiệu, dữ liệu, phương pháp, kết quả, phân tích lỗi, kết luận.
- [ ] Chuẩn bị slide thuyết trình + chạy thử demo trước khi nộp.

## Đánh giá (metrics)

| Metric | Định nghĩa |
|---|---|
| Word accuracy | Tỉ lệ từ khôi phục dấu đúng / tổng số từ |
| Sentence accuracy | Tỉ lệ câu khôi phục đúng toàn bộ / tổng số câu |
| Accuracy trên từ đa nghĩa | Accuracy riêng cho nhóm từ có ≥ 2 dạng có dấu (vd: co, ma, an) |

## Rủi ro & cách xử lý

- **GPU Kaggle giới hạn thời gian:** giảm kích thước corpus hoặc số epoch; lưu checkpoint sau mỗi epoch.
- **Từ hiếm / tên riêng không có trong train:** mô hình sẽ sai — ghi nhận trong phần phân tích lỗi, không cố "chữa" bằng tay.
- **Dữ liệu nhiễu (teencode, viết tắt):** lọc ở bước làm sạch; ghi rõ trong báo cáo.

## Deliverables

- Code (notebook + script) chạy được từ đầu đến cuối trên Kaggle.
- Báo cáo (PDF) + slide thuyết trình.
- Demo chạy được (Gradio/Streamlit hoặc notebook demo).
