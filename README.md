# Khôi phục dấu tiếng Việt bằng phân loại token với Transformer

Đề án cuối môn Xử lý ngôn ngữ tự nhiên (NLP): nhập câu tiếng Việt không
dấu, trả về câu có dấu đúng. Ví dụ: `toi dang hoc tieng viet` →
`Tôi đang học tiếng Việt`.

## Kết quả chính

Bốn hệ thống so sánh trên cùng tập test 10.000 câu Wikipedia
(239.189 từ), cùng một định nghĩa metric:

| Hệ thống | Word accuracy | Sentence accuracy | Từ đa nghĩa |
|---|---:|---:|---:|
| Từ điển | 77,4% | 2,6% | 74,6% |
| Bigram + Viterbi | 91,9% | 30,8% | 91,3% |
| PhoBERT fine-tune | 88,5% | 15,6% | 87,4% |
| Transformer tự build (argmax) | 93,2% | 34,2% | 93,0% |
| **Transformer tự build + Viterbi** | **95,0%** | **46,3%** | **94,9%** |

Kết quả bất ngờ: Transformer tự build (encoder 4 lớp, 9,5M tham số,
train 20 epoch) tốt nhất, vượt cả PhoBERT fine-tune (6 epoch, 88,5%).
Giải mã Viterbi + bigram (beta 2,0, chọn trên validation) cộng thêm
+1,8 điểm word và +12,1 điểm sentence so với argmax, đưa hệ tốt nhất
lên **95,0% — đạt mục tiêu tham khảo 95%**. Fine-tune thêm 3 epoch
trên 150.000 câu phụ đề phim (trộn 50.000 câu Wikipedia) đạt 96,2%
trên test phụ đề mà vẫn giữ 95,0% trên test Wikipedia. Phân tích lỗi:
55,1% số lỗi chỉ sai thanh điệu, 88,6% rơi vào từ đa nghĩa (phân tích
của bản 12 epoch 16/11; bản Viterbi chỉ sai 11.896 từ, chưa phân tích
lại chi tiết). Chi tiết đầy đủ trong
[`reports/bao-cao-khoi-phuc-dau-tieng-viet.md`](reports/bao-cao-khoi-phuc-dau-tieng-viet.md).

## Dữ liệu

Corpus 100.000 câu Wikipedia tiếng Việt (`wikimedia/wikipedia`,
`20231101.vi`), chia train/val/test 80.000/10.000/10.000 (seed 42),
2.387.449 từ. Cặp song song tạo bằng cách bỏ dấu bản gốc làm input, giữ
bản gốc làm nhãn. Cả hai mô hình neural dùng công thức **mẫu dấu**
(diacritic pattern): nhãn của một từ là tuple các đơn vị có dấu tại các
vị trí mang dấu (ví dụ `được` → `("đ", "ư", "ợ")`), 2.326 nhãn.

## Cấu trúc repo

| Đường dẫn | Nội dung |
|---|---|
| `notebooks/01-data-and-baselines.ipynb` | Dữ liệu + baseline từ điển + bigram (Kaggle Notebook 1) |
| `notebooks/02-phobert-finetuning.ipynb` | PhoBERT fine-tune, công thức mẫu dấu (Kaggle Notebook 2) |
| `notebooks/03-self-built-transformer.ipynb` | Transformer tự build + so sánh + phân tích lỗi + demo Gradio + fine-tune văn nói (Kaggle Notebook 3, 14 cell) |
| `scripts/restore_document.py` | Khôi phục dấu cho cả file tài liệu (.txt/.docx/.pdf) bằng checkpoint đã train — mặc định Viterbi beta 2,0 |
| `reports/bao-cao-khoi-phuc-dau-tieng-viet.md` | Báo cáo đầy đủ (tiếng Việt, 10 trang) |
| `PLAN.md` | Kế hoạch đề án |
| `STEP_BY_STEP.md` | Các bước thực hiện chi tiết |
| `results.md` | Nhật ký kết quả từng chặng (số liệu đo thực tế) |
| `scripts/build_pdf.py` | Build báo cáo PDF từ file `.md` (fpdf2 + font Noto Sans) |
| `scripts/render_slides.py` | Render slide kết quả (trang 11–12) dạng PNG |
| `docs/slide-10-tu-build-model.md` | Ghi chú thuyết trình trang 10–12 |

## Chạy lại trên Kaggle

Các notebook chạy trên Kaggle (GPU T4 x2, Internet ON), toàn bộ comment
trong code bằng tiếng Việt:

- Notebook 1 — dữ liệu + baseline: https://www.kaggle.com/code/lcaoho/vn-diacritic-restoration-data-and-baselines
- Notebook 2 — PhoBERT: https://www.kaggle.com/code/lcaoho/vietnamese-diacritic-restoration-phobert
- Notebook 3 — Transformer tự build: https://www.kaggle.com/code/lcaoho/vn-diacritic-restoration-self-built-transformer

Số kiểm chứng khi chạy lại: 1.501 bài viết, 100.000 câu, 2.387.449 từ,
5.461 từ đa nghĩa (train), 2.326 nhãn mẫu dấu.

Thứ tự chạy: Notebook 1 → 2 → 3, mỗi lần một notebook (Kaggle chỉ cho
1 phiên GPU tương tác). Notebook 1 chạy CPU cũng được; Notebook 2 và 3
cần GPU T4 x2 + Internet ON. Tổng thời gian khoảng 1 tiếng 45 phút
(PhoBERT ~70 phút với 6 epoch, tự build ~25 phút với 20 epoch + vài
phút chọn beta Viterbi ở Cell 5). Để bảng so sánh ở Notebook 3 dùng toàn
số mới, tải file `phobert_predictions.csv` từ Output của Notebook 2
rồi upload vào Notebook 3 (hoặc đưa vào Kaggle Dataset) trước khi chạy
Cell 8 — không có file thì bảng vẫn chạy nhưng hàng PhoBERT là số cũ.

## Demo

Demo Gradio chạy trên checkpoint tốt nhất của hệ tự build (Cell 9 của
notebook 3, đã chuyển sang giải mã Viterbi beta 2,0 để cho kết quả đúng
nhất): nhập câu không dấu, nhận câu có dấu trong ~1 giây. Cell 13 là
demo dùng model fine-tune trên phụ đề + Viterbi, hiểu được câu văn nói
(vd "noi len em oi" → "nói lên em ơi"). Link public Gradio chỉ sống khi
phiên Kaggle còn chạy; chạy lại Cell 9 hoặc Cell 13 để có link mới.

## Lưu ý

- Ngày 17/11 code đã qua một vòng review và sửa lỗi (seed cho train,
  assert khớp dữ liệu, masking mẫu tương thích khi dự đoán, PhoBERT
  chọn checkpoint tốt nhất theo val — chi tiết trong `results.md`).
- Ngày 18/11 tinh chỉnh thêm (chi tiết trong `results.md`): PhoBERT
  6 epoch thay cho 3, tự build 20 epoch thay cho 12, và giải mã
  Viterbi + bigram cho hệ tự build (beta chọn trên val, chỉ giữ nếu
  val cải thiện). Đã chạy lại tối 21/11: PhoBERT 88,5%, tự build
  argmax 93,2%, Viterbi 95,0% — xem `results.md` mục cuối.
- File PDF của báo cáo và file PowerPoint không nằm trong repo này
  (GitHub push qua công cụ chỉ nhận file text UTF-8); bản PDF và slide
  nằm cùng thư mục làm việc của đề án.
- Mục tiêu tham khảo 95% word accuracy **đã đạt được** (95,0% nhờ
  Viterbi). Các công trình công bố đạt khoảng 97–98% với corpus và mô
  hình lớn hơn.
