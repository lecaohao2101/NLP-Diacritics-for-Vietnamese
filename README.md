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
| PhoBERT fine-tune | 83,4% | 6,7% | 81,8% |
| **Transformer tự build** | **92,5%** | **30,8%** | **92,4%** |

Kết quả bất ngờ: Transformer tự build (encoder 4 lớp, 9,5M tham số,
train 12 epoch) tốt nhất, vượt cả PhoBERT fine-tune (chỉ train 3 epoch,
loss còn 1,95, đầu phân loại 2.326 lớp chưa học đủ). Phân tích lỗi: 55,1%
số lỗi chỉ sai thanh điệu, 88,6% rơi vào từ đa nghĩa. Chi tiết đầy đủ
trong [`reports/bao-cao-khoi-phuc-dau-tieng-viet.md`](reports/bao-cao-khoi-phuc-dau-tieng-viet.md).

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
| `notebooks/03-self-built-transformer.ipynb` | Transformer tự build + so sánh + phân tích lỗi + demo Gradio (Kaggle Notebook 3) |
| `reports/bao-cao-khoi-phuc-dau-tieng-viet.md` | Báo cáo đầy đủ (tiếng Việt, 9 trang) |
| `PLAN.md` | Kế hoạch đề án |
| `STEP_BY_STEP.md` | Các bước thực hiện chi tiết |
| `results.md` | Nhật ký kết quả từng chặng (số liệu đo thực tế) |
| `scripts/build_pdf.py` | Build báo cáo PDF từ file `.md` (fpdf2 + font Noto Sans) |
| `scripts/render_slides.py` | Render slide kết quả (trang 11–12) dạng PNG |
| `docs/slide-10-tu-build-model.md` | Ghi chú thuyết trình trang 10–12 |

## Chạy lại trên Kaggle

Các notebook chạy trên Kaggle (GPU T4 x2, Internet ON), toàn bộ comment
trong code bằng tiếng Việt:

- Notebook 1 — dữ liệu + baseline: https://www.kaggle.com/code/lcaoho/vietnamese-diacritic-restoration-week-1-starter
- Notebook 2 — PhoBERT: https://www.kaggle.com/code/lcaoho/vietnamese-diacritic-restoration-phobert
- Notebook 3 — Transformer tự build: https://www.kaggle.com/code/lcaoho/vn-diacritic-restoration-self-built-transformer

Số kiểm chứng khi chạy lại: 1.501 bài viết, 100.000 câu, 2.387.449 từ,
5.461 từ đa nghĩa (train), 2.326 nhãn mẫu dấu.

## Demo

Demo Gradio chạy trên checkpoint tốt nhất của hệ tự build (Cell 9 của
notebook 3): nhập câu không dấu, nhận câu có dấu trong ~1 giây. Link
public Gradio chỉ sống khi phiên Kaggle còn chạy; chạy lại Cell 9 để có
link mới.

## Lưu ý

- File PDF của báo cáo và file PowerPoint 12 trang không nằm trong repo
  này (GitHub push qua công cụ chỉ nhận file text UTF-8); bản PDF và
  slide nằm cùng thư mục làm việc của đề án.
- Mục tiêu tham khảo 95% word accuracy chưa đạt được (tốt nhất 92,5%).
  Các công trình công bố đạt khoảng 97–98% với corpus và mô hình lớn hơn.
