# Step by step — Đề án Khôi phục dấu tiếng Việt

Toàn bộ quy trình từ lúc chưa có gì đến lúc nộp bài. Mỗi bước ghi rõ **làm gì, dùng file/code nào, xong thì kiểm tra gì**.

> Giả định: cá nhân, 4 tuần (PLAN.md). Nếu 5 tuần thì giãn Tuần 3 thành 2 tuần.
> Hướng so sánh mặc định: Baseline vs PhoBERT. Phần tự build Transformer là **nhánh tùy chọn** (Giai đoạn 3B) — chỉ làm nếu bạn chốt làm 3 hệ.

---

## Giai đoạn 0 — Chuẩn bị (nửa ngày)

**Bước 0.1.** Tạo notebook mới trên Kaggle, bật GPU (Settings → Accelerator → GPU T4 x2).
**Bước 0.2.** Bật Internet ON trong Settings (cần để tải Wikipedia + PhoBERT).
**Bước 0.3.** Chạy thử 1 cell `import torch; print(torch.cuda.is_available())` — phải in `True`.

✅ Kiểm tra: GPU nhận, internet bật.

---

## Giai đoạn 1 — Dữ liệu (Tuần 1)

Code đã có sẵn: `week1_starter_kaggle.md` (Cell 1–3) và `week1_real_corpus_kaggle.md` (Cell 4–6).

**Bước 1.1.** Viết hàm `strip_diacritics` (Cell 1): bỏ dấu bằng `unicodedata` NFD, xử lý riêng đ/Đ.
- Kiểm tra: 4/4 test strip đúng.

**Bước 1.2.** Chạy thử trên corpus demo 12 câu (Cell 2–3): tạo cặp song song, chia train/test, baseline từ điển.
- Kiểm tra: word accuracy demo 10/22 = 45,5%. Thấp là đúng — corpus demo quá nhỏ. **Không đưa số này vào báo cáo.**

**Bước 1.3.** Tải corpus thật (Cell 4): stream Wikipedia tiếng Việt, tối đa 2.000 bài → giữ 20.000 câu (5–40 từ, khử trùng, lọc URL/số nhiều).
- Kiểm tra: in ra 5 câu mẫu, đọc bằng mắt xem có sạch không.

**Bước 1.4.** Tạo cặp song song + chia 80/10/10, seed 42 (Cell 5). In thống kê: tổng số từ, tỉ lệ từ có dấu, danh sách từ đa nghĩa phổ biến.
- Kiểm tra: train/val/test không trùng câu; tỉ lệ từ có dấu ~50–70% là bình thường.

**Bước 1.5.** (Sau khi 20k câu đã ổn) Tăng quy mô lên 100k câu: `TARGET_SENTENCES = 100000`, `N_ARTICLES ≈ 10000`, chạy lại Cell 4–5.
- Kiểm tra: số câu thực tế đạt ~100k; lưu train/val/test ra file CSV để dùng lại ở các giai đoạn sau.

✅ Xong Tuần 1 khi có: file train/val/test thật + bảng thống kê corpus.

---

## Giai đoạn 2 — Baseline (Tuần 2)

**Bước 2.1.** Xây từ điển tần suất trên tập train: đếm mỗi từ không dấu ứng với dạng có dấu nào bao nhiêu lần, giữ dạng hay gặp nhất.
- Kiểm tra: in 10 từ đa nghĩa phổ biến nhất kèm số đếm (vd "co" → có/cô/cỏ...).

**Bước 2.2.** Thêm n-gram: đếm bigram/trigram của dạng có dấu trong train. Khi dự đoán, chọn dạng có dấu làm cho câu có xác suất cao nhất.
- Kiểm tra: chạy thử câu "co ay rat dep" — từ điển thuần cho "có ấy", n-gram phải sửa thành "cô ấy".

**Bước 2.3.** Dự đoán toàn bộ tập test bằng baseline (từ chưa gặp ở train → giữ nguyên không dấu).

**Bước 2.4.** Đo 3 metric trên test: word accuracy, sentence accuracy, accuracy trên từ đa nghĩa.
- Kiểm tra: ghi 3 số này vào bảng kết quả — đây là **mốc so sánh** cho mọi model sau.

**Bước 2.5.** Phân tích lỗi baseline: lấy 10–20 câu sai, phân loại lỗi (từ đa nghĩa? từ hiếm? tên riêng?).
- Kiểm tra: viết được 3–4 dòng mô tả "baseline sai ở đâu" để đưa vào báo cáo.

✅ Xong Tuần 2 khi có: 3 số baseline trên test + phân tích lỗi.

---

## Giai đoạn 3 — Fine-tune PhoBERT (Tuần 3)

**Bước 3.1.** Chuẩn bị dữ liệu dạng token classification: mỗi câu không dấu là input, mỗi token có nhãn là dạng có dấu tương ứng.
- Lưu ý: PhoBERT dùng BPE subword — một từ có thể tách thành nhiều subword; chỉ tính loss trên subword đầu tiên của mỗi từ, các subword còn lại gán nhãn -100 (bỏ qua).
- Kiểm tra: in 2–3 câu đã gắn nhãn, đối chiếu input/label bằng mắt.

**Bước 3.2.** Load `vinai/phobert-base` + thêm classification head (số nhãn = số dạng có dấu khác nhau trong train).
- Kiểm tra: in số lượng nhãn, số tham số model.

**Bước 3.3.** Cấu hình train: 2–3 epochs, learning rate ~2e-5, batch 16–32, max length 128, lưu checkpoint sau mỗi epoch.
- Kiểm tra: chạy thử 50 step xem loss có giảm không trước khi train full.

**Bước 3.4.** Train full trên Kaggle GPU, theo dõi loss trên train và val.
- Kiểm tra: val accuracy sau mỗi epoch — nếu val đi xuống trong khi train vẫn lên là overfit, dừng sớm.

**Bước 3.5.** Đánh giá trên test: đo đúng 3 metric như baseline (word / sentence / từ đa nghĩa), trên **cùng tập test**.
- Kiểm tra: điền 3 số PhoBERT vào bảng kết quả cạnh 3 số baseline.

✅ Xong Tuần 3 khi có: model PhoBERT đã fine-tune + 3 số trên test.

---

## Giai đoạn 3B — Tự build model (TÙY CHỌN, chỉ nếu chốt làm 3 hệ)

**Bước 3B.1.** Thiết kế Transformer encoder nhỏ từ đầu: 4–6 lớp, hidden 256, 4–8 attention heads, dropout 0,1. Input là từ không dấu (embedding tự học), output là dạng có dấu cho mỗi vị trí.
- Điểm khởi đầu, chỉnh theo validation — không phải con số cố định.

**Bước 3B.2.** Train from scratch trên cùng tập train: optimizer Adam, lr 1e-3, 10–20 epochs (model nhỏ train nhanh hơn PhoBERT).
- Kiểm tra: loss giảm đều; val accuracy sau mỗi epoch.

**Bước 3B.3.** Đánh giá trên cùng tập test, đúng 3 metric, điền vào bảng kết quả.

✅ Xong 3B khi có: 3 số của model tự build trên test.

---

## Giai đoạn 4 — Đánh giá & phân tích lỗi (Tuần 4, đầu tuần)

**Bước 4.1.** Hoàn thiện bảng kết quả: mỗi hệ thống (baseline / tự build / PhoBERT) một hàng, 3 cột metric.

**Bước 4.2.** So sánh từng cặp: PhoBERT hơn baseline bao nhiêu điểm? Hơn ở metric nào nhiều nhất? (Thường là accuracy từ đa nghĩa.)

**Bước 4.3.** Phân tích lỗi của PhoBERT: lấy 20–30 câu PhoBERT còn sai, phân loại (từ hiếm, tên riêng, câu dài, ngữ cảnh mơ hồ thật sự).
- Kiểm tra: viết được đoạn "model còn sai ở đâu và vì sao" cho báo cáo.

**Bước 4.4.** Lấy 5–10 ví dụ minh họa tốt: câu baseline sai nhưng PhoBERT đúng (khoe điểm mạnh), và 2–3 câu cả hai cùng sai (trung thực về giới hạn).

✅ Xong khi có: bảng kết quả đầy đủ + phân tích lỗi + ví dụ minh họa.

---

## Giai đoạn 5 — Demo (Tuần 4, giữa tuần)

**Bước 5.1.** Viết demo Gradio/Streamlit: ô nhập câu không dấu → hiện câu đã khôi phục dấu (tô màu các từ đã sửa).
**Bước 5.2.** Chạy thử demo với 10 câu tự nghĩ (câu đời thường, câu có từ đa nghĩa, câu dài).
**Bước 5.3.** Chụp màn hình demo để đưa vào báo cáo + slide.

✅ Xong khi có: demo chạy được + ảnh chụp màn hình.

---

## Giai đoạn 6 — Báo cáo & nộp bài (Tuần 4, cuối tuần)

**Bước 6.1.** Viết báo cáo theo cấu trúc: Giới thiệu → Dữ liệu → Phương pháp (baseline + PhoBERT + tự build nếu có) → Kết quả (bảng 3 metric) → Phân tích lỗi → Kết luận.
- Quy tắc: mọi con số trong báo cáo phải lấy từ bảng kết quả Giai đoạn 4, không dùng số demo 45,5%.

**Bước 6.2.** Cập nhật slide: điền số thật vào bảng ở slide 7, thêm 1–2 ví dụ minh họa từ Giai đoạn 4.

**Bước 6.3.** Chạy lại toàn bộ notebook từ đầu đến cuối một lần (Restart & Run All) để chắc chắn không còn lỗi.

**Bước 6.4.** Kiểm tra lần cuối trước khi nộp:
- [ ] Code chạy được từ đầu, không lỗi
- [ ] Số liệu trong báo cáo khớp với output của code
- [ ] Slide có số thật, không còn ô trống
- [ ] Demo chạy được

---

## Tóm tắt timeline

| Tuần | Việc chính | Sản phẩm bàn giao |
|---|---|---|
| 0 | Chuẩn bị Kaggle | Notebook chạy được, GPU OK |
| 1 | Dữ liệu | train/val/test + thống kê corpus |
| 2 | Baseline | 3 số baseline + phân tích lỗi |
| 3 | PhoBERT (+ tự build nếu chốt) | Model + 3 số trên test |
| 4 | Đánh giá, demo, báo cáo | Bảng kết quả, demo, báo cáo, slide |
