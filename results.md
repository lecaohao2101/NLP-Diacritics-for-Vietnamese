# Kết quả đo — Khôi phục dấu tiếng Việt

Nhật ký số đo theo từng chặng, chạy trên Kaggle notebook
"Vietnamese Diacritic Restoration - Week 1 Starter"
(https://www.kaggle.com/code/lcaoho/vietnamese-diacritic-restoration-week-1-starter).
Chỉ ghi số đo thật từ output của notebook, không ghi số ước lượng.

## Chặng 1 — Corpus 20k câu, baseline từ điển (16/11/2026, Version 2)

- Cell 4: đọc 208 bài Wikipedia, giữ lại 20.000 câu (5–40 từ, khử trùng).
  Config `wikimedia/wikipedia` / `20231101.vi` chạy được, không lỗi config.
  Chỉ có warning HF Hub unauthenticated (không phải lỗi).
- Cell 5: train 16.000 | val 2.000 | test 2.000.
  Tổng số từ: 479.315 | từ có dấu: 387.687 (80,9%).
  Từ không dấu có ≥2 dạng có dấu: 2.587 / 22.760.
  Đa nghĩa hàng đầu: 'va' (và 8.450, vả 13...), 'la' (là 6.948...),
  'cua' (của 6.824, cửa 114...), 'co' (có 4.752, cò 630, cỏ 207...).
- Cell 6: từ điển học được 19.623 từ không dấu.
  - Word accuracy: 38.199/48.491 = 78,8%
  - Sentence accuracy: 57/2.000 = 2,9%
  - Accuracy trên từ đa nghĩa: 29.552/39.449 = 74,9%
- Lưu ý: số 20k chỉ là mốc chạy thử, KHÔNG phải số báo cáo.
  Số báo cáo lấy từ corpus 100k (chặng 2) trở đi, cùng một tập test
  cho cả 3 hệ thống.

## Chặng 2 — Corpus 100k câu, baseline từ điển (16/11/2026, Version 3)

- Cell 4 (đổi N_ARTICLES=10000, TARGET_SENTENCES=100000): đọc 1.501 bài
  Wikipedia, giữ lại 100.000 câu.
- Cell 5: train 80.000 | val 10.000 | test 10.000.
  Tổng số từ: 2.387.449 | từ có dấu: 1.933.227 (81,0%).
  Từ không dấu có ≥2 dạng có dấu: 6.078 / 78.862.
- Cell 6: từ điển học được 68.067 từ không dấu.
  - Word accuracy: 185.248/239.189 = 77,4%
  - Sentence accuracy: 261/10.000 = 2,6%
  - Accuracy trên từ đa nghĩa: 156.257/209.279 = 74,7%
- Cell 7: đã lưu train.csv, val.csv, test.csv vào /kaggle/working
  (Output 24,7 MiB). Đây là tập dữ liệu CHUNG cho cả 3 hệ thống —
  mọi số báo cáo sau này đều đo trên test 10.000 câu này.

## Chặng 3 — Baseline bigram + Viterbi (16/11/2026, Version 4)

- Cell 7 đã thay bằng bản có comment tiếng Việt (hành vi giữ nguyên).
- Cell 8: bigram với làm mịn add-alpha 0,1, giải mã Viterbi trên toàn câu.
  Trên cùng tập test 10.000 câu:
  - Word accuracy: 219.952/239.189 = 92,0%
  - Sentence accuracy: 3.107/10.000 = 31,1%
  - Accuracy trên từ đa nghĩa: 190.961/209.279 = 91,2%
- So với baseline từ điển (77,4% / 2,6% / 74,7%): ngữ cảnh (từ đứng
  trước) cải thiện cả 3 metric, mạnh nhất ở sentence accuracy (2,6% ->
  31,1%) vì lỗi từ đa nghĩa giảm mạnh.

## Chặng 4a — PhoBERT, công thức candidate-index (16/11/2026, Version 1 — THỬ NGHIỆM, ĐÃ THAY THẾ)

- Notebook mới "Vietnamese Diacritic Restoration - PhoBERT".
  transformers 5.20.2 (bản này bỏ warmup_ratio — Cell 4 lọc tham số
  theo inspect.signature); tokenizer PhoBERT là slow tokenizer nên
  Cell 3/5 dùng tokenize từng từ + kiểm chứng 200/200 câu khớp.
- Công thức: nhãn = vị trí của dạng đúng trong danh sách ứng viên của
  từ (MAX_CANDS = 21). Train 2 epoch, 2.500 bước, loss 1,71 -> 0,96.
- Kết quả trên test 10.000 câu:
  - Word accuracy: 203.889/239.189 = 85,2%
  - Sentence accuracy: 937/10.000 = 9,4%
  - Accuracy trên từ đa nghĩa: 173.632/207.644 = 83,6%
- VẤN ĐỀ: thấp hơn hẳn baseline bigram (92,0% / 31,1% / 91,2%).
  Nguyên nhân: nhãn candidate-index mang nghĩa khác nhau cho mỗi từ
  (nhãn 1 = "vả" với "va" nhưng = "lả" với "la") nên model phải ghi nhớ
  riêng từng từ, không chuyển giao được quy luật dấu giữa các từ.
  QUYẾT ĐỊNH: đổi sang công thức mẫu dấu (pattern) — nhãn = mẫu dấu
  của từ (vd "được" -> "đượ", "có" -> "ó"), cùng một mẫu có nghĩa giống
  nhau cho mọi từ, model học được quy luật dấu từ ngữ cảnh.
  Số liệu 4a giữ lại làm bài học thử nghiệm, KHÔNG dùng làm số báo cáo.
- Số liệu phụ: từ có dạng đúng chưa thấy trong train: train 0, val
  6.250, test 6.432; từ trong test chưa thấy trong train: 5.754.

## Ghi chú tiền xử lý — chuẩn hóa NFC (16/11/2026)

- Assert kiểm chứng mẫu dấu (Cell 2, notebook PhoBERT) lỗi ở cặp
  ('Res', 'Rês'): từ "Rês" trong Wikipedia được lưu ở dạng NFD
  ("e" + dấu mũ tách riêng) nên dài 4 ký tự trong khi từ không dấu
  "Res" dài 3 ký tự — mã hóa mẫu dấu theo vị trí bị lệch.
- Cách sửa: chuẩn hóa câu về NFC khi tạo cặp song song
  (unicodedata.normalize("NFC", s)) ở cả Notebook 1 (Cell 5) và
  notebook PhoBERT (Cell 1). NFC gộp mỗi ký tự có dấu thành 1 ký tự
  duy nhất. Tập câu và cách chia train/val/test không đổi (chỉ chuẩn
  hóa dạng lưu trữ của ký tự), nên các số baseline đo lại chỉ có thể
  lệch rất nhỏ (vài từ hiếm bị tách dấu gộp lại thành 1 dạng duy nhất).

## Cập nhật chặng 2–3 — số liệu sau chuẩn hóa NFC (16/11/2026, Version 5)

- Notebook 1 sau khi thêm chuẩn hóa NFC ở Cell 5 (chỉ chuẩn hóa dạng
  lưu trữ của ký tự, tập câu/cách chia không đổi):
  - Dictionary baseline: KHÔNG ĐỔI (77,4% / 2,6% / 74,7%).
  - Bigram baseline: word 219.912/239.189 = 91,9% (trước: 92,0%),
    sentence 3.099/10.000 = 31,0% (trước: 31,1%), từ đa nghĩa
    190.921/209.279 = 91,2% (không đổi). Số MỚI này là số chính thức.
- Đính chính ghi chú NFC ở trên: NFC một mình KHÔNG sửa được trường
  hợp ('Res', 'Rês') — dấu mũ ở đó là ký tự tách rời mà NFC không gộp
  được (không phải U+0302 thông thường). Cách sửa đúng: mã hóa mẫu dấu
  bằng đơn vị NFD (tách từ thành ký tự gốc + dấu đi kèm, gộp mỗi đơn
  vị về NFC), mẫu dấu lưu dạng tuple; assert kiểm chứng vòng tròn trên
  toàn bộ train + đếm từ dị thường.

## Chẩn đoán lỗi ('Res', 'Rês') — kết luận (16/11/2026)

- 3 lần sửa đều lỗi ở cùng cặp từ. Phân tích: từ "Rês" lưu dạng NFD
  ("e" + U+0302); bản mã hóa đơn vị NFD đã tạo đúng mẫu ("ê",) nhưng
  assert so sánh chuỗi trực tiếp: kết quả khôi phục ở dạng NFC ("Rês"
  gộp) != gold ở dạng NFD ("Re" + dấu tách rời) dù 2 dạng tương đương
  về mặt chính tả. Kết luận: bước chuẩn hóa NFC ở Cell 1 đã KHÔNG có
  hiệu lực trong notebook PhoBERT (nếu có thì cả 2 bản mã hóa đều đã
  chạy qua), và assert cần so sánh ở dạng NFC thay vì so sánh chuỗi thô.
- Cách sửa: (1) viết lại dòng pairs ở Cell 1 notebook PhoBERT thành
  bản NFC và KIỂM CHỨNG bằng cách đọc lại nội dung cell sau khi sửa;
  (2) assert kiểm chứng so sánh NFC 2 phía + in chi tiết mã code-point
  nếu lỗi; (3) giữ mã hóa mẫu dấu theo đơn vị NFD dạng tuple.

## Chẩn đoán lỗi ('Res', 'Rês') — nguyên nhân THẬT (16/11/2026)

- Chẩn đoán bằng mã code-point cho thấy từ lỗi thật ra là 'Rêš'
  (R + ê U+00EA + š U+0161), không phải 'Rês'; NFC đã có hiệu lực
  (0 từ chưa ở dạng NFC trong train). Nguyên nhân thật: PHỤ ÂM CÓ DẤU
  NGOÀI TIẾNG VIỆT trong tên riêng nước ngoài/phiên âm tiếng Phạn
  (š, ç, Ś, ś, ṣ trong "Rêš", "François", "Śulba", "Śvetāśvatara-
  Upaniṣad", "Spiš") — strip_diacritics bỏ dấu của chúng nhưng
  MODIFIABLE chỉ gồm nguyên âm + đ nên mẫu dấu không khôi phục được.
  524 từ như vậy trong train (trên ~1,9 triệu từ), + 21 từ dị thường
  (số đơn vị NFD khác số ký tự gốc).
- Cách sửa (quyết định của assistant): HỌC MODIFIABLE TỪ DỮ LIỆU —
  ký tự gốc được coi là có thể mang dấu nếu từng thấy nó đi kèm một
  đơn vị khác chính nó trong train; bảng này đồng thời làm bảng
  valid_forms cho Cell 5. Cách này tự động bao quát cả dấu tiếng Việt
  lẫn dấu phụ âm nước ngoài mà không cần liệt kê tay, và giữ mẫu dấu
  gọn (chỉ gồm vị trí thật sự có thể đổi dấu).

## Chặng 4b — PhoBERT, mẫu dấu với MODIFIABLE học từ dữ liệu (16/11/2026, Version 2 — THỬ NGHIỆM, ĐÃ THAY THẾ)

- Assert vòng tròn ĐÃ QUA (0 từ lỗi, 21 từ dị thường), nhưng học
  MODIFIABLE từ dữ liệu kéo theo 85 ký tự "có thể mang dấu" — gồm cả
  chữ Thái, Khmer, Devanagari, Hy Lạp, Cyrillic, kana Nhật trong
  Wikipedia — nên mẫu dấu gần như trùng với cả từ: NUM_LABELS =
  42.525 (10 mẫu phổ biến nhất là ('của',), ('là',), ('có',)... —
  nguyên cả từ). Đầu phân loại 42.525 lớp, loss khởi đầu 18,75.
- Train 3 epoch, 3.750 bước, loss 18,75 -> 4,34. Kết quả test 10k:
  - Word accuracy: 186.057/239.189 = 77,8%
  - Sentence accuracy: 287/10.000 = 2,9%
  - Accuracy trên từ đa nghĩa: 156.420/207.644 = 75,3%
  - Accuracy trên từ chưa thấy trong train: 4.809/5.754 = 83,6%
    (điểm sáng duy nhất — model đoán được dấu cho từ mới).
  Tệ hơn cả bản candidate-index (85,2%) và bigram (91,9%). Số 4b chỉ
  giữ làm bài học thử nghiệm, KHÔNG dùng làm số báo cáo.
- QUYẾT ĐỊNH: MODIFIABLE để CỐ ĐỊNH theo tiếng Việt (nguyên âm
  aeiouy + đ); từ nước ngoài không mã hóa được bằng mẫu dấu tiếng Việt
  thì bỏ qua khi train (nhãn -100), tỉ lệ phải < 1% (assert kiểm tra);
  thêm guard ở Cell 4: dừng train nếu NUM_LABELS >= 5000.
- Lưu ý cho chặng so sánh: amb_set ở notebook PhoBERT tính trên train
  (5.461 từ) còn Notebook 1 tính trên toàn bộ pairs (6.078 từ) nên
  mẫu số accuracy từ đa nghĩa khác nhau (207.644 vs 209.279) — chặng
  so sánh sẽ tính lại cả 3 metric từ file dự đoán theo MỘT định nghĩa
  chung.
- Version 2 đã lưu nhưng bản commit chạy lại trên Kaggle còn Queued
  (hàng đợi GPU phía Kaggle, không phải lỗi notebook); mọi số trên lấy
  từ lần Run All tương tác đã chạy xong.

## Chặng 4c — PhoBERT, mẫu dấu tiếng Việt CỐ ĐỊNH (16/11/2026, đang chạy lại)

- MODIFIABLE cố định theo tiếng Việt (nguyên âm + đ) đã chạy thử:
  từ không mã hóa được 545/1.909.902 = 0,029% (assert < 1% đã qua),
  0 từ chưa NFC. Nhưng NUM_LABELS KHÔNG về "vài trăm" như dự đoán
  mà là 9.166 — mẫu dấu là tuple các đơn vị nguyên âm nên từ có nhiều
  nguyên âm tạo ra hàng nghìn tổ hợp; chốt chặn 5000 đã chặn train.
- QUYẾT ĐỊNH: chỉ giữ mẫu dấu xuất hiện >= 5 lần trong train (đuôi
  dài các mẫu 1-2 lần chiếm ít token, model không đủ ví dụ để học;
  từ có mẫu hiếm mang nhãn -100 khi train), nâng chốt chặn lên 10000
  (chỉ bắt trường hợp bùng nổ kiểu 42.525). Đang chạy lại toàn bộ.

## Chặng 4c — PhoBERT, mẫu dấu + giới hạn tần suất (16/11/2026, Version 3 — SỐ CHÍNH THỨC CỦA PHOBERT)

- Chỉ giữ mẫu xuất hiện >= 5 lần trong train: NUM_LABELS 9.166 ->
  2.326 (6.840 mẫu bị loại, ảnh hưởng 10.255 từ); từ không mã hóa
  được 545 (0,029%); từ đa nghĩa 5.461.
- Train 3 epoch, 3.750 bước, loss 12,31 -> 1,95. Kết quả test 10k:
  - Word accuracy: 199.500/239.189 = 83,4%
  - Sentence accuracy: 671/10.000 = 6,7%
  - Accuracy trên từ đa nghĩa: 169.922/207.644 = 81,8%
  - Accuracy trên từ chưa thấy trong train: 4.916/5.754 = 85,4%
- So sánh các lần chạy PhoBERT: candidate-index 85,2% > mẫu dấu
  83,4% > mẫu dấu với MODIFIABLE học từ data 77,8%. Nguyên nhân có
  thể: đầu phân loại candidate-index chỉ có 21 lớp nên học đủ trong
  3 epoch, còn đầu mẫu dấu có 2.326 lớp (đuôi dài ít ví dụ) nên chưa
  học đủ. QUYẾT ĐỊNH: dừng lặp PhoBERT ở đây; lấy 83,4% (Version 3,
  công thức mẫu dấu) làm số chính thức của hệ PhoBERT, ghi 85,2%
  (candidate-index) làm thử nghiệm so sánh trong báo cáo; không train
  thêm vì khoảng cách tới bigram (91,9%) quá lớn để 1-2 epoch bù lại.

## TẠM DỪNG (18:08 ngày 16/11/2026)

- User tắt máy. Đã dừng browser task. Chặng 5 (tự build Transformer)
  mới giao cho browser lúc ~18:00, chưa rõ đã tạo được notebook chưa
  hay mới tới bước nào — khi user mở máy lại, KIỂM TRA trạng thái
  notebook "Vietnamese Diacritic Restoration - Self-built Transformer"
  trên Kaggle trước rồi mới chạy tiếp (Cells 1-2 copy từ notebook
  PhoBERT + verify 1501 bài / 100000 câu / 2387449 từ và NUM_LABELS
  2326; Cells 3-5 đã soạn sẵn trong lệnh giao lúc 18:00).
- Đã xong và đã lưu an toàn trên Kaggle: Notebook 1 (Version 5),
  PhoBERT (Version 3, word 83,4%).

## Chặng 6 — bản nháp đã soạn offline (16/11/2026, CHƯA chạy)

- File `stage6_comparison_kaggle.md`: 4 cell, comment tiếng Việt, cú pháp
  đã kiểm chứng local + Viterbi chạy thử đúng trên dữ liệu đồ chơi.
  Cell 1 tải lại corpus 100k (số kiểm chứng 100.000 câu / 2.387.449 từ /
  5.461 từ đa nghĩa); Cell 2 tính lại dự đoán từ điển + bigram và assert
  khớp số đã báo (từ điển 185.248/239.189, 261/10.000; bigram
  219.912/239.189, 3.099/10.000 sau NFC); Cell 3 nạp thêm
  `phobert_predictions.csv` (đã lưu ở Version 3) +
  `selfbuilt_test_predictions.csv` (chặng 5) và tính lại cả 3 metric theo
  MỘT định nghĩa chung, assert PhoBERT tái hiện đúng Version 3
  (199.500/239.189, 671/10.000, 169.922/207.644); Cell 4 phân tích lỗi
  4 loại + 25 câu sai mẫu mỗi hệ + ví dụ minh họa (STEP_BY_STEP 4.3–4.4).
  Lưu ý: accuracy từ đa nghĩa tính lại của từ điển/bigram sẽ lệch nhẹ so
  với số đã báo (74,7%/91,2%) do đổi mẫu số 209.279 -> 207.644 — đó là
  đổi định nghĩa, không phải lỗi. Chưa chạy vì chặng 5 chưa xong và máy
  user đang tắt; chạy sau khi có file dự đoán của chặng 5. Không cần GPU.

## TIẾP TỤC (20:40 ngày 16/11/2026)

- User mở máy lại, giao tiếp tục. File nháp chặng 5
  (stage5_selfbuilt_kaggle.md) đã kiểm chứng cú pháp 5 cell + logic
  vòng tròn local: "hoa"->"hoà" khớp, "Rêš"/"François"/"Śulba" ->
  None (nhãn -100), đúng chẩn đoán chặng 4.
- QUYẾT ĐỊNH khi chạy thật: Cells 1-2 lấy BẢN COPY từ notebook
  PhoBERT (đã kiểm chứng, bảo đảm dữ liệu giống hệt) thay vì Cell 2
  trong file nháp (cách lọc câu khác nên chưa chắc ra đúng 2.387.449
  từ); Cells 3-5 dùng thiết kế của file nháp (12 epoch, giữ
  checkpoint tốt nhất theo val word-accuracy, vocab >= 2 lần) nhưng
  đổi tên hàm/biến theo PhoBERT (to_signature/sig_to_id/signatures/
  apply_signature) và MAX_LEN = 64 (dữ liệu PhoBERT cho câu tới 60
  từ, bản nháp lọc tối đa 40 từ nên MAX_LEN 40 sẽ lỗi). Bản thích
  nghi này đã kiểm chứng cú pháp local (/tmp/stage5_adapted_cells.py).
- Browser task mới đã giao: kiểm tra trạng thái notebook self-built
  trước (chưa rõ lần giao 18:00 làm tới đâu), rồi chạy toàn bộ.

## Chặng 5 — Transformer tự build (16/11/2026, Version 1 — HỆ TỐT NHẤT)

- Notebook: https://www.kaggle.com/code/lcaoho/vn-diacritic-restoration-self-built-transformer
  (slug thật là vn-diacritic-restoration-..., KHÔNG phải
  vietnamese-diacritic-restoration-... như dự đoán; URL dự đoán báo
  lỗi quyền, không tạo notebook thừa). Cells 1-2 là bản copy PhoBERT
  (đã có sẵn từ phiên 18:05-18:07, đối chiếu khớp); số kiểm chứng
  khớp: 1501 bài / 100.000 câu / 2.387.449 từ, 545 từ không mã hóa
  được, NUM_LABELS 2.326, 5.461 từ đa nghĩa.
- Cell 3: vocab 22.413 từ, model 9.510.934 tham số (encoder 4 lớp,
  hidden 256, 8 heads, dropout 0,1).
- Cell 4: train 12 epoch, 478 giây; loss 1,09 -> 0,26; val word-acc
  83,5% -> 91,1% (tốt nhất epoch 12; epoch 11 cũng 91,1% sau làm
  tròn). Checkpoint tốt nhất theo val.
- Cell 5, test 10k:
  - Word accuracy: 221.271/239.189 = 92,5%
  - Sentence accuracy: 3.075/10.000 = 30,8%
  - Accuracy trên từ đa nghĩa: 191.774/207.644 = 92,4%
- KẾT QUẢ BẤT NGỜ: tự build VƯỢT bigram (91,9%/31,0%/91,2%) ở word
  và từ đa nghĩa, gần bằng ở sentence; vượt xa PhoBERT
  (83,4%/6,7%/81,8%). Giả thuyết: train 12 epoch (PhoBERT chỉ 3,
  loss còn 1,95); mức từ (không mất thông tin subword như PhoBERT
  chỉ dự đoán ở subword đầu); đầu phân loại học cùng encoder từ đầu
  với lr 1e-3 (PhoBERT fine-tune lr 2e-5, đầu 2.326 lớp chưa học đủ).
- File: selfbuilt_best.pt + selfbuilt_test_predictions.csv đã lưu.
  Version 1 đã lưu, commit đang Queued.

## Chặng 6 — So sánh + phân tích lỗi (16/11/2026, Version 2)

- Từ điển tái tạo: 77,4%/2,6%/74,6% (word/sentence khớp Notebook 1;
  từ đa nghĩa 74,6% vs 74,7% do mẫu số train 207.644 vs 209.279).
- Bigram tái tạo: 91,9%/30,8%/91,3% (word khớp Notebook 1; sentence
  30,8% vs 31,0% — thấp hơn 0,2pp, có thể do tập ứng viên chỉ từ
  train thay vì toàn bộ pairs; từ đa nghĩa 91,3% vs 91,2%).
- BẢNG SO SÁNH THỐNG NHẤT (cùng định nghĩa, dùng cho báo cáo):
  Từ điển 77,4/2,6/74,6; Bigram 91,9/30,8/91,3;
  PhoBERT 83,4/6,7/81,8; Tự build 92,5/30,8/92,4.
- Phân tích lỗi tự build: 17.918 từ sai; 55,1% chỉ sai thanh điệu;
  6,3% từ chưa thấy trong train; 88,6% từ đa nghĩa. Top nhầm lẫn:
  nhưng->những 73, từ->tự 45, hoá->hóa 45, khí->khi 42, còn->con 41,
  trọng->trong 39, tự->từ 38, cổ->có 38, nền->nên 35, loài->loại 32
  — cả 10 cặp đều chỉ khác thanh điệu.
- File: bigram_test_predictions.csv + comparison_chart.png đã lưu.
  Version 2 đã lưu, commit đang Queued.

## Chặng 7 — Demo Gradio (16/11/2026, Version 3)

- Cell 9: Gradio demo dùng checkpoint tốt nhất (selfbuilt_best.pt),
  share=True, prevent_thread_lock=True. Link public:
  https://5c5532dd203ae07dc6.gradio.live (link tạm, tối đa 1 tuần
  theo Gradio, sống khi phiên Kaggle còn chạy; phiên tương tác vẫn
  đang chạy, KHÔNG dừng).
- Kiểm thử qua link public, 3 câu mẫu (20/27 từ đúng, ~74% — thấp
  hơn 92,5% trên test vì các câu ví dụ dồn nhiều từ đa nghĩa):
  (a) "xin chao cac ban, toi ten la hao" -> "xin chào các bản, tôi
  tên là hào" (7/8; lỗi "ban"->"bản", đúng là "bạn");
  (b) "ha noi la thu do cua viet nam" -> "hà nội là thủ đô của viết
  năm" (6/8; lỗi "Việt Nam"->"viết năm");
  (c) "toi dang hoc tieng viet va rat thich mon nay" -> "tới dạng
  học tiếng viết và rất thích môn này" (7/11; lỗi "Tôi"->"tới",
  "đang"->"dạng", "Việt"->"viết").
  Nhận xét: model hay chọn nghĩa phổ biến hơn; các lỗi đều là nhầm
  từ đa nghĩa/thanh điệu, đúng dạng lỗi chủ đạo ở chặng 6. Model
  không đổi chữ hoa/thường (chỉ khôi phục dấu) nên "hà nội" giữ
  nguyên chữ thường như input.
- Lưu ý kỹ thuật: lần lưu Version 3 bị chặn do "GPU Session cap
  reached" (2 suất batch bị 2 commit Queued của V1+V2 chiếm) — đã
  dừng commit Queued của V1 (chuyển sang Cancelled, V1 vẫn còn trong
  lịch sử phiên bản) rồi lưu lại thành công. V2 và V3 vẫn Queued.
- Ảnh chụp trang demo (câu test c) lưu ở:
  workspace/.hatch-browser/outputs/browser-task:5f233f7f-6c48-5e27-a54c-7d770d8e85e0/browser-page-20261116T142919.539005208Z-0.pdf

## Chặng 8 — Báo cáo + slide (16/11/2026 — ĐỀ ÁN HOÀN THÀNH)

- Báo cáo tiếng Việt 9 trang:
  nlp-diacritic-restoration/bao-cao-khoi-phuc-dau-tieng-viet.md
  (+ build_pdf.py tái tạo PDF từ .md, font Noto Sans) và PDF cùng tên
  (đã copy sang your_files/khoi-phuc-dau-tieng-viet/). Mọi số liệu đã
  kiểm chứng lại bằng phép tính độc lập trước khi viết.
- Slide: thêm trang 11 (kết quả thực nghiệm) + trang 12 (phân tích lỗi
  & demo), render bằng render_slides.py cùng style trang 10.
  File đủ 12 trang:
  your_files/khoi-phuc-dau-tieng-viet/khoi-phuc-dau-tieng-viet-cover-12slides.pptx
  (bản 10 trang giữ nguyên). Ghi chú thuyết trình ở trang 10 đã cập
  nhật (slide-10-tu-build-model.md) vì kỳ vọng ban đầu sai.
- Số liệu chính thức (bảng thống nhất, test 10k câu): Từ điển
  77,4/2,6/74,6; Bigram 91,9/30,8/91,3; PhoBERT 83,4/6,7/81,8;
  Tự build 92,5/30,8/92,4. Mục tiêu 95% chưa đạt; SOTA ~97–98%.

## Review code + sửa lỗi (17/11/2026)

Review toàn bộ code trong repo (3 notebook + 2 script) và sửa các điểm
sau. **Mọi số liệu trong báo cáo đo ngày 16/11, trước các sửa đổi này**
— chạy lại code đã sửa có thể cho số lệch nhẹ (xem ghi chú từng mục).

Notebook 1 (dữ liệu + baseline):
- Metric từ đa nghĩa chuyển sang định nghĩa train-only (amb_set tính
  trên train, giống Notebook 2/3) thay vì tính trên toàn bộ pairs
  (kể cả test). Số in ra ở Cell 6/8 giờ khớp bảng thống nhất
  (74,6% / 91,3% thay cho 74,7% / 91,2% cũ).
- Cell 3 thêm cảnh báo: số 45,5% chỉ là demo trên 12 câu mẫu.
- Cell 7 sửa comment: CSV train/val/test không đọc được từ notebook
  khác (mỗi notebook có /kaggle/working riêng); Notebook 2/3 tái tạo
  tập dữ liệu bằng cùng code + seed 42.

Notebook 2 (PhoBERT):
- Cell 1 thêm assert tổng số từ = 2.387.449 (dừng ngay nếu dữ liệu
  lệch khỏi Notebook 1, thay vì chỉ in ra).
- Cell 4 đánh giá val mỗi epoch và giữ checkpoint tốt nhất theo eval
  loss (eval_strategy="epoch", load_best_model_at_end=True); trước
  đây val không được dùng và model epoch cuối được lấy mù quáng.
  warmup_ratio đổi thành warmup_steps=750 (transformers 5.20.2 không
  có warmup_ratio). Số 83,4% trong báo cáo đo bằng model epoch cuối
  (trước sửa đổi) — chạy lại có thể lệch nhẹ.

Notebook 3 (tự build):
- Cell 1 sửa tiêu đề comment (trước đây ghi nhầm "Notebook 2 —
  Fine-tune PhoBERT" do copy) + thêm assert tổng số từ.
- Cell 2 thêm hàm skeleton()/compatible_ids() để tra cứu mẫu dấu
  tương thích với một từ.
- Cell 3 thêm torch.manual_seed(42) (+ cuda): trước đây train không
  tái tạo được (khởi tạo trọng số, dropout, thứ tự batch đều ngẫu
  nhiên). Sửa comment MAX_LEN (câu dài nhất 40 từ, không phải 60).
- Cell 5 + Cell 9 dùng masking mẫu tương thích khi dự đoán (như
  PhoBERT), thay cho argmax trần + fallback về từ không dấu. Masking
  chỉ có thể làm số tăng lên, không giảm — số 92,5% trong báo cáo đo
  bằng argmax trần.
- Cell 5 bỏ đoạn tính lại amb_set (dùng chung Cell 2); Cell 6 bỏ số
  hard-code 239189 (đếm thật); Cell 8 tính số PhoBERT từ file dự
  đoán nếu có, không thì dùng số hard-code kèm cảnh báo rõ ràng;
  Cell 7 ghi chú về token </s> và cách phá tie (nguyên nhân chênh
  0,2pp so với Notebook 1).

Script:
- build_pdf.py + render_slides.py chạy được từ repo: tìm file .md ở
  ./ hoặc reports/, tự tạo comparison_chart.png bằng matplotlib khi
  file PNG gốc không có trong repo. Sửa docstring (Noto Sans, không
  phải DejaVu), thêm cảnh báo khi thiếu ảnh.

Chưa sửa (cần chạy lại trên Kaggle, ngoài phạm vi đợt này):
- Notebook 3 trên GitHub vẫn không có output trong file .ipynb (0/9
  cell) — cần chạy lại notebook rồi tải lại file.
- Val accuracy 91,1% thấp hơn test 92,5% (1,4pp) — chưa rõ nguyên
  nhân đầy đủ; có thể do val loại từ -100 còn test tính cả, và
  fallback về từ không dấu ở eval cũ có thể đúng tình cờ với từ
  không dấu. Đáng kiểm tra khi chạy lại.
