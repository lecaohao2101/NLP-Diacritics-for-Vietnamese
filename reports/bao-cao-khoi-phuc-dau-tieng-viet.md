# Khôi phục dấu tiếng Việt bằng phân loại token với Transformer

**Đề án cuối môn Xử lý ngôn ngữ tự nhiên (NLP)**
**Ngày hoàn thành:** 16/11/2026

---

## 1. Giới thiệu

### 1.1. Bài toán

Khôi phục dấu tiếng Việt là bài toán biến một câu tiếng Việt không dấu thành
câu có dấu đúng:

- Input: `toi dang hoc tieng viet`
- Output: `Tôi đang học tiếng Việt`

Bài toán có ý nghĩa thực tiễn: người dùng thường gõ không dấu vì nhanh, nhưng
văn bản không dấu gây khó đọc và dễ hiểu sai, đặc biệt trong các hệ thống
tìm kiếm, dịch máy và trợ lý ảo.

### 1.2. Vì sao khó

Một từ không dấu có thể tương ứng với nhiều dạng có dấu khác nhau, và chỉ
ngữ cảnh mới phân biệt được. Ví dụ `co` có thể là *có, cô, cọ, cỏ, cò, cơ*;
`ma` có thể là *ma, má, mà, mạ, mã*. Trong corpus của đề án này, 6.078 trên
78.862 từ không dấu (7,7%) có từ 2 dạng có dấu trở lên. Ngoài ra còn có các
trường hợp chỉ khác nhau ở thanh điệu (ví dụ *nhưng* và *những*), đòi hỏi
mô hình phải hiểu ngữ cảnh ở mức câu.

### 1.3. Mục tiêu và kết quả chính

Kế hoạch ban đầu đặt mục tiêu tham khảo là word accuracy ≥ 95% và kỳ vọng
PhoBERT fine-tune sẽ là hệ mạnh nhất. Kết quả thực tế đo được trên tập test
10.000 câu:

- Hệ tốt nhất là **Transformer tự build: word accuracy 92,5%** — chưa đạt
  mục tiêu 95% nhưng vượt cả bigram (91,9%) và PhoBERT (83,4%).
- Kết quả này **ngược với kỳ vọng ban đầu**: PhoBERT không những không mạnh
  nhất mà còn thua bigram tới 8,5 điểm phần trăm. Báo cáo phân tích nguyên
  nhân ở Mục 5.4.

Đề án xây dựng và so sánh **4 hệ thống trên cùng một tập test, cùng một
định nghĩa metric**: từ điển, bigram + Viterbi, PhoBERT fine-tune và
Transformer tự build, kèm theo một demo Gradio chạy được.

---

## 2. Dữ liệu

### 2.1. Nguồn và quy mô

Corpus lấy từ **Wikipedia tiếng Việt** (`wikimedia/wikipedia`,
`20231101.vi`, đọc dạng streaming), gồm 1.501 bài viết. Sau khi lọc
(câu 5–40 từ, bỏ câu chứa link, bỏ câu có quá 15% ký tự số, bỏ câu trùng),
giữ lại **100.000 câu**, chia theo tỉ lệ 80/10/10 với seed 42:

| Tập | Số câu |
|---|---:|
| Train | 80.000 |
| Validation | 10.000 |
| Test | 10.000 |

Tổng số từ: **2.387.449**, trong đó 1.933.227 từ (81,0%) mang dấu. Cặp song
song được tạo bằng cách bỏ dấu bản gốc làm input, giữ bản gốc làm nhãn.
Hàm bỏ dấu dùng chuẩn hóa NFD (tách dấu tổ hợp rồi bỏ), xử lý riêng `đ/Đ`
vì hai ký tự này không tách được bằng NFD. Cả hai phía input và nhãn đều
được chuẩn hóa về NFC, vì trong Wikipedia có những chỗ dấu được lưu ở dạng
tách rời (NFD) thay vì dạng tổ hợp.

### 2.2. Từ đa nghĩa

- Trên toàn bộ corpus: 6.078/78.862 từ không dấu có ≥ 2 dạng có dấu.
- Trên train (định nghĩa dùng cho metric ở Mục 3.6): 5.461 từ đa nghĩa,
  tương ứng 207.644 từ trên tập test.

### 2.3. Dữ liệu nhiễu

Wikipedia chứa tên riêng nước ngoài và phiên âm tiếng Phạn có **phụ âm mang
dấu ngoài tiếng Việt** (ví dụ *š, ç, Ś, ś, ṣ* trong `Rêš`, `François`,
`Śulba`). Những từ này chiếm 545/1.909.902 từ trong train (0,029%) và không
thể mã hóa bằng mẫu dấu tiếng Việt (Mục 3.1); chúng được bỏ qua khi train
(nhãn −100) thay vì ép vào một mẫu sai.

---

## 3. Phương pháp

### 3.1. Công thức mẫu dấu (diacritic pattern)

Cả hai mô hình neural đều dùng chung một cách mã hóa nhãn. Nhãn của một từ
là **mẫu dấu** của nó: tuple các đơn vị có dấu tại những vị trí mang dấu.
Ví dụ:

- `được` → `("đ", "ư", "ợ")`
- `có` → `("ó",)`
- `hoàn` → `("o", "à")`
- `con` → `()` (không có vị trí mang dấu)

Một từ được tách thành các đơn vị (ký tự gốc + dấu kết hợp đi kèm) bằng
NFD, rồi mẫu dấu là dãy các đơn vị tại các vị trí mang dấu. Vị trí mang dấu
được cố định theo tiếng Việt: nguyên âm (`a, e, i, o, u, y`) và `đ`.
Cách mã hóa này có một lợi thế quan trọng: **cùng một mẫu dấu được dùng
chung cho nhiều từ khác nhau** (ví dụ mẫu `("ó",)` dùng cho cả `có`, `bó`,
`mó`), nên mô hình học được quy luật dấu và chuyển giao được sang từ chưa
thấy. Chỉ giữ các mẫu xuất hiện **≥ 5 lần** trong train; các từ có mẫu hiếm
mang nhãn −100 (bỏ qua khi train). Kết quả: **2.326 nhãn**.

Trước khi chốt công thức này, đề án đã thử hai phương án khác và ghi lại
làm bài học (Mục 4.3):

- **Candidate-index** (nhãn = vị trí của dạng đúng trong danh sách ứng viên
  của từ): nhãn mang nghĩa khác nhau cho mỗi từ (nhãn 1 là `vả` với từ `va`
  nhưng là `lả` với từ `la`), nên mô hình không chuyển giao được quy luật.
- **Học tập mang dấu từ dữ liệu**: khiến tập mang dấu phình ra 85 ký tự,
  gồm cả chữ Thái, Khmer, Devanagari, Hy Lạp trong Wikipedia, làm số nhãn
  nổ lên 42.525 và mẫu dấu gần như trùng cả từ.

### 3.2. Baseline 1 — Từ điển

Mỗi từ không dấu được gán dạng có dấu **phổ biến nhất** trong train. Từ
chưa thấy trong train giữ nguyên dạng không dấu. Đây là mốc tham chiếu
thấp nhất, không dùng ngữ cảnh.

### 3.3. Baseline 2 — Bigram + Viterbi

Mô hình bigram trên các dạng có dấu trong train, làm mịn cộng alpha
(α = 0,1), giải mã bằng thuật toán Viterbi trên toàn câu: với mỗi từ không
dấu, các dạng có dấu đã thấy trong train là ứng viên, và Viterbi chọn dãy
dạng có xác suất bigram cao nhất.

### 3.4. PhoBERT fine-tune

Mô hình `vinai/phobert-base` (đã được VinAI pretrain sẵn trên ~20 GB văn
bản tiếng Việt) được fine-tune cho bài toán phân loại token: mỗi từ không
dấu là một token, nhãn là mẫu dấu của nó (Mục 3.1). Cấu hình: 3 epoch,
learning rate 2e-5, batch 32, fp16, chạy trên Kaggle GPU T4 x2.

Hai lưu ý kỹ thuật: tokenizer của PhoBERT là tokenizer chậm (không hỗ trợ
`word_ids()`), nên các từ được tokenize riêng từng từ (kèm cache) và nhãn
gắn vào subword đầu tiên của mỗi từ; thư viện transformers 5.20.2 trên
Kaggle không còn tham số `warmup_ratio` trong `TrainingArguments` nên tham
số này bị bỏ qua.

### 3.5. Transformer tự build

Một Transformer encoder nhỏ, train từ đầu ở **mức từ** (không dùng subword):

- Từ vựng: 22.413 từ không dấu xuất hiện ≥ 2 lần trong train.
- Kiến trúc: embedding 256 + positional encoding, encoder 4 lớp, 8 heads,
  dropout 0,1; tổng cộng **9.510.934 tham số**.
- Huấn luyện: Adam, lr 1e-3, 12 epoch, giữ checkpoint tốt nhất theo word
  accuracy trên validation. Loss giảm từ 1,09 xuống 0,26; accuracy trên
  validation tăng từ 83,5% lên **91,1%**. (Ở lần chạy 16/11, số validation
  này đo trên nhãn mẫu, loại từ không mã hóa được; code từ 17/11 đo word
  accuracy thật trên validation, cùng cách đo với test.)

### 3.6. Metric

Cả 4 hệ thống được đo trên **cùng tập test 10.000 câu** (239.189 từ) với
3 metric:

| Metric | Định nghĩa |
|---|---|
| Word accuracy | Tỉ lệ từ khôi phục dấu đúng / tổng số từ |
| Sentence accuracy | Tỉ lệ câu khôi phục đúng toàn bộ / tổng số câu |
| Accuracy trên từ đa nghĩa | Accuracy riêng cho nhóm từ có ≥ 2 dạng có dấu (tập đa nghĩa tính trên train, 207.644 từ trên test) |

Riêng PhoBERT còn được đo thêm accuracy trên từ chưa thấy trong train
(5.754 từ trên test): **85,4%** — cao hơn cả word accuracy tổng thể của nó,
cho thấy PhoBERT chuyển giao tốt sang từ mới nhờ pretrain.

---

## 4. Kết quả

### 4.1. Bảng so sánh chính

| Hệ thống | Word accuracy | Sentence accuracy | Từ đa nghĩa |
|---|---:|---:|---:|
| Từ điển | 77,4% (185.248/239.189) | 2,6% (261/10.000) | 74,6% (154.991/207.644) |
| Bigram | 91,9% (219.857/239.189) | 30,8% (3.079/10.000) | 91,3% (189.600/207.644) |
| PhoBERT | 83,4% (199.500/239.189) | 6,7% (671/10.000) | 81,8% (169.922/207.644) |
| **Tự build** | **92,5% (221.271/239.189)** | **30,8% (3.075/10.000)** | **92,4% (191.774/207.644)** |

![So sánh 4 hệ thống trên tập test 10.000 câu](comparison_chart.png)

### 4.2. Nhận xét

- **Transformer tự build là hệ tốt nhất**, vượt bigram ở word accuracy
  (+0,6 điểm phần trăm) và từ đa nghĩa (+1,1 điểm), hòa ở sentence
  accuracy (30,8% — chỉ chênh 4 câu). Đây là kết quả bất ngờ so với kỳ
  vọng ban đầu (PhoBERT được kỳ vọng mạnh nhất).
- **Bigram là baseline rất mạnh**: chỉ dùng thống kê 2-gram mà đã đạt
  91,9%, vượt xa PhoBERT. Văn bản Wikipedia có nhiều cụm từ lặp lại, và
  Viterbi với tập ứng viên chính xác khai thác tốt đặc điểm này.
- **PhoBERT đạt 83,4%**, thấp hơn cả bigram. Phân tích nguyên nhân ở
  Mục 5.4.
- Sentence accuracy của mọi hệ đều thấp (tốt nhất 30,8%) vì chỉ cần sai
  một từ là cả câu sai; với trung bình ~24 từ/câu và word accuracy 92,5%,
  xác suất một câu hoàn toàn đúng theo lý thuyết đã thấp.
- Mục tiêu tham khảo 95% word accuracy **chưa đạt được** (tốt nhất
  92,5%). Các công trình công bố trên bài toán này đạt khoảng 97–98%
  (Mục 7) với corpus lớn hơn nhiều và mô hình lớn hơn; đó là hướng phát
  triển, không phải kết quả của đề án này.

### 4.3. Các thử nghiệm PhoBERT khác (không vào bảng chính)

| Thử nghiệm | Word | Sentence | Từ đa nghĩa | Bài học |
|---|---:|---:|---:|---|
| Candidate-index (2 epoch) | 85,2% | 9,4% | 83,6% | Nhãn không chuyển giao được giữa các từ |
| MODIFIABLE học từ dữ liệu (3 epoch, 42.525 nhãn) | 77,8% | 2,9% | 75,3% | Học tập mang dấu từ dữ liệu kéo chữ nước ngoài vào, nổ số nhãn |
| Mẫu dấu, MODIFIABLE cố định, giới hạn ≥ 5 lần (3 epoch, 2.326 nhãn) | 83,4% | 6,7% | 81,8% | Công thức chính thức của hệ PhoBERT |

Đáng chú ý, candidate-index (85,2%) còn cao hơn công thức mẫu dấu (83,4%),
vì đầu phân loại 21 lớp của nó học đủ trong 3 epoch, trong khi đầu 2.326
lớp của công thức mẫu dấu còn thiếu huấn luyện (loss cuối epoch 3 vẫn
1,95). Đây là bằng chứng cho giả thuyết ở Mục 5.4 rằng PhoBERT bị thiếu
huấn luyện chứ không phải kiến trúc yếu hơn.

### 4.4. Ghi chú về tính nhất quán của số liệu

Số của từ điển và bigram trong Bảng 4.1 là số **tái tạo ở chặng so sánh**
(cùng định nghĩa tập đa nghĩa trên train cho cả 4 hệ). Số gốc đo ở
notebook dữ liệu: từ điển 77,4% / 2,6% / 74,7%; bigram 91,9% / 31,0% /
91,2% (mẫu số từ đa nghĩa 209.279, tính trên toàn bộ corpus). Chênh lệch
tối đa 0,2 điểm phần trăm, do định nghĩa tập đa nghĩa và tập ứng viên
khác nhau — không ảnh hưởng tới thứ hạng các hệ. Ngày 17/11 notebook dữ
liệu đã sửa sang định nghĩa train-only (xem `results.md`) nên chạy lại
sẽ cho thẳng số của Bảng 4.1; số gốc 74,7% / 91,2% chỉ còn giá trị lịch
sử. Tương tự, code đánh giá của hệ tự build đã thêm masking mẫu tương
thích (17/11) nên chạy lại có thể cho số cao hơn 92,5% một chút — số
trong bảng này đo bằng argmax trần ngày 16/11.

---

## 5. Phân tích lỗi

### 5.1. Lỗi của hệ tốt nhất (Transformer tự build)

Trên tập test, hệ tự build sai **17.918 / 239.189 từ**. Phân loại lỗi:

| Loại lỗi | Số lượng | Tỉ lệ trên số lỗi |
|---|---:|---:|
| Chỉ sai thanh điệu (cùng nguyên âm, khác dấu thanh) | 9.878 | 55,1% |
| Sai ở từ đa nghĩa | 15.870 | 88,6% |
| Sai ở từ chưa thấy trong train | 1.126 | 6,3% |

Hai con số đầu cho thấy **lỗi tập trung ở việc phân biệt thanh điệu giữa
các dạng của cùng một từ không dấu**, chứ không phải ở từ mới hay từ hiếm.

### 5.2. Các cặp nhầm lẫn phổ biến nhất

| Đúng → Dự đoán | Số lần |
|---|---:|
| nhưng → những | 73 |
| từ → tự | 45 |
| hoá → hóa | 45 |
| khí → khi | 42 |
| còn → con | 41 |
| trọng → trong | 39 |
| tự → từ | 38 |
| cổ → có | 38 |
| nền → nên | 35 |
| loài → loại | 32 |

Cả 10 cặp đều **chỉ khác nhau ở thanh điệu**. Đặc biệt cặp `hoá → hóa`
chỉ là hai cách viết chính tả khác nhau của cùng một từ (dấu hỏi đặt ở
`o` hay `a`), cho thấy một phần lỗi đến từ sự không nhất quán chính tả
trong chính Wikipedia, không phải lỗi hiểu ngữ cảnh.

### 5.3. Ví dụ từ demo

Demo Gradio (Mục 6) chạy thử 3 câu mẫu cho 20/27 từ đúng (~74%) — thấp hơn
92,5% trên test vì các câu ví dụ dồn nhiều từ đa nghĩa:

- `xin chao cac ban, toi ten la hao` → `xin chào các bản, tôi tên là hào`
  (lỗi duy nhất: `ban` → `bản`, đúng là `bạn`).
- `ha noi la thu do cua viet nam` → `hà nội là thủ đô của viết năm`
  (cụm `Việt Nam` bị khôi phục thành `viết năm`).
- `toi dang hoc tieng viet va rat thich mon nay` → `tới dạng học tiếng
  viết và rất thích môn này` (lỗi: `Tôi` → `tới`, `đang` → `dạng`,
  `Việt` → `viết`).

Nhận xét: mô hình có xu hướng chọn **nghĩa phổ biến hơn** trong train
(`viết` thay cho `Việt`, `năm` thay cho `Nam`, `tới` thay cho `Tôi`),
đúng dạng lỗi chủ đạo ở Mục 5.1. Ngoài ra mô hình chỉ khôi phục dấu, không
đổi chữ hoa/thường, nên `hà nội` giữ nguyên chữ thường như input.

### 5.4. Vì sao PhoBERT thua mô hình tự build?

Đây là kết quả bất ngờ nhất của đề án. Các giả thuyết sau đây dựa trên
quan sát trong quá trình huấn luyện (chưa được kiểm chứng bằng thử nghiệm
đối chứng, và được ghi rõ là giả thuyết):

1. **Thiếu huấn luyện.** PhoBERT chỉ train 3 epoch, loss cuối vẫn 1,95
   (từ 12,31), trong khi tự build train 12 epoch, loss xuống 0,26.
   Đầu phân loại 2.326 lớp với lr 2e-5 chưa học đủ — bằng chứng là công
   thức candidate-index với đầu 21 lớp lại đạt cao hơn (85,2%).
2. **Mức dự đoán.** PhoBERT dự đoán ở mức subword (nhãn gắn vào subword
   đầu tiên của từ), trong khi tự build dự đoán ở mức từ, không mất
   thông tin.
3. **Đầu phân loại học cùng encoder.** Tự build học đầu phân loại và
   encoder cùng nhau từ đầu với lr 1e-3; PhoBERT fine-tune một encoder
   đã pretrain với lr thấp, đầu phân loại mới thêm vào học chậm hơn.

Hướng kiểm chứng trực tiếp: train PhoBERT thêm epoch (5–10) và/hoặc tăng
lr riêng cho đầu phân loại, rồi đo lại.

### 5.5. Hạn chế

- Mô hình không xử lý **viết hoa** (chỉ khôi phục dấu).
- Từ nước ngoài mang dấu ngoài tiếng Việt (0,029%) không mã hóa được.
- Chỉ đo trên văn bản Wikipedia; chưa đo trên miền khác (báo chí, văn
  nói, teencode), nơi tỉ lệ lỗi có thể cao hơn.
- Mục tiêu 95% word accuracy chưa đạt được (Mục 4.2).

---

## 6. Demo

Demo Gradio chạy trên checkpoint tốt nhất của hệ tự build, đặt trong
Cell 9 của notebook Transformer tự build:

- Link public: https://5c5532dd203ae07dc6.gradio.live
  (link tạm của Gradio, tồn tại tối đa 1 tuần và chỉ sống khi phiên
  Kaggle còn chạy; chạy lại Cell 9 để có link mới).
- Cách dùng: nhập câu không dấu, bấm Submit, nhận câu đã khôi phục dấu
  trong khoảng 1 giây/câu. Demo có sẵn 3 câu ví dụ.

---

## 7. Kết luận và hướng phát triển

Đề án đã xây dựng và so sánh công bằng 4 hệ thống khôi phục dấu tiếng
Việt trên cùng một tập test 10.000 câu Wikipedia. Kết quả chính: **Transformer
tự build đạt word accuracy 92,5%, tốt nhất trong 4 hệ**, vượt bigram
(91,9%) và PhoBERT fine-tune (83,4%). Phân tích lỗi cho thấy hơn một nửa
số lỗi chỉ sai ở thanh điệu, và gần 9/10 lỗi rơi vào từ đa nghĩa — đó là
hướng cải thiện rõ ràng nhất.

Hướng phát triển:

- Train PhoBERT nhiều epoch hơn và tăng learning rate cho đầu phân loại
  để kiểm chứng giả thuyết ở Mục 5.4.
- Mở rộng corpus lên 500.000 câu (mức cao của kế hoạch ban đầu).
- Thử mô hình seq2seq (mT5/T5) — các công trình công bố đạt 97–98% word
  accuracy trên bài toán này (T5 đạt 97,81% trong benchmark nom-vn;
  XLM-RoBERTa token classification đạt 97%).
- Thêm giải mã có ràng buộc (constrained decoding) hoặc CRF để giảm lỗi
  thanh điệu.
- Xử lý viết hoa và đo thêm trên miền ngoài Wikipedia.

---

## Tài liệu tham khảo

- Anh, N. & Trang, N. (2020). TDP: Hybrid Transformer Decoder for
  Vietnamese diacritic restoration. *PACLIC*.
- Popovac, M. et al. CNN cho khôi phục dấu tiếng Việt, 97,73% word
  accuracy.
- PhoPunct (2026). PhoBERT-large + Bi-LSTM + CRF. *PACLIC 40*.
- vietai (2019). Transformer Seq2Seq cho khôi phục dấu tiếng Việt. GitHub.
- duongntbk. `restore_vietnamese_diacritics` (~720.000 câu từ Kaggle Old
  Newspapers). GitHub.
- ngwkhai (2025). `Neural-Machine-Translation` (12,5 triệu cặp, Transformer
  BLEU 81,31). GitHub.
- peterhung. `vietnamese-accent-marker-xlm-roberta` (token classification,
  97% accuracy so với HMM 91%). Hugging Face.
- Toshiiiii1. Mô hình T5, 97,81% word accuracy trong benchmark nom-vn.
  Hugging Face.

---

## Phụ lục: Tái tạo kết quả

Code chạy trên Kaggle (tài khoản `lcaoho`), GPU T4 x2, Internet ON.
Toàn bộ comment trong code bằng tiếng Việt.

| Notebook | Nội dung | Link |
|---|---|---|
| Vietnamese Diacritic Restoration - Week 1 Starter | Dữ liệu + baseline từ điển + bigram (Version 5) | https://www.kaggle.com/code/lcaoho/vietnamese-diacritic-restoration-week-1-starter |
| Vietnamese Diacritic Restoration - PhoBERT | PhoBERT fine-tune, công thức mẫu dấu (Version 3) | https://www.kaggle.com/code/lcaoho/vietnamese-diacritic-restoration-phobert |
| VN Diacritic Restoration - Self-built Transformer | Transformer tự build + so sánh + phân tích lỗi + demo Gradio (Version 3) | https://www.kaggle.com/code/lcaoho/vn-diacritic-restoration-self-built-transformer |

Số kiểm chứng khi chạy lại: 1.501 bài viết, 100.000 câu, 2.387.449 từ,
5.461 từ đa nghĩa (train), 2.326 nhãn mẫu dấu. Lưu ý: các commit Version
trên Kaggle có thể còn ở trạng thái Queued do hàng đợi GPU; số liệu trong
báo cáo này lấy từ các lần chạy tương tác đã xác minh, và mọi notebook đều
lưu lại đầy đủ output của lần chạy đó.
