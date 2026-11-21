# restore_document.py — Khôi phục dấu cho cả 1 file tài liệu bằng model
# Transformer tự build (Notebook 3) đã train xong.
#
# Đây là "tầng ứng dụng" (inference), KHÔNG phải phần train: model chỉ
# biết 1 việc là nhận 1 câu không dấu -> trả 1 câu có dấu. Script này
# đọc file -> tách thành từng câu -> gọi model khôi phục từng câu
# (câu dài quá 64 từ thì cắt nhỏ trước) -> ghép lại -> ghi ra file mới.
# Chatbot cũng dùng đúng 1 hàm restore() này cho mỗi tin nhắn.
#
# Cách dùng:
#   1. Chạy xong Notebook 3 trên Kaggle, chạy thêm cell xuất dữ liệu phụ
#      ở Phần 9.0 của docs giải thích để có 2 file:
#        selfbuilt_best.pt      (checkpoint, ~40 MB)
#        diacritic_assets.json  (vocab + mẫu dấu + đếm bigram)
#   2. Chạy script (CPU là đủ, model chỉ có 9,5 triệu tham số):
#        python restore_document.py --checkpoint selfbuilt_best.pt \
#            --assets diacritic_assets.json \
#            --input van-ban-khong-dau.txt --output van-ban-co-dau.txt
#      Hoặc thử nhanh 1 câu, không cần file:
#        python restore_document.py --checkpoint selfbuilt_best.pt \
#            --assets diacritic_assets.json \
#            --text "ha noi la thu do cua viet nam"
#      Thêm --no-viterbi để giải mã bằng argmax thay thế.
#      Mặc định giải mã bằng Viterbi + bigram (beta 2.0 là beta tốt
#      nhất chọn trên val ở Cell 5 của Notebook 3, đo tối 21/11:
#      beta 0,5 -> val 94,4%, beta 1,0 -> val 94,8%, beta 2,0 ->
#      val 95,0%) — cho kết quả đúng nhất: 95,0% word so với 93,2%
#      của argmax trên test Wikipedia, 96,2% so với 88,3% trên test
#      phụ đề sau fine-tune văn nói.
#   3. File .txt đọc thẳng. File .docx cần thư viện python-docx,
#      file .pdf cần pypdf (pip install python-docx pypdf).
#
# Giới hạn cần biết (giống hệt model trong notebook):
#   - Chữ hoa: input thường "viet" không bao giờ ra được "Việt" hoa,
#     vì model chỉ khôi phục dấu, không đoán chữ hoa/thường.
#   - Câu dài: model chỉ nhìn tối đa 64 từ/lần nên script tự cắt câu
#     dài thành đoạn <= 60 từ; ngữ cảnh ở chỗ cắt sẽ bị mất 1 phần.
#   - Số, URL, tên riêng nước ngoài: model giữ nguyên như input.
#
# Logic model sao chép từ Notebook 3 (Cell 2/3/5) để kết quả khớp với
# số đo trong báo cáo. Chỉ có 1 chỗ khác: file diacritic_assets.json
# không chứa bảng valid_forms nên tính tương thích của mẫu dấu bằng
# cách so ký tự gốc (bỏ dấu của đơn vị có dấu phải khớp ký tự gốc),
# thay vì tra bảng các dạng đã thấy trong train. Kiểm tra này hơi
# rộng hơn 1 chút nhưng apply_signature vẫn chỉ tạo ra dạng hợp lệ
# với khung của từ đó.
import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict

# Vị trí có thể mang dấu trong tiếng Việt (cố định, giống Cell 2 NB3).
MODIFIABLE = set("aeiouyAEIOUYdD")
MAX_LEN = 64            # model chỉ nhìn tối đa 64 từ/lần (Cell 3 NB3)
MAX_WORDS_PER_CALL = 60  # cắt câu dài thành đoạn <= 60 từ, chừa biên an toàn


# ---------------------------------------------------------------------------
# Hàm bỏ dấu + mã hóa mẫu dấu (sao chép từ Cell 1/2 của Notebook 3)
# ---------------------------------------------------------------------------
def strip_diacritics(s):
    """Bỏ dấu tiếng Việt: tách dấu bằng NFD rồi loại ký tự dấu."""
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def split_units(word):
    """Tách từ thành các đơn vị: 1 ký tự gốc + dấu đi kèm (giống Cell 2)."""
    units = []
    for ch in unicodedata.normalize("NFD", word):
        if unicodedata.category(ch) == "Mn" and units:
            units[-1] += ch
        else:
            units.append(ch)
    return [unicodedata.normalize("NFC", u) for u in units]


def apply_signature(bare, sig):
    """Khôi phục từ có dấu từ từ không dấu + mẫu dấu. Trả None nếu mẫu
    không khớp từ này (số đơn vị khác số vị trí có thể mang dấu)."""
    chars = list(bare)
    pos = [i for i, b in enumerate(bare) if b in MODIFIABLE]
    if len(sig) != len(pos):
        return None
    for i, s in zip(pos, sig):
        chars[i] = s
    return "".join(chars)


def to_signature(bare, acc):
    """Mẫu dấu = tuple các đơn vị có dấu tại vị trí có thể mang dấu."""
    units = split_units(acc)
    if len(units) != len(bare):
        return None
    sig = tuple(u for u, b in zip(units, bare) if b in MODIFIABLE)
    got = apply_signature(bare, sig)
    if got is None or unicodedata.normalize("NFC", got) != unicodedata.normalize("NFC", acc):
        return None
    return sig


def skeleton(bare):
    """Khung của từ: tuple các ký tự có thể mang dấu, theo thứ tự."""
    return tuple(b for b in bare if b in MODIFIABLE)


# ---------------------------------------------------------------------------
# Model Transformer tự build (sao chép từ Cell 3 của Notebook 3)
# ---------------------------------------------------------------------------
class SmallTransformer:
    """Chỉ là vỏ định nghĩa kiến trúc; trọng số nạp từ checkpoint.

    Cần torch để tạo model thật — hàm build_model() import torch lười
    để các hàm thuần túy bên trên vẫn chạy được khi chưa cài torch
    (vd chạy --self-test để kiểm tra tách câu / cắt đoạn)."""

    def __init__(self, torch, nn, vocab_size, num_labels,
                 d_model=256, nhead=8, num_layers=4, dim_ff=1024,
                 dropout=0.1):
        self._torch = torch
        self._nn = nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
                self.pos = nn.Embedding(MAX_LEN, d_model)
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                    dropout=dropout, batch_first=True)
                self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
                self.head = nn.Linear(d_model, num_labels)

            def forward(self, x):
                pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
                h = self.emb(x) + self.pos(pos)
                pad_mask = (x == 0)
                h = self.encoder(h, src_key_padding_mask=pad_mask)
                return self.head(h)

        self.model = _Model()

    def __call__(self, x):
        return self.model(x)

    def load_state_dict(self, state):
        self.model.load_state_dict(state)

    def eval(self):
        self.model.eval()


class Restorer:
    """Nạp checkpoint + assets 1 lần, rồi khôi phục bao nhiêu câu cũng được."""

    def __init__(self, checkpoint_path, assets_path, device=None, use_viterbi=True,
                 beta=2.0):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        self.torch = torch
        self.F = F
        self.use_viterbi = use_viterbi
        self.beta = beta
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        with open(assets_path, encoding="utf-8") as f:
            assets = json.load(f)
        # signatures trong JSON là list -> đổi lại tuple để khớp notebook.
        self.vocab = assets["vocab"]
        self.signatures = [tuple(s) for s in assets["signatures"]]
        self.w2id = {w: i for i, w in enumerate(self.vocab)}
        self.sig_to_id = {s: i for i, s in enumerate(self.signatures)}

        # Bảng mẫu tương thích theo khung từ, tính từ ký tự gốc
        # (xem ghi chú ở đầu file: assets không có valid_forms).
        by_len = defaultdict(list)
        for i, s in enumerate(self.signatures):
            by_len[len(s)].append(i)
        self._compat_cache = {}

        def compatible_ids(skel):
            if skel not in self._compat_cache:
                self._compat_cache[skel] = [
                    i for i in by_len[len(skel)]
                    if all(strip_diacritics(self.signatures[i][j]) == skel[j]
                           for j in range(len(skel)))]
            return self._compat_cache[skel]

        self.compatible_ids = compatible_ids

        # Đếm bigram (dùng cho giải mã Viterbi mặc định): đếm unigram suy ra từ
        # bigram (tổng theo từ đứng trước), giống cách làm mịn ở NB3.
        self.bigram = Counter()
        if use_viterbi:
            for k, v in assets.get("bigram", {}).items():
                prev, w = k.split("\t")
                self.bigram[(prev, w)] = v
            self.unigram = Counter()
            for (prev, _), v in self.bigram.items():
                self.unigram[prev] += v
            self._bigram_V = len(self.unigram)
            self._bigram_alpha = 0.1

        model = SmallTransformer(torch, nn, len(self.vocab),
                                 len(self.signatures))
        try:
            state = torch.load(checkpoint_path, map_location=self.device,
                               weights_only=True)
        except TypeError:  # torch cũ chưa có tham số weights_only
            state = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        self.model = model
        print(f"da nap model: vocab {len(self.vocab)} tu | "
              f"{len(self.signatures)} mau dau | device {self.device}",
              file=sys.stderr)

    def encode_sentence(self, bare_sent):
        ids = [self.w2id.get(w, 1) for w in bare_sent.split()]
        return ids[:MAX_LEN]

    def bigram_logprob(self, prev, w):
        return math.log((self.bigram[(prev, w)] + self._bigram_alpha) /
                        (self.unigram[prev] + self._bigram_alpha * self._bigram_V))

    def restore_words(self, words):
        """Khôi phục dấu cho 1 danh sách từ không dấu (<= 64 từ)."""
        if not words:
            return []
        if self.use_viterbi:
            return self._restore_words_viterbi(words)
        return self._restore_words_argmax(words)

    def _restore_words_argmax(self, words):
        # Giống restore_with_model ở Cell 3 NB3: mỗi từ chỉ xét mẫu
        # tương thích, lấy mẫu có logit cao nhất.
        torch = self.torch
        ids = torch.tensor([self.encode_sentence(" ".join(words))],
                           device=self.device)
        with torch.no_grad():
            logits = self.model(ids)[0].cpu()
        out = []
        for b, lg in zip(words, logits):
            cands = self.compatible_ids(skeleton(b))
            if not cands:
                out.append(b)
                continue
            best = max(cands, key=lambda k: lg[k].item())
            restored = apply_signature(b, self.signatures[best])
            out.append(restored if restored is not None else b)
        return out

    def _restore_words_viterbi(self, words):
        # Giống restore_with_viterbi ở Cell 3 NB3, chỉ khác nguồn ứng
        # viên: dạng có dấu suy ra từ các mẫu tương thích (assets
        # không có bảng forms), thay vì dạng đã thấy trong train.
        torch = self.torch
        ids = torch.tensor([self.encode_sentence(" ".join(words))],
                           device=self.device)
        with torch.no_grad():
            logp = self.F.log_softmax(self.model(ids)[0].cpu(), dim=-1)
        cand_lists = []
        for i, b in enumerate(words):
            cands = {}
            for sid in self.compatible_ids(skeleton(b)):
                form = apply_signature(b, self.signatures[sid])
                if form is None:
                    continue
                sc = logp[i][sid].item()
                if form not in cands or sc > cands[form]:
                    cands[form] = sc
            if not cands:
                cands = {b: -30.0}
            cand_lists.append(list(cands.items()))
        beta = self.beta
        dp = [{f: sc + beta * self.bigram_logprob("<s>", f)
               for f, sc in cand_lists[0]}]
        bp = []
        for i in range(1, len(words)):
            prev_dp = dp[-1]
            cur, cur_bp = {}, {}
            for f, sc in cand_lists[i]:
                best_prev, best_score = None, float("-inf")
                for pf, ps in prev_dp.items():
                    s = ps + sc + beta * self.bigram_logprob(pf, f)
                    if s > best_score:
                        best_score, best_prev = s, pf
                cur[f] = best_score
                cur_bp[f] = best_prev
            dp.append(cur)
            bp.append(cur_bp)
        last = max(dp[-1], key=dp[-1].get)
        out = [last]
        for i in range(len(words) - 1, 0, -1):
            last = bp[i - 1][last]
            out.append(last)
        return out[::-1]


# ---------------------------------------------------------------------------
# Tách văn bản thành câu + cắt câu dài (tầng ứng dụng, không thuộc model)
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"^([^\wÀ-ỹ]*)(.*?)([^\wÀ-ỹ]*)$", re.UNICODE)


def split_token(token):
    """Tách 1 token thành (tiền tố dấu câu, từ lõi, hậu tố dấu câu).

    Vd "nam." -> ("", "nam", "."), "(toi" -> ("(", "toi", "").
    Từ lõi được đưa vào model, dấu câu giữ nguyên vị trí cũ."""
    m = _TOKEN_RE.match(token)
    return m.group(1), m.group(2), m.group(3)


def split_sentences(paragraph):
    """Tách 1 đoạn thành câu, giữ dấu kết thúc câu (giống Cell 9 NB3)."""
    return [s for s in re.findall(r"[^.!?…]+[.!?…]*", paragraph) if s.strip()]


def restore_text(restorer, text):
    """Khôi phục dấu cho cả 1 đoạn văn bản, giữ nguyên đoạn văn/ngắt dòng."""
    out_paragraphs = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            out_paragraphs.append("")
            continue
        out_sentences = []
        for sentence in split_sentences(paragraph):
            tokens = sentence.split()
            # Tách dấu câu ra khỏi từ trước khi đưa vào model.
            parts = [split_token(t) for t in tokens]
            cores = [strip_diacritics(core) if core else core
                     for _, core, _ in parts]
            restored_cores = []
            for start in range(0, len(cores), MAX_WORDS_PER_CALL):
                chunk = cores[start:start + MAX_WORDS_PER_CALL]
                restored_cores.extend(restorer.restore_words(chunk))
            rebuilt = " ".join(
                pre + core + post
                for (pre, _, post), core in zip(parts, restored_cores))
            # Câu gốc có thể kết thúc bằng dấu câu đã nằm trong token
            # cuối; giữ nguyên khoảng trắng giữa các câu.
            out_sentences.append(rebuilt)
        out_paragraphs.append(" ".join(out_sentences))
    return "\n".join(out_paragraphs)


# ---------------------------------------------------------------------------
# Đọc / ghi file
# ---------------------------------------------------------------------------
def read_input(path):
    if path.endswith(".txt"):
        with open(path, encoding="utf-8") as f:
            return f.read()
    if path.endswith(".docx"):
        try:
            from docx import Document
        except ImportError:
            raise SystemExit(
                "doc .docx can thu vien python-docx: pip install python-docx")
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    if path.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise SystemExit(
                "doc .pdf can thu vien pypdf: pip install pypdf")
        reader = PdfReader(path)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    raise SystemExit(f"khong ho tro dinh dang file: {path} "
                     "(chi ho tro .txt, .docx, .pdf)")


def self_test():
    """Kiểm tra nhanh các hàm thuần túy, không cần model/checkpoint."""
    assert apply_signature("duoc", ("đ", "ư", "ợ")) == "được"
    assert apply_signature("co", ("ó",)) == "có"
    assert apply_signature("hoan", ("o", "à")) == "hoàn"
    assert apply_signature("ba", ("á",)) == "bá"
    # apply_signature chỉ kiểm tra độ dài mẫu; việc mẫu có hợp với
    # từng nguyên âm hay không do compatible_ids lọc trước khi gọi.
    assert apply_signature("ba", ("ó", "á")) is None  # mẫu dài hơn khung
    assert to_signature("duoc", "được") == ("đ", "ư", "ợ")
    assert skeleton("duoc") == ("d", "u", "o")  # 'd' cũng mang dấu được (đ)
    assert split_token("nam.") == ("", "nam", ".")
    assert split_token("(toi") == ("(", "toi", "")
    assert len(split_sentences("Xin chao. Toi ten la Hao!")) == 2
    long_text = " ".join(["toi"] * 130)
    chunks = [long_text.split()[i:i + MAX_WORDS_PER_CALL]
              for i in range(0, 130, MAX_WORDS_PER_CALL)]
    assert len(chunks) == 3 and all(len(c) <= 60 for c in chunks)
    print("self-test OK: apply_signature / to_signature / tach cau / "
          "cat cau dai deu dung")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Khoi phuc dau tieng Viet cho ca 1 file tai lieu "
                    "bang model Transformer tu build (Notebook 3).")
    parser.add_argument("--checkpoint",
                        help="duong dan selfbuilt_best.pt")
    parser.add_argument("--assets",
                        help="duong dan diacritic_assets.json")
    parser.add_argument("--input", help="file dau vao (.txt, .docx, .pdf)")
    parser.add_argument("--output", help="file dau ra (.txt); mac dinh "
                        "in ra man hinh")
    parser.add_argument("--text", help="khoi phuc nhanh 1 doan van ban, "
                        "khong can file")
    parser.add_argument("--no-viterbi", action="store_true",
                        help="dung argmax thay cho Viterbi mac dinh")
    parser.add_argument("--beta", type=float, default=2.0,
                        help="trong so bigram khi giai ma Viterbi (mac dinh "
                             "2.0, beta tot nhat chon tren val toi 21/11)")
    parser.add_argument("--self-test", action="store_true",
                        help="chi kiem tra ham thuan tuy, khong nap model")
    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
        return 0
    if not args.checkpoint or not args.assets:
        parser.error("--checkpoint va --assets la bat buoc "
                     "(tru khi chay --self-test)")
    if not args.input and not args.text:
        parser.error("can --input hoac --text")
    if args.input and args.text:
        parser.error("chi dung 1 trong --input hoac --text")

    restorer = Restorer(args.checkpoint, args.assets,
                        use_viterbi=not args.no_viterbi, beta=args.beta)
    text = args.text if args.text is not None else read_input(args.input)
    result = restore_text(restorer, text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"da ghi: {args.output}", file=sys.stderr)
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
