# NOTE · Chạy kiểm tra (01/10/2026)

Phiếu ngắn để bạn tự chạy kiểm tra cho 3 phần mới nhất: **auto-detailer mặt/tay**,
**9 vùng sửa**, **khung 💅 Chi tiết mắt & móng**. Bản đầy đủ hơn (kèm giải thích từng
bước) ở [`PHIEU_CHAY_THU.md`](PHIEU_CHAY_THU.md).

---

## A. Kiểm tra code (máy local hoặc sandbox — KHÔNG cần GPU, ~1 phút)

### A0. Dựng môi trường test (chỉ làm khi chưa có `.venv`)

```bash
cd ai-anime
python3 -m venv .venv
.venv/bin/pip install "gradio==6.15.2" pillow nbformat packaging \
  "huggingface_hub==0.36.2" black torch
```

> `torch` chỉ để chạy test tái hiện lỗi inference-mode; không có GPU vẫn cài được bản CPU.

### A1. Toàn bộ test

```bash
.venv/bin/python -m unittest discover -s tests
```

**Mong đợi:**

```text
Ran 81 tests in ~7s
OK
```

0 test bị bỏ qua. Kiểm tra lại bằng:

```bash
.venv/bin/python -m unittest discover -s tests -v 2>&1 | grep -c "ok$"      # → 81
.venv/bin/python -m unittest discover -s tests -v 2>&1 | grep -c "skipped"  # → 0
```

### A2. Chạy riêng từng nhóm (nhanh, để biết lỗi nằm ở đâu)

| Lệnh | Mong đợi | Kiểm tra gì |
| --- | --- | --- |
| `.venv/bin/python -m unittest tests.test_colab_studio.NotebookTests` | `Ran 6 tests … OK` | Notebook khớp `build()`, 10 ô, hash checkpoint/LoRA, `ultralytics` chỉ cảnh báo |
| `.venv/bin/python -m unittest tests.test_colab_studio.PromptLibraryTests` | `Ran 7 tests … OK` | Thư viện prompt `.txt` |
| `.venv/bin/python -m unittest tests.test_colab_studio.ProfessionalPromptTests` | `Ran 8 tests … OK` | Sắp xếp prompt, 8 bộ negative, bộ kiểm tra |
| `.venv/bin/python -m unittest tests.test_colab_studio.RuntimeValidationTests` | `Ran 20 tests … OK` | Runtime, hires, inpaint, **18 sự kiện UI**, đường link công khai |
| `.venv/bin/python -m unittest tests.test_colab_studio.AutoDetailerTests` | `Ran 8 tests … OK` | Auto-detailer: dò vùng → inpaint → dán lại, hash weight |
| `.venv/bin/python -m unittest tests.test_colab_studio.RepairAndLookPromptTests` | `Ran 6 tests … OK` | 9 vùng sửa + màu mắt/kiểu móng/màu sơn móng |

### A3. Định dạng + notebook phải khớp bản sinh

```bash
.venv/bin/black --check .                 # → "5 files would be left unchanged"
.venv/bin/python scripts/build_colab_studio.py
```

Rồi kiểm notebook **khớp đúng bản sinh** (đừng dùng `git diff`: notebook khác commit cũ
là chuyện bình thường khi chưa commit):

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from scripts.build_colab_studio import build

n = json.loads(Path("WAI_Illustrious_Studio_Colab.ipynb").read_text(encoding="utf-8"))
print("notebook == build():", n == build())
PY
```

**Mong đợi:** `notebook == build(): True`. Nếu ra `False` nghĩa là bạn sửa
`colab/studio.py` (hoặc `scripts/build_colab_studio.py`) mà chưa sinh lại notebook —
chạy lại `build_colab_studio.py` rồi commit **cả hai** file.

### A4. Web (demo Cloudflare, không liên quan model WAI)

```bash
cd web && npm ci && npm test
```

**Mong đợi:** `# tests 15` · `# pass 15` · `# fail 0`.
Nếu báo `Cannot find package 'tsx'` thì chỉ là thiếu `node_modules` → chạy `npm ci` trước.

---

## B. Kiểm tra trong Colab — KHÔNG tốn GPU (2 phút)

Mở notebook của nhánh này, **Run all** tới ô 7 (chưa cần tạo ảnh):

- [ ] Ô 2 in `✅ Thư viện Studio đã sẵn sàng`. Nếu có dòng
      `⚠️ Chưa cài được ultralytics` → auto-detailer sẽ báo lỗi khi bật, phần còn lại vẫn chạy.
- [ ] Thấy khung **🔎 Tự sửa mặt / bàn tay** (4 ô, mặc định **Tắt**) và khung
      **💅 Chi tiết mắt & móng** (4 ô, mặc định **Không thêm**).
- [ ] Tab **✎ Sửa vùng ảnh** → ô **Chi tiết cần sửa** có **9 lựa chọn**: Bàn tay,
      Móng tay / móng chân, Chân / bàn chân, Mắt, Khuôn mặt, Răng, Tóc, Da, Tùy chỉnh.
- [ ] Chọn lần lượt 5 vùng mới → mỗi lần bấm **Thêm gợi ý sửa vùng vào prompt đang hiển thị**:
      hai ô dài ra đúng cặp thẻ của vùng đó; **bấm lần hai không nhân đôi**.
- [ ] Khung 💅 để cả 4 ô `Không thêm` rồi bấm nút → báo
      `Chưa chọn chi tiết nào…` và **không** đổi ô prompt.
- [ ] Chọn `Hai màu (heterochromia)` + `Hạnh nhân (almond)` + `Đỏ` + `Đen` → bấm nút.
      Ô prompt phải có đúng:
      `heterochromia, multicolored eyes, almond-shaped nails, red nails, nail polish, painted toenails, black nail polish`
      — và **ô negative không đổi**.
- [ ] Xóa tay một thẻ trong ô prompt → nó **không tự quay lại** khi bấm tạo ảnh
      (nguyên tắc không thẻ ngầm).
- [ ] Bấm **🩺 Kiểm tra prompt & thông số** → đọc số token; quá 75 thì bỏ bớt thẻ.
- [ ] Bấm **Sắp xếp prompt theo thứ tự chuẩn** → thẻ mắt/móng nằm trong nhóm **Ngoại hình**.

---

## C. Kiểm tra trên GPU (phần quyết định — code không tự chứng minh được)

Cần T4 trở lên. Giữ **cùng seed** giữa các lượt để chỉ một biến thay đổi.

### C1. Auto-detailer (phép thử B5)

Prompt chân dung B4, seed `2024`, 1024x1024, 28 bước, CFG 6:

- [ ] Lượt 1: detailer **Tắt** → ghi lại mặt/tay lỗi gì.
- [ ] Lượt 2: **Mặt + tay**, strength 0.4, ngưỡng 0.3, 2 vùng, cùng seed.
      Thanh trạng thái phải có `auto-detailer đã sửa mặt … · tin cậy 0,xx`.
- [ ] Lượt 3: **Mặt** thôi, strength 0.3 (nếu lượt 2 làm đổi nét mặt).
- [ ] So 3 ảnh ở 100%: ngón tay/mắt khá hơn mà **bố cục, nền, áo, tông màu không đổi**,
      không có viền cứng quanh vùng sửa.
- [ ] (Tuỳ chọn) `sha256sum /content/wai_detailer_cache/*.pt` phải khớp bảng trong `README.md`.

### C2. Kiểu móng + màu sơn (phép thử B6)

Prompt cận tay, seed `3110`, 1024x1024, 28 bước, CFG 6:

```text
masterpiece, best quality, amazing quality, 1girl, solo, adult woman, close-up, hands, fingers spread, hand on hip, elegant dress, indoor cafe background, warm light, cel shading, anime illustration, absurdres
```

- [ ] Lượt 1: `Tự nhiên, ngắn` + `Không sơn`.
- [ ] Lượt 2: `Hạnh nhân (almond)` + `Đỏ`.
- [ ] Lượt 3: `Dài nhọn (stiletto)` + `Kim tuyến`.
- [ ] Hình dạng móng đổi đúng ý? Vẫn **5 ngón**? Màu sơn đều?
- [ ] Lượt 4: thêm **auto-detailer = Tay** (strength 0.35) nếu ngón/móng còn lỗi.

### C3. Ghi lại

```text
seed | kích thước | steps | CFG | LoRA + weight | detailer (mức/strength/vùng) | thẻ mắt+móng | nhận xét
```

---

## D. Nếu có lỗi, gửi lại đúng những thứ này

1. Traceback **nguyên văn** của ô bị lỗi (ô 2/4/5/6/7), che thông tin riêng.
2. Dòng `Chế độ:` và VRAM ở ô 6.
3. Prompt + negative **nguyên văn** hai ô lúc bấm tạo (bật metadata PNG rồi đọc trong file).
4. Seed, kích thước, steps, CFG, lựa chọn detailer và 4 ô mắt/móng.
5. Ảnh PNG (nếu có) — đừng gửi link `gradio.live` công khai.

**Điều test CPU KHÔNG chứng minh được:** detector dò đúng mặt/tay trên ảnh của bạn,
thẻ móng/màu mắt ra đúng ý trên WAI v17, VRAM/thời gian thật trên T4. Ba thứ đó chỉ
đo được ở mục C.
