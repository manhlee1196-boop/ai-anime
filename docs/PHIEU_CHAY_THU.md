# Phiếu chạy thử · quy trình prompt/negative mới

Phiếu này để bạn tự kiểm tra 3 công cụ mới và đo xem negative có thật sự giúp ảnh đẹp
hơn trên GPU của bạn. **Phần A không tốn GPU** (chỉ cần giao diện mở lên). Phần B/C mới
cần tạo ảnh.

> Thời gian: Phần 0 ≈ 10 phút (chủ yếu chờ tải checkpoint 6,94 GB) · Phần A ≈ 3 phút ·
> Phần B ≈ 20 phút GPU · Phần C ≈ 10 phút GPU.
> Đánh dấu `[x]` khi đạt. Nếu một bước sai, dừng lại và gửi tôi nội dung ở **Phần E**.

---

## Phần 0 · Mở đúng bản có tính năng mới

- [ ] Mở notebook **của nhánh này** (bản `main` chưa có 3 nút mới):

  <https://colab.research.google.com/github/manhlee1196-boop/ai-anime/blob/arena/01a0f38a-ai-anime/WAI_Illustrious_Studio_Colab.ipynb>

  Nếu link báo lỗi: vào GitHub → nhánh `arena/01a0f38a-ai-anime` → mở
  `WAI_Illustrious_Studio_Colab.ipynb` → **Open in Colab**; hoặc tải file `.ipynb` về máy
  rồi trong Colab chọn **File → Upload notebook**.
- [ ] **Runtime → Change runtime type → T4 GPU** (hoặc mạnh hơn) → **Runtime → Run all**.
- [ ] Ô 2 in `✅ Thư viện Studio đã sẵn sàng`.
- [ ] Ô 4 và ô 5 in **SHA-256 đã xác minh** của checkpoint WAI v17 và các LoRA đã bật.
- [ ] Ô 6 in VRAM trước/sau và `Chế độ sau khi nạp` (GPU trực tiếp / CPU offload).
- [ ] Ô cuối in `Running on public URL` → mở link `https://….gradio.live`.
- [ ] Trong giao diện thấy accordion **🧭 Quy trình chuẩn · khung prompt + negative tối ưu**
      (ngay dưới nút *Thêm trigger `perfect eyes`*). **Không thấy = bạn đang mở nhầm bản cũ.**

> Link `gradio.live` **không có đăng nhập**: ai biết link đều dùng được GPU của bạn. Đừng
> chia sẻ, và dừng runtime khi xong.

---

## Phần A · Kiểm tra 3 nút mới (không tạo ảnh, không tốn GPU)

Mở accordion **🧭 Quy trình chuẩn**.

### A1. Nút "Sắp xếp prompt theo thứ tự chuẩn"

- [ ] Xóa sạch ô **Prompt gửi model**, dán đúng prompt cố ý viết lộn xộn này:

```text
cherry blossoms, 1girl, detailed eyes, score_9, standing in a modern office, soft sunlight, long_hair, masterpiece, best quality, amazing quality, highest quality, cel shading, 1girl, text, (blue eyes:1.4)
```

- [ ] Để **Khung prompt theo loại ảnh** = *Nhân vật · 1 nhân vật*, bấm
      **Sắp xếp prompt theo thứ tự chuẩn**.
- [ ] Ô prompt phải thành **đúng** chuỗi này (15 thẻ, `1girl` trùng đã bị bỏ, `absurdres`
      được thêm vào cuối):

```text
masterpiece, best quality, amazing quality, highest quality, 1girl, detailed eyes, (blue eyes:1.4), standing in a modern office, score_9, long_hair, text, cherry blossoms, soft sunlight, cel shading, absurdres
```

- [ ] Dòng ghi chú bên dưới phải báo: `Chất lượng (4), Chủ thể (1), Ngoại hình (2),
      Tư thế / hành động (1), Thẻ khác của bạn (3), Bối cảnh (1), Ánh sáng / màu (1),
      Phong cách (1)` · *Thêm thẻ neo còn thiếu: `absurdres`* · *Bỏ 1 thẻ trùng: `1girl`*.
- [ ] Bấm nút đó **lần nữa** → chuỗi **không đổi** (idempotent).
- [ ] `(blue eyes:1.4)` vẫn còn nguyên dấu ngoặc và trọng số (công cụ không phá cú pháp).

### A2. Nút "🩺 Kiểm tra prompt & thông số" (prompt đang lỗi)

- [ ] Đổi thông số: **steps = 40**, **CFG = 9**, **Kích thước = 512x512**,
      **Độ phân giải cao = 1.5×**, **Hires strength = 0.65**.
- [ ] Thêm vào cuối ô *Negative gửi model*: `, masterpiece, 1girl`
      (cố ý tạo mâu thuẫn với prompt).
- [ ] Bấm **🩺 Kiểm tra prompt & thông số** → báo cáo phải có **đúng 5 dòng ⚠️**:
      1. thừa thẻ chất lượng (4 thẻ),
      2. thẻ trùng `1girl`,
      3. `1girl`, `masterpiece` vừa nằm trong prompt vừa nằm trong negative,
      4. `score_9` là cú pháp họ Pony,
      5. trọng số `blue eyes` ngoài khoảng an toàn.
- [ ] Và **7 dòng ℹ️**: `text` nằm nhầm bên prompt dương · `detailed eyes` vô dụng trên
      Illustrious · 2 thẻ dùng dấu gạch dưới · steps 40 · CFG 9 · 512×512 · hires 0.65.
- [ ] Dòng đầu ghi `15 thẻ · ≈35 token (ước lượng)`.

### A3. Nút "Nạp negative đã chọn"

- [ ] **Negative tối ưu theo mục đích** = *Chuẩn nhà phát hành WAI v17 (ngắn nhất)*,
      **Cách áp dụng** = *Ghi đè ô negative*, bấm **Nạp negative đã chọn**.
- [ ] Ô negative phải thành đúng 5 thẻ:

```text
bad quality, worst quality, worst detail, sketch, censor
```

- [ ] Đổi sang *Sửa tay / chân / tỷ lệ cơ thể*, cách áp dụng = *Nối thêm thẻ còn thiếu*,
      bấm nạp → ô negative dài ra nhưng **không lặp** thẻ đã có.
- [ ] Đổi lại *Ghi đè*, chọn *Illustrious chuẩn* → ô negative đúng 15 thẻ.

### A4. Vòng lặp "sửa đến khi sạch"

- [ ] Dán prompt sạch này vào ô prompt:

```text
masterpiece, best quality, amazing quality, 1girl, solo, adult woman, long dark hair, gentle smile, white shirt, standing under cherry blossoms, petals falling, upper body, soft rim light, spring dusk, cel shading, anime illustration, absurdres
```

- [ ] Negative = *Chuẩn nhà phát hành*; thông số: steps **28**, CFG **6**, kích thước
      **832x1216**, hires **1.5×**, strength **0.4**.
- [ ] Bấm 🩺 → báo cáo **không còn ⚠️/ℹ️**, chỉ 4 dòng ✅:

```text
✅ Prompt ≈ 35 token, vừa trong một khối 75 token của SDXL.
✅ Đã có thẻ phong cách/chất liệu vẽ trong prompt.
✅ Đã khai báo chủ thể: `1girl`, `solo`.
✅ Negative 5 thẻ — độ dài hợp lý.
```

### A5. Không có thẻ ngầm

- [ ] Gõ prompt tùy ý (ví dụ `1girl, solo, watercolor`), xóa mọi thứ bạn không muốn,
      bật **Nhúng prompt vào metadata PNG**, tạo 1 ảnh, tải PNG về.
- [ ] Xem metadata PNG (Properties → Details, hoặc kéo ảnh vào <https://exif.tools>) →
      prompt/negative trong file phải **giống nguyên văn** hai ô lúc bạn bấm tạo.

---

## Phần B · Tạo ảnh thật: 4 phép thử A/B (đây mới là phần quyết định)

Mỗi phép thử **giữ nguyên seed** để chỉ negative/thông số thay đổi. Ghi kết quả vào bảng
ở Phần D. Chụp/ghi lại seed bằng cách bật metadata PNG.

### B1. Chân dung · so 3 bộ negative

Prompt (18 thẻ, ≈39 token, đã kiểm không có ⚠️), kích thước **832x1216**, steps 28,
CFG 6, **seed 12345**, hires Tắt:

```text
masterpiece, best quality, amazing quality, 1girl, solo, adult woman, long dark hair, gentle smile, white dress, holding a transparent umbrella, standing under cherry blossoms, petals falling, upper body, soft rim light, spring dusk, cel shading, anime illustration, absurdres
```

- [ ] Lượt 1 — negative `publisher` (5 thẻ).
- [ ] Lượt 2 — negative `core` (15 thẻ).
- [ ] Lượt 3 — negative `portrait` (17 thẻ).
- [ ] So 3 ảnh ở 100%: mắt, răng, ngón tay cầm ô, nhiễu nền. Bộ nào tốt nhất cho **bạn**?

### B2. Chống chất 3D/nhựa

- [ ] Cùng prompt B1 + seed 12345, negative `anime2d` (14 thẻ, có `realistic, 3d, plastic skin`).
- [ ] So với lượt 1 của B1: da/tóc có bớt "nhựa" không? Nét vẽ có rõ là 2D hơn không?

### B3. Phong cảnh · chặn người lọt khung

Prompt (13 thẻ, ≈33 token), **1216x832**, steps 26, CFG **5.5**, seed 777:

```text
masterpiece, best quality, amazing quality, no humans, wide shot, coastal village on a hillside at dawn, terraced gardens, enormous pastel clouds, sea shimmering, expansive detailed background, soft morning light, anime background, absurdres
```

- [ ] Lượt 1 — negative `core`. Lượt 2 — negative `scene` (có `1girl, 1boy, solo, people, crowd`).
- [ ] Có nhân vật/bóng người lọt vào ảnh không? Phối cảnh đường chân trời ổn không?

### B4. Bàn tay · negative + LoRA + inpaint

Prompt (18 thẻ, ≈45 token), **1024x1024**, steps 28, CFG 6, seed 2024:

```text
masterpiece, best quality, amazing quality, 1girl, solo, adult woman, short black hair, white shirt, both hands clasped in front of chest, holding a coffee cup, fingers interlaced, sitting at a desk, upper body, modern office, warm desk lamp light, cel shading, anime illustration, absurdres
```

- [ ] Lượt 1 — negative `core`, **tắt** LoRA Anatomy.
- [ ] Lượt 2 — negative `anatomy` (17 thẻ), **bật** LoRA Anatomy weight 0.55, cùng seed.
- [ ] Đếm ngón tay ở 100%. Nếu còn lỗi: bấm **Dùng ảnh mới nhất để sửa vùng** → tô
      **chỉ** vùng bàn tay → chọn `hands` → **Thêm gợi ý sửa vùng vào prompt** → sửa/xóa
      thẻ vừa thêm → strength **0.45** → chạy.
- [ ] So ảnh trước/sau inpaint: ngón tay khá hơn mà **mặt và áo không đổi**?

---

## Phần C · Hires & phóng to

- [ ] C1: prompt B1, seed 12345, hires **Tắt** rồi **1.5× / strength 0.4** → chi tiết tóc/vải
      tăng, bố cục và nét mặt **không đổi**.
- [ ] C2: cùng seed, strength **0.35** so với **0.5** → 0.5 thêm chi tiết nhưng có đổi nét không?
- [ ] C3: tab **⤢ Phóng to ảnh** với ảnh 1024×1024 vừa tạo, hệ số **2×** → ảnh ra
      2048×2048 (hoặc bị giảm hệ số kèm thông báo nếu vượt ≈4,2 MP).
- [ ] C4: nếu OOM → giảm về 1.5× hoặc kích thước gốc nhỏ hơn, và ghi lại VRAM ở ô 6.

---

## Phần D · Bảng ghi kết quả (điền khi chạy)

| Phép thử | Negative | Seed | Thông số | Nhận xét (mắt/tay/nét/nhiễu) | Chọn |
| --- | --- | --- | --- | --- | --- |
| B1-1 | publisher | 12345 | 832x1216, 28, CFG 6 | | |
| B1-2 | core | 12345 | như trên | | |
| B1-3 | portrait | 12345 | như trên | | |
| B2 | anime2d | 12345 | như trên | | |
| B3-1 | core | 777 | 1216x832, 26, CFG 5.5 | | |
| B3-2 | scene | 777 | như trên | | |
| B4-1 | core, tắt Anatomy | 2024 | 1024x1024, 28, CFG 6 | | |
| B4-2 | anatomy, Anatomy 0.55 | 2024 | như trên | | |
| B4-3 | inpaint hands 0.45 | 2024 | — | | |
| C1/C2 | — | 12345 | hires 0.35 / 0.4 | | |

Ghi thêm: chế độ nạp ở ô 6 (GPU trực tiếp / CPU offload), thời gian mỗi ảnh, và VRAM.

---

## Phần E · Nếu có lỗi, gửi tôi đúng những thứ này

1. **Traceback đầy đủ** của ô báo lỗi (ô 2/4/5/6/7/8) — **che link `gradio.live`** và thông tin riêng.
2. Với lỗi trong giao diện: tên nút đã bấm + **nguyên văn hai ô prompt/negative** +
   steps/CFG/kích thước/hires/LoRA weight + nội dung ô báo cáo 🩺.
3. Với ảnh xấu: file PNG (kèm metadata nếu bạn bật) + seed + mô tả chỗ lỗi (khoanh càng tốt).
4. Dòng `Chế độ sau khi nạp` và VRAM ở ô 6.

**Nhắc lại trước khi tắt phiên:** tải **tất cả** ảnh cần giữ về máy — ảnh, checkpoint và
LoRA trong `/content` mất khi runtime ngắt, và link `gradio.live` cũng chết theo.

---

## Nếu bạn không có GPU ngay lúc này

Ba công cụ mới đã được kiểm thử CPU trong repo (67 test Python đạt), nên bạn có thể kiểm
phần logic mà không cần Colab:

```bash
python -m venv .venv && . .venv/bin/activate
pip install "gradio==6.15.2" "pillow>=10" nbformat packaging "huggingface_hub==0.36.2" torch
python -m unittest discover -s tests      # kỳ vọng: Ran 67 tests ... OK
python scripts/build_colab_studio.py      # sinh lại notebook (phải khớp bản trong repo)
```

Phần **chất lượng ảnh** thì không có cách nào thay thế GPU thật.
