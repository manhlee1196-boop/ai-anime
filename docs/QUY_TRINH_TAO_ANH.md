# Quy trình tạo ảnh chuẩn trong WAI Studio (ô 7 của notebook Colab)

Tài liệu này giải thích accordion **🧭 Quy trình chuẩn · khung prompt + negative tối ưu** và cách dùng nó cùng các tính năng khác của Studio. Mọi con số dưới đây lấy **trực tiếp từ mã** trong `colab/studio.py` (được nhúng nguyên văn vào ô 7 của `WAI_Illustrious_Studio_Colab.ipynb`).

> **Thư viện prompt trên điện thoại:** danh sách prompt trong accordion 📚 là **danh sách chạm (radio) cuộn được** — chạm một dòng là nạp ngay, hoặc chọn dòng rồi bấm **⬇️ Nạp prompt đã chọn**.

> **Nguyên tắc quan trọng:** cả ba nút trong accordion **chỉ ghi nội dung hiển thị** vào ô *Prompt gửi model*, ô *Negative gửi model* hoặc ô báo cáo. Không có thẻ nào được ghép ngầm khi bạn bấm tạo ảnh — bạn xem, sửa hoặc xóa trước khi tạo.

---

## 1. Ba nút trong **🧭 Quy trình chuẩn**

### 1.1. Sắp xếp prompt theo thứ tự chuẩn
Đọc prompt hiện tại, tách thành từng thẻ (phân tách bằng dấu phẩy), rồi:

- **Khử trùng lặp** theo phần lõi của thẻ (bỏ trọng số `(thẻ:1.1)` và ngoặc khi so sánh; ví dụ `(long hair:1.1)` và `long hair` bị coi là một).
- **Phân nhóm** mỗi thẻ vào 12 nhóm theo đúng thứ tự ưu tiên CLIP:

  `Chất lượng → Nhãn phân loại → Chủ thể → Ngoại hình → Trang phục → Tư thế/hành động → Thẻ khác của bạn → Bố cục/góc máy → Bối cảnh → Ánh sáng/màu → Phong cách → Độ nét (cuối prompt)`

  Thẻ không khớp nhóm nào nằm ở nhóm *Thẻ khác của bạn*, giữ nguyên vị trí tương đối.
- **Thêm thẻ neo còn thiếu** của khung đã chọn — chỉ thêm khi nhóm đó **đang trống**, để không nhồi thừa thẻ chất lượng.
- Trả về chuỗi mới **kèm ghi chú**: số thẻ đã sắp xếp, phân nhóm, thẻ neo đã thêm, thẻ trùng đã bỏ.
- Nếu kết quả dài hơn 2200 ký tự, nút báo lỗi và **không** ghi gì (bỏ bớt thẻ rồi bấm lại).

### 1.2. Nạp negative đã chọn
Chọn một trong **8 bộ negative** ở dropdown, chọn cách áp dụng rồi bấm nút:

- **Ghi đè ô negative** — thay toàn bộ nội dung ô negative bằng bộ đã chọn.
- **Nối thêm thẻ còn thiếu** — chỉ thêm những thẻ chưa xuất hiện trong ô hiện tại.

Bảng bộ negative (số thẻ lấy từ `NEGATIVE_PRESETS`):

| Nhãn trong dropdown | Dùng khi | Số thẻ |
| --- | --- | ---: |
| Chuẩn nhà phát hành WAI v17 (ngắn nhất) | Mặc định cho hầu hết ảnh; đúng khuyến nghị nhà phát hành | 5 |
| Illustrious chuẩn · chất lượng + lỗi vẽ | Muốn negative đầy hơn cho lỗi vẽ/chất lượng | 15 |
| Sửa tay / chân / tỷ lệ cơ thể | Ảnh hay lỗi bàn tay, bàn chân, tỷ lệ | 17 |
| Giữ chất anime 2D (chống 3D/thực) | Ảnh bị ra chất 3D/nhựa/thực | 14 |
| Chân dung · mặt, mắt, răng | Chân dung cận mặt | 17 |
| Phong cảnh · không có nhân vật | Phong cảnh, tránh lọt người | 16 |
| An toàn nội dung (mọi nhân vật trưởng thành) | Muốn đẩy các thẻ nội dung người lớn/`loli`/`shota` sang negative | 12 |
| Inpaint / sửa vùng (rất ngắn) | Sửa vùng nhỏ, denoise thấp, cần negative ngắn | 8 |

Bộ **Chuẩn nhà phát hành WAI v17** = `bad quality, worst quality, worst detail, sketch, censor` — theo khuyến nghị “đừng thêm quá nhiều thẻ chất lượng/thẩm mỹ và đừng viết negative quá dài vì sẽ làm giảm chất lượng, ảnh dễ mờ” (xem [Nguồn](#nguồn)).

### 1.3. 🩺 Kiểm tra prompt & thông số
Chỉ đọc và báo cáo, **không sửa gì**. Các mục được kiểm:

| Nhóm kiểm tra | Ngưỡng trong mã | Ý nghĩa |
| --- | --- | --- |
| Độ dài prompt | ước lượng token theo khối **75 token** của SDXL (`<= 75` = ✅, `<= 150` = ⚠️, hơn nữa = ⚠️ nặng) | Phần vượt khối bị đẩy sang khối 75 token sau và mất ưu tiên; gợi ý dùng `BREAK` (viết hoa) để tách khối có chủ đích |
| Thẻ chất lượng | > 3 thẻ = ⚠️, 0 thẻ = gợi ý thêm `masterpiece, best quality, amazing quality` | Nhà phát hành cảnh báo thừa thẻ chất lượng/thẩm mỹ làm mờ ảnh |
| Thẻ trùng lặp | mọi thẻ trùng theo phần lõi | Lặp thẻ tốn token mà không tăng trọng số |
| Thẻ vừa dương vừa âm | thẻ xuất hiện ở cả prompt và negative | Hai lệnh ngược nhau làm model dao động |
| Cú pháp Pony | `score_…`, `source_…` | WAI-illustrious không dùng hệ Pony |
| Thẻ chữ/ký hiệu | `text`, `watermark`, `signature`, `logo`, `username`, `artist name` trong prompt dương | Nên đưa sang negative nếu không muốn chữ trong ảnh |
| `detailed eyes` | có trong prompt dương | Được cộng đồng báo gần như không tác dụng trên họ Illustrious; dùng LoRA mắt + `perfect eyes` hoặc mô tả cụ thể |
| Thẻ phong cách | thiếu `cel shading`, `anime illustration`… | Dễ ra chất 3D/nhựa khi thiếu thẻ chất lượng + phong cách |
| Số lượng chủ thể | thiếu `1girl`, `1boy`, `solo`, `no humans`… | Thiếu dễ thừa nhân vật |
| Trọng số | ngoài khoảng **0.5–1.2** (nhất là > 1.2) | Quá 1.2 dễ cháy nét/mất bố cục; ưu tiên mô tả bằng từ ngữ |
| Dấu gạch dưới | `long_hair`… | Illustrious đọc cả dạng cách (`long hair`) và tốn ít token hơn |
| Negative | trống = gợi ý tối thiểu; > 40 thẻ hoặc > 150 token = ⚠️ | Negative quá dài làm giảm chất lượng theo nhà phát hành |
| Thông số | steps **15–30**, CFG **5–7**, kích thước theo preset, hires strength **0.35–0.5** | Lệch khỏi khuyến nghị sẽ được nhắc |

Số token là **ước lượng heuristic** (mỗi từ ≈ 1 token, cộng thêm cho `_`, số, dấu câu, từ dài) — đủ để cảnh báo vượt khối 75 token, không phải số token chính xác của tokenizer CLIP.

---

## 2. Khung prompt theo loại ảnh

Dropdown **Khung prompt theo loại ảnh** dùng cho nút *Sắp xếp prompt theo thứ tự chuẩn*. Bốn khung và thẻ neo (chỉ thêm khi nhóm đang trống):

| Khung | Thẻ neo thêm khi thiếu |
| --- | --- |
| Nhân vật · 1 nhân vật | `masterpiece, best quality, amazing quality` · `1girl, solo` · `anime illustration, cel shading` · `absurdres` |
| Chân dung cận mặt | như trên + `close-up, looking at viewer` · `soft lighting` |
| Phong cảnh · không nhân vật | như trên + `no humans` · `wide shot` · `detailed background, scenery` · `anime background` |
| Hành động / key visual | như trên + `dynamic pose` · `dynamic angle, depth of field` · `dramatic lighting` · `anime key visual, cel shading` |

---

## 3. Thông số khuyến nghị (nhà phát hành WAI v17)

| Thông số | Trong UI/notebook | Khuyến nghị v17 |
| --- | --- | --- |
| Sampler | Euler a (cố định trong mã) | Euler a |
| Steps | 10–45 (mặc định 25) | **15–30** |
| CFG | 1–12 (mặc định 6) | **5–7** |
| Kích thước gốc | preset 512² … 1024×1344 | ≥ 1024×1024, ví dụ 1024×1344 |
| Hires fix | `Tắt` / `1.5×` / `2×`, strength 0.2–0.7 (mặc định 0.4) | `1.5`, denoise **0.35–0.5** |
| Prompt dương | do bạn viết | mở đầu `masterpiece, best quality, amazing quality` (không nhiều hơn) |
| Negative | do bạn viết (nút nạp nhanh) | `bad quality, worst quality, worst detail, sketch, censor` |

Studio khác nhà phát hành ở hai điểm có chủ đích: sampler được ghim Euler a và bước hires dùng ảnh → ảnh với tỷ lệ 1.5×/2× thay cho upscaler R-ESRGAN 4x+ Anime6B.

---

## 4. Gợi ý phong cách (🎨) và auto-detailer

**🎨 Gợi ý phong cách** chỉ **thêm thẻ mô tả vào hai ô đang hiển thị** (không thay thế prompt, không phải preset phong cách):

- **Màu mắt** (17 lựa chọn) → `blue eyes`, `aqua eyes`, …
- **Kiểu dáng móng tay** (10) → `natural short nails`, `almond nails`, …
- **Màu sơn móng tay** (13) / **Màu sơn móng chân** (10) → `red nails`, `black toenails`, …

**Tự sửa mặt/tay (auto-detailer)** — tùy chọn, cần `ultralytics` (ô 2 cài, chỉ cảnh báo nếu lỗi):

- Mức: `Tắt` / `Mặt` / `Tay` / `Mặt + tay` (mặc định Tắt).
- Tham số: strength **0.2–0.7** (mặc định 0.4), ngưỡng phát hiện **0.1–0.9** (mặc định 0.3, thấp = dễ tìm hơn), số vùng tối đa **1–4** (mặc định 2).
- Weight YOLOv8 tải một lần vào `/content/wai_detailer_cache`: `face_yolov8n.pt` (6.230.011 byte, SHA-256 `70b640f8…`), `hand_yolov8n.pt` (6.237.883 byte, SHA-256 `3991202e…`) từ `Bingsu/adetailer` @ `c310c216`; sai hash thì file bị xóa và không nạp.
- Cách dùng: bật mức, tạo ảnh như bình thường. Studio phát hiện vùng, nới bounding box 25%, phóng vùng nhỏ lên tối thiểu 512 px, inpaint lại đúng vùng rồi dán về với mép mềm 12 px. Đây là gợi ý chỉnh sửa — **không đảm bảo** hết mọi lỗi ngón/mặt.

---

## 5. Nguồn

Các khuyến nghị ở mục 3 và bảng negative lấy từ hướng dẫn của nhà phát hành WAI-illustrious (đã đối chiếu công khai khi viết tài liệu này):

- Trang model tại commit được ghim: [LyliaEngine/waiIllustriousSDXL_v170 · README](https://huggingface.co/LyliaEngine/waiIllustriousSDXL_v170/blob/main/README.md) — Steps 15–30, CFG 5–7, Euler a, prompt dương `,masterpiece,best quality,amazing quality,`, negative `bad quality,worst quality,worst detail,sketch,censor,`, cảnh báo không thêm quá nhiều thẻ chất lượng/thẩm mỹ và không viết negative quá dài, 4 nhãn an toàn `general/sensitive/nsfw/explicit`.
- Bản sao nội dung trang model Civitai (version 2883731): [civarchive.com/models/827184](https://civarchive.com/models/827184?modelVersionId=2883731) — cùng nội dung khuyến nghị, kèm mục hires `1.5`, denoise 0.35–0.5.
- Bài tổng hợp thực hành v17: [lilting.ch — WAI-Illustrious v17 hands-on](https://lilting.ch/en/articles/wai-illustrious-v17-review) — so sánh thông số v15–v16 (25–40 steps) với v17 (15–30 steps) và cách dùng cùng negative ngắn.

Phần thứ tự thẻ, khối 75 token và các thẻ Illustrious bám tốt là **thực hành phổ biến của cộng đồng Illustrious** được ghi lại trong mã (`colab/studio.py`, khối *Căn cứ đã đối chiếu*), không phải tuyên bố chính thức của nhà phát hành.

## 6. Chưa được kiểm chứng

Mã và tài liệu này đã qua kiểm thử CPU (`python -m unittest discover -s tests -v` — 58 test, 1 bỏ qua vì cần PyTorch thật) và `scripts/build_colab_studio.py` tái tạo notebook **chính xác**. Tuy nhiên **chưa** chạy trên GPU Colab: chưa đo chất lượng ảnh, tốc độ, VRAM hay hiệu quả thật của auto-detailer/hires fix. Xem `VERIFICATION.md` để biết đầy đủ giới hạn.
