# Quy trình tạo ảnh chuyên nghiệp · WAI-illustrious v17 (Colab Studio)

Tài liệu này là kết quả đối chiếu cách làm việc của người tạo ảnh AI có kinh nghiệm
(nhà phát hành model, hướng dẫn Illustrious XL, quy trình production của hoạ sĩ AI)
với quy trình hiện có trong repo, rồi **chuẩn hoá thành 9 bước** và bổ sung ba công cụ
mới trong giao diện Studio.

> **Muốn chạy thử ngay?** Mở **[PHIEU_CHAY_THU.md](PHIEU_CHAY_THU.md)** — phiếu kiểm tra
> từng bước, có kết quả mong đợi để đối chiếu và 5 phép thử A/B trên GPU.

**Trạng thái kiểm chứng:** các con số khuyến nghị (steps, CFG, hires strength, thứ tự
thẻ, negative) lấy từ nhà phát hành WAI-illustrious v17 và hướng dẫn cộng đồng
Illustrious XL — xem mục **Nguồn**. Chúng là *khuyến nghị đã công bố*, không phải kết
quả đo trên GPU; repo này **chưa chạy thật trên GPU Colab** (xem [`../VERIFICATION.md`](../VERIFICATION.md)).
Ba công cụ mới (sắp xếp prompt, negative tối ưu, kiểm tra prompt) đã qua kiểm thử CPU
trong `tests/`; chất lượng ảnh thật vẫn phải đánh giá bằng mắt trên máy của bạn.

---

## 0. SOP 9 bước (bảng điều hành)

| # | Bước | Làm gì | Công cụ trong Studio | Tiêu chí "đạt" để sang bước sau |
| --- | --- | --- | --- | --- |
| 1 | Chuẩn bị phiên | Chạy ô 1→6, kiểm SHA-256, xem VRAM/chế độ nạp | Notebook | Ô 2 in `✅ Thư viện Studio đã sẵn sàng`, ô 6 in chế độ + VRAM |
| 2 | Brief & tham khảo | Viết 1 câu mô tả ảnh + chọn 2–3 ảnh tham khảo bố cục/ánh sáng | Ngoài Studio | Có chủ thể, hành động, bối cảnh, ánh sáng, phong cách, tỷ lệ khung |
| 3 | Viết prompt | Viết theo **thứ tự chuẩn 10 nhóm**, ≤ 75 token | Nút **Sắp xếp prompt theo thứ tự chuẩn** | Prompt đọc được theo nhóm, không trùng thẻ |
| 4 | Chọn negative | Chọn 1 bộ theo **mục đích**, thêm ≤ 5 thẻ riêng | Nút **Nạp negative đã chọn** (8 bộ) | Negative ≤ ~40 thẻ, không mâu thuẫn prompt |
| 5 | Kiểm tra | Chạy bộ kiểm tra prompt/thông số | Nút **🩺 Kiểm tra prompt & thông số** | Không còn mục ⚠️ |
| 6 | Dò seed | Tạo 3–4 ảnh cùng prompt, đổi seed, chọn 1 | *Văn bản → ảnh*, `Số ảnh` 1–4, seed | Có 1 ảnh đúng bố cục; **ghi lại seed** |
| 7 | Tự sửa mặt/tay | Bật auto-detailer, strength 0,4 · ngưỡng 0,3 · 1–2 vùng | Khung **🔎 Tự sửa mặt / bàn tay** | Mặt/tay hết lỗi mà bố cục, nền và nét mặt không đổi (mục 7) |
| 8 | Phóng nét | Hires 1.5× (2× nếu cần), strength 0.35–0.5 | Khung **🔍 Ảnh độ phân giải cao** / tab **⤢ Phóng to ảnh** | Chi tiết tăng, bố cục/nét mặt không đổi (mục 8) |
| 9 | Sửa cục bộ + QA | Inpaint vùng tay/mắt/chân còn sót, chạy checklist | Tab **✎ Sửa vùng ảnh** + nút gợi ý | Qua checklist mục 10, tải PNG về máy |

Nguyên tắc xuyên suốt: **thay đổi một biến mỗi lượt** (hoặc prompt, hoặc CFG, hoặc
seed, hoặc LoRA weight). Đổi nhiều thứ cùng lúc thì không biết cái gì đã tạo ra khác
biệt, và không lặp lại được kết quả đẹp.

---

## 1. Chuẩn bị phiên Colab (làm đúng một lần)

1. `Runtime → Change runtime type → T4 GPU` (hoặc mạnh hơn) → `Run all`.
2. Ô 2 phải in `✅ Thư viện Studio đã sẵn sàng`; chỉ có dòng `ERROR: pip's dependency
   resolver...` thì vẫn được coi là cảnh báo.
3. Ô 4/5 in SHA-256 đã xác minh của checkpoint và LoRA; ô 6 in `Chế độ sau khi nạp`
   (GPU trực tiếp hay CPU offload) và VRAM.
4. Quyết định LoRA **trước** khi tạo ảnh: Anatomy Helper (tay/chân/tư thế) và Perfect
   Eyes (mắt, trigger `perfect eyes`). Bật/tắt sau khi nạp model thì phải restart runtime.

> Mẹo chuyên nghiệp: coi LoRA như *công cụ sửa có kiểm soát*, không phải mặc định luôn
> bật. Với ảnh phong cảnh hoặc ảnh không thấy tay, tắt Anatomy Helper thường cho nét
> vẽ tự do hơn.

## 2. Brief & tài liệu tham khảo

Trước khi gõ prompt, viết 6 dòng này ra (giấy hoặc file ghi chú):

- **Chủ thể:** ai/mấy người, độ tuổi trưởng thành, giới tính, dáng người.
- **Hành động/tư thế:** đang làm gì, tay ở đâu, nhìn đâu.
- **Bối cảnh:** ở đâu, thời điểm nào, thời tiết.
- **Ánh sáng:** nguồn sáng, hướng, màu, cứng hay mềm.
- **Phong cách:** chất liệu vẽ (cel shading, watercolor, film still…), tham chiếu hoạ sĩ/thời kỳ nếu cần.
- **Khung hình:** cận mặt / nửa người / toàn thân / phong cảnh → chọn preset kích thước tương ứng.

Có ảnh tham khảo thì mô tả **bố cục và ánh sáng** của ảnh đó bằng từ khoá, đừng cố mô tả
cả bức ảnh. Illustrious học từ Danbooru nên từ khoá càng gần tag Danbooru càng chính xác.

## 3. Viết prompt theo thứ tự chuẩn

### 3.1 Ngân sách 75 token

SDXL/Illustrious mã hoá prompt theo **khối 75 token**; phần vượt quá bị đẩy sang khối
sau và mất ưu tiên. Hệ quả thực dụng:

- Giữ khoảng **20–40 thẻ quan trọng nhất**. Prompt càng dài thì mỗi thẻ càng yếu.
- Thẻ quan trọng nhất phải nằm **đầu** prompt.
- Muốn tách ý có chủ đích (ví dụ nhiều chủ thể), dùng `BREAK` **viết hoa** để kết thúc
  khối hiện tại.
- `long hair` tốn ít token hơn `long_hair`; Illustrious đọc được cả hai.

Nút **🩺 Kiểm tra prompt & thông số** in số token *ước lượng* (heuristic, không dùng
tokenizer CLIP thật) để bạn biết mình đang ở đâu so với mốc 75.

### 3.2 Thứ tự 10 nhóm (đúng thứ tự CLIP ưu tiên)

```
1. Chất lượng      masterpiece, best quality, amazing quality
2. Nhãn phân loại  general (hoặc để trống) — họ Illustrious dùng general/sensitive/nsfw/explicit
3. Chủ thể         1girl, solo  /  1boy, solo  /  2girls  /  no humans
4. Ngoại hình      long dark hair, blue eyes, gentle smile, glasses
5. Trang phục      white shirt, navy blazer, knee-length skirt
6. Tư thế/hành động sitting at a desk, holding documents, looking at viewer
7. Bố cục/góc máy   upper body, close-up, from above, depth of field, dynamic angle
8. Bối cảnh        modern office at night, rain on the window, city lights
9. Ánh sáng/màu    warm desk lamp light, soft rim light, pastel palette
10. Phong cách      cel shading, anime illustration  →  chốt bằng absurdres
```

Nút **Sắp xếp prompt theo thứ tự chuẩn** tự xếp lại thẻ của bạn theo đúng thứ tự này,
bỏ thẻ trùng, và **thêm thẻ neo còn thiếu** của khung đã chọn (nhân vật / chân dung /
phong cảnh / hành động). Kết quả ghi thẳng vào ô *Prompt gửi model* để bạn xem và sửa —
không có thẻ nào được thêm ngầm lúc tạo ảnh.

### 3.3 Nhấn mạnh có kỷ luật

- Dùng `(thẻ:1.05)`–`(thẻ:1.2)`; trên 1.2 dễ cháy nét, phá bố cục, kéo theo artefact.
- **Không lặp thẻ** để "tăng trọng số": chỉ tốn token.
- Ưu tiên mô tả bằng từ (`sharp focus on eyes`, `golden hour rim light`) thay vì cân
  nặng thẻ.
- Muốn *giảm* một thứ, đưa nó vào negative — hiệu quả hơn `(thẻ:0.5)`.

### 3.4 Tag Danbooru hay câu văn?

Illustrious XL (nên WAI v17) hỗ trợ cả hai, nhưng **tag Danbooru vẫn ổn định và chính
xác hơn**; câu văn dài dễ bị hiểu sai. Cách dùng lai hiệu quả nhất: tag cho mọi thứ có
tag, cụm mô tả ngắn cho những thứ Danbooru không có (`leaning against a glass wall at
dusk`). Viết prompt bằng **tiếng Anh** — dịch tự động sang ngôn ngữ khác làm rơi nghĩa.

### 3.5 Bốn khung mẫu

| Khung | Thẻ neo được thêm khi thiếu | Dùng khi |
| --- | --- | --- |
| **Nhân vật** | `masterpiece, best quality, amazing quality` · `1girl, solo` · `cel shading, anime illustration` · `absurdres` | Ảnh có một nhân vật, thấy rõ nửa người trở lên |
| **Chân dung cận mặt** | như trên + `close-up, looking at viewer` · `soft lighting` | Avatar, ảnh khoe mắt/tóc |
| **Phong cảnh** | chất lượng + `no humans` · `wide shot` · `detailed background, scenery` · `anime background` · `absurdres` | Background, key visual không người |
| **Hành động** | chất lượng + `1girl, solo` · `dynamic pose` · `dynamic angle, depth of field` · `dramatic lighting` · `anime key visual, cel shading` | Ảnh chuyển động, poster |

## 4. Negative prompt tối ưu

### 4.1 Bốn nguyên tắc

1. **Ngắn thắng dài.** Nhà phát hành WAI v17 ghi rõ: đừng thêm quá nhiều thẻ chất
   lượng/thẩm mỹ và **đừng viết negative quá dài**, vì sẽ *giảm chất lượng và làm ảnh
   mờ hơn*. Một danh sách 100 thẻ kiểu SD 1.5 là phản tác dụng trên model này.
2. **Chọn theo mục đích, không phải một bộ vạn năng.** Mỗi preset trong Studio cố ý
   ≤ 20 thẻ và nhắm một loại việc.
3. **Không mâu thuẫn với prompt dương.** Thẻ nào vừa có trong prompt vừa có trong
   negative sẽ khiến model dao động (bộ kiểm tra sẽ báo ⚠️).
4. **Họ Illustrious bám negative rất tốt** (khác họ Pony), nên negative là công cụ bố
   cục thật sự: muốn bỏ người khỏi phong cảnh thì đưa `1girl, 1boy` vào negative; muốn
   giữ nét 2D thì đưa `realistic, 3d` vào negative.

### 4.2 Tám bộ negative trong Studio

| Bộ (id) | Nội dung | Khi nào dùng |
| --- | --- | --- |
| `publisher` · Chuẩn nhà phát hành WAI v17 | `bad quality, worst quality, worst detail, sketch, censor` | Mặc định an toàn, ít rủi ro nhất; đúng bộ nhà phát hành công bố |
| `core` · Illustrious chuẩn | `worst quality, low quality, bad quality, lowres, jpeg artifacts, bad anatomy, bad hands, extra digit, fewer digits, watermark, signature, text, artistic error, very displeasing, oldest` | Bộ dùng hằng ngày: chất lượng + lỗi vẽ + lỗi giải phẫu nhẹ |
| `anatomy` · Tay/chân/tỷ lệ | `bad anatomy, bad hands, deformed hands, extra digit, fewer digits, fused fingers, conjoined fingers, extra limbs, missing limbs, extra arms, extra legs, bad proportions, bad perspective, deformed feet, extra toes, fused toes, long neck` | Tay/chân phức tạp; dùng kèm LoRA Anatomy hoặc inpaint |
| `anime2d` · Giữ chất 2D | `realistic, photorealistic, 3d, cgi, render, plastic skin, realistic skin texture, dull colors, monochrome, greyscale, sketch, traditional media, jpeg artifacts, bad anatomy` | Khi ảnh ra "nhựa"/3D — vấn đề hay gặp ở v17 khi thiếu thẻ chất lượng |
| `portrait` · Mặt, mắt, răng | `bad face, poorly drawn face, deformed eyes, asymmetrical eyes, cross-eyed, extra eyes, dull eyes, bad teeth, crooked teeth, skin blemishes, acne, bad anatomy, bad hands, extra digit, jpeg artifacts, watermark, text` | Ảnh cận mặt; kết hợp LoRA mắt + trigger `perfect eyes` |
| `scene` · Phong cảnh | `1girl, 1boy, solo, people, crowd, bad perspective, bad proportions, lowres, blurry, worst quality, low quality, jpeg artifacts, text, watermark, signature, logo` | Background/key visual; chặn người lọt khung và lỗi phối cảnh |
| `sfw` · An toàn nội dung | `nsfw, explicit, nude, loli, shota, child, aged_down, censored, censor bars, worst quality, low quality, jpeg artifacts` | WAI có nhãn `general/sensitive/nsfw/explicit`: đưa nhãn không mong muốn vào negative. **Không** thay thế bộ lọc nội dung/tuổi hoàn chỉnh |
| `inpaint` · Sửa vùng | `bad quality, worst quality, lowres, blurry, jpeg artifacts, watermark, text, bad anatomy` | Khi inpaint: negative ngắn để denoise thấp giữ nét vùng xung quanh |

Hai chế độ áp dụng: **Ghi đè ô negative** (khuyến nghị khi đổi loại ảnh) và **Nối thêm
thẻ còn thiếu** (khi muốn giữ negative hiện tại và bổ sung).

### 4.3 Bảng tra "ảnh đang lỗi gì → thêm gì vào negative"

| Triệu chứng | Thêm vào negative | Ghi chú |
| --- | --- | --- |
| Thừa/thiếu/dính ngón tay | `extra digit, fewer digits, fused fingers, bad hands` | Negative chỉ giảm xác suất; bật **auto-detailer** (mục 7.1) hoặc inpaint cho tay phức tạp |
| Chân/bàn chân sai | `deformed feet, extra toes, fused toes, bad anatomy` | LoRA Anatomy thiên về tay/chân/tư thế, không "chữa" mọi ca |
| Ảnh ra như 3D/nhựa | `realistic, 3d, cgi, plastic skin` + thêm thẻ chất lượng/phong cách ở prompt | Cộng đồng WAI xác nhận `realistic` trong negative là cách nhanh nhất |
| Mắt lệch/mờ | `deformed eyes, asymmetrical eyes, cross-eyed, dull eyes` | `detailed eyes` trong prompt **gần như không tác dụng** trên họ Illustrious (cộng đồng báo) — dùng LoRA mắt, rồi auto-detailer `Mặt` nếu vẫn lệch |
| Mặt/tay lỗi dù prompt đúng | *không thêm negative* — bật **Tự sửa mặt/tay** = `Mặt + tay`, strength 0,3–0,45, cùng seed | Vẽ lại đúng vùng nhỏ hiệu quả hơn negative; xem mục 7.1 |
| Ra chữ/watermark | `text, watermark, signature, logo, username` | Không đảm bảo hết; thử lại seed khác hiệu quả hơn |
| Ảnh đơn sắc/xám | `monochrome, greyscale` | |
| Người lọt vào ảnh phong cảnh | `1girl, 1boy, solo, people, crowd` | |
| Ảnh trông "trẻ" hơn ý muốn | `loli, shota, child, aged_down` + `adult woman`/`mature female` trong prompt | Dataset Danbooru lệch về nhân vật trẻ; negative một mình không đủ |
| Nhiều khung nhìn/nhân bản | `multiple views, extra arms, extra legs` | |
| Nét vẽ bị "sketch"/chưa tô màu | `sketch, lineart, monochrome` | Cẩn thận nếu bạn *muốn* phong cách sketch |

### 4.4 Copy-paste nhanh

```text
# Mặc định (nhà phát hành)
bad quality, worst quality, worst detail, sketch, censor

# Hằng ngày, nhân vật
worst quality, low quality, bad quality, lowres, jpeg artifacts, bad anatomy, bad hands,
extra digit, fewer digits, watermark, signature, text, artistic error, very displeasing, oldest

# Chống 3D, giữ anime 2D
realistic, photorealistic, 3d, cgi, plastic skin, dull colors, monochrome, greyscale,
worst quality, low quality, jpeg artifacts

# Phong cảnh không người
1girl, 1boy, solo, people, crowd, bad perspective, bad proportions, lowres, blurry,
worst quality, low quality, jpeg artifacts, text, watermark, signature, logo
```

## 5. Thông số (khuyến nghị WAI v17)

| Thông số | Khuyến nghị | Ghi chú |
| --- | --- | --- |
| Sampler | **Euler a** (hoặc DPM++ 2M Karras) | Studio dùng Euler a |
| Steps | **15–30** (mặc định 25) | Trên 30 chủ yếu tốn thời gian |
| CFG | **5–7** (mặc định 6) | Trên 7 dễ "cháy"; một số checkpoint Illustrious còn khuyên không quá 6 |
| CLIP skip | 2 | Chuẩn của họ Illustrious |
| Kích thước gốc | ~1 MP: `1024×1024`, `832×1216`, `1216×832` | Tạo thẳng ở 512/768 cho kết quả kém; muốn ảnh lớn dùng hires |
| Hires | **1.5×**, strength **0.35–0.5** | Nhà phát hành gợi ý 1.5×, denoise 0.35–0.5; Studio giới hạn ≈4,2 MP |
| Batch | 1–4 ảnh, đổi seed | Dò seed rồi mới tinh chỉnh |
| LoRA weight | Anatomy ~0.55, Eyes ~0.45 (thử nghiệm) | Giữ seed cố định khi so sánh bật/tắt LoRA |

## 6. Dò seed và lọc ảnh (bước người mới hay bỏ qua)

1. Chốt prompt + negative + thông số. Đặt `Số ảnh = 4`, seed `-1`.
2. Xem 4 ảnh, chọn **đúng một** ảnh có bố cục/tư thế đúng ý (đừng chọn ảnh đẹp nhất về
   nét — nét sửa được bằng hires/inpaint, bố cục thì không).
3. Lấy seed của ảnh đó (bật metadata PNG hoặc ghi từ giao diện), tạo lại 2–3 lần để
   xác nhận seed tái lập được.
4. Từ seed cố định, **đổi một biến** mỗi lượt: thêm thẻ ánh sáng, đổi CFG 5.5 ↔ 6.5,
   đổi LoRA weight. Ghi lại mỗi lần đổi.
5. Ảnh bị lỗi nhỏ → sang bước 7/8/9, đừng tạo lại từ đầu.

Đây chính là "XY grid" của người dùng A1111/ComfyUI, làm thủ công: một trục là seed,
một trục là thông số bạn đang thử.

## 7. Sửa mặt/tay: auto-detailer (tự động) rồi inpaint (thủ công)

### 7.1 Auto-detailer — để máy tự tìm mặt/bàn tay

Khung **🔎 Tự sửa mặt / bàn tay** (dưới khung hires, dùng cho *Văn bản → ảnh* và
*Ảnh → ảnh*) là bản tương đương ADetailer:

| Ô | Nên để | Ghi chú |
| --- | --- | --- |
| Tự sửa mặt/tay | `Tắt` → `Mặt` → `Mặt + tay` | Mặc định `Tắt`; mỗi loại vùng thêm một lượt inpaint |
| Detailer strength | **0,3–0,45** | 0,25 giữ gần như nguyên nét; ≥ 0,5 vẽ lại mạnh, dễ đổi nét mặt |
| Ngưỡng phát hiện | **0,3** (0,2 nếu bỏ sót) | Thấp = dò dễ hơn nhưng dễ nhận nhầm vùng không phải mặt/tay |
| Số vùng tối đa | **1–2** | Vùng được chọn theo độ tin cậy cao nhất; 4 vùng = 4 lượt inpaint |

Cách chạy: tạo ảnh → nếu mặt/tay lỗi thì bật **Mặt + tay** và **tạo lại cùng seed**
(auto-detailer dùng lại seed gốc nên phần còn lại của ảnh gần như không đổi) → so hai
bản. Vùng dò được nới thêm 25%, cạnh chia hết cho 8 và tối thiểu 512 px, được inpaint
bằng **đúng prompt/negative bạn đang thấy** rồi dán lại với viền mềm 12 px; bố cục, nền,
trang phục ngoài vùng đó không bị vẽ lại. Detailer chạy **trước** hires fix.

Giới hạn cần biết: detector là YOLOv8 (`Bingsu/adetailer`, có tập huấn ảnh anime,
mAP50 mặt ≈ 0,66 · tay ≈ 0,77 theo model card) nên **vẫn bỏ sót** tay bị che/mất nét,
và nó **không sửa được** lỗi ở chân, trang phục hay hậu cảnh — dùng inpaint thủ công.
Nếu nét mặt bị đổi, hạ strength; nếu mặt bị “lệch tông” so với ảnh, tắt và dùng inpaint.

### 7.2 Inpaint thủ công — kiểm soát tối đa

Tab **✎ Sửa vùng ảnh** cho vùng lỗi mà auto-detailer bỏ sót hoặc không hỗ trợ:

1. Bấm **Dùng ảnh mới nhất để sửa vùng** (hoặc tải ảnh lên), tô **vùng nhỏ** quanh lỗi.
2. Chọn vùng trong **Chi tiết cần sửa** — `Bàn tay`, `Móng tay / móng chân`,
   `Chân / bàn chân`, `Mắt`, `Khuôn mặt`, `Răng`, `Tóc`, `Da`, `Tùy chỉnh` — rồi bấm
   **Thêm gợi ý sửa vùng vào prompt đang hiển thị**: cặp thẻ dương/âm của vùng đó được
   thêm vào hai ô để bạn **sửa/xóa** trước khi chạy (bấm lại không nhân đôi).
3. Denoise strength **0.35–0.55**: thấp giữ nguyên nét cũ, cao vẽ lại nhiều hơn.
4. Sửa **từng vùng một**, chạy lại, rồi mới sang vùng khác. Tô cả khuôn mặt + thân người
   trong một mask sẽ làm đổi cả trang phục.
5. Inpaint không hỗ trợ hires; ảnh lớn sẽ bị thu về cạnh dài 1024 px khi sửa vùng.

### 7.3 Màu mắt & móng (khung 💅 Chi tiết mắt & móng)

Khung này dùng chung cho *Văn bản → ảnh* và *Ảnh → ảnh*: chọn xong bấm **Thêm chi tiết
mắt/móng vào prompt đang hiển thị**, thẻ được ghi **thẳng vào ô prompt** để sửa/xóa.

| Ô | Lựa chọn | Thẻ được nạp (cú pháp Danbooru, họ Illustrious bám tốt) |
| --- | --- | --- |
| Màu mắt | xanh dương / xanh dương nhạt / xanh ngọc / xanh lá / nâu / hổ phách / đỏ / hồng / tím / vàng / xám / bạc / đen / hai màu / gradient / phát sáng | `blue eyes`, `aqua eyes`, `amber eyes`, `heterochromia, multicolored eyes`, `gradient eyes`, `glowing eyes, detailed pupils`… |
| Kiểu dáng móng tay | tự nhiên ngắn / tròn ngắn / vuông / bầu dục / hạnh nhân / stiletto / coffin / móng dài / móng sắc | `short round nails`, `square nails`, `almond-shaped nails`, `stiletto nails, long fingernails`, `coffin nails`, `sharp fingernails` |
| Màu sơn móng tay | đỏ / đen / hồng / trắng / xanh dương / tím / nude / gradient / kim tuyến / French / vẽ hoa văn / không sơn | `red nails, nail polish`, `glitter nails`, `french nails`, `nail art`, `natural nails` |
| Màu sơn móng chân | đỏ / đen / hồng / trắng / xanh dương / tím / nude / kim tuyến / không sơn | `painted toenails, red nail polish`…, `natural toenails` |

Mẹo dùng cho đúng:

- **Phải thấy được vùng đó** thì model mới vẽ: móng tay cần tay đủ lớn trong khung
  (thêm `hands, fingers spread` hoặc dùng *Chân dung cận*), móng chân cần thấy bàn chân
  (`barefoot, feet`). Ảnh bán thân mà đòi `painted toenails` thì thẻ gần như vô nghĩa.
- Mỗi thẻ đều chiếm token trong **khối 75** (mục 3.1): chỉ chọn 1 màu mắt + 1 kiểu +
  1 màu sơn, đừng nạp cả bốn ô nếu ảnh không cận tay/chân.
- Muốn **đổi** màu mắt/móng của ảnh đã có: dùng *Sửa vùng ảnh* (tô đúng vùng mắt/móng,
  chọn `Mắt` hoặc `Móng tay / móng chân`, strength 0,4–0,55) hoặc auto-detailer `Mặt`
  cho mắt; inpaint vùng nhỏ giữ được nét cũ tốt hơn tạo lại cả ảnh.
- Thẻ kiểu/màu là **mô tả mong muốn**, không bảo đảm model vẽ đúng 100% — nếu móng ra sai
  hình dạng, thử `close-up` bàn tay và giảm CFG xuống 5,5–6.

## 8. Hires & upscale

- **Trong lúc tạo:** khung **🔍 Ảnh độ phân giải cao** → `1.5×` (hoặc `2×`), strength
  0.35–0.5. Ảnh được tạo ở ~1 MP, phóng Lanczos rồi tinh chỉnh bằng img2img cùng
  prompt/seed.
- **Ảnh đã có:** tab **⤢ Phóng to ảnh** (không tạo lại từ đầu), cùng giới hạn ≈4,2 MP.
- Cộng đồng anime thường dùng upscaler chuyên dụng (R-ESRGAN 4x+ Anime6B, 4x-AnimeSharp)
  trước bước refine; Studio hiện dùng Lanczos + refine, nên nếu bạn cần upscale "sạch"
  hơn cho in ấn thì xuất PNG rồi upscale ngoài bằng model anime chuyên dụng.
- Strength > 0.5 bắt đầu **thêm chi tiết mới** (đổi nét mặt, đổi hoa văn vải). Muốn giữ
  đúng ảnh cũ, ở mức 0.35.

## 9. Nhân vật nhất quán giữa nhiều ảnh

Xếp theo độ tin cậy (đánh giá của cộng đồng, không phải đo trong repo này):

1. **LoRA nhân vật tự huấn luyện** (20–50 ảnh) — gần như 100%, tốn công.
2. **IP-Adapter FaceID / ControlNet pose** — cần ComfyUI/A1111, Studio Colab này chưa có.
3. **Mô tả nhân vật cố định + seed cố định** — mức cơ bản: viết **đúng một đoạn mô tả
   nhân vật** (màu/kiểu tóc, màu mắt, trang phục, phụ kiện) và **dán nguyên văn** vào
   mọi prompt, chỉ đổi tư thế/bối cảnh/ánh sáng. Giữ cùng LoRA, cùng CFG ±0.5.

Trong phạm vi Studio: dùng cách 3, cộng với **thư viện prompt** (nạp file `.txt`) để mọi
ảnh của một nhân vật dùng chung một đoạn mô tả.

## 10. Checklist QA trước khi xuất ảnh

Phóng to 100% và kiểm lần lượt:

- [ ] **Tay:** đủ ngón, không dính, không thừa khớp; đồ vật cầm đúng hướng. Còn lỗi →
      auto-detailer `Tay` (mục 7.1) rồi inpaint.
- [ ] **Chân/bàn chân:** đủ ngón, cổ chân không gãy, giày không chảy.
- [ ] **Mặt:** hai mắt cân, đồng tử cùng hướng, răng không vỡ, tai đối xứng. Còn lỗi →
      auto-detailer `Mặt` (mục 7.1) rồi inpaint.
- [ ] **Auto-detailer (nếu có bật):** vùng được sửa **không** để lại viền cứng, không đổi
      màu da/tông sáng so với phần còn lại; nếu có, hạ strength hoặc tắt.
- [ ] **Chữ/ký hiệu:** không có chữ vô nghĩa, watermark, logo lạ.
- [ ] **Trang phục:** không tan vào da, không thừa/thiếu lớp áo, hoa văn không rối.
- [ ] **Bối cảnh:** phối cảnh đường chân trời, phản xạ/đổ bóng đúng hướng sáng.
- [ ] **Nét:** không mờ cục bộ, không "nhựa" 3D ngoài ý muốn, không JPEG artifacts.
- [ ] **Bố cục:** chủ thể không bị cắt ở khớp, còn khoảng thở đúng ý.
- [ ] **Nội dung:** mọi nhân vật trưởng thành; tuân thủ giấy phép model và điều khoản
      Colab/Gradio.

Xong: **tắt** `Nhúng prompt vào metadata PNG` nếu định chia sẻ ảnh, rồi tải PNG về máy
trước khi runtime Colab ngắt.

## 11. Ghi log phiên làm việc

Mỗi ảnh "đạt" nên ghi 6 trường để tái lập được:

```text
seed | kích thước | steps | CFG | LoRA + weight | prompt & negative (đúng nguyên văn)
```

Bật `Nhúng prompt vào metadata PNG` khi làm việc riêng để PNG tự chứa thông tin; tắt
trước khi đăng công khai.

---

## Phụ lục A · Ba prompt mẫu theo thứ tự chuẩn

### A0. Ví dụ nhân vật hoàn chỉnh (phân tích từng khối)

Prompt đã kiểm bằng chính công cụ trong repo: **34 thẻ · ≈70 token** (72 sau khi thêm
trigger `perfect eyes`) — còn dư chỗ trong khối 75 token; **🩺 báo 0 ⚠️ / 0 ℹ️ / 4 ✅**;
nút *Sắp xếp prompt theo thứ tự chuẩn* **không đổi gì** vì prompt đã đúng thứ tự.

```text
masterpiece, best quality, amazing quality, general, 1girl, solo, adult woman, mature female, long straight black hair, purple eyes, gentle smile, mole under mouth, white dress shirt, navy blazer, red necktie, black pencil skirt, black gloves, thighhighs, standing, arms crossed, looking at viewer, upper body, straight-on, depth of field, sharp focus, modern office lobby, glass wall, city lights, night, warm rim light, cinematic lighting, cel shading, anime illustration, absurdres
```

| Khối | Thẻ | Ghi chú |
| --- | --- | --- |
| 1. Chất lượng | `masterpiece, best quality, amazing quality` | Đúng 3 thẻ nhà phát hành khuyên; thêm nữa dễ mờ ảnh |
| 2. Nhãn phân loại | `general` | Họ Illustrious dùng `general/sensitive/nsfw/explicit` |
| 3. Chủ thể | `1girl, solo, adult woman, mature female` | `mature female` chống dataset kéo nhân vật trẻ lại |
| 4. Ngoại hình | `long straight black hair, purple eyes, gentle smile, mole under mouth` | **Khối nhận dạng** — dán nguyên văn vào mọi ảnh của nhân vật này |
| 5. Trang phục | `white dress shirt, navy blazer, red necktie, black pencil skirt, black gloves, thighhighs` | Mô tả từng món, đừng gộp "office uniform" |
| 6. Tư thế | `standing, arms crossed, looking at viewer` | Đổi khối này khi đổi ảnh, giữ khối 4 |
| 7. Bố cục | `upper body, straight-on, depth of field, sharp focus` | `full body` / `close-up` / `cowboy shot` tuỳ ý |
| 8. Bối cảnh | `modern office lobby, glass wall, city lights, night` | |
| 9. Ánh sáng | `warm rim light, cinematic lighting` | |
| 10. Phong cách | `cel shading, anime illustration` | Chống chất 3D/nhựa của v17 |
| 11. Độ nét | `absurdres` | Chốt cuối prompt |

**Negative** (bộ `core`, 15 thẻ):

```text
worst quality, low quality, bad quality, lowres, jpeg artifacts, bad anatomy, bad hands, extra digit, fewer digits, watermark, signature, text, artistic error, very displeasing, oldest
```

**Thông số:** `832x1216` · steps 28 · CFG 6 · Euler a · seed cố định (ví dụ `12345`) ·
hires `1.5×` strength `0.4` · LoRA Anatomy 0.55 + Eyes 0.45, bấm nút thêm trigger
`perfect eyes`.

**Muốn cùng nhân vật ở cảnh khác:** giữ nguyên khối 3+4+5+10+11, chỉ thay khối 6/7/8/9 —
ví dụ đổi `standing, arms crossed, looking at viewer` thành `sitting, holding a coffee cup,
looking away` và `modern office lobby, glass wall, city lights, night` thành
`rooftop at sunset, wind, orange sky`.

```text
# 1. Chân dung dưới hoa anh đào (832×1216, 25 steps, CFG 6, hires 1.5× / 0.4)
masterpiece, best quality, amazing quality, 1girl, solo, adult woman, long dark hair,
gentle smile, white dress, holding a transparent umbrella, standing under cherry blossoms,
petals falling, upper body, soft rim light, spring dusk, cel shading, anime illustration,
absurdres
Negative: bad quality, worst quality, worst detail, sketch, censor, deformed eyes, bad hands

# 2. Văn phòng đêm, ánh sáng đèn bàn (1216×832, 28 steps, CFG 6)
masterpiece, best quality, amazing quality, 1girl, solo, adult woman, glasses, messy bun,
white shirt, oversized knit cardigan, typing at a laptop, focused expression, upper body,
dark office at night, rain streaks on the window, warm desk lamp light, cinematic lighting,
anime illustration, absurdres
Negative: worst quality, low quality, bad quality, lowres, jpeg artifacts, bad anatomy,
bad hands, extra digit, text, watermark, artistic error, very displeasing

# 3. Phong cảnh không người (1216×832, 26 steps, CFG 5.5)
masterpiece, best quality, amazing quality, no humans, wide shot, coastal village on a
hillside at dawn, terraced gardens, enormous pastel clouds, sea shimmering, expansive
detailed background, soft morning light, anime background, absurdres
Negative: 1girl, 1boy, solo, people, crowd, bad perspective, lowres, blurry, worst quality,
low quality, jpeg artifacts, text, watermark, signature, logo
```

## Phụ lục B · Ba lỗi quy trình phổ biến

| Lỗi | Vì sao sai | Sửa |
| --- | --- | --- |
| Viết negative 100+ thẻ "cho chắc" | Nhà phát hành WAI v17: negative quá dài làm giảm chất lượng, ảnh mờ | Chọn 1 bộ theo mục đích, thêm ≤ 5 thẻ riêng |
| Nhồi 8–10 thẻ chất lượng | Thừa thẻ chất lượng/thẩm mỹ cũng làm mờ ảnh | `masterpiece, best quality, amazing quality` là đủ |
| Đổi prompt + seed + CFG cùng lúc | Không biết nguyên nhân, không lặp lại được | Một biến mỗi lượt, ghi log |
| Tạo thẳng ở 512×512 rồi upscale | SDXL được huấn luyện quanh ~1 MP | Tạo ở ~1 MP rồi hires 1.5–2× |
| Inpaint cả nửa người | Đổi luôn mặt/trang phục | Mask nhỏ từng vùng, strength 0.35–0.55 |

## Nguồn

**Nhà phát hành / trang model**

- WAI-illustrious-SDXL v17.0 trên Civitai (version 2883731): quality head
  `masterpiece, best quality, amazing quality`; negative `bad quality, worst quality,
  worst detail, sketch, censor`; steps 15–30; CFG 5–7; Euler a; hires 1.5× với denoise
  0.35–0.5; nhãn an toàn `general/sensitive/nsfw/explicit` (đưa `nsfw` vào negative để
  lọc); và cảnh báo *không thêm quá nhiều thẻ chất lượng/thẩm mỹ hay negative quá dài
  vì sẽ giảm chất lượng, ảnh mờ hơn*. Bản lưu nội dung: civarchive.com/models/827184?modelVersionId=2883731
  và ?modelVersionId=2167369.

**Hướng dẫn prompt họ Illustrious XL**

- SeaArt — *Illustrious XL v1.0/v1.1 & v2.0 Update And User Guide*: thứ tự prompt 10
  nhóm; `absurdres, highres` đặt cuối; negative rất hiệu quả trên Illustrious
  (`worst quality, bad quality, very displeasing, displeasing, oldest`, `artistic error`,
  `lowres, jpeg artifacts, censor, watermark, bad hands, bad anatomy, traditional media`);
  NLP dùng được nhưng tag Danbooru vẫn ổn định hơn.
- tensor.art — *Comprehensive Guide of Illustrious XL*: khai báo số chủ thể
  (`1girl`, `2boys`, `vehicle focus`); negative ảnh hưởng mạnh ở độ phân giải cao.
- artificialguy.com — *Pony & Illustrious Prompt Syntax*: `score_`/`source_` là của họ
  Pony, không dùng cho Illustrious; `BREAK` tách khối CLIP; clip skip 2.
- Reddit r/StableDiffusion — *Token limit on NoobAI/Illustrious models*: SDXL đọc theo
  khối 75 token; prompt ngắn dễ làm việc hơn; `BREAK` để kiểm soát khối.
- Civitai — *Arctenox's Simple Prompt Guide for Illustrious*: bắt đầu bằng
  `masterpiece, best quality` / negative `worst quality, low quality`; khuyên không vượt
  CFG 6 với nhiều checkpoint Illustrious; thêm `loli, shota, child, aged_down` vào
  negative khi muốn nhân vật trưởng thành.

**Phản hồi cộng đồng trên trang WAI-illustrious (kinh nghiệm, không phải thông số chính thức)**

- Thêm `realistic` (và `3d`) vào negative để giảm chất 3D/nhựa của v17; v17 thiếu thẻ
  chất lượng thì dễ ra chất nhựa.
- `detailed eyes` gần như không tác dụng trên model họ Illustrious; `empty eyes` trong
  negative làm mắt long lanh hơn.
- Negative kiểu "lazy": `flat colors`, `rough sketch`, `simple sketch`, `lineart`,
  `pencil drawing` để đẩy ảnh về phía painterly; mục tiêu là *giảm xác suất lỗi về mức
  chấp nhận được*, không phải loại tuyệt đối.
- Trọng số tối đa ~1.2, phần lớn prompt nên không cân nặng; ưu tiên từ mô tả.
- Với lỗi chi tiết ở 1024²: hires fix, Kohya Deep Shrink hoặc ADetailer.
- Upscale anime: R-ESRGAN 4x+ Anime6B / 4x-AnimeSharp / 2x-AnimeSharpV4, refine denoise
  ~0.35.

**Quy trình production & nhất quán nhân vật**

- digitalzoomstudio.net — *Stable Diffusion Character Consistency (2026)*: xếp hạng
  IP-Adapter FaceID > ControlNet/InstantID > ADetailer + prompt anchoring > LoRA tự huấn
  luyện > seed cố định + mô tả nặng; batch 4–8 để dò, hires denoise 0.3–0.4.
- myimageupscaler.com — *Cartoon to Realistic AI: A Pro Workflow Guide (2026)*: thứ tự
  hậu kỳ layered (dọn artefact → sửa mặt → sharpen → upscale → color grade → QA), xem
  100% trước khi xuất.
- reddit r/StableDiffusion — *what anime style workflow do you use?*: quy trình phổ biến
  là model chính ~28 steps + hires fix với upscaler anime (2×) + denoise ~0.35.
