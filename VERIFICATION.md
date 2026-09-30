# Kiểm tra nguồn và vận hành WAI Studio (26/09/2026, cập nhật 30/09/2026)

**Phạm vi chính:** notebook Google Colab `WAI_Illustrious_Studio_Colab.ipynb`, notebook nâng cao dùng chung các ô chuẩn bị model/LoRA, và các tài nguyên mà chúng ghim. Cloudflare trong `web/` là ứng dụng *khác*, không chạy checkpoint WAI nếu không có backend GPU bên ngoài.

## 1. Danh tính tài nguyên: đối chiếu hai nguồn độc lập

Đã đọc lại metadata API của **phiên bản gốc trên Civitai** và **file LFS tại commit được ghim trên Hugging Face**. Kết quả đối chiếu đúng *toàn bộ* SHA-256 và số byte ghi trong hai notebook:

| Tài nguyên / phiên bản Civitai | File gốc trên Civitai | File mirror HF tại commit cố định | Số byte | SHA-256 |
| --- | --- | --- | ---: | --- |
| [WAI-illustrious SDXL v17.0 · 2883731](https://civitai.com/api/v1/model-versions/2883731) | ID `2763986` · checkpoint SDXL, pruned FP16 | [LyliaEngine/waiIllustriousSDXL_v170 · `32be7bfdcd406db70df663b9cee3313957deb68f`](https://huggingface.co/api/models/LyliaEngine/waiIllustriousSDXL_v170/revision/32be7bfdcd406db70df663b9cee3313957deb68f?blobs=true) · `waiIllustriousSDXL_v170.safetensors` | 6.938.040.682 | `f116b0c78ff441467b0cdc8f1936e1ed18ea31e9997c7b132b1b8db533f0bd04` |
| [Anatomy Helper V1 · 1318504](https://civitai.com/api/v1/model-versions/1318504) | ID `1223247` · LoRA Illustrious | [ench100/bodyandface · `bed49d45df95c0695aedad3b2aa6aff389fb3777`](https://huggingface.co/api/models/ench100/bodyandface/revision/bed49d45df95c0695aedad3b2aa6aff389fb3777?blobs=true) · `anatomy_helper.safetensors` | 228.473.940 | `bf6a950036b7599212a2c68d65f3ba07b28689067e167915d2a0ecb2018c26ca` |
| [Eyes for Illustrious V1 · 2066663](https://civitai.com/api/v1/model-versions/2066663) | ID `1963176` · LoRA Illustrious, trigger `perfect eyes` | [Muapi/eyes-for-illustrious-lora-perfect-anime-eyes · `1abbc862f53f5101962ebf1c337513aff91bd206`](https://huggingface.co/api/models/Muapi/eyes-for-illustrious-lora-perfect-anime-eyes/revision/1abbc862f53f5101962ebf1c337513aff91bd206?blobs=true) · `eyes-for-illustrious-lora-perfect-anime-eyes.safetensors` | 228.457.660 | `97c1a083ffe6b4d45c545196eabd01c754936b996ade0c9db6d072f3bd340c55` |

**Lưu ý xác thực:** kho HF là bản lưu của bên khác; trùng hash **không** chứng minh chủ kho là tác giả Civitai. Thậm chí thẻ chung của kho HF Anatomy ghi `base_model: lodestones/Chroma`, trong khi *file cụ thể* cùng SHA-256 với LoRA Illustrious do Civitai phát hành. Tương tự, thẻ chung của kho HF checkpoint có từ `lora` dù file gốc được Civitai phân loại checkpoint. **Không dùng mô tả/thẻ của kho mirror để xác minh biến thể; dùng ID phiên bản gốc, tên file, kích thước và SHA-256 của đúng file.** Tên file Eyes trên Civitai và mirror HF khác nhau nhưng SHA-256 trùng.

API HF ghi ba kho `gated: false` tại thời điểm kiểm tra. Đó là trạng thái metadata, **không bảo đảm** mọi mạng Colab tải được file vào mọi thời điểm. Chưa tải 7 GB checkpoint và hai LoRA thật trong sandbox để tự tính lại hash byte từ file nhận được; vì vậy đây là **đối chiếu metadata phát hành**, không phải kiểm tra độc lập bản tải thật.

## 2. Luồng `/content` và prompt có thể sửa

- Hai notebook **không gọi `google.colab.drive.mount`**, không đặt cờ sao chép/lưu Drive. `MODEL_PATH` mặc định `/content/wai_model_cache/waiIllustriousSDXL_v170.safetensors`, LoRA bật nhưng hai đường dẫn thủ công để rỗng, kết quả `/content/wai_outputs`. Ô 3 từ chối đường dẫn ngoài `/content` và `/content/drive`, kể cả cache local trỏ bằng symlink. Không còn fallback sang Drive khi đĩa thiếu; báo lỗi để người dùng giải phóng đĩa/tắt LoRA.
- Ô 4 dùng `hf_hub_download(..., revision=commit, local_dir=/content/wai_model_cache, token=False)` khi chưa có checkpoint và **nạp chính file được tải**, không tạo bản sao thứ hai; nếu file đã có trong cùng runtime thì dùng lại và kiểm đầy đủ hash. Studio **chỉ chấp nhận WAI v17 trùng toàn bộ SHA-256**, bao gồm đường dẫn thủ công/cached; bản nâng cao cho phép checkpoint SDXL khác nhưng nêu rõ chưa xác thực phiên bản. Ô 5 tự tải LoRA được bật vào `/content/wai_lora_cache`, xác nhận size + SHA-256 + header; file thủ công vẫn bắt buộc đúng hash. File đã có sai hash bị từ chối, không tự ghi đè. Ô 6 kiểm lại LoRA trước khi nạp/khôi phục sau OOM.
- Studio chọn phong cách/đổi ý tưởng gốc/negative gốc/bật tắt LoRA mắt sẽ **điền lại hai ô prompt cuối, có thể sửa trực tiếp**. Lệnh tạo text, img2img và inpaint dùng **đúng chuỗi hai ô tại thời điểm bấm tạo**, kể cả khi người dùng xóa thẻ preset hoặc `perfect eyes`: runtime không ghép lại preset hay ẩn thêm gợi ý sửa. Nút sửa vùng thêm từ vào **hai ô đang hiển thị**, không thay đổi ngầm lúc suy luận; adult preset vẫn yêu cầu xác nhận 18+ và kiểm một số từ khóa vị thành niên trong prompt dương.
- Kiểm thử file/hàm kích thước nhỏ và mock giúp phát hiện sai luồng, **không** thay cho tải và xác minh ba file lớn thực tế từ Colab. Ba mã hash/size trong bảng vẫn là đối chiếu metadata độc lập với file thực nhận.

### 2b. Ba công cụ quy trình mới (30/09/2026)

Accordion **🧭 Quy trình chuẩn · khung prompt + negative tối ưu** thêm ba sự kiện Gradio
(nâng tổng số sự kiện từ 14 lên 17), tất cả đều `queue=False`, `api_visibility="private"`
và **chỉ ghi vào ô đang hiển thị**:

- `structure_prompt(prompt, kind)` — xếp lại thẻ theo thứ tự CLIP ưu tiên, bỏ thẻ trùng
  (so khớp sau khi đã bóc cú pháp `(thẻ:1.2)`), giữ nguyên trọng số, và thêm thẻ neo của
  khung đã chọn **chỉ khi nhóm tương ứng trống**. Kết quả ghi vào ô *Prompt gửi model*
  cùng ghi chú phân nhóm; hàm idempotent (chạy lại không đổi). Từ chối prompt sau sắp xếp
  dài hơn 2200 ký tự — đúng giới hạn `_parameters` đang áp dụng.
- `apply_negative_preset(preset_id, negative, mode)` — 8 bộ negative theo mục đích, mỗi bộ
  ≤ 20 thẻ; chế độ *ghi đè* hoặc *nối thêm không trùng*. Test duyệt cả 8 bộ × 2 chế độ và
  khẳng định kết quả không vượt 1700 ký tự (giới hạn runtime). Bộ `publisher` đúng nguyên
  văn negative nhà phát hành WAI v17 công bố.
- `run_prompt_check(...)` — chỉ **đọc** prompt/negative/steps/CFG/kích thước/hires rồi trả
  Markdown báo cáo; không đổi giá trị nào và không đụng gallery. Số token là **ước lượng
  heuristic** (`estimate_tokens`), không dùng tokenizer CLIP thật, nên chỉ dùng để so với
  mốc 75 token chứ không phải số chính xác.

Ba nút này không can thiệp hàng rào nội dung: `_parameters` vẫn kiểm
`UNDERAGE_PROMPT`/`ADULT_PROMPT` trên đúng chuỗi hai ô lúc bấm tạo, như trước.

Các con số khuyến nghị trong `docs/QUY_TRINH_TAO_ANH.md` (steps 15–30, CFG 5–7, hires
0,35–0,5, thứ tự thẻ, negative ngắn) là **khuyến nghị do nhà phát hành và cộng đồng công
bố**, được trích nguồn trong tài liệu; **không phải kết quả đo trên GPU** trong repo này.

## 3. Kiểm thử mã đã thực hiện cho thay đổi này

| Kiểm tra | Kết quả | Giới hạn |
| --- | --- | --- |
| `python -m unittest discover -s tests` trong venv mới (30/09/2026): Gradio 6.15.2, Pillow 12.3.0, `huggingface_hub` 0.36.2, `torch` 2.14.1 CPU, nbformat, packaging | **66 test: 66 đạt, 0 bỏ qua** (cài thêm `torch` CPU nên bài trước đây bị skip đã chạy). Gồm 8 test mới cho negative preset/sắp xếp prompt/bộ kiểm tra, 1 test UI Gradio cho ba nút quy trình, và toàn bộ bài cũ về hash/đĩa/OOM/inpaint/hires | Model WAI và GPU đều được **giả lập**; chưa đo VRAM, thời gian hay chất lượng ảnh |
| `cd web && npm ci && npm test` | **15 test: 15 đạt** (thêm `tests/negative-presets.test.mjs`: 6 preset negative duy nhất/không trùng/không vượt giới hạn ô nhập, và chip nạp đúng nội dung vào ô negative đang hiển thị, không gọi endpoint suy luận) | Không triển khai Worker, không tạo ảnh WAI thật |
| `npm run format:check`, `npm run build`, `npx wrangler deploy --dry-run` | Đạt (Vite build 1583 module; dry-run đọc 10 file asset, binding `AI`/`ASSETS`) | Dry-run không triển khai và không gọi Workers AI |
| `black --check colab tests scripts`; `nbformat.validate` hai notebook; notebook Studio khớp `build()` | Đạt (đã chạy `black` cho 3 file Python rồi sinh lại notebook) | Cấu trúc đúng không chứng minh Colab tải model hay tạo ảnh chất lượng |
| Báo cáo trước (26/09/2026) | 44 test: 43 đạt, 1 bỏ qua; `pip check` với Gradio 6.15.2 + `huggingface-hub==0.36.2` đạt | Đã được thay bằng lượt chạy 66 test ở trên |

Sandbox hiện **không có GPU**. Thử tải qua Hugging Face trực tiếp tại sandbox từng gặp TLS EOF; đây **không chứng minh** mạng Colab cũng gặp lỗi. Không tải 6,94 GB checkpoint/two LoRA để thực sự so hash byte ở đây, không thử tốc độ tải hoặc chất lượng ảnh thực tế. Tốc độ còn tùy mạng Colab; loại bỏ Drive chỉ bỏ thao tác gắn/sao chép/lưu, **không bảo đảm tải qua mạng nhanh hơn**.

## 4. Phép thử cuối cùng cần chạy trên Colab của bạn

1. Mở [notebook Studio trên nhánh hiện tại](https://colab.research.google.com/github/manhlee1196-boop/ai-anime/blob/main/WAI_Illustrious_Studio_Colab.ipynb), chọn GPU và **Runtime → Restart runtime → Run all**. Không cần cấp quyền Google Drive. Để trống hai đường dẫn LoRA ở ô 3 nếu dùng bản tự tải. Link Gradio của phiên cũ **không tự cập nhật**.
2. Ô 1 thấy GPU; ô 2 in `✅ Thư viện Studio đã sẵn sàng`; ô 4/5 in xác minh SHA-256 WAI v17 và LoRA; ô 6 in chế độ nạp và VRAM. Ô 4 có thể mất thời gian tải + hash toàn bộ 6,94 GB, **không còn dòng “Sao chép model từ Drive”**. Nếu thiếu dung lượng/mạng lỗi, dừng xử lý tại ô đó; đừng coi việc chạy các ô sau là đã xác nhận thành công.
3. Ở giao diện, mở accordion **🧭 Quy trình chuẩn**: bấm **Sắp xếp prompt theo thứ tự chuẩn** với prompt của bạn rồi đọc ghi chú phân nhóm; chọn một bộ negative (bắt đầu bằng `publisher` hoặc `core`) và bấm **Nạp negative đã chọn**; bấm **🩺 Kiểm tra prompt & thông số** và xử lý hết các mục ⚠️. Sau đó xem hai ô **“Prompt gửi model”** / **“Negative gửi model”**: sửa hoặc xóa bất kỳ thẻ nào, bật metadata PNG nếu muốn tự đối chiếu, rồi tạo ảnh và tải PNG. Thử **hai lần liên tiếp** cùng seed (ví dụ `12345`) để xác nhận seed tái lập được, rồi đổi **một** biến mỗi lượt (CFG 5,5 ↔ 6,5; steps 20 ↔ 30; bật/tắt một LoRA). Dùng tab sửa vùng: chọn tay/chân/mắt, bấm nút thêm gợi ý rồi sửa chúng trước khi inpaint; thử không bấm để đảm bảo không có thẻ ẩn. Nội dung người lớn chỉ khi mọi nhân vật đều trưởng thành và đã tick xác nhận 18+. Hiệu quả thực tế — negative nào thật sự giảm lỗi tay/chất 3D trên máy bạn — **chỉ đánh giá được ở bước này**.
4. **Tải tất cả ảnh cần giữ về máy trước khi runtime ngắt**: ảnh, checkpoint và LoRA trong `/content` sẽ mất cùng phiên. Nếu lỗi, gửi traceback ô 2/4/5/6/8, che thông tin riêng và URL `gradio.live` vì ai biết URL đều có thể dùng GPU của bạn.

**Bổ sung hires fix (ảnh lớn 1,5×/2×):** test CPU kiểm tra kích thước đích (bội số 8, giới hạn ≈4,2 MP), hai lượt pipe (tạo gốc rồi ảnh → ảnh cùng seed/prompt), metadata, từ chối khi dùng cho sửa vùng, tab Phóng to ảnh (không tạo lại ảnh gốc) và đường sự kiện Gradio. Pipeline diffusers/VAE tiling là giả lập; **chưa đo VRAM, thời gian hay chất lượng ảnh 2048×2048 trên T4 thật**.

**Kết luận:** SHA-256 ghim khớp metadata Civitai/HF của đúng file; mã local-only, luồng prompt cuối và ba công cụ quy trình mới đã qua kiểm thử cục bộ (66 test Python, 15 test web). **Chưa xác nhận** tạo ảnh WAI thật trên GPU Colab, cũng chưa đo việc negative preset/thứ tự thẻ có thật sự cải thiện ảnh trên máy bạn — các khuyến nghị đó là công bố của nhà phát hành và cộng đồng, đã ghi nguồn trong `docs/QUY_TRINH_TAO_ANH.md`. Xác nhận 18+ chỉ là khai báo của người dùng cộng với kiểm một số từ khóa, **không xác minh tuổi/bộ lọc hoàn chỉnh**.
