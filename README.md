# WAI Studio cá nhân — giao diện tạo ảnh qua Google Colab

[![Mở WAI Studio trong Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manhlee1196-boop/ai-anime/blob/main/WAI_Illustrious_Studio_Colab.ipynb)

**[WAI_Illustrious_Studio_Colab.ipynb](WAI_Illustrious_Studio_Colab.ipynb)** chạy WAI-illustrious v17 và hai LoRA tùy chọn trên GPU Colab; Gradio tạo **link tạm không cần đăng nhập**. Không cần Google Drive hoặc API token; link Gradio chính không cần Cloudflare, còn ô 9 có thể tạo Cloudflare Quick Tunnel dự phòng. Checkpoint và LoRA được kiểm **toàn bộ SHA-256** trước khi nạp. [Báo cáo kiểm tra nguồn và vận hành](VERIFICATION.md) phân biệt điều đã thử cục bộ với việc **chưa thử tạo ảnh bằng GPU Colab thật**.

1. Mở notebook từ nút trên → **Runtime → Change runtime type → T4 GPU** (hoặc mạnh hơn) → **Runtime → Run all**. Lần đầu notebook tải checkpoint ~6,94 GB và tối đa ~457 MB LoRA **trực tiếp vào `/content`**, không gắn Drive/không sao chép file qua lại. Cần khoảng **9 GiB đĩa trống**; tốc độ vẫn tùy mạng Colab và bước kiểm hash. Chọn/tắt LoRA, `VRAM_MODE` và đường dẫn tùy chỉnh **dưới `/content`** ở ô 3 trước khi nạp. Để `ANATOMY_LORA_PATH`/`EYE_LORA_PATH` rỗng nếu muốn tải tự động. **Cả 9 ô code được thu gọn thành form của Colab** — mỗi ô chỉ còn một thanh tiêu đề kèm nút **Run** (kể cả ô 7 chứa ~1.500 dòng mã giao diện và ô 9 tạo link dự phòng qua Cloudflare Quick Tunnel), nên trang ngắn và chạy tuần tự dễ; muốn xem hoặc sửa code của ô nào thì bấm biểu tượng `>_` (⋮ → *Show code*) ở góc ô đó, output vẫn hiện bình thường.
2. Ô 8 tạo link `https://….gradio.live` để mở giao diện tạo ảnh bằng văn bản, ảnh → ảnh hoặc mask sửa tay/chân/mắt (kèm hires `1.5×`/`2×`, tab phóng to ảnh và **auto-detailer** nếu bật). Nếu Gradio Share không tạo link hoặc `gradio.live` báo 504/bị ngắt, ô 9 tạo URL `https://….trycloudflare.com` tới cùng giao diện đang chạy; không cần tài khoản Cloudflare/API token hay nạp lại model. Ô 9 chạy tự động khi **Run all** và dùng link công khai, không có đăng nhập — đừng chia sẻ. **Không có selector phong cách: bạn tự viết phong cách trong prompt** (hoặc nạp từ thư viện prompt / dùng accordion **🧭 Quy trình chuẩn** và **🎨 Gợi ý phong cách**). Khi gõ prompt, gợi ý tag/cụm prompt Việt/English hiện theo cụm cuối; chọn một dòng rồi bấm **Thêm tag đã chọn** để thêm tên tiếng Anh chuẩn vào cuối prompt mà vẫn giữ phần bạn đã viết — không tự chèn khi gõ. Điều chỉnh kích thước, steps, CFG, seed, strength, số ảnh, LoRA và **hai ô prompt thực sự gửi model**.
3. Ảnh chỉ lưu dưới **`/content/wai_outputs`** (hoặc thư mục `/content` được chọn); model ở `/content/wai_model_cache`, LoRA ở `/content/wai_lora_cache`. **Tải ảnh về trước khi phiên Colab kết thúc**: cả ảnh và weights cục bộ sẽ mất khi runtime ngắt, phiên sau cần tải lại. Link cũng ngừng hoạt động khi Colab ngắt.

**Ảnh độ phân giải lớn (hires fix):** trên giao diện `gradio.live`, khung **🔍 Ảnh độ phân giải cao** (luôn hiển thị, ngay trên các tab) cho chọn **Độ phân giải cao** = `1.5×` hoặc `2×` cho tab *Văn bản → ảnh* và *Ảnh → ảnh*. Studio tạo ảnh ở kích thước đã chọn (SDXL học tốt nhất quanh ~1 MP), phóng to bằng Lanczos rồi chạy thêm một lượt ảnh → ảnh cùng prompt/seed/LoRA ở **Hires strength** (mặc định 0,4; thấp giữ bố cục, cao thêm chi tiết nhưng dễ đổi nét). Ví dụ `1024×1024` → `2048×2048`, `832×1216` → `1248×1824`. Kích thước cuối luôn chia hết cho 8 và **tối đa ≈4,2 MP**: nếu vượt (như `1344×1024` × 2) hệ số tự giảm cho vừa và thanh trạng thái báo rõ. VAE tiling được bật để giảm đỉnh VRAM; vẫn tốn thêm thời gian, VRAM có thể phải chuyển sang CPU offload, nếu OOM hãy chọn `1.5×` hoặc kích thước gốc nhỏ hơn. PNG và metadata (nếu bật) ghi kích thước cuối cùng và thông số hires. Tab **⤢ Phóng to ảnh** phóng to một ảnh có sẵn (tải lên hoặc nút *Dùng ảnh mới nhất để phóng to*) mà không tạo lại từ đầu, cùng giới hạn ≈4,2 MP. Tab *Sửa vùng ảnh* không hỗ trợ hires và vẫn thu ảnh về cạnh dài tối đa 1024 px. Chưa thử trên GPU Colab thật (xem `VERIFICATION.md`).

**Prompt do bạn viết, gửi nguyên văn:** Studio không còn preset phong cách — hai ô **“Prompt gửi model · tự viết phong cách của bạn”** và **“Negative gửi model · ngón tay / ngón chân”** là **chính xác** những gì được gửi ở cả ba chế độ, không tự ghép thêm thẻ theo LoRA hay vùng sửa khi bấm tạo. Mọi phong cách (`anime illustration, cel shading`, `watercolor`, `cinematic lighting`…) là từ khóa bạn tự viết hoặc nạp từ **thư viện prompt**. Hai nút hỗ trợ đều **hiển thị trong ô để bạn sửa/xóa**: **Thêm trigger `perfect eyes` cho LoRA mắt** và, trong tab **Sửa vùng ảnh**, **Thêm gợi ý sửa vùng vào prompt đang hiển thị**; bật LoRA mắt vẫn tải và dùng adapter kể cả khi bạn xóa trigger. Negative mẫu nhắm ngón thừa/thiếu/dính. Phong cách chỉ là từ khóa trên **cùng một** checkpoint anime, không phải model riêng; prompt không bảo đảm sửa hết lỗi ngón.

**Bảo mật:** link `gradio.live` và Cloudflare Quick Tunnel (`trycloudflare.com`) đều công khai, **không có tài khoản/mật khẩu**: ai biết link đều có thể dùng GPU của bạn. Không chia sẻ link hay lưu công khai notebook có output chứa link; dừng runtime để ngắt link. File checkpoint/LoRA bị chặn tải qua đường file Gradio; prompt/negative chỉ được nhúng vào PNG nếu bạn tự bật metadata (mặc định tắt).

**Nếu ô 4 lâu:** tải 6,94 GB qua mạng và kiểm SHA-256 toàn bộ file có thể mất thời gian (có in tiến độ kiểm); **không còn bước sao chép Drive**. Nếu thất bại, xem mạng/đĩa Colab và chạy lại; file đúng còn trong cùng runtime sẽ được dùng lại, file hỏng bị từ chối. Không tắt kiểm hash để tăng tốc.

**Nếu RAM hệ thống gần đầy nhưng VRAM gần trống:** mở notebook mới nhất rồi **Runtime → Restart runtime → Run all** để giải phóng model cũ. `VRAM_MODE=auto` ưu tiên GPU trực tiếp khi VRAM trống ≥ **12,5 GiB + 0,4 GiB/LoRA** (hai LoRA: 13,3 GiB); không đủ/OOM mới thử CPU offload, vẫn dùng GPU từng phần nhưng tốn RAM hệ thống. Xem VRAM và chế độ sau khi nạp ở ô 6. Nếu OOM, giảm kích thước, tắt LoRA hoặc chọn `low_vram` rồi restart và Run all.

**Nếu ảnh thứ hai báo `Setting requires_grad=True on inference tensor outside InferenceMode`:** lấy notebook mới nhất và **Runtime → Restart runtime → Run all**; bản Studio giữ thao tác đổi LoRA trong `torch.inference_mode()` kể cả sau offload. Nếu còn lỗi, gửi traceback nhưng che URL Gradio.

**Nếu ô 2 báo xung đột thư viện:** lấy bản mới nhất, **Runtime → Restart runtime → Run all**. Dòng `ERROR: pip's dependency resolver...` có thể chỉ là cảnh báo; ô 2 chỉ được coi đã sẵn sàng khi in **`✅ Thư viện Studio đã sẵn sàng`**. Nếu không có dòng này hoặc có traceback, dừng và gửi đầy đủ traceback (che thông tin riêng). Việc cài `ultralytics==8.4.170` (cho auto-detailer) nằm trong khối `try/except` riêng: máy nào không cài được vẫn chạy Studio bình thường, chỉ auto-detailer báo lỗi khi bật.

---

# WAI-illustrious trên Google Colab (notebook nâng cao, không cần link)

[![Mở trong Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manhlee1196-boop/ai-anime/blob/main/WAI_Illustrious_Colab.ipynb)

**[WAI_Illustrious_Colab.ipynb](WAI_Illustrious_Colab.ipynb)** không mở URL chia sẻ, vẫn tạo ảnh bằng checkpoint WAI-illustrious SDXL với hai LoRA tùy chọn và inpainting ô 8.

## Chạy nhanh

1. Chọn GPU trong Runtime, chạy các ô **1 → 7**. `AUTO_DOWNLOAD=True` tải checkpoint v17 thiếu từ Hugging Face và ô 5 chỉ tải LoRA đã bật; không dùng Drive. Sửa prompt/negative/seed trực tiếp ở ô 7 (không có selector phong cách tại notebook này); PNG ở `/content/wai_outputs` và cần tải về trước khi ngắt phiên.
2. Có checkpoint riêng? Đặt `MODEL_PATH` thành **đường dẫn `.safetensors` dưới `/content`** ở ô 3; không ghi đè file đã có. Chỉ xác nhận là WAI v17 nếu trùng kích thước + SHA-256; nếu không, notebook chỉ kiểm header SDXL, hãy tắt hai LoRA khi dùng model khác. Nếu đổi LoRA/chế độ bộ nhớ sau khi nạp, restart và Run all để không giữ hai model trong RAM.
3. Ảnh còn lỗi tay/chân/mắt? Dùng ô 8 để khoanh mask inpaint vùng nhỏ; lưu một PNG khác, không ghi đè ảnh gốc.

## Nguồn và phiên bản đã đối chiếu

| Tài nguyên | Bản phát hành gốc / mục đích | Bản lưu tải tự động, ghim commit | Kích thước chính xác (byte) | SHA-256 toàn bộ file |
| --- | --- | --- | ---: | --- |
| Checkpoint **WAI-illustrious SDXL v17**, pruned FP16 | [Civitai version 2883731](https://civitai.com/api/v1/model-versions/2883731) | [HF LyliaEngine, commit `32be7bfd…`](https://huggingface.co/LyliaEngine/waiIllustriousSDXL_v170/blob/32be7bfdcd406db70df663b9cee3313957deb68f/waiIllustriousSDXL_v170.safetensors) | 6.938.040.682 | `f116b0c78ff441467b0cdc8f1936e1ed18ea31e9997c7b132b1b8db533f0bd04` |
| LoRA **Anatomy Helper V1** (Illustrious; tác giả mô tả hỗ trợ bàn tay, bàn chân, tư thế) | [Civitai version 1318504, file 1223247](https://civitai.com/api/v1/model-versions/1318504) | [HF ench100, commit `bed49d45…`](https://huggingface.co/ench100/bodyandface/blob/bed49d45df95c0695aedad3b2aa6aff389fb3777/anatomy_helper.safetensors) | 228.473.940 | `bf6a950036b7599212a2c68d65f3ba07b28689067e167915d2a0ecb2018c26ca` |
| LoRA **Eyes for Illustrious (Perfect anime eyes) V1** (Illustrious; trigger `perfect eyes`) | [Civitai version 2066663, file 1963176](https://civitai.com/api/v1/model-versions/2066663) | [HF Muapi, commit `1abbc862…`](https://huggingface.co/Muapi/eyes-for-illustrious-lora-perfect-anime-eyes/blob/1abbc862f53f5101962ebf1c337513aff91bd206/eyes-for-illustrious-lora-perfect-anime-eyes.safetensors) | 228.457.660 | `97c1a083ffe6b4d45c545196eabd01c754936b996ade0c9db6d072f3bd340c55` |

**Cách xác minh:** Đã đối chiếu metadata bản **Illustrious**/SHA-256 đầy đủ trên API Civitai với SHA-256 từ trang file/LFS HF ở đúng commit ghi trên. Hai kho HF là **mirror của bên thứ ba**, không khẳng định tài khoản HF thuộc tác giả Civitai. SHA-256 công bố ở hai nguồn **khớp nhau theo metadata**; notebook băm toàn bộ file thực nhận sau khi tải (kể cả đường dẫn LoRA do bạn cung cấp) và từ chối file sai hash. Không coi `AutoV2` 10 ký tự hoặc tên file là bằng chứng đầy đủ. Nếu nguồn đổi nội dung, file tải dở/hỏng hoặc sai biến thể (Pony/Anima/SD 1.5), sẽ **không nạp**. Không khẳng định đã kiểm nghiệm hiệu quả hình ảnh trên Colab; SHA **không** chứng minh tay/chân/mắt được sửa chính xác ở mọi ảnh. Xem điều khoản của từng model trước khi sử dụng/phân phối.

Bạn có thể tự kiểm tra SHA-256 trong Colab: `sha256sum /content/wai_model_cache/waiIllustriousSDXL_v170.safetensors /content/wai_lora_cache/*.safetensors`; đối chiếu cột hash trên.

## Tải và lưu tài nguyên

- Checkpoint ~6,94 GB tải một lần vào **`/content/wai_model_cache/waiIllustriousSDXL_v170.safetensors`**, kiểm đầy đủ SHA-256, rồi nạp trực tiếp **cùng file**. Trong cùng runtime dùng lại file đã xác minh mà không tải/sao chép lần nữa. Thiếu khoảng 9 GiB đĩa trống thì báo lỗi; không chuyển sang lưu trữ khác. `AUTO_DOWNLOAD=False` yêu cầu file có sẵn tại `MODEL_PATH`.
- Hai LoRA mỗi file ~228 MB. Khi bật, ô 5 dùng lại file được xác minh trong **`/content/wai_lora_cache`** hoặc tải file còn thiếu từ commit HF ghim; có thể dùng file Civitai gốc tại `ANATOMY_LORA_PATH`/`EYE_LORA_PATH` **dưới `/content`** nếu đúng kích thước + SHA-256. File sai hash bị từ chối, không tự ghi đè. Nếu đĩa ít, tắt LoRA không cần. Không gắn hay lưu Google Drive ở bất kỳ ô nào.
- `diffusers==0.35.2`, `transformers==4.52.4`, `accelerate==1.10.1`, `peft==0.17.1` nạp checkpoint và LoRA cùng PyTorch/CUDA của Colab. Cấu hình/tokenizer SDXL được lấy khi cần; không tải trọng số SDXL base hoặc checkpoint inpaint riêng. Không đưa weights/ảnh vào repo. **`/content` bị xóa khi phiên Colab kết thúc**; muốn giữ ảnh hãy tải xuống.

## Tối ưu và sửa ảnh

- FP16 + SDPA của PyTorch; Euler a; mặc định 25 bước / CFG 6 / 1024×1024. Mỗi lượt một ảnh. `VRAM_MODE=auto` **ưu tiên GPU trực tiếp** nếu VRAM trống trước khi nạp ≥ **12,5 GiB + 0,4 GiB cho mỗi LoRA bật** (hai LoRA: 13,3 GiB); thấp hơn thì CPU offload (chậm hơn, tốn RAM hệ thống dù suy luận vẫn dùng GPU từng phần). Bật VAE slicing và tiling ở cả hai chế độ để giảm đỉnh VRAM giải mã ảnh. Nếu nạp/tạo ảnh OOM ở `auto`, notebook nạp lại với offload rồi thử lại **cùng seed** khi tạo ảnh, dùng file đã xác minh. Ô 6 in VRAM trước/sau khi nạp và chế độ cuối; muốn giảm ảnh hưởng của pipeline cũ, hãy khởi động lại runtime trước khi chạy lại toàn bộ notebook. Mặc định strength anatomy `0.55`, mắt `0.45` (các mức thử nghiệm, không phải đảm bảo chất lượng); muốn so sánh, giữ seed và tắt từng LoRA.
- **Ô 8: inpaint đúng vị trí.** Sau khi thấy ảnh ở ô 7, nhập `BOXES="x1,y1,x2,y2"` (tọa độ pixel ảnh; góc trên-trái đến góc dưới-phải, ví dụ `100,300,240,490`), hoặc tạo ảnh mask **trắng = sửa / đen = giữ nguyên**, đúng kích thước ảnh, nhập `MASK_PATH`. Chỉ dùng **một** trong hai cách. `SOURCE_IMAGE` để trống lấy ảnh ô 7; có thể trỏ tới PNG/JPG khác. Chọn `TARGET=hands/legs/eyes`, chỉnh `REFINE_STRENGTH` (thử 0.35–0.55), seed và chạy ô 8. Nên sửa từng vùng nhỏ riêng để không làm đổi mặt/trang phục; có thể sửa PNG kết quả lần trước ở lượt tiếp theo. Sử dụng `AutoPipelineForInpainting.from_pipe(pipe)` để **chia sẻ** checkpoint SDXL hiện có; với UNet 4 kênh, mask định hướng tạo lại vùng trắng, sau đó chỉ ghép vùng đã chọn với viền mềm. Không tự nhận diện vị trí lỗi; khoanh sai chỗ hoặc strength cao có thể làm ảnh xấu hơn. Nếu inpaint OOM, auto sẽ thử offload, nếu vẫn OOM hãy chọn ảnh nhỏ hơn/`low_vram`.
- Anatomy Helper được mô tả tập huấn luyện có nhiều ảnh bàn chân và có thể kéo theo thiên lệch phong cách/nội dung; bạn có thể **tắt LoRA này** hoặc chỉ dùng inpaint thủ công. LoRA mắt dùng trigger `perfect eyes`: notebook nâng cao tự thêm khi bật; Studio **không tự thêm** mà có nút **Thêm trigger `perfect eyes` cho LoRA mắt** để bạn bấm rồi sửa/xóa tùy ý. LoRA tay/chân chủ yếu hướng tới **bàn tay/bàn chân và tư thế**, không chứng nhận chữa lỗi ống chân; dùng inpaint ô 8 cho các lỗi chân còn lại.
- Nếu hết **RAM hệ thống** khi nạp checkpoint, cần Colab high-RAM; notebook không thể cấp thêm GPU/RAM. Ảnh chỉ nằm dưới `/content` và mất khi runtime kết thúc (hãy tải xuống trước). `EMBED_METADATA=True` nhúng prompt/nguồn ảnh vào PNG; tắt trước khi chia sẻ nếu không muốn lộ thông tin.
- **Nạp thư viện prompt trong giao diện Studio (ô 7, đã tối ưu cho điện thoại).** Danh sách prompt hiện là **danh sách chạm (radio) cuộn được** thay cho dropdown lọc: mỗi dòng cao ≥42 px, khung giới hạn ~44% chiều cao màn hình nên không còn cảnh bàn phím ảo che danh sách; **chạm một dòng là nạp ngay**, và nếu thao tác chạm không như ý thì chọn dòng rồi bấm **⬇️ Nạp prompt đã chọn**. Chi tiết định dạng đọc file bên dưới. Mở accordion **📚 Thư viện prompt · nạp danh sách từ file text**, tải file `.txt`/`.md`/`.json` (tối đa 2 MB) hoặc dán nội dung, hệ thống đọc ra danh sách để lọc và **chọn một dòng là prompt được nạp thẳng vào ô *Prompt gửi model***. Định dạng được nhận: `PROMPT 01 - Tên tiếng Việt` rồi đoạn prompt bên dưới, bảng một dòng `Tên tiếng Việt | Nội dung prompts tiếng Anh`, JSON `[{"title", "prompt"}]`, hoặc các đoạn prompt cách nhau dòng trống. Các dòng `Negative:`, `Steps:`, `CFG:`, `Size: 832x1216`, `Seed:` trong mỗi prompt cũng được áp dụng và bị kẹp về dải giao diện cho phép (steps 10–45, CFG 1–12, kích thước theo preset); prompt dài hơn 2000 ký tự bị cắt bớt. File chỉ được đọc trong phiên Colab, không tải lên dịch vụ nào; chưa có file thì bấm **Nạp thư viện mẫu** (12 prompt) để xem định dạng.
- **Giao diện Studio gọn một màn hình.** Ô 7 xếp lại thành hai cột: cột trái là prompt/negative + trigger `perfect eyes` + các accordion **đóng sẵn** (📚 Thư viện prompt, 🧭 Quy trình chuẩn, 🎨 Gợi ý phong cách, ⚙️ Thông số ảnh và LoRA) + ba tab chế độ; cột phải là gallery kết quả. Điều khiển ghép thành từng cặp trên một hàng (steps/CFG, seed/số ảnh, checkbox LoRA + cường độ), nút phụ dùng cỡ nhỏ, nút tạo ảnh cỡ lớn; ghi chú phụ chuyển thành chữ nhỏ màu nhạt hoặc dòng `info` dưới ô nhập nên không còn các đoạn văn dài chiếm chỗ. Hiện có **21 sự kiện** Gradio: 4 nút tạo ảnh, 3 nút *Dùng ảnh mới nhất*, 2 sự kiện gợi ý tag inline, trigger mắt, gợi ý sửa vùng, gợi ý phong cách, sắp xếp prompt, nạp negative, kiểm tra prompt, 5 sự kiện thư viện prompt và nút **⬇️ Nạp prompt đã chọn**. Hành vi vẫn như cũ: đúng hai ô *Prompt/Negative gửi model* là nội dung gửi model, không thẻ ẩn.
- **🧭 Quy trình chuẩn · khung prompt + negative tối ưu (ô 7).** Ba nút hỗ trợ, tất cả **chỉ ghi vào nội dung hiển thị** để bạn sửa: **Sắp xếp prompt theo thứ tự chuẩn** (chất lượng → chủ thể → ngoại hình → trang phục → tư thế → bố cục → bối cảnh → ánh sáng → phong cách → `absurdres`, bỏ thẻ trùng, thêm thẻ neo của khung đã chọn), **Nạp negative đã chọn** (8 bộ: chuẩn nhà phát hành WAI v17, Illustrious chuẩn, tay/chân, giữ chất 2D, chân dung, phong cảnh, an toàn nội dung, inpaint; ghi đè hoặc nối thêm) và **🩺 Kiểm tra prompt & thông số** (ước lượng token so với khối 75 token của SDXL, thẻ chất lượng thừa/trùng/vừa dương vừa âm, cú pháp Pony, trọng số > 1.2, negative quá dài, steps/CFG/kích thước/hires ngoài khuyến nghị). Accordion **🎨 Gợi ý phong cách** chỉ thêm các thẻ mô tả (màu mắt, kiểu/màu móng) vào hai ô đang hiển thị. Quy trình đầy đủ: [`docs/QUY_TRINH_TAO_ANH.md`](docs/QUY_TRINH_TAO_ANH.md).
- **Tự sửa mặt/tay (auto-detailer, tùy chọn).** Ô 2 **cài thêm `ultralytics==8.4.170`** (chỉ cảnh báo nếu thất bại, các chế độ khác vẫn chạy). Khi bật mục **Tự sửa mặt/tay** trong *Thông số ảnh và LoRA*, Studio tải weight YOLOv8 (`face_yolov8n.pt`, `hand_yolov8n.pt`) từ `Bingsu/adetailer` @ `c310c216` vào **`/content/wai_detailer_cache`**, **kiểm SHA-256 đầy đủ** (khớp metadata HF đã đối chiếu) rồi phát hiện mặt/bàn tay và inpaint lại đúng vùng đó với strength/ngưỡng/số vùng bạn chọn. Weight sai hash bị xóa và không nạp; cần mạng ở lần bật đầu tiên. Tính năng này chỉ là gợi ý chỉnh sửa, **không đảm bảo** hết lỗi ngón/mặt.

Kiểm tra cấu trúc notebook và hành vi tải/hash/nạp LoRA bằng mock CPU: `python -m unittest discover -s tests -v` (hiện **58 test: 57 đạt, 1 bỏ qua** vì cần PyTorch thật). `colab/studio.py` là bản mã nguồn tương ứng của ô 7 và `python scripts/build_colab_studio.py` tái tạo **chính xác** notebook từ nó — sửa giao diện ở `colab/studio.py` rồi chạy lại script. Việc tải thực tế checkpoint/LoRA/weight YOLOv8, khả năng chạy với GPU Colab và chất lượng ảnh/tay-chân-mắt **chưa thể xác nhận** trong môi trường kiểm thử CPU này.

---

# Mirai Studio — Cloudflare SDXL (tùy chọn, không cần cho Colab WAI)

**Nếu bạn chỉ dùng WAI với liên kết cá nhân Colab, dừng ở hướng dẫn đầu trang; không cần triển khai phần này.** Mã ứng dụng ở **[`web/`](web/)** gồm giao diện React/Vite, Worker API và Cloudflare Workers AI binding. Đây là ứng dụng khác với notebook Colab ở trên: **Cloudflare Workers AI chạy SDXL Base 1.0 / SDXL Lightning do Cloudflare lưu trữ, KHÔNG chạy checkpoint WAI-illustrious v17**. Muốn dùng đúng WAI cần GPU bên ngoài; xem mục bên dưới. Ảnh mẫu trong giao diện là ảnh minh họa đóng gói sẵn, **không phải ảnh ứng dụng vừa tạo**.

## Tính năng

- Văn bản → ảnh, ảnh → ảnh và inpainting: tô mask trực tiếp, tẩy/hoàn tác/xóa hoặc nạp mask PNG **trắng = vùng sửa, đen = phần giữ**. Mask hướng dẫn model; với Workers AI, không cam kết mọi pixel ngoài mask hoàn toàn không đổi.
- Prompt, negative prompt, gợi ý phong cách; SDXL Base/Lightning hoặc WAI qua GPU tùy chọn; kích thước/tỷ lệ, steps, CFG, seed, strength, tạo lần lượt 1–4 ảnh; tùy chọn sampler, CLIP skip và hai LoRA **chỉ khi backend GPU WAI hỗ trợ**.
- Lưu ảnh trong **IndexedDB của trình duyệt** (20 ảnh gần nhất không yêu thích; ảnh yêu thích được giữ), xem phóng to, sao chép prompt, dùng ảnh kết quả để sửa tiếp, tải PNG/JPG/WebP. Không có server lưu bộ sưu tập. Prompt/ảnh nguồn vẫn được gửi đến **Cloudflare Workers AI hoặc backend GPU bạn cấu hình** để suy luận; tránh đưa dữ liệu nhạy cảm nếu chưa tin tưởng nhà cung cấp. Xóa dữ liệu trang web hoặc dùng chế độ riêng tư có thể làm mất lịch sử — hãy tải ảnh quan trọng về máy.
- Bảo vệ API tạo ảnh bằng secret `APP_ACCESS_TOKEN` trên Worker. Mã nhập từ giao diện lưu trong `sessionStorage` của phiên trình duyệt; **không** đưa Cloudflare API token vào giao diện hay Git.

## Xem giao diện ở máy cá nhân

Cần Node.js 20.19+ (hoặc 22.12+) và npm.

```bash
cd web
npm ci
npm run dev
```

Mở URL Vite in ra. Bản xem trước này **chỉ hiển thị giao diện và ảnh mẫu**; `/api/generate` trả 503 minh bạch, không giả lập kết quả tạo ảnh. Muốn thử Worker cục bộ: sau khi có quyền vào tài khoản Cloudflare, có thể chạy `npm run worker:dev` và cấu hình secret bằng `.dev.vars` (không commit); Workers AI vẫn cần kết nối dịch vụ của Cloudflare và hạn mức tài khoản. AI binding có thể báo `not supported` khi Wrangler không truy cập được Cloudflare: khi đó chỉ kiểm tra được routing/asset, **không thể tạo ảnh cục bộ**. Binding AI vẫn có thể tính phí trong dev. Không dùng Vite như một inference server.

## Triển khai lên tài khoản Cloudflare của bạn

> **Chưa triển khai hộ:** cần tài khoản Cloudflare có Workers AI và hạn mức/quyền sử dụng, đăng nhập Wrangler trên máy của bạn. Yêu cầu tạo ảnh có thể tính phí; khóa API được bảo vệ nhưng ai biết mã truy cập đều có thể tiêu thụ hạn mức. Không chia sẻ mã này.

```bash
cd web
npm ci
npx wrangler login
npm run deploy             # build giao diện rồi triển khai Worker + assets
npx wrangler secret put APP_ACCESS_TOKEN  # tự đặt mã dài, ngẫu nhiên tại lời nhắc
```

`web/wrangler.jsonc` khai báo binding `AI`, static assets và ưu tiên Worker cho `/api/*`; có thể sửa `name` nếu tên Worker trùng. **Chưa đặt `APP_ACCESS_TOKEN` thì API từ chối mọi yêu cầu tạo ảnh (503)**. Sau khi đặt secret, mở URL Worker do Wrangler hiển thị, chọn **Cloudflare AI**, nhập **chính mã APP_ACCESS_TOKEN vừa đặt** qua nút *Mã truy cập*, rồi tạo ảnh. `/api/config` cho biết nguồn đã cấu hình nhưng không xuất giá trị secret. Nên dùng Cloudflare Access/rate limiting bổ sung nếu công khai URL cho nhiều người; mã đơn lẻ chỉ là lớp bảo vệ tối thiểu.

Kiểm tra sau triển khai: `/api/config` phải trả JSON `cloudflare: true`, `configured: true`. Thử một ảnh 512×512 với prompt đơn giản, tải file và kiểm tra PNG. Nếu lỗi hạn mức/binding, xem nhật ký `npx wrangler tail`. **Chưa có tài khoản/triển khai hoặc GPU trong môi trường kiểm thử repo này, nên chưa xác nhận ảnh tạo thực tế.**

### Giới hạn và ý nghĩa thông số

| Tùy chọn | Phạm vi giao diện/API | Lưu ý |
| --- | --- | --- |
| Kích thước | Cạnh 512–1344 px, chia hết cho 8; tối đa 1,5 MP, tỷ lệ tối đa 1,75:1 | Ảnh nguồn được thu phóng về cạnh dài tối đa 1024; quá dài sẽ yêu cầu cắt trước. |
| Steps | Workers AI: 1–20; WAI GPU: 1–45 | Lightning thường 4–8 bước; nhiều bước hơn tăng thời gian/chi phí. |
| CFG / Guidance | 1–12 | Giá trị cao chưa chắc đẹp hơn. |
| Seed | -1 (ngẫu nhiên) hoặc 0–4294967295 | Lượt tiếp theo tăng seed 1; cùng seed **không** đảm bảo ảnh giống nhau giữa model/backends. |
| Strength | 0,20–0,95 cho img2img/inpaint | Thấp giữ ảnh nguồn tốt hơn; cao thay đổi nhiều hơn. |
| Ảnh đầu vào | PNG/JPEG/WebP ≤12 MB khi chọn file; Worker nhận ảnh chuẩn hóa ≤2 MB và mask PNG ≤1 MB | Body JSON tối đa 5 MB; tùy chọn mask tải vào phải đúng kích thước ảnh nguồn. |
| Batch | 1–4 yêu cầu nối tiếp | Dừng chờ ở trình duyệt có thể không dừng tác vụ và chi phí trên server. |

Worker chỉ chuyển **tham số thật sự có trong schema** của SDXL tới Workers AI: `prompt`, `negative_prompt`, `width`, `height`, `num_steps`, `guidance`, `seed`, cùng `image_b64` và `strength` khi img2img; inpaint gửi `image` và `mask` dạng mảng byte. **Không gửi LoRA, sampler hoặc CLIP skip tới Workers AI**. [SDXL Base schema](https://developers.cloudflare.com/workers-ai/models/stable-diffusion-xl-base-1.0/) · [SDXL Lightning schema](https://developers.cloudflare.com/workers-ai/models/stable-diffusion-xl-lightning/). Model/cước/hạn mức của Cloudflare có thể thay đổi; kiểm tra tài khoản trước khi dùng nhiều ảnh.

## Tùy chọn: WAI v17 trên GPU bên ngoài

**Cloudflare Worker không thể nạp checkpoint WAI ~6,94 GB** trong môi trường Worker; nhánh này chỉ là cổng HTTPS tới **máy GPU do bạn tự triển khai**. Không có máy GPU mặc định hoặc endpoint WAI dùng chung. Nếu chỉ cần Workers AI SDXL, **không cần** cấu hình nhánh này.

Nếu đã có một server GPU đủ RAM/VRAM, máy đó phải nạp checkpoint **WAI v17** và hai LoRA **đúng SHA-256 trong bảng ở trên**, tự thực hiện inference và cung cấp endpoint. Worker **không tự tải/xác minh** file trên máy GPU; hãy áp dụng kiểm tra hash trong notebook/backend của bạn. Khi bật LoRA mắt, giao diện tự thêm trigger `perfect eyes` vào prompt gửi đi nếu chưa có. Trên máy quản lý Cloudflare:

```bash
cd web
npx wrangler secret put GPU_BACKEND_URL    # ví dụ: https://gpu.example.org/api
npx wrangler secret put GPU_BACKEND_TOKEN  # Bearer token riêng của máy GPU
```

Worker POST tới `${GPU_BACKEND_URL}/generate` qua HTTPS, thêm `Authorization: Bearer <GPU_BACKEND_TOKEN>`, body JSON:

```json
{
  "mode": "inpaint",
  "model": "wai-v17",
  "prompt": "anime portrait, perfect eyes",
  "negative_prompt": "bad anatomy",
  "width": 1024,
  "height": 1024,
  "steps": 25,
  "cfg": 6,
  "seed": 123,
  "strength": 0.45,
  "scheduler": "euler_a",
  "clip_skip": 2,
  "loras": {
    "anatomy": { "enabled": true, "weight": 0.55 },
    "eyes": { "enabled": true, "weight": 0.45 }
  },
  "image_b64": "iVBORw0KGgo...",
  "mask_b64": "iVBORw0KGgo..."
}
```

`image_b64` chỉ có ở img2img/inpaint; `mask_b64` chỉ có ở inpaint. Worker chuyển về **chuỗi base64 thuần** (không có tiền tố data URL) trước khi gửi đến GPU. GPU backend **phải** tự xác thực Bearer token, kiểm tra lại kích thước/định dạng/LoRA, xử lý sampler/CLIP skip theo khả năng hoặc **từ chối** tham số không hỗ trợ, thực hiện sinh ảnh thật rồi trả **raw PNG/JPEG/WebP bytes** với `Content-Type` tương ứng. JSON chứa URL/base64 **không** được chấp nhận làm kết quả. Đặt timeout, giới hạn lượt, lọc nội dung và bảo vệ dữ liệu ở backend theo nhu cầu. Nếu muốn giữ nguyên pixel ngoài mask, backend phải tự ghép/composite như notebook; Worker không làm thay. Không có backend GPU đi kèm: bật WAI trong giao diện khi chưa có endpoint sẽ báo không sẵn sàng.

## Kiểm thử mã

```bash
python -m unittest discover -s tests -v   # notebook, dùng mock CPU
cd web
npm ci
npm test                              # API auth/validation/routes + IndexedDB mock
npm run build                         # sản phẩm Vite
npx wrangler deploy --dry-run         # kiểm tra Worker/binding, không triển khai
```

Các bài test và build **không phải** là bằng chứng checkpoint WAI đã chạy trên GPU hoặc Workers AI đã sinh ảnh thật trên một tài khoản Cloudflare.

### Kho thẻ Danbooru / e621 cho prompt

Trong `WAI_Illustrious_Studio_Colab.ipynb`, mở tab **🏷️ Kho thẻ**.
Nguồn là `danbooru_e621_merged_2026-10-01_pt20-ia-dd-ed-spc.csv` ở thư mục gốc
(349.714 dòng, không có header: tên thẻ, mã danh mục, số lượt, bí danh, nhãn tiếng Việt).
Cột thứ năm là chú giải hiển thị; nhãn dịch được lập chỉ mục để tìm kiếm, còn tên thẻ và bí danh tiếng Anh vẫn được giữ nguyên.

- Tìm quét toàn bộ 349.714 thẻ bằng tiếng Việt hoặc English; dấu gạch dưới, gạch nối và khoảng trắng tương đương, có thể gõ tiếng Việt không dấu.
- Ngay dưới ô **Prompt**, khi gõ tiếng Việt hoặc English sẽ có tối đa 8 gợi ý cho cụm cuối (sau dấu phẩy/chấm phẩy/xuống dòng). Lần đầu trong runtime có thể cần chờ nạp CSV; dữ liệu sau đó được dùng lại. Chọn nhãn Việt — tag gốc, rồi bấm **Thêm tag đã chọn** để thêm tag tiếng Anh chuẩn vào cuối prompt; phần đang viết vẫn được giữ nguyên và không tự chèn khi gõ.
- Có thể dán prompt nhiều cụm, phân tách bằng dấu phẩy, chấm phẩy hoặc xuống dòng; nếu cụm dài không khớp nguyên văn, tìm tiếp theo các từ khóa riêng trong cụm.
- Lọc theo danh mục của nguồn, kết hợp chủ đề; sắp xếp phổ biến nhất hoặc A–Z. Chủ đề được suy đoán từ tên thẻ, không phải nhãn chính thức của CSV.
- Chọn nhiều kết quả, chọn Prompt hoặc Negative prompt rồi nhấn **Thêm thẻ đã chọn**. Prompt nhận tên thẻ tiếng Anh chuẩn; nhãn Việt chỉ dùng để hiển thị/tìm kiếm.
- Kho được nạp một lần vào RAM mỗi runtime; mỗi trang có 60 kết quả. Đây là từ khóa tạo prompt, không phải checkpoint, LoRA hay bộ ảnh huấn luyện; nội dung có thể chứa từ khóa nhạy cảm.

#### Dùng kho thẻ trong Colab / gradio.live

`WAI_Illustrious_Studio_Colab.ipynb` cũng có bộ chọn **🏷️ Kho thẻ Danbooru / e621 · danh mục & chủ đề**
ngay dưới Prompt / Negative. Chạy notebook rồi mở link Gradio ở ô 8:

Gợi ý inline ở ngay dưới Prompt dò cụm cuối khi bạn gõ. Chọn kết quả rồi bấm **Thêm tag đã chọn** để nối tên tiếng Anh chuẩn vào prompt mà không xóa phần tiếng Việt; muốn tìm nhiều cụm cùng lúc thì mở tab kho thẻ:

1. Mở mục kho thẻ và nhấn **Tìm / tải kho thẻ**. Colab tải bản CSV gốc ~9 MB từ GitHub ở commit cố định và kiểm tra SHA-256; bản trong repo đã có thêm cột nhãn tiếng Việt (~13,7 MB), cả hai dạng đều được hỗ trợ.
2. Nhập cụm tiếng Việt hoặc English, hoặc dán các tag trong prompt cách nhau bằng dấu phẩy/xuống dòng; bộ tìm kiếm quét toàn bộ kho, bỏ qua dấu tiếng Việt và tìm cả tên/bí danh. Chọn danh mục/chủ đề và cách sắp xếp nếu cần; nhấn **Tìm** để cập nhật. Mỗi trang có 60 kết quả; nhập số trang rồi nhấn Tìm.
3. Chọn nhiều thẻ trong kết quả, chọn Prompt hoặc Negative prompt, nhấn **Thêm thẻ đã chọn**. Nội dung hiển thị trực tiếp trong ô tương ứng; không có thẻ nào được thêm ngầm khi tạo ảnh. Đổi kết quả tìm kiếm sẽ xóa lựa chọn chưa thêm.

Nếu GitHub không truy cập được, dùng bảng **Files** bên trái Colab để tải đúng CSV lên `/content/`, rồi nhấn Tìm lại.
Kho chỉ nạp một lần vào RAM mỗi runtime, dùng chung dữ liệu chỉ đọc; prompt/lựa chọn vẫn riêng từng phiên Gradio.
Không đưa toàn bộ danh sách 349.714 thẻ xuống trình duyệt. Lỗi kho thẻ không chặn viết prompt hoặc tạo ảnh.
Sau khi cập nhật notebook, cần chạy lại ô 7–8 để giao diện Gradio đang chạy nhận tính năng mới.
Mã nằm trong `colab/studio.py`; tái tạo notebook bằng `python scripts/build_colab_studio.py`.

#### Bố cục tab ngang của Gradio Studio

Giao diện mặc định mở **✦ Tạo ảnh**; các mục trước đây xếp dọc được gom thành 6 tab:

| Tab | Nội dung |
| --- | --- |
| ✦ Tạo ảnh | Văn bản → ảnh, Ảnh → ảnh, Phóng to ảnh, Sửa vùng ảnh |
| 🏷️ Kho thẻ | CSV Danbooru/e621, tìm kiếm, danh mục, chủ đề |
| 📚 Thư viện | Nạp file/dán danh sách hoặc chọn prompt mẫu |
| 🧭 Quy trình | Khung prompt, negative tối ưu, kiểm tra prompt/thông số |
| ⚙️ Thông số | Steps, CFG, seed, số ảnh, LoRA, metadata PNG |
| ✨ Chi tiết | Hires fix, auto-detailer, chi tiết mắt/móng |

Prompt/Negative dùng chung nằm ngoài tab; chuyển tab không tạo bản sao hoặc đặt lại giá trị.
Khung kết quả vẫn ở bên cạnh trên màn hình rộng, xuống dưới trên màn hình nhỏ.
Thanh tab cuộn ngang trên điện thoại. Các nút dùng ảnh mới nhất tự mở đúng tab đích
sau khi nạp ảnh thành công. Chạy lại ô 7–8 của notebook đã cập nhật để áp dụng.

Kho thẻ Gradio hiển thị nhãn tiếng Việt trước tên gốc, ví dụ **Tóc dài — long_hair**,
**Mắt xanh dương — blue_eyes**. Có thể tìm bằng nhãn Việt có dấu/không dấu, tag hoặc
bí danh tiếng Anh; nhiều cụm trong một prompt được tìm riêng trên toàn bộ catalog.
CSV có cột chú giải Việt ở vị trí thứ năm. Từ điển/quy tắc dịch bao phủ thẻ thông dụng
và các tổ hợp rõ nghĩa; đây **không phải bản dịch máy đầy đủ cho 349.714 thẻ**. Tên
họa sĩ/nhân vật/tác phẩm được giữ nguyên kèm nhãn loại; mục chưa dịch được ghi rõ.
Khi thêm kết quả, prompt nhận đúng tên thẻ tiếng Anh gốc — nhãn Việt không gửi vào model.
Tạo lại cột nhãn bằng `python scripts/add_vietnamese_tag_captions.py`; sau khi chỉnh từ điển
`TAG_VI_LABELS`/quy tắc trong `colab/studio.py`, chạy script trên, cập nhật
`TAG_CSV_SHA256` theo hash được in ra rồi chạy `python scripts/build_colab_studio.py`.
