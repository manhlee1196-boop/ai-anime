# WAI Studio cá nhân — giao diện tạo ảnh qua Google Colab

[![Mở WAI Studio trong Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manhlee1196-boop/ai-anime/blob/main/WAI_Illustrious_Studio_Colab.ipynb)

**[WAI_Illustrious_Studio_Colab.ipynb](WAI_Illustrious_Studio_Colab.ipynb)** chạy WAI-illustrious v17 và hai LoRA tùy chọn trên GPU Colab; Gradio tạo **link tạm không cần đăng nhập**. Không cần Google Drive hoặc API token; link Gradio chính không cần Cloudflare, còn ô 9 có thể tạo Cloudflare Quick Tunnel dự phòng. Checkpoint và LoRA được kiểm **toàn bộ SHA-256** trước khi nạp. [Báo cáo kiểm tra nguồn và vận hành](VERIFICATION.md) phân biệt điều đã thử cục bộ với việc **chưa thử tạo ảnh bằng GPU Colab thật**.

1. Mở notebook từ nút trên → **Runtime → Change runtime type → T4 GPU** (hoặc mạnh hơn) → **Runtime → Run all**. Lần đầu notebook tải checkpoint ~6,94 GB và tối đa ~457 MB LoRA **trực tiếp vào `/content`**, không gắn Drive/không sao chép file qua lại. Cần khoảng **9 GiB đĩa trống**; tốc độ vẫn tùy mạng Colab và bước kiểm hash. Chọn/tắt LoRA, `VRAM_MODE` và đường dẫn tùy chỉnh **dưới `/content`** ở ô 3 trước khi nạp. Để `ANATOMY_LORA_PATH`/`EYE_LORA_PATH` rỗng nếu muốn tải tự động. **Cả 9 ô code được thu gọn thành form của Colab** — mỗi ô chỉ còn một thanh tiêu đề kèm nút **Run** (kể cả ô 7 chứa ~1.500 dòng mã giao diện và ô 9 tạo link dự phòng qua Cloudflare Quick Tunnel), nên trang ngắn và chạy tuần tự dễ; muốn xem hoặc sửa code của ô nào thì bấm biểu tượng `>_` (⋮ → *Show code*) ở góc ô đó, output vẫn hiện bình thường.
2. Ô 8 mở Gradio nội bộ, **ô 9 tạo URL `https://….trycloudflare.com` tới cùng giao diện — đây là đường khuyến nghị** để mở Studio tạo ảnh bằng văn bản, ảnh → ảnh hoặc mask sửa tay/chân/mắt (có phóng to và **auto-detailer** nếu bật); không cần tài khoản Cloudflare/API token hay nạp lại model. Mặc định **tắt** liên kết `gradio.live` vì phải đi qua máy chủ trung gian công cộng (dễ nghẽn, kẹt giao diện); cần thì đặt `GRADIO_SHARE = True` ở đầu ô 8. Ô 9 chạy tự động khi **Run all** và dùng link công khai, không có đăng nhập — đừng chia sẻ. Studio có ô prompt/negative cố định, nhóm điều khiển theo tác vụ và tab riêng cho **Kho thẻ**, **Thư viện**, **Quy trình**, **Thông số** và **Chi tiết**; trên điện thoại các cột xếp dọc, thanh tab cuộn ngang. **Không có selector phong cách: bạn tự viết phong cách trong prompt** (hoặc nạp từ tab **📚 Thư viện** / dùng công cụ trong **🧭 Quy trình**). Khi gõ từ khóa chủ đề ở cuối prompt (ví dụ `mắt`/`eyes`), các tag tiếng Anh khớp trong CSV hiện ra; chọn một tag để thay từ khóa. Danh sách này **không hiển thị toàn bộ** số kết quả khớp: nó cắt ở N thẻ có heat cao nhất, mặc định `16` (khi khớp 559 thẻ, dropdown chỉ hiện 16 thẻ hot nhất + ghi rõ `16 tag từ CSV trong 559 kết quả`); chỉnh N bằng ô **Số gợi ý** `16/32/64/150` ngay cạnh dropdown — tab **🏷️ Kho thẻ** (phân trang 60 dòng, lọc theo nhóm/chủ đề) mới là nơi xem hết. Autocomplete theo **Semi-Auto Tag Complete** của [Character Select SAA](https://github.com/mirabarukaso/character_select_stand_alone_app): `*đuôi` và `*giữa*` tìm theo hậu tố/trung tố, `@tên` chỉ lọc thẻ họa sĩ, mỗi dòng gợi ý có nhãn danh mục `[G] [A] [©] [C] [M]` (Danbooru) hoặc `<G> <A> <©> <C> <S> <M> <L>` (e621), và `Ctrl+↑/↓` (hoặc nút `±0,1`) chỉnh trọng số tag đang chọn theo bước `0,1`. CSV được tải/đọc và xác minh SHA-256 ngay khi Studio khởi động, dùng chung với tab Kho thẻ; nếu tải thất bại, Studio sẽ thử lại khi bạn gõ hoặc tìm trong kho. Điều chỉnh kích thước, steps, CFG, seed, strength, số ảnh, LoRA và **hai ô prompt thực sự gửi model**.
3. Ảnh chỉ lưu dưới **`/content/wai_outputs`** (hoặc thư mục `/content` được chọn); model ở `/content/wai_model_cache`, LoRA ở `/content/wai_lora_cache`. **Tải ảnh về trước khi phiên Colab kết thúc**: cả ảnh và weights cục bộ sẽ mất khi runtime ngắt, phiên sau cần tải lại. Link cũng ngừng hoạt động khi Colab ngắt.

**Ảnh độ phân giải lớn (hires fix):** trong tab **✨ Chi tiết**, nhóm **🔍 Ảnh độ phân giải cao** cho chọn `1.25×`, `1.5×`, `1.75×` hoặc `2×` cho *Văn bản → ảnh* và *Ảnh → ảnh*. Studio tạo ảnh gốc (~1 MP), chạy Real-ESRGAN 4x+ Anime6B bằng tile rồi lấy mẫu xuống đúng kích thước đã chọn; sau đó WAI img2img tinh chỉnh với cùng prompt/seed/LoRA theo **Hires strength** (mặc định 0,4; thấp giữ bố cục, cao thêm chi tiết nhưng dễ đổi nét). Weight (~18 MB) chỉ tải lần đầu vào `/content/wai_upscaler_cache`, được kiểm tra kích thước và SHA-256 trước khi nạp. SHA hiện ghim theo metadata của mirror Hugging Face; do sandbox lỗi TLS khi tải asset GitHub chính thức nên **chưa đối chiếu độc lập digest với file chính thức** (xem `VERIFICATION.md`). Kích thước đầu ra chia hết cho 8, tối đa ≈4,2 MP; nếu vượt, hệ số tự giảm và thanh trạng thái báo rõ. Ví dụ `1024×1024` ở `2×` → `2048×2048`, `832×1216` ở `1.5×` → `1248×1824`. VAE tiling giảm VRAM cho lượt img2img; Real-ESRGAN cũng xử lý theo tile và có thể rơi về CPU, nhưng vẫn tốn thời gian/bộ nhớ; nếu OOM hãy chọn hệ số nhỏ hơn hoặc kích thước gốc nhỏ hơn. PNG và metadata (nếu bật) ghi kích thước cuối, hệ số và tên upscaler. Tab **⤢ Phóng to ảnh** dùng cùng Anime6B + img2img cho ảnh tải lên hoặc ảnh **chọn từ ô nguồn** ở khung Kết quả, không tạo ảnh gốc mới; cùng giới hạn ≈4,2 MP. Tab *Sửa vùng ảnh* không hỗ trợ hires và vẫn thu ảnh về cạnh dài tối đa 1024 px. Chưa thử inference/GPU/UI thật trong sandbox (xem `VERIFICATION.md`).

**Prompt do bạn viết, gửi nguyên văn:** Studio không còn preset phong cách — hai ô **“Prompt gửi model · tự viết phong cách của bạn”** và **“Negative gửi model · ngón tay / ngón chân”** là **chính xác** những gì được gửi ở cả ba chế độ, không tự ghép thêm thẻ theo LoRA hay vùng sửa khi bấm tạo. Mọi phong cách (`anime illustration, cel shading`, `watercolor`, `cinematic lighting`…) là từ khóa bạn tự viết hoặc nạp từ **thư viện prompt**. Hai nút hỗ trợ đều **hiển thị trong ô để bạn sửa/xóa**: **Thêm trigger `perfect eyes` cho LoRA mắt** và, trong tab **Sửa vùng ảnh**, **Thêm gợi ý sửa vùng vào prompt đang hiển thị**; bật LoRA mắt vẫn tải và dùng adapter kể cả khi bạn xóa trigger. Negative mẫu nhắm ngón thừa/thiếu/dính. Phong cách chỉ là từ khóa trên **cùng một** checkpoint anime, không phải model riêng; prompt không bảo đảm sửa hết lỗi ngón.

**Bảo mật:** Cloudflare Quick Tunnel (`trycloudflare.com`) — và `gradio.live` nếu bạn bật `GRADIO_SHARE` — đều là link công khai, **không có tài khoản/mật khẩu**: ai biết link đều có thể dùng GPU của bạn. Không chia sẻ link hay lưu công khai notebook có output chứa link; dừng runtime để ngắt link. File checkpoint/LoRA bị chặn tải qua đường file Gradio; prompt/negative chỉ được nhúng vào PNG nếu bạn tự bật metadata (mặc định tắt).

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
- **Nạp thư viện prompt trong giao diện Studio (ô 7, đã tối ưu cho điện thoại).** Danh sách prompt hiện là **danh sách chạm (radio) cuộn được** thay cho dropdown lọc: mỗi dòng cao ≥42 px, khung giới hạn ~44% chiều cao màn hình nên không còn cảnh bàn phím ảo che danh sách; **chạm một dòng là nạp ngay**, và nếu thao tác chạm không như ý thì chọn dòng rồi bấm **⬇️ Nạp prompt đã chọn**. Chi tiết định dạng đọc file bên dưới. Mở tab **📚 Thư viện**, tải file `.txt`/`.md`/`.json` (tối đa 2 MB) hoặc dán nội dung, hệ thống đọc ra danh sách; **chạm một dòng để nạp prompt vào ô *Prompt gửi model***. Định dạng được nhận: `PROMPT 01 - Tên tiếng Việt` rồi đoạn prompt bên dưới, bảng một dòng `Tên tiếng Việt | Nội dung prompts tiếng Anh`, JSON `[{"title", "prompt"}]`, hoặc các đoạn prompt cách nhau dòng trống. Các dòng `Negative:`, `Steps:`, `CFG:`, `Size: 832x1216`, `Seed:` trong mỗi prompt cũng được áp dụng và bị kẹp về dải giao diện cho phép (steps 10–45, CFG 1–12, kích thước theo preset); prompt dài hơn 2000 ký tự bị cắt bớt. File chỉ được đọc trong phiên Colab, không tải lên dịch vụ nào; chưa có file thì bấm **Nạp thư viện mẫu** (12 prompt) để xem định dạng.
- **Giao diện Studio được làm mới.** Header làm rõ model và cảnh báo link công khai; prompt/negative và gallery được gom thành hai panel có phân cấp rõ trên màn hình rộng, tự xếp dọc trên điện thoại. Các tab tác vụ được giữ gọn và cuộn ngang khi thiếu chỗ; nút chính, nút phụ và ghi chú có thứ bậc nhất quán. Tất cả callback và luồng tạo ảnh hiện có được giữ nguyên; đúng hai ô *Prompt/Negative gửi model* vẫn là nội dung gửi model, không thẻ ẩn.
- **Trang vẫn bấm được khi đang tạo ảnh và khi chuyển tab.** Phần tìm tag trong catalog và đọc danh sách ảnh nguồn chạy trong hàng đợi riêng của Gradio (`demo.queue(max_size=64)`), không còn chạy trực tiếp trên event loop: trước đây mỗi phím gõ có thể chặn toàn bộ HTTP/SSE của trang tới ~1,5 giây nên mọi tab như bị "đơ". Job GPU vẫn chỉ một lượt một (`concurrency_id="wai_gpu"`) để an toàn VRAM; bấm một ảnh trong thư viện chỉ đọc bộ nhớ, còn `↻` mới quét `/content/wai_outputs` (kết quả giữ lại 5 giây và xếp theo mốc thời gian trong tên file nên không `stat()` từng ảnh — trên ổ mạng của Colab mỗi stat tốn hàng chục ms). Gõ nhanh cũng chỉ tìm một lần: lượt quét catalog cũ tự hủy khi bạn gõ tiếp.

  **Mã UI nằm ở ô 7** — chạy lại riêng ô 8/ô 9 không cập nhật giao diện. Header của Studio có chip **"Bản dựng …"**; không thấy chip đó (hoặc thấy ngày cũ) là phiên Colab vẫn chạy bản cũ, hãy chạy lại ô 7 rồi ô 8, ô 9.

  **Studio tự báo sự kiện nào chậm**: `colab/studio.py` bọc mọi handler UI và in vào output của Ô 8 dòng `⏱ #n gallery.change «Ảnh đã tạo» hết 1,4 s s` (ngưỡng 0,3 s) hoặc `⏳ vẫn đang chạy sau 10 s`; phía trình duyệt có watchdog đo **bấm → vẽ xong** (`window.__waiBlockLog` trong DevTools) và hiện "Trang bị chặn X s khi chuyển tab …". Khi báo lỗi, kèm hai số này là tìm ra nguyên nhân ngay.
  **Gõ tiếng Việt để tìm tag không còn dựng bảng tra giữa cú gõ**: index từ khóa theo nhãn Việt và bảng tên thẻ đã chuẩn hoá được dựng một lần ở Ô 7/Ô 8 (in ra "Đã dựng index N từ khóa từ nhãn tiếng Việt"), nên cú gõ đầu tiên sau khi mở trang nhanh như các cú sau (~1 ms thay vì ~1 s).
  Khi chuyển tab cũng nhẹ hơn: observer phím tắt `Ctrl+↑/↓` chỉ quan sát vùng ô prompt rồi tự ngắt (trước đây nó chạy theo mọi thay đổi DOM của cả trang), và tab **✎ Sửa vùng** chỉ tải một ảnh thay vì hai bản PNG full-size vì `editor_value_for` không còn nhân bản `composite`.
- **🧭 Quy trình chuẩn · khung prompt + negative tối ưu (ô 7).** Ba nút hỗ trợ, tất cả **chỉ ghi vào nội dung hiển thị** để bạn sửa: **Sắp xếp prompt theo thứ tự chuẩn** (chủ thể → nhãn phân loại → ngoại hình/chi tiết nhân vật → trang phục → tư thế → bố cục → bối cảnh → ánh sáng → phong cách → thẻ khác → chất lượng → `absurdres`; bỏ thẻ trùng, thêm thẻ neo của khung đã chọn), **Nạp negative đã chọn** (8 bộ: chuẩn nhà phát hành WAI v17, Illustrious chuẩn, tay/chân, giữ chất 2D, chân dung, phong cảnh, an toàn nội dung, inpaint; ghi đè hoặc nối thêm) và **🩺 Kiểm tra prompt & thông số** (ước lượng token so với khối 75 token của SDXL, thẻ chất lượng thừa/trùng/vừa dương vừa âm, cú pháp Pony, trọng số > 1.2, negative quá dài, steps/CFG/kích thước/hires ngoài khuyến nghị). Nhóm **💅 Chi tiết mắt & móng** trong tab **✨ Chi tiết** chỉ thêm các thẻ mô tả (màu mắt, kiểu/màu móng) vào hai ô đang hiển thị. Quy trình đầy đủ: [`docs/QUY_TRINH_TAO_ANH.md`](docs/QUY_TRINH_TAO_ANH.md).
- **Tự sửa mặt/tay (auto-detailer, tùy chọn).** Ô 2 **cài thêm `ultralytics==8.4.170`** (chỉ cảnh báo nếu thất bại, các chế độ khác vẫn chạy). Khi bật mục **Tự sửa mặt/tay** trong tab **⚙️ Thông số**, Studio tải weight YOLOv8 (`face_yolov8n.pt`, `hand_yolov8n.pt`) từ `Bingsu/adetailer` @ `c310c216` vào **`/content/wai_detailer_cache`**, **kiểm SHA-256 đầy đủ** (khớp metadata HF đã đối chiếu) rồi phát hiện mặt/bàn tay và inpaint lại đúng vùng đó với strength/ngưỡng/số vùng bạn chọn. Weight sai hash bị xóa và không nạp; cần mạng ở lần bật đầu tiên. Tính năng này chỉ là gợi ý chỉnh sửa, **không đảm bảo** hết lỗi ngón/mặt.

Kiểm tra cấu trúc notebook và hành vi tải/hash/nạp LoRA bằng mock CPU: `python -m unittest discover -s tests -v` (hiện **186 test: 158 đạt, 28 bỏ qua** — phần bỏ qua do thiếu `torch`/`diffusers` và notebook legacy không còn trong checkout; UI Gradio 6.15.2 và Pillow đã được cài nên các test dựng giao diện và tuyến sự kiện thật chạy được). `colab/studio.py` là bản mã nguồn tương ứng của ô 7 và `python scripts/build_colab_studio.py` tái tạo **chính xác** notebook từ nó — sửa giao diện ở `colab/studio.py` rồi chạy lại script. Việc tải thực tế checkpoint/LoRA/weight YOLOv8, khả năng chạy với GPU Colab và chất lượng ảnh/tay-chân-mắt **chưa thể xác nhận** trong môi trường kiểm thử CPU này.

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
- Ngay dưới ô **Prompt**, gõ từ khóa ở cuối prompt (ví dụ `mắt`/`eyes`, `tóc`/`hair`, `nơ`/`bow` hoặc `váy`/`dress`) để tìm tag **tiếng Anh** trực tiếp trong CSV đã được nạp lúc Studio khởi động. Gợi ý chỉ lấy từ catalog, chọn một dòng sẽ thay từ khóa cuối bằng tên thẻ canonical; không tự chèn khi bạn chưa chọn. CSV dùng chung với tab Kho thẻ và được xác minh SHA-256 trước khi dùng.
- Ba kiểu truy vấn và một bộ lọc theo **Semi-Auto Tag Complete** của [Character Select SAA](https://github.com/mirabarukaso/character_select_stand_alone_app): `chữ đầu` tìm tiền tố, `*đuôi` tìm hậu tố (ví dụ `*hair`), `*giữa*` tìm ở giữa tên thẻ, còn `@tên` **chỉ lọc thẻ họa sĩ** (nhóm 1 của Danbooru và 8 của e621) và dấu `@` không được chèn vào prompt. Mỗi dòng gợi ý mở đầu bằng nhãn danh mục: `[G] [A] [©] [C] [M]` cho Danbooru, `<G> <A> <©> <C> <S> <M> <L>` cho e621.
- `Ctrl+↑` / `Ctrl+↓` trong ô prompt, hoặc hai nút **`+0,1` / `−0,1`** bên cạnh dropdown, chỉnh trọng số của tag đang bôi đen / tag ở con trỏ (nếu không có lựa chọn thì chỉnh cụm cuối prompt) theo bước `0,1` trong dải `0,1–2,0`; về `1,0` thì dấu `(tag:…)` được gỡ. Cả hai cách đều sửa text hiển thị nên bạn xem/xóa được trước khi bấm tạo.
- Có thể dán prompt nhiều cụm, phân tách bằng dấu phẩy, chấm phẩy hoặc xuống dòng; nếu cụm dài không khớp nguyên văn, tìm tiếp theo các từ khóa riêng trong cụm.
- Lọc theo danh mục của nguồn, kết hợp chủ đề; sắp xếp phổ biến nhất hoặc A–Z. Chủ đề được suy đoán từ tên thẻ, không phải nhãn chính thức của CSV.
- Chọn nhiều kết quả, chọn Prompt hoặc Negative prompt rồi nhấn **Thêm thẻ đã chọn**. Prompt nhận tên thẻ tiếng Anh chuẩn; nhãn Việt chỉ dùng để hiển thị/tìm kiếm.
- Kho được nạp một lần vào RAM mỗi runtime; mỗi trang có 60 kết quả. Đây là từ khóa tạo prompt, không phải checkpoint, LoRA hay bộ ảnh huấn luyện; nội dung có thể chứa từ khóa nhạy cảm.

#### Dùng kho thẻ trong Colab / gradio.live

`WAI_Illustrious_Studio_Colab.ipynb` cũng có bộ chọn **🏷️ Kho thẻ Danbooru / e621 · danh mục & chủ đề**
ngay dưới Prompt / Negative. Chạy notebook rồi mở link Gradio ở ô 8:

Khi `build_app()` khởi động ở ô 8, Studio nạp CSV gợi ý vào RAM một lần: dùng file đã có nếu khớp SHA-256 hoặc tải bản ở commit cố định rồi xác minh trước khi phân tích. Catalog dùng chung cho autocomplete và tab Kho thẻ; prompt gõ `mắt`, `eyes`, `tóc`, `hair`… sẽ tìm trong dữ liệu đó, hiện các tên thẻ tiếng Anh canonical kèm nhãn danh mục để chọn. Chọn tag thay cụm cuối prompt. Muốn thử kiểu tìm của SAA, gõ `*hair` (hậu tố), `*hair*` (chứa giữa) hoặc `@tên` (chỉ họa sĩ, nhóm 1 và 8). Muốn đổi nhấn mạnh, đặt con trỏ trong một tag hoặc bôi đen vài tag rồi bấm `Ctrl+↑`/`Ctrl+↓` (hoặc hai nút `±0,1`); trọng số bước `0,1`, kẹp trong `0,1–2,0`. Nếu CSV chưa tải được, tạo ảnh vẫn hoạt động và autocomplete sẽ báo lỗi/thử lại khi có yêu cầu. Muốn xem nhiều hơn N gợi ý hot nhất thì tăng ô **Số gợi ý** (16 → 150) cạnh dropdown gợi ý — vẫn chọn được thẻ nằm ngoài top-16; muốn tìm mọi thẻ trong CSV hoặc thêm nhiều tag cùng lúc thì mở tab kho thẻ:

1. Khi khởi động, Colab tự đọc bản CSV có sẵn hoặc tải bản CSV gốc ~9 MB từ GitHub ở commit cố định và kiểm tra SHA-256; bản trong repo đã có thêm cột nhãn tiếng Việt (~13,7 MB), cả hai dạng đều được hỗ trợ. Nếu bước này thất bại, tải đúng CSV lên `/content/` qua bảng **Files** rồi dùng **Tìm / tải kho thẻ** để thử lại.
2. Nhập cụm tiếng Việt hoặc English, hoặc dán các tag trong prompt cách nhau bằng dấu phẩy/xuống dòng; bộ tìm kiếm quét toàn bộ kho, bỏ qua dấu tiếng Việt và tìm cả tên/bí danh. Chọn danh mục/chủ đề và cách sắp xếp nếu cần; nhấn **Tìm** để cập nhật. Mỗi trang có 60 kết quả; nhập số trang rồi nhấn Tìm.
3. Chọn nhiều thẻ trong kết quả, chọn Prompt hoặc Negative prompt, nhấn **Thêm thẻ đã chọn**. Nội dung hiển thị trực tiếp trong ô tương ứng; không có thẻ nào được thêm ngầm khi tạo ảnh. Đổi kết quả tìm kiếm sẽ xóa lựa chọn chưa thêm.

Nếu GitHub không truy cập được, dùng bảng **Files** bên trái Colab để tải đúng CSV lên `/content/`, rồi nhấn Tìm lại.
Kho chỉ nạp một lần vào RAM mỗi runtime, dùng chung dữ liệu chỉ đọc; prompt/lựa chọn vẫn riêng từng phiên Gradio.
Không đưa toàn bộ danh sách 349.714 thẻ xuống trình duyệt. Lỗi kho thẻ không chặn viết prompt hoặc tạo ảnh.
Sau khi cập nhật notebook, cần chạy lại ô 7–8 để giao diện Gradio đang chạy nhận tính năng mới.
Mã nằm trong `colab/studio.py`; tái tạo notebook bằng `python scripts/build_colab_studio.py`.

#### Bố cục và trải nghiệm Studio

Giao diện có header sản phẩm/cảnh báo bảo mật, panel **Soạn prompt** và panel **Kết quả**. Trên màn hình rộng hai panel nằm cạnh nhau; dưới 900 px chúng xếp dọc. Prompt/Negative luôn ở ngoài tab và giữ nguyên khi chuyển tác vụ; thanh tab cuộn ngang, vùng bấm đủ lớn cho thao tác cảm ứng.

**Chọn ảnh nào để sửa / phóng.** Khung **02 · Kết quả** có ô **Ảnh sẽ nạp vào tab sửa / phóng**: mặc định là ảnh vừa tạo xong, nhưng bấm một ảnh trong thư viện (hoặc chọn trong dropdown — nhãn gồm số thứ tự, tên file, `Seed … · kích thước`) sẽ chỉ định đúng ảnh đó. Nút `↻` đọc lại toàn bộ PNG trong `/content/wai_outputs`, nên vẫn chọn được ảnh của các lượt tạo trước khi trang được tải lại. Bấm **↪ Nạp ảnh đã chọn vào cả ba tab** để đưa ảnh vào ◈ Biến đổi, ⤢ Phóng to và ✎ Sửa vùng cùng lúc, hoặc dùng ba nút `→ ◈ Biến đổi` / `→ ⤢ Phóng to` / `→ ✎ Sửa vùng` để nạp rồi mở ngay một tab. Không chọn ảnh nào thì Studio báo lỗi rõ, không âm thầm lấy ảnh khác.

Sáu tab chính, mặc định mở **✦ Tạo ảnh**:

| Tab | Nội dung |
| --- | --- |
| ✦ Tạo ảnh | Bốn chế độ lồng bên trong: Văn bản → ảnh, Ảnh → ảnh, Phóng to ảnh, Sửa vùng ảnh |
| 🏷️ Kho thẻ | CSV Danbooru/e621, tìm kiếm, danh mục, chủ đề |
| 📚 Thư viện | Nạp file/dán danh sách hoặc chọn prompt mẫu |
| 🧭 Quy trình | Khung prompt, negative tối ưu, kiểm tra prompt/thông số |
| ⚙️ Thông số | Steps, CFG, seed, số ảnh, LoRA, metadata PNG |
| ✨ Chi tiết | Hires fix, auto-detailer, chi tiết mắt/móng |

Các nút dùng ảnh mới nhất tự mở đúng chế độ sau khi nạp ảnh thành công. Bố cục chỉ thay đổi cách trình bày; callback và đường xử lý ảnh hiện có được giữ nguyên. Chạy lại ô 7–8 của notebook đã cập nhật để áp dụng.

Kho thẻ Gradio hiển thị nhãn tiếng Việt trước tên gốc, ví dụ **Tóc dài — long_hair**,
**Mắt xanh dương — blue_eyes**. Có thể tìm bằng nhãn Việt có dấu/không dấu, tag hoặc
bí danh tiếng Anh; nhiều cụm trong một prompt được tìm riêng trên toàn bộ catalog.
CSV có cột chú giải Việt ở vị trí thứ năm. Từ điển gồm **2.964 mục** cùng quy tắc ghép
(783 danh từ chính × 1.521 bổ ngữ × 36 màu) và mười bảy quy tắc cụm — `wearing_/holding_/looking_at_/no_`
(kèm phó từ hướng: `looking_down_at_viewer`), `<A>_<giới từ>_<B>`, `<món đồ>_only`,
`<bộ phận>_<hướng>`, `<danh từ>_<trạng thái>`, `see_through_<x>`/`floating_<x>`, hậu tố `-less`
họ loài/nội thất (`canine_ears`, `office_chair`), lượng từ + loại từ (`three_tails` →
`Ba cái đuôi`, `multiple_arms` → `Nhiều cánh tay`), động từ đặt sau danh từ (`dress_pull` → `Kéo váy`,
`tail_lick` → `Liếm đuôi`), nội động từ (`melting_tail` → `Đuôi đang tan chảy`), gạch nối và
`(giải nghĩa)` (`see-through_dress`, `pearl_(gem)`) cùng số nhiều (`curved_horns`), khung đồng phục theo tên riêng
(`tokiwadai_school_uniform` → `Đồng phục trường Tokiwadai`, giữ nguyên và viết hoa tên riêng), sở hữu cách
(`fool's_hat` → `Mũ của chú hề`) và động từ mặc/cởi (`undressing_another` → `Đang cởi đồ người khác`),
liên từ bị lược trong cụm tương tác (`talking_to_viewer` → `Nói chuyện với người xem`), lượng từ ghi bằng
chữ số (`9_tails` → `Chín cái đuôi`, `2_penises` → `Hai dương vật`) và bước chuốt nhãn gộp từ lặp do ghép
(`hair_scrunchie` từ `Dây buộc tóc tóc` → `Dây buộc tóc`, `cream` từ `màu màu kem` → `Màu kem`) nhưng vẫn giữ
từ láy thật (`dragonfly_print` → `Họa tiết chuồn chuồn`); họ giới tính/loài và động từ quan hệ
(`dominant_female` → `Nhân vật nữ chiếm ưu thế`, `felid_humanoid` → `Dạng người họ mèo`,
`male_penetrating_female` → `Nam thâm nhập nữ`, `brother_and_sister` → `Anh em trai và chị em gái`) —
phủ **29.619 thẻ**,
ưu tiên chi tiết nhân vật
(tóc, mắt, mặt, tai/đuôi, trang phục, biểu cảm);
đây **không phải bản dịch máy đầy đủ cho 349.714 thẻ**. Tên
họa sĩ/nhân vật/tác phẩm được giữ nguyên kèm nhãn loại; mục chưa dịch được ghi rõ.
Khi thêm kết quả, prompt nhận đúng tên thẻ tiếng Anh gốc — nhãn Việt không gửi vào model.
Tạo lại cột nhãn bằng `python scripts/add_vietnamese_tag_captions.py`; sau khi chỉnh từ điển
`TAG_VI_LABELS`/quy tắc trong `colab/studio.py`, chạy script trên, cập nhật
`TAG_CSV_SHA256` theo hash được in ra rồi chạy `python scripts/build_colab_studio.py`.

#### File dịch tiếng Việt độc lập · `danbooru_e621_merged_vi_vn.csv`

Cùng bộ từ điển trong `colab/studio.py` được xuất ra **`danbooru_e621_merged_vi_vn.csv`**
ở thư mục gốc repo — "file dịch" (translate file) để công cụ bên ngoài tìm thẻ bằng tiếng Việt,
theo đúng định dạng mà Semi-Auto Tag Complete của Character Select SAA dùng cho tiếng Trung
(`data/danbooru_e621_merged_zh_cn.csv`).

- Mỗi dòng là `tag,category,translation`, **không có dòng tiêu đề**, UTF-8 không BOM, xuống dòng LF,
  mỗi thẻ xuất hiện đúng một lần theo thứ tự phổ biến giảm dần của CSV nguồn.
- Từ điển gồm **2.964 mục dịch cố định** + bộ ghép **783 danh từ chính × 1.521 bổ ngữ × 36 màu**
  và mười bảy quy tắc cụm: `wearing_hat` → **Đội mũ** (động từ chọn theo loại món đồ: Đội/Mặc/Đeo/Thắt/Đi/Mang),
  `holding_sword` → **Cầm kiếm**, `no_gloves` → **Không có găng tay**, `bandaid_on_face` →
  **Băng cá nhân trên khuôn mặt**, `hat_with_ribbon` → **Mũ kèm ruy băng**, `hairless` →
  **Không có tóc**, `hat_only` → **Chỉ đội mũ**, `skirt_down` → **Kéo chân váy xuống**,
  `looking_down_at_viewer` → **Nhìn xuống người xem**, `tail_raised` → **Đuôi dựng lên**,
  `hairless_cat` → **Mèo không lông**, `floating_sleeves` → **Tay áo lơ lửng**,
  `see_through_shirt` → **Áo sơ mi xuyên thấu**, `three_tails` → **Ba cái đuôi**,
  `multiple_arms` → **Nhiều cánh tay**, `curved_horns` → **Sừng cong**, `melting_tail` →
  **Đuôi đang tan chảy**, `see-through_dress` → **Váy liền xuyên thấu**, `pearl_(gem)` → **Ngọc trai**, `tokiwadai_school_uniform` →
  **Đồng phục trường Tokiwadai**, `fool's_hat` → **Mũ của chú hề**, `national_soccer_team_uniform` →
  **Đồng phục đội tuyển bóng đá quốc gia**, `talking_to_viewer` → **Nói chuyện với người xem**,
  `blurred_background` → **Nền mờ**, `bust_portrait` → **Ảnh chân dung bán thân**.
  Mỗi mục danh từ mới mở khóa cả họ thẻ nên độ phủ tăng nhanh hơn số từ phải viết tay.
- Trường dịch **không chứa dấu phẩy**: bộ nạp JavaScript của SAA cắt dòng bằng `line.split(',', 3)`
  nên phần sau trường thứ ba bị bỏ; script tự đổi `,` thành `;`, bỏ nháy và làm phẳng xuống dòng.
- Chỉ thẻ **có bản dịch thật** được ghi. Mục chưa dịch (`Chưa có bản dịch`), từ loại thuần
  (`Tác phẩm`, `Nhân vật`, `Họa sĩ`) và tên họa sĩ (nhóm 1 và 8 — SAA bỏ qua khi nạp file dịch)
  bị loại, nên tệp không chứa dòng vô nghĩa. Tên riêng được giữ nguyên theo chủ trương của repo.
- Hiện tại: **29.619/349.714 thẻ** (1,18 MB), phủ **96,8%** trong 500 thẻ phổ biến nhất, **96,7%**
  trong 1.000, **94,2%** trong 2.000, **71,6%** trong 5.000 và **57,5%** trong 10.000 thẻ đầu. Theo
  `python scripts/tag_vi_audit.py report` (chỉ đếm các danh mục dịch được 0/5/7/12/14, nên tên họa sĩ và
  tên tác phẩm không bị tính vào mẫu số), độ phủ **số thẻ · lượt dùng** của từng nhóm là: **Ngoại hình**
  5.989/8.714 · **68,7% / 99,2%**, **Trang phục & phụ kiện** 3.201/4.088 · **78,3% / 99,4%**, **Biểu cảm &
  tư thế** 576/1.032 · 55,8% / 99,3%, **Bối cảnh & thiên nhiên** 784/1.760 · 44,5% / 98,3%, **Ánh sáng &
  màu sắc** 862/1.722 · 50,1% / 96,0%, **Bố cục & kỹ thuật** 238/446 · 53,4% / **99,8%**. Sáu nhóm còn
  **6.112** thẻ trống, chủ yếu là thẻ ký hiệu (`?`, `^^^`, `:<`, `0_0`) hoặc tên riêng — phần không dịch
  được này giữ nguyên tiếng Anh theo chủ trương của repo, nên vẫn **không** phải bản dịch máy cho toàn bộ catalog.
- Từ điển trong mã **luôn thắng** cột chú giải cũ của CSV (`_prefer_vietnamese_label`), nên thêm
  bản dịch vào `colab/studio.py` là Studio và file dịch nhận ngay — không phải tạo lại CSV 13,7 MB
  hay đổi `TAG_CSV_SHA256`.
- Tạo lại sau khi thêm từ điển: `python scripts/build_vietnamese_translate_file.py`.
  Thêm `--bom` nếu muốn mở bằng Excel, `--check` để kiểm tra tệp đang có khớp với từ điển
  (dùng trong CI; lệch nhau sẽ trả mã lỗi 1).
- Studio của repo **không** nạp tệp này — nó đọc cột thứ năm của CSV đã ghim. Tệp sinh ra để
  chia sẻ cho tool khác. SAA hiện hard-code đường dẫn `data/danbooru_e621_merged_zh_cn.csv`,
  nên muốn thử trong SAA thì chép tệp này đè lên đường dẫn đó trong bản SAA của bạn (thay lớp
  tiếng Trung, không ảnh hưởng tìm kiếm tiếng Anh).
