# WAI Studio cá nhân — giao diện tạo ảnh qua Google Colab

[![Mở WAI Studio trong Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manhlee1196-boop/ai-anime/blob/arena/01a0d84b-ai-anime/WAI_Illustrious_Studio_Colab.ipynb)

**[WAI_Illustrious_Studio_Colab.ipynb](WAI_Illustrious_Studio_Colab.ipynb)** dành cho người dùng Colab cá nhân: GPU Google Colab chạy **đúng checkpoint WAI-illustrious v17 và LoRA đã xác minh**, sau đó Gradio tạo **liên kết giao diện tạm thời không cần đăng nhập**. Không cần Cloudflare, Node.js, server GPU khác hay nhập token Cloudflare. Đây là lựa chọn nên dùng nếu bạn chỉ muốn tự tạo ảnh.

1. Bấm nút Colab phía trên → **Runtime → Change runtime type → T4 GPU** (hoặc GPU mạnh hơn) → **Runtime → Run all**. Nếu cần lưu model/ảnh cho lần sau, cho phép gắn Google Drive. Lần đầu tải ~6,94 GB checkpoint và tối đa ~457 MB LoRA; các ô kiểm tra SHA-256 trước khi nạp. Mặc định hai LoRA bật; tắt/bật tài nguyên và chọn `VRAM_MODE` ở **ô 3** trước khi chạy nếu muốn tối ưu.
2. Khi Colab in URL `https://….gradio.live` ở ô cuối, mở URL đó; **không cần tài khoản hoặc mật khẩu**. Giao diện có text-to-image, img2img, tô hoặc tải mask PNG để inpaint; chọn **Anime chuẩn**, **Bán thực 2.5D**, **Tùy chỉnh** hoặc **Anime NSFW 18+** (chỉ nhân vật trưởng thành, cần xác nhận riêng), rồi chỉnh prompt/negative, kích thước, steps, CFG, seed, strength, số ảnh và cường độ LoRA đã nạp.
3. Ảnh mặc định lưu ở `MyDrive/AI/outputs`; nếu Drive không lưu được thì lưu dự phòng ở `/content/wai_outputs`, hãy tải về trước khi Colab hết phiên. **Link không hoạt động nếu Colab ngắt** và không phải hosting lâu dài. `share=True` tạo URL có thể truy cập từ Internet qua proxy Gradio: **bất kỳ ai biết URL đều có thể dùng giao diện và GPU của bạn**. Không chia sẻ URL hoặc lưu công khai output chứa URL; dừng runtime để ngắt link. Model/LoRA bị chặn tải qua đường file Gradio, nhưng URL không có xác thực.

**Phong cách và ngón tay/ngón chân:** preset áp dụng từ khóa ở **đầu** prompt/negative trước khi suy luận, ưu tiên phong cách khi prompt dài; giao diện hiện ô **Xem prompt sau khi áp dụng phong cách** và cập nhật khi bạn đổi phong cách, prompt, negative hoặc LoRA mắt. Nội dung bạn gõ không bị ghi đè. **Bán thực 2.5D** tăng gợi ý đổ bóng/độ sâu và chặn nét vẽ phẳng, nhưng vẫn dùng **cùng checkpoint WAI v17 thiên về anime**, không thể bảo đảm ảnh bán thật như model chuyên dụng. Negative chung nhắm lỗi ngón *thừa/thiếu/dính/dị dạng*; các phong cách thường (kể cả `Tùy chỉnh`) tự thêm `nsfw, explicit` vào negative. **Anime NSFW 18+** là lựa chọn tách biệt, không thêm hai thẻ chặn này và chỉ dùng cho nhân vật trưởng thành; người dùng phải tự xác nhận 18+ trước khi tạo. Notebook từ chối một số từ khóa rõ ràng về vị thành niên trong prompt 18+, nhưng kiểm tra từ khóa **không phải bộ lọc hoàn chỉnh hay xác minh tuổi**. Hãy tuân thủ điều khoản của Colab/Gradio và giấy phép model. Link Gradio không có đăng nhập: đừng chia sẻ, nhất là khi dùng chế độ 18+. Prompt/negative không bảo đảm hình ảnh đẹp hoặc sửa hết lỗi ngón; nếu còn lỗi hãy khoanh vùng nhỏ bằng tab **Sửa vùng ảnh**. Prompt/negative thực tế chỉ nhúng vào PNG khi bạn tự bật metadata (tắt mặc định).

**So sánh hai phong cách:** mở notebook mới nhất, chọn **Runtime → Restart runtime → Run all** để không dùng link Gradio cũ. Giữ **cùng prompt, negative, seed cố định (khác `-1`), kích thước, steps và cường độ LoRA**, rồi tạo một ảnh Anime chuẩn và một ảnh Bán thực 2.5D. Xem hai ô prompt áp dụng: ở Bán thực phải xuất hiện `semi-realistic anime art, 2.5d illustration` phía trước và negative có `flat cel shading`. Nếu prompt gốc yêu cầu `cel shading`/`anime lineart` hoặc LoRA thiên anime quá mạnh, hai ảnh vẫn có thể trông gần nhau. Không có GPU/model thật trong môi trường kiểm thử repo nên cần so sánh thực tế trên Colab.

**Nếu RAM hệ thống gần đầy nhưng VRAM gần trống:** mở lại notebook mới nhất và chọn **Runtime → Restart runtime → Run all** để giải phóng pipeline cũ (không chạy lại riêng ô 6 khi Gradio còn giữ model cũ). Để `VRAM_MODE=auto` ở ô 3. Ở ô 6 xem `VRAM trống trước/sau khi nạp` và `Chế độ sau khi nạp`; giao diện cũng hiển thị chế độ đang dùng. Bản cũ dễ chọn CPU offload trên T4 khi bật hai LoRA; bản mới ưu tiên GPU trực tiếp nếu còn ≥ **13,3 GiB** VRAM với hai LoRA, áp dụng VAE tiling và giữ cơ chế thử lại bằng CPU offload khi OOM. CPU offload vẫn dùng GPU từng phần nên có thể thấy 5–6 GB VRAM *đang chạy* nhưng nhiều RAM hệ thống; GPU trống khi không tạo ảnh không chứng minh ảnh chạy hoàn toàn trên CPU. Nếu bản mới vẫn chuyển offload hoặc RAM tăng, gửi log ô 6 và lúc tạo ảnh (che URL Gradio); kết quả thực tế phụ thuộc GPU và VRAM khả dụng của phiên Colab.

**Nếu ảnh thứ hai báo `Setting requires_grad=True on inference tensor outside InferenceMode`:** mở lại notebook mới nhất từ nút Colab phía trên, chọn **Runtime → Restart runtime → Run all** (không chỉ chạy lại ô 8); bản Studio mới đặt thao tác đổi LoRA trong `torch.inference_mode()` cả sau khi CPU offload. Nếu vẫn lỗi, gửi traceback ở ô 8 và che URL Gradio.

**Nếu ô 2 báo xung đột thư viện:** mở lại notebook từ nút Colab ở trên để lấy phiên bản mới (Gradio 6.15.2); chọn **Runtime → Restart runtime → Run all** thay vì chỉ chạy lại ô cũ. Dòng `ERROR: pip's dependency resolver does not currently take into account...` có thể chỉ là cảnh báo sau khi cài thành công. Ô 2 chỉ được coi là đã sẵn sàng khi in **`✅ Thư viện Studio đã sẵn sàng`**; nếu không có dòng này hoặc có traceback, dừng và gửi toàn bộ traceback (che URL Gradio/thông tin riêng trước khi chia sẻ). Notebook mới ghim Pydantic và Starlette tương thích với các gói Colab được báo xung đột.

Nếu không muốn gắn Drive, đặt `MOUNT_DRIVE=False` và đổi `MODEL_PATH`, `OUTPUT_DIR` ở ô 3 thành đường dẫn dưới `/content/`; hãy tải ảnh về trước khi phiên Colab kết thúc. Khi đổi danh sách LoRA/chế độ bộ nhớ sau khi đã nạp model, **Restart runtime → Run all**; không chạy lại riêng ô nạp để tránh giữ hai bản checkpoint trong RAM. Các phiên bản/hash tài nguyên và giới hạn GPU được giải thích ở phần notebook nâng cao dưới đây. Việc chạy inference GPU thực tế vẫn cần bạn tự thực hiện trong tài khoản Colab; môi trường kiểm thử repo không có GPU hoặc link Gradio đang hoạt động.

---

# WAI-illustrious trên Google Colab (notebook nâng cao, không cần link)

[![Mở trong Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manhlee1196-boop/ai-anime/blob/arena/01a0d84b-ai-anime/WAI_Illustrious_Colab.ipynb)

Notebook: **[WAI_Illustrious_Colab.ipynb](WAI_Illustrious_Colab.ipynb)** — tạo ảnh anime bằng checkpoint SDXL WAI-illustrious, có **hai LoRA tùy chọn cho tay/chân/mắt** và **sửa vùng ảnh bằng inpainting**. Không cần WebUI hay mở liên kết chia sẻ công khai.

## Chạy nhanh

1. Mở notebook bằng nút Colab trên (hoặc Colab → **File → Upload notebook**). Chọn **Runtime → Change runtime type → GPU** (T4 hoặc mạnh hơn).
2. Chạy lần lượt các ô **1 → 7**, chấp thuận gắn Google Drive. `AUTO_DOWNLOAD=True` tự tải checkpoint v17 nếu `MODEL_PATH` chưa có; ô 5 tự tải **chỉ những LoRA đã bật** (`USE_ANATOMY_LORA`, `USE_EYE_LORA`; mặc định cả hai). Chỉnh cường độ trong ô 3 nếu cần, rồi chạy lại các ô 3 → 7 theo thứ tự (đặc biệt ô 5 xác minh lại LoRA và ô 6 nạp lại model). Đổi prompt/seed ở ô 7 để tạo ảnh mới. PNG mặc định lưu ở `MyDrive/AI/outputs`.
3. Nếu **đã có checkpoint riêng**, sửa `MODEL_PATH` trong ô 3; file sẵn có được ưu tiên, **không bị ghi đè**. Nếu đó không phải WAI/Illustrious SDXL, hãy tắt hai LoRA để tránh ghép sai dòng model. File riêng chỉ được xác thực là WAI v17 khi trùng **cả kích thước lẫn SHA-256** với bản ghim; nếu không, notebook chỉ kiểm tra định dạng SDXL, **không xác thực phiên bản**.
4. Nếu ảnh còn lỗi tay/chân/mắt, dùng **ô 8** để khoanh vùng cần sửa rồi lưu một PNG khác. Đây là tùy chọn; ảnh gốc không bị ghi đè.

## Nguồn và phiên bản đã đối chiếu

| Tài nguyên | Bản phát hành gốc / mục đích | Bản lưu tải tự động, ghim commit | Kích thước chính xác (byte) | SHA-256 toàn bộ file |
| --- | --- | --- | ---: | --- |
| Checkpoint **WAI-illustrious SDXL v17**, pruned FP16 | [Civitai version 2883731](https://civitai.com/api/v1/model-versions/2883731) | [HF LyliaEngine, commit `32be7bfd…`](https://huggingface.co/LyliaEngine/waiIllustriousSDXL_v170/blob/32be7bfdcd406db70df663b9cee3313957deb68f/waiIllustriousSDXL_v170.safetensors) | 6.938.040.682 | `f116b0c78ff441467b0cdc8f1936e1ed18ea31e9997c7b132b1b8db533f0bd04` |
| LoRA **Anatomy Helper V1** (Illustrious; tác giả mô tả hỗ trợ bàn tay, bàn chân, tư thế) | [Civitai version 1318504, file 1223247](https://civitai.com/api/v1/model-versions/1318504) | [HF ench100, commit `bed49d45…`](https://huggingface.co/ench100/bodyandface/blob/bed49d45df95c0695aedad3b2aa6aff389fb3777/anatomy_helper.safetensors) | 228.473.940 | `bf6a950036b7599212a2c68d65f3ba07b28689067e167915d2a0ecb2018c26ca` |
| LoRA **Eyes for Illustrious (Perfect anime eyes) V1** (Illustrious; trigger `perfect eyes`) | [Civitai version 2066663, file 1963176](https://civitai.com/api/v1/model-versions/2066663) | [HF Muapi, commit `1abbc862…`](https://huggingface.co/Muapi/eyes-for-illustrious-lora-perfect-anime-eyes/blob/1abbc862f53f5101962ebf1c337513aff91bd206/eyes-for-illustrious-lora-perfect-anime-eyes.safetensors) | 228.457.660 | `97c1a083ffe6b4d45c545196eabd01c754936b996ade0c9db6d072f3bd340c55` |

**Cách xác minh:** Đã đối chiếu metadata bản **Illustrious**/SHA-256 đầy đủ trên API Civitai với SHA-256 từ trang file/LFS HF ở đúng commit ghi trên. Hai kho HF là **mirror của bên thứ ba**, không khẳng định tài khoản HF thuộc tác giả Civitai. Cùng SHA-256 chứng minh **file ở mirror trùng từng byte** với file phát hành được Civitai công bố; notebook cũng băm toàn bộ file sau khi tải (kể cả đường dẫn LoRA do bạn cung cấp) và từ chối file sai hash. Không coi `AutoV2` 10 ký tự hoặc tên file là bằng chứng đầy đủ. Nếu nguồn đổi nội dung, file tải dở/hỏng hoặc sai biến thể (Pony/Anima/SD 1.5), sẽ **không nạp**. Không khẳng định đã kiểm nghiệm hiệu quả hình ảnh trên Colab; SHA **không** chứng minh tay/chân/mắt được sửa chính xác ở mọi ảnh. Xem điều khoản của từng model trước khi sử dụng/phân phối.

Bạn có thể tự kiểm tra file đã tải trong Colab: `sha256sum '/content/drive/MyDrive/AI/loras/anatomy_helper.safetensors'` (hoặc đường dẫn file tương ứng); so với cột SHA-256 trên.

## Tải và lưu tài nguyên

- Checkpoint ~6,94 GB. Mặc định tải vào cache `/content`, dùng ngay **không nhân đôi file trên ổ tạm**, rồi sao chép bản xác minh tới `MyDrive/AI/models/WAI-illustrious.safetensors` nếu `PERSIST_MODEL_TO_DRIVE=True`. Nếu /content thiếu khoảng 9 GiB trống, thử tải **trực tiếp vào Drive**. File có sẵn trên Drive chỉ được sao chép về /content nếu `CACHE_MODEL_LOCAL=True` **và** đủ chỗ; nếu không nạp thẳng từ Drive.
- Hai LoRA mỗi file ~228 MB, tổng ~457 MB. Bản HF được ghim commit; khi bật, ô 5 ưu tiên file đã xác minh ở `MyDrive/AI/loras`, sau đó tải cache `/content/wai_lora_cache`. `PERSIST_LORAS_TO_DRIVE=True` lưu bản sao đã **kiểm tra lại hash** trên Drive để phiên sau không tải lại. Khi /content thiếu chỗ nhưng Drive còn chỗ, tải LoRA thẳng vào Drive. Không sao chép file Drive về /content chỉ để dùng LoRA. Có thể nhập file Civitai gốc ở `ANATOMY_LORA_PATH`/`EYE_LORA_PATH` nếu tải HF không được; vẫn phải đúng hash phiên bản bảng trên. File có sẵn sai hash sẽ **bị từ chối, không bị tự ghi đè**.
- Cài `diffusers==0.35.2`, `transformers==4.52.4`, `accelerate==1.10.1`, `peft==0.17.1` để nạp nhiều LoRA; vẫn dùng PyTorch/CUDA của Colab. Cấu hình/tokenizer SDXL do Diffusers lấy khi cần, **không** tải trọng số SDXL base hoặc checkpoint inpainting riêng. Không đưa trọng số hay ảnh tạo ra vào repo. `/content` bị xóa khi phiên Colab kết thúc.

## Tối ưu và sửa ảnh

- FP16 + SDPA của PyTorch; Euler a; mặc định 25 bước / CFG 6 / 1024×1024. Mỗi lượt một ảnh. `VRAM_MODE=auto` **ưu tiên GPU trực tiếp** nếu VRAM trống trước khi nạp ≥ **12,5 GiB + 0,4 GiB cho mỗi LoRA bật** (hai LoRA: 13,3 GiB); thấp hơn thì CPU offload (chậm hơn, tốn RAM hệ thống dù suy luận vẫn dùng GPU từng phần). Bật VAE slicing và tiling ở cả hai chế độ để giảm đỉnh VRAM giải mã ảnh. Nếu nạp/tạo ảnh OOM ở `auto`, notebook nạp lại với offload rồi thử lại **cùng seed** khi tạo ảnh, dùng file đã xác minh. Ô 6 in VRAM trước/sau khi nạp và chế độ cuối; muốn giảm ảnh hưởng của pipeline cũ, hãy khởi động lại runtime trước khi chạy lại toàn bộ notebook. Mặc định strength anatomy `0.55`, mắt `0.45` (các mức thử nghiệm, không phải đảm bảo chất lượng); muốn so sánh, giữ seed và tắt từng LoRA.
- **Ô 8: inpaint đúng vị trí.** Sau khi thấy ảnh ở ô 7, nhập `BOXES="x1,y1,x2,y2"` (tọa độ pixel ảnh; góc trên-trái đến góc dưới-phải, ví dụ `100,300,240,490`), hoặc tạo ảnh mask **trắng = sửa / đen = giữ nguyên**, đúng kích thước ảnh, nhập `MASK_PATH`. Chỉ dùng **một** trong hai cách. `SOURCE_IMAGE` để trống lấy ảnh ô 7; có thể trỏ tới PNG/JPG khác. Chọn `TARGET=hands/legs/eyes`, chỉnh `REFINE_STRENGTH` (thử 0.35–0.55), seed và chạy ô 8. Nên sửa từng vùng nhỏ riêng để không làm đổi mặt/trang phục; có thể sửa PNG kết quả lần trước ở lượt tiếp theo. Sử dụng `AutoPipelineForInpainting.from_pipe(pipe)` để **chia sẻ** checkpoint SDXL hiện có; với UNet 4 kênh, mask định hướng tạo lại vùng trắng, sau đó chỉ ghép vùng đã chọn với viền mềm. Không tự nhận diện vị trí lỗi; khoanh sai chỗ hoặc strength cao có thể làm ảnh xấu hơn. Nếu inpaint OOM, auto sẽ thử offload, nếu vẫn OOM hãy chọn ảnh nhỏ hơn/`low_vram`.
- Anatomy Helper được mô tả tập huấn luyện có nhiều ảnh bàn chân và có thể kéo theo thiên lệch phong cách/nội dung; bạn có thể **tắt LoRA này** hoặc chỉ dùng inpaint thủ công. LoRA mắt sử dụng trigger `perfect eyes` (notebook tự thêm khi bật). LoRA tay/chân chủ yếu hướng tới **bàn tay/bàn chân và tư thế**, không chứng nhận chữa lỗi ống chân; dùng inpaint ô 8 cho các lỗi chân còn lại.
- Nếu hết **RAM hệ thống** khi nạp checkpoint, cần Colab high-RAM; notebook không thể cấp thêm GPU/RAM. Nếu Drive ngắt khi lưu, ảnh dự phòng nằm ở `/content/wai_outputs` (hãy tải xuống trước khi phiên hết). Prompt mẫu hướng tới nội dung lành mạnh nhưng **không đảm bảo bộ lọc**. `EMBED_METADATA=True` nhúng prompt/nguồn ảnh vào PNG; tắt trước khi chia sẻ nếu không muốn lộ thông tin.

Kiểm tra cấu trúc notebook và hành vi tải/hash/nạp LoRA bằng mock CPU: `python -m unittest discover -s tests -v`. Việc tải thực tế checkpoint/LoRA, khả năng chạy với GPU Colab và chất lượng tay/chân/mắt **chưa thể xác nhận** trong môi trường kiểm thử CPU này.

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
