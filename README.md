# WAI Studio cá nhân — giao diện tạo ảnh qua Google Colab

[![Mở WAI Studio trong Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manhlee1196-boop/ai-anime/blob/arena/01a0d84b-ai-anime/WAI_Illustrious_Studio_Colab.ipynb)

**[WAI_Illustrious_Studio_Colab.ipynb](WAI_Illustrious_Studio_Colab.ipynb)** chạy WAI-illustrious v17 và hai LoRA tùy chọn trên GPU Colab; Gradio tạo **link tạm không cần đăng nhập**. Không cần Google Drive, Cloudflare hoặc API token. Checkpoint và LoRA được kiểm **toàn bộ SHA-256** trước khi nạp. [Báo cáo kiểm tra nguồn và vận hành](VERIFICATION.md) phân biệt điều đã thử cục bộ với việc **chưa thử tạo ảnh bằng GPU Colab thật**.

1. Mở notebook từ nút trên → **Runtime → Change runtime type → T4 GPU** (hoặc mạnh hơn) → **Runtime → Run all**. Lần đầu notebook tải checkpoint ~6,94 GB và tối đa ~457 MB LoRA **trực tiếp vào `/content`**, không gắn Drive/không sao chép file qua lại. Cần khoảng **9 GiB đĩa trống**; tốc độ vẫn tùy mạng Colab và bước kiểm hash. Chọn/tắt LoRA, `VRAM_MODE` và đường dẫn tùy chỉnh **dưới `/content`** ở ô 3 trước khi nạp. Để `ANATOMY_LORA_PATH`/`EYE_LORA_PATH` rỗng nếu muốn tải tự động.
2. Khi ô cuối in URL `https://….gradio.live`, mở link để tạo ảnh bằng văn bản, ảnh → ảnh hoặc tô/tải mask sửa tay/chân/mắt. Chọn **Anime chuẩn**, **Bán thực 2.5D**, **Tùy chỉnh** hoặc **Anime NSFW 18+** (cần xác nhận mọi nhân vật đều 18+). Điều chỉnh kích thước, steps, CFG, seed, strength, số ảnh, LoRA và **hai ô prompt thực sự gửi model**.
3. Ảnh chỉ lưu dưới **`/content/wai_outputs`** (hoặc thư mục `/content` được chọn); model ở `/content/wai_model_cache`, LoRA ở `/content/wai_lora_cache`. **Tải ảnh về trước khi phiên Colab kết thúc**: cả ảnh và weights cục bộ sẽ mất khi runtime ngắt, phiên sau cần tải lại. Link cũng ngừng hoạt động khi Colab ngắt.

**Prompt có thể chỉnh trực tiếp:** đổi phong cách, ý tưởng gốc, negative gốc hoặc trạng thái LoRA mắt sẽ *điền lại/ghi đè* hai ô **“Prompt gửi model · sửa được”** và **“Negative gửi model · sửa được”**. Chọn xong rồi sửa/xóa/bổ sung bất kỳ thẻ nào ngay trong hai ô đó; **bấm tạo sẽ gửi chính xác nội dung hiện tại**, không tự ghép preset, `perfect eyes` hoặc gợi ý sửa vùng một lần nữa. Nút **Áp dụng lại phong cách** reset hai ô từ ý tưởng gốc. Trong tab **Sửa vùng ảnh**, muốn gợi ý tay/chân/mắt thì bấm **Thêm gợi ý sửa vùng vào prompt đang hiển thị**, rồi có thể sửa/xóa trước khi tạo. Nếu bật LoRA mắt, `perfect eyes` được *gợi ý* khi áp dụng preset, có thể xóa; bật LoRA vẫn tải và dùng adapter. Tên preset trong trạng thái/metadata chỉ là lựa chọn UI, **không thêm thẻ sau khi bạn sửa hai ô cuối**. Preset Bán thực 2.5D chỉ điều hướng **cùng checkpoint anime**, không thể bảo đảm khác biệt như một model 2.5D riêng. Negative mẫu nhắm ngón thừa/thiếu/dính; preset thường điền `nsfw, explicit` vào negative, còn NSFW 18+ không điền, nhưng vẫn yêu cầu xác nhận và chặn một số từ khóa vị thành niên trong prompt dương (không phải bộ lọc nội dung/tuổi hoàn chỉnh). Tuân thủ giấy phép model, điều khoản Colab/Gradio; prompt không bảo đảm sửa hết lỗi ngón.

**Bảo mật:** `share=True` tạo URL công khai, **không có tài khoản/mật khẩu**: ai biết link đều có thể dùng GPU của bạn. Không chia sẻ link hay lưu công khai notebook có output chứa link; dừng runtime để ngắt link. File checkpoint/LoRA bị chặn tải qua đường file Gradio; prompt/negative chỉ được nhúng vào PNG nếu bạn tự bật metadata (mặc định tắt).

**Nếu ô 4 lâu:** tải 6,94 GB qua mạng và kiểm SHA-256 toàn bộ file có thể mất thời gian (có in tiến độ kiểm); **không còn bước sao chép Drive**. Nếu thất bại, xem mạng/đĩa Colab và chạy lại; file đúng còn trong cùng runtime sẽ được dùng lại, file hỏng bị từ chối. Không tắt kiểm hash để tăng tốc.

**Nếu RAM hệ thống gần đầy nhưng VRAM gần trống:** mở notebook mới nhất rồi **Runtime → Restart runtime → Run all** để giải phóng model cũ. `VRAM_MODE=auto` ưu tiên GPU trực tiếp khi VRAM trống ≥ **12,5 GiB + 0,4 GiB/LoRA** (hai LoRA: 13,3 GiB); không đủ/OOM mới thử CPU offload, vẫn dùng GPU từng phần nhưng tốn RAM hệ thống. Xem VRAM và chế độ sau khi nạp ở ô 6. Nếu OOM, giảm kích thước, tắt LoRA hoặc chọn `low_vram` rồi restart và Run all.

**Nếu ảnh thứ hai báo `Setting requires_grad=True on inference tensor outside InferenceMode`:** lấy notebook mới nhất và **Runtime → Restart runtime → Run all**; bản Studio giữ thao tác đổi LoRA trong `torch.inference_mode()` kể cả sau offload. Nếu còn lỗi, gửi traceback nhưng che URL Gradio.

**Nếu ô 2 báo xung đột thư viện:** lấy bản mới nhất, **Runtime → Restart runtime → Run all**. Dòng `ERROR: pip's dependency resolver...` có thể chỉ là cảnh báo; ô 2 chỉ được coi đã sẵn sàng khi in **`✅ Thư viện Studio đã sẵn sàng`**. Nếu không có dòng này hoặc có traceback, dừng và gửi đầy đủ traceback (che thông tin riêng).

---

# WAI-illustrious trên Google Colab (notebook nâng cao, không cần link)

[![Mở trong Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manhlee1196-boop/ai-anime/blob/arena/01a0d84b-ai-anime/WAI_Illustrious_Colab.ipynb)

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
- Anatomy Helper được mô tả tập huấn luyện có nhiều ảnh bàn chân và có thể kéo theo thiên lệch phong cách/nội dung; bạn có thể **tắt LoRA này** hoặc chỉ dùng inpaint thủ công. LoRA mắt dùng trigger `perfect eyes`: notebook nâng cao tự thêm khi bật; Studio chỉ điền gợi ý vào prompt khi áp dụng preset để người dùng sửa/xóa nếu muốn. LoRA tay/chân chủ yếu hướng tới **bàn tay/bàn chân và tư thế**, không chứng nhận chữa lỗi ống chân; dùng inpaint ô 8 cho các lỗi chân còn lại.
- Nếu hết **RAM hệ thống** khi nạp checkpoint, cần Colab high-RAM; notebook không thể cấp thêm GPU/RAM. Ảnh chỉ nằm dưới `/content` và mất khi runtime kết thúc (hãy tải xuống trước). Prompt mẫu hướng tới nội dung lành mạnh nhưng **không đảm bảo bộ lọc**. `EMBED_METADATA=True` nhúng prompt/nguồn ảnh vào PNG; tắt trước khi chia sẻ nếu không muốn lộ thông tin.

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
