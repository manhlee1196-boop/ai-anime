# Kiểm tra nguồn và vận hành WAI Studio (26/09/2026)

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

## 2. Kiểm tra logic xác minh và dùng tài nguyên

- Ô 3 mặc định `ANATOMY_LORA_PATH = ""`, `EYE_LORA_PATH = ""`, hai cờ LoRA bật: ô 5 tự dùng file trên Drive đã kiểm tra hoặc tải file còn thiếu từ repo **và commit ghim**. Không cần nhập đường dẫn thủ công. File thủ công vẫn phải khớp kích thước + SHA-256, không chỉ đúng tên.
- Ô 4 băm checkpoint tải về; bản Studio **không cho qua ô chuẩn bị nếu checkpoint không đúng toàn bộ SHA-256 WAI v17**, kể cả file do người dùng cung cấp hoặc cache sao chép sai. Notebook nâng cao cho phép SDXL riêng nhưng ghi rõ **không xác thực là WAI** nếu khác hash.
- Ô 5 kiểm tra size + SHA-256 và header safetensors cho mỗi LoRA; ô 6 kiểm tra lại file ngay trước khi nạp hoặc nạp lại sau OOM. File sai hash bị từ chối, **không tự ghi đè**.
- Đã tạo các file safetensors nhỏ hợp lệ bằng thư viện thật và chạy chính các hàm kiểm tra trong notebook: header đúng được chấp nhận; thay đổi **một byte** ở nội dung checkpoint hoặc LoRA bị từ chối bởi SHA-256. Đây là phép thử cơ chế xác minh, **không thay cho kiểm tra ba file model thật**.

## 3. Kiểm thử chức năng đã thực hiện

| Kiểm tra | Kết quả | Giới hạn |
| --- | --- | --- |
| Cài bộ thư viện notebook vào môi trường Python riêng (`gradio==6.15.2`, `gradio-client==2.5.0`, `diffusers==0.35.2`, `transformers==4.52.4`, `accelerate==1.10.1`, `peft==0.17.1`, `huggingface-hub==0.36.2`); `pip check`; import SDXL/SDPA | Cài được, **không có dependency conflict** trong môi trường thử | Không giống hệt gói có sẵn trên Colab của bạn; ô 2 vẫn phải in `✅ Thư viện Studio đã sẵn sàng` |
| `python -m unittest discover -s tests -v` | **45/45 đạt, 0 bỏ qua**; gồm fake GPU T4 với 2 LoRA, OOM→offload cùng seed, LoRA/hash, phong cách/18+, ảnh thứ 2 với **PyTorch CPU thật**, các sự kiện Gradio 6 và HTTP nội bộ chặn đường file checkpoint | Model WAI/GPU trong các bài tạo ảnh vẫn được giả lập; test PyTorch chỉ tái hiện lỗi tensor inference-mode |
| Kiểm tra `nbformat` cả hai notebook, bản Studio đúng mã nguồn generator; Black và `git diff --check` | Đạt | Cấu trúc đúng không chứng minh model tạo ảnh đẹp hoặc tải file thành công trên Colab |
| Cloudflare tùy chọn: `npm ci`, `npm test`, `npm run build`, `npm run format:check` | **14/14 test đạt**, build/format đạt | Không triển khai Cloudflare; Workers AI và backend WAI bên ngoài **chưa** được thử thật |

Sandbox hiện **không có GPU** (`torch.cuda.is_available() == False`). Thử tải ngay cả `README.md` nhỏ bằng `hf_hub_download` tới Hugging Face trong sandbox này gặp `SSLZeroReturnError` / TLS EOF; dịch vụ duyệt web vẫn đọc được các API metadata công khai. Đây là giới hạn kết nối của sandbox, **không chứng minh mạng Colab của bạn gặp cùng lỗi**. Không thể nói đã chạy thành công checkpoint WAI 6,94 GB + hai LoRA để tạo ảnh thật, hoặc khẳng định mức VRAM/tốc độ/chất lượng thực tế.

## 4. Phép thử cuối cùng cần chạy trên chính Colab của bạn

1. Mở [notebook Studio trên nhánh hiện tại](https://colab.research.google.com/github/manhlee1196-boop/ai-anime/blob/arena/01a0d84b-ai-anime/WAI_Illustrious_Studio_Colab.ipynb), chọn T4 GPU hoặc GPU mạnh hơn và **Runtime → Restart runtime → Run all**. Để trống hai đường dẫn LoRA ở ô 3 nếu muốn tải tự động. Nếu chạy từ link Gradio cũ, mã mới không tự cập nhật.
2. Ô 1 phải thấy GPU; ô 2 phải in `✅ Thư viện Studio đã sẵn sàng`; ô 4 in thông báo **xác minh SHA-256 v17**; ô 5 in `Đã xác minh anatomy` / `Đã xác minh eyes` (nếu bật cả hai); ô 6 in `Chế độ sau khi nạp` và VRAM. Nếu một ô lỗi, dừng tại đó, đừng coi ô sau là đã thử thành công.
3. Ô 8 tạo link tạm **không đăng nhập**. Trong UI dùng prompt thường, seed cố định (ví dụ `12345`), 512×512, 15–20 steps; thử **2 ảnh liên tiếp**, tải PNG, rồi đổi Anime chuẩn ↔ Bán thực 2.5D và kiểm tra hai ô **Xem prompt sau khi áp dụng phong cách**. Ảnh thực/hiệu quả LoRA/phong cách chỉ có thể đánh giá bằng bước này. Test kiểm tra *đúng prompt được gửi*, không chứng minh ảnh 2.5D trông khác rõ trên checkpoint anime.
4. Nếu cần hỗ trợ, gửi **log ô 2, 4, 5, 6 và traceback lúc tạo ảnh** cùng ảnh chụp kết quả không nhạy cảm. **Che đường dẫn riêng; đừng gửi URL `gradio.live`**: bất kỳ ai có URL đều dùng được GPU của bạn.

**Kết luận:** ba tham chiếu tài nguyên và hash ghim **khớp metadata của cả Civitai lẫn HF**; cơ chế kiểm tra file và luồng giao diện **đạt kiểm thử cục bộ**. Trạng thái “đã tạo ảnh WAI thật trên Colab” **chưa được xác nhận** và không thể suy ra từ bộ test giả lập. Hiệu quả sửa tay/mắt và mức độ bán thực 2.5D **không được bảo đảm** bằng hash hay kiểm thử phần mềm. Preset 18+ chỉ yêu cầu xác nhận và kiểm tra một số từ khóa, **không xác minh tuổi hay lọc nội dung hoàn chỉnh**.
