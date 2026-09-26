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

## 2. Luồng `/content` và prompt có thể sửa

- Hai notebook **không gọi `google.colab.drive.mount`**, không đặt cờ sao chép/lưu Drive. `MODEL_PATH` mặc định `/content/wai_model_cache/waiIllustriousSDXL_v170.safetensors`, LoRA bật nhưng hai đường dẫn thủ công để rỗng, kết quả `/content/wai_outputs`. Ô 3 từ chối đường dẫn ngoài `/content` và `/content/drive`, kể cả cache local trỏ bằng symlink. Không còn fallback sang Drive khi đĩa thiếu; báo lỗi để người dùng giải phóng đĩa/tắt LoRA.
- Ô 4 dùng `hf_hub_download(..., revision=commit, local_dir=/content/wai_model_cache, token=False)` khi chưa có checkpoint và **nạp chính file được tải**, không tạo bản sao thứ hai; nếu file đã có trong cùng runtime thì dùng lại và kiểm đầy đủ hash. Studio **chỉ chấp nhận WAI v17 trùng toàn bộ SHA-256**, bao gồm đường dẫn thủ công/cached; bản nâng cao cho phép checkpoint SDXL khác nhưng nêu rõ chưa xác thực phiên bản. Ô 5 tự tải LoRA được bật vào `/content/wai_lora_cache`, xác nhận size + SHA-256 + header; file thủ công vẫn bắt buộc đúng hash. File đã có sai hash bị từ chối, không tự ghi đè. Ô 6 kiểm lại LoRA trước khi nạp/khôi phục sau OOM.
- Studio chọn phong cách/đổi ý tưởng gốc/negative gốc/bật tắt LoRA mắt sẽ **điền lại hai ô prompt cuối, có thể sửa trực tiếp**. Lệnh tạo text, img2img và inpaint dùng **đúng chuỗi hai ô tại thời điểm bấm tạo**, kể cả khi người dùng xóa thẻ preset hoặc `perfect eyes`: runtime không ghép lại preset hay ẩn thêm gợi ý sửa. Nút sửa vùng thêm từ vào **hai ô đang hiển thị**, không thay đổi ngầm lúc suy luận; adult preset vẫn yêu cầu xác nhận 18+ và kiểm một số từ khóa vị thành niên trong prompt dương.
- Kiểm thử file/hàm kích thước nhỏ và mock giúp phát hiện sai luồng, **không** thay cho tải và xác minh ba file lớn thực tế từ Colab. Ba mã hash/size trong bảng vẫn là đối chiếu metadata độc lập với file thực nhận.

## 3. Kiểm thử mã đã thực hiện cho thay đổi này

| Kiểm tra | Kết quả | Giới hạn |
| --- | --- | --- |
| `python -m unittest discover -s tests -v` với Gradio 6.15.2, Pillow và Hugging Face Hub ghim trong venv | **44 test: 43 đạt, 1 bỏ qua**; gồm local-only download/cache/hash, đĩa thiếu, 3 luồng prompt qua chính sự kiện Gradio 6, download PNG và chặn file weights, OOM/offload, regression lần tạo thứ hai bằng fake inference-mode | Bài dùng **PyTorch CPU thật** cần gói `torch`, chưa được chạy lại trong venv hiện tại; model WAI/GPU đều được giả lập |
| Kiểm tra `nbformat` hai notebook, notebook Studio khớp `build()`, kiểm cú pháp các ô Python; Black, `git diff --check` | Đạt | Cấu trúc đúng không chứng minh Colab tải model hay tạo ảnh chất lượng |
| `pip check` trong venv thử Gradio 6.15.2 + `huggingface-hub==0.36.2` | Đạt | Không phải toàn bộ tập thư viện Colab. Báo cáo trước đã kiểm cài tập Diffusers/PEFT/PyTorch CPU trong venv khác, nhưng không thay thế kiểm tra ô 2 trên Colab của bạn |
| Cloudflare tùy chọn ở `web/` | Không sửa trong thay đổi này; báo cáo trước ghi 14/14 test đạt | Không triển khai Worker hay tạo ảnh WAI thật |

Sandbox hiện **không có GPU**. Thử tải qua Hugging Face trực tiếp tại sandbox từng gặp TLS EOF; đây **không chứng minh** mạng Colab cũng gặp lỗi. Không tải 6,94 GB checkpoint/two LoRA để thực sự so hash byte ở đây, không thử tốc độ tải hoặc chất lượng ảnh thực tế. Tốc độ còn tùy mạng Colab; loại bỏ Drive chỉ bỏ thao tác gắn/sao chép/lưu, **không bảo đảm tải qua mạng nhanh hơn**.

## 4. Phép thử cuối cùng cần chạy trên Colab của bạn

1. Mở [notebook Studio trên nhánh hiện tại](https://colab.research.google.com/github/manhlee1196-boop/ai-anime/blob/arena/01a0d84b-ai-anime/WAI_Illustrious_Studio_Colab.ipynb), chọn GPU và **Runtime → Restart runtime → Run all**. Không cần cấp quyền Google Drive. Để trống hai đường dẫn LoRA ở ô 3 nếu dùng bản tự tải. Link Gradio của phiên cũ **không tự cập nhật**.
2. Ô 1 thấy GPU; ô 2 in `✅ Thư viện Studio đã sẵn sàng`; ô 4/5 in xác minh SHA-256 WAI v17 và LoRA; ô 6 in chế độ nạp và VRAM. Ô 4 có thể mất thời gian tải + hash toàn bộ 6,94 GB, **không còn dòng “Sao chép model từ Drive”**. Nếu thiếu dung lượng/mạng lỗi, dừng xử lý tại ô đó; đừng coi việc chạy các ô sau là đã xác nhận thành công.
3. Ở giao diện, chọn **Bán thực 2.5D** và xem hai ô **“Prompt gửi model · sửa được”**. Xóa thẻ phong cách/`perfect eyes`, nhập một prompt riêng và negative riêng, bật metadata PNG nếu muốn tự đối chiếu; tạo ảnh và tải PNG. Thử **hai lần liên tiếp** cùng seed (ví dụ `12345`), so với Anime chuẩn giữ thông số tương đương. Dùng tab sửa vùng: chọn tay/chân/mắt, bấm nút thêm gợi ý rồi sửa chúng trước khi inpaint; thử không bấm để đảm bảo không có thẻ ẩn. Thử adult preset chỉ khi mọi nhân vật đều trưởng thành. Hiệu quả thực tế chỉ có thể đánh giá ở bước này.
4. **Tải tất cả ảnh cần giữ về máy trước khi runtime ngắt**: ảnh, checkpoint và LoRA trong `/content` sẽ mất cùng phiên. Nếu lỗi, gửi traceback ô 2/4/5/6/8, che thông tin riêng và URL `gradio.live` vì ai biết URL đều có thể dùng GPU của bạn.

**Kết luận:** SHA-256 ghim khớp metadata Civitai/HF của đúng file; mã local-only và luồng prompt cuối đã qua kiểm thử cục bộ. Chưa xác nhận tạo ảnh WAI thật trên GPU Colab hoặc tốc độ/hiệu quả sửa ngón và bán thực 2.5D. Preset 18+ chỉ xác nhận của người dùng và kiểm một số từ khóa, **không xác minh tuổi/bộ lọc hoàn chỉnh**.
