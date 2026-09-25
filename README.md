# WAI-illustrious trên Google Colab

[![Mở trong Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manhlee1196-boop/ai-anime/blob/arena/01a0d84b-ai-anime/WAI_Illustrious_Colab.ipynb)

Notebook: **[WAI_Illustrious_Colab.ipynb](WAI_Illustrious_Colab.ipynb)** — tạo ảnh từ checkpoint SDXL WAI-illustrious bằng Diffusers. Không cần WebUI hay mở liên kết chia sẻ công khai.

## Chạy nhanh

1. Mở notebook bằng nút Colab ở trên (hoặc vào Colab → **File → Upload notebook**). Chọn **Runtime → Change runtime type → GPU** (T4 hoặc GPU mạnh hơn).
2. Chạy các ô **1 → 6**, chấp thuận gắn Google Drive. **Không cần chuẩn bị model trước:** nếu `MODEL_PATH` chưa tồn tại, `AUTO_DOWNLOAD=True` sẽ tải WAI-illustrious v17 từ [mirror Hugging Face được ghim phiên bản](https://huggingface.co/LyliaEngine/waiIllustriousSDXL_v170/blob/32be7bfdcd406db70df663b9cee3313957deb68f/waiIllustriousSDXL_v170.safetensors), kiểm tra SHA-256 đối chiếu với [metadata Civitai của tác giả](https://civitai.com/api/v1/model-versions/2883731). Checkpoint ~6,94 GB; cần Internet và dung lượng đĩa/Drive. File model **không** được lưu trong repo.
3. Nếu **đã có checkpoint riêng**, sửa `MODEL_PATH` trong ô 3; notebook ưu tiên dùng file đó, **không ghi đè** và không bắt buộc là v17. Thay prompt/kích thước/seed trong ô 6, chạy lại ô 6 để tạo ảnh mới. PNG mặc định lưu ở `MyDrive/AI/outputs`.

## Tải và lưu tài nguyên

- Mặc định model tải vào cache `/content`, sử dụng trực tiếp từ cache **không sao chép thừa**, rồi lưu bản đã xác minh tại `MyDrive/AI/models/WAI-illustrious.safetensors` để phiên sau không tải lại. Nếu không lưu được Drive, phiên hiện tại vẫn chạy bằng bản tạm. Có thể đặt `PERSIST_MODEL_TO_DRIVE=False` nếu không muốn lưu checkpoint vào Drive.
- Cần khoảng **9 GiB** đĩa trống cho tải vào `/content` (file ~6,94 GB + dự phòng). Nếu `/content` quá ít chỗ nhưng đã gắn Drive và bật lưu model, notebook thử tải **trực tiếp vào Drive**. Khi file đã có trên Drive, `CACHE_MODEL_LOCAL=True` chỉ sao chép vào `/content` nếu còn đủ đĩa; nếu không sẽ nạp trực tiếp từ Drive. `/content` bị xoá khi phiên Colab kết thúc.
- Notebook tự cài Diffusers, Transformers, Accelerate, Safetensors và bộ tải HF Xet; **không cài lại PyTorch/CUDA** của Colab. Cấu hình/tokenizer SDXL do Diffusers tự lấy khi cần, không tải thêm trọng số SDXL base. Checkpoint WAI đã kèm VAE/text encoder; không tải VAE riêng.

## Tối ưu khi tạo ảnh

- FP16 + attention SDPA của PyTorch; Euler a, mặc định 25 bước / CFG 6 / 1024×1024. Sinh từng ảnh để giảm đỉnh VRAM. `VRAM_MODE=auto` dùng GPU trực tiếp khi VRAM trống ≥ 14 GiB; nếu thiếu, dùng CPU offload và VAE tiling (chậm hơn). Khi tạo ảnh bị OOM ở chế độ `auto`, notebook **tự nạp lại bằng offload và thử lần nữa cùng seed**.
- Nếu vẫn hết VRAM, chọn `low_vram` hoặc kích thước 768×1024/1024×768. GPU dư VRAM có thể thử preset 1024×1344/1344×1024. Nếu **RAM hệ thống** không đủ để nạp checkpoint, cần dùng runtime high-RAM; notebook không thể tự cấp thêm GPU/RAM. Khi Drive ngắt lúc lưu ảnh, notebook lưu dự phòng ở `/content/wai_outputs` (hãy tải xuống trước khi hết phiên).
- Model có thể tạo nội dung không mong muốn; prompt mẫu không phải bộ lọc bảo đảm an toàn. Kiểm tra ảnh và tuân thủ điều khoản sử dụng model/Colab. `EMBED_METADATA=True` nhúng prompt vào PNG, có thể tắt trước khi chia sẻ.

Kiểm tra notebook không cần GPU: `python -m unittest discover -s tests -v`. Kiểm tra thực tế tốc độ/chất lượng ảnh cần GPU Colab và tải checkpoint ~6,94 GB.
