# WAI-illustrious trên Google Colab

[![Mở trong Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manhlee1196-boop/ai-anime/blob/arena/01a0d84b-ai-anime/WAI_Illustrious_Colab.ipynb)

Notebook: **[WAI_Illustrious_Colab.ipynb](WAI_Illustrious_Colab.ipynb)** — tạo ảnh từ checkpoint SDXL `WAI-illustrious.safetensors` bằng Diffusers. Không cần khởi chạy WebUI hay mở liên kết chia sẻ công khai.

## Chạy nhanh

1. Mở notebook bằng nút Colab ở trên (hoặc vào Colab → **File → Upload notebook** và chọn file `.ipynb`). Chọn **Runtime → Change runtime type → GPU** (T4 hoặc GPU mạnh hơn).
2. Tải **checkpoint SDXL đầy đủ** từ [trang WAI-illustrious-SDXL](https://civitai.com/models/827184/wai-illustrious-sdxl), theo điều khoản của model; lưu tại `MyDrive/AI/models/WAI-illustrious.safetensors`. Nếu tên file khác, sửa `MODEL_PATH` trong ô 3 của notebook. Repo **không** phân phối trọng số model.
3. Chạy các ô **1 → 5**, chấp thuận gắn Google Drive. Thay prompt/kích thước/seed trong ô 5 rồi chạy lại để tạo ảnh mới. PNG mặc định lưu vào `MyDrive/AI/outputs`.

## Tối ưu & lưu ý

- FP16 + attention SDPA của PyTorch; không cài lại PyTorch hoặc xformers. Euler a, mặc định 25 bước / CFG 6 / 1024×1024. Chỉ tạo một ảnh mỗi lượt để hạn chế thiếu VRAM.
- `VRAM_MODE=auto` dùng GPU trực tiếp khi còn ít nhất 14 GiB VRAM; GPU ít VRAM hơn sẽ dùng CPU offload và VAE tiling (tiết kiệm VRAM nhưng chậm hơn). Nếu vẫn gặp lỗi bộ nhớ, khởi động lại runtime rồi chọn `low_vram` hoặc giảm kích thước ảnh.
- `CACHE_MODEL_LOCAL=True` chép model từ Drive vào `/content` để nạp nhanh hơn; cần dung lượng đĩa trống xấp xỉ kích thước file **cộng 2 GiB**. Có thể tắt nếu ổ Colab không đủ chỗ. Bộ nhớ `/content` bị xoá khi phiên Colab kết thúc; ảnh lưu ở Drive vẫn còn.
- Notebook cần Internet lần đầu để lấy cấu hình/tokenizer; không tự tải checkpoint. Hãy dùng file `.safetensors` đầy đủ, không phải LoRA/UNet-only/FP8. Seed và thông số có thể được lưu trong metadata PNG; tắt `EMBED_METADATA` nếu không muốn chia sẻ prompt.
- Colab cấp GPU theo hạn mức riêng. Model có thể tạo nội dung không mong muốn; prompt mẫu không phải bộ lọc bảo đảm an toàn. Hãy kiểm tra kết quả và tuân thủ điều khoản sử dụng model/Colab.
