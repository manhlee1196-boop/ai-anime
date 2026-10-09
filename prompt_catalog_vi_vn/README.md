# File dịch tiếng Việt cho catalog prompt

Định dạng SAA-compatible: `tag,category,translation`, không header, UTF-8
không BOM, xuống dòng LF. Các nhóm thông thường chỉ ghi bản dịch thật.
Riêng `99_khac.csv`, tag thiếu bản dịch thật nhận nhãn dịch máy dự phòng; dấu `_`
được đổi thành khoảng trắng, từ chưa biết được giữ nguyên để không bịa nghĩa.

| File | Tổng dòng | Dịch thật | Dịch máy | Tag nguồn | Chưa có bản dịch thật |
| --- | ---: | ---: | ---: | ---: | ---: |
| `01_chu_the.csv` | 472 | 472 | 0 | 938 | 466 |
| `04_loai_lore.csv` | 684 | 684 | 0 | 5,757 | 5,073 |
| `05_trang_phuc_quan_ao.csv` | 5,685 | 5,685 | 0 | 7,136 | 1,451 |
| `06_phu_kien.csv` | 2,573 | 2,573 | 0 | 3,366 | 793 |
| `07_ngoai_hinh.csv` | 7,719 | 7,719 | 0 | 11,540 | 3,821 |
| `08_tu_the_bieu_cam.csv` | 1,056 | 1,056 | 0 | 2,639 | 1,583 |
| `09_boi_canh.csv` | 441 | 441 | 0 | 1,198 | 757 |
| `10_phong_cach.csv` | 46 | 46 | 0 | 268 | 222 |
| `11_anh_sang_mau_sac.csv` | 1,665 | 1,665 | 0 | 3,722 | 2,057 |
| `12_bo_cuc_ky_thuat.csv` | 255 | 255 | 0 | 644 | 389 |
| `13_vat_the.csv` | 724 | 724 | 0 | 2,082 | 1,358 |
| `99_khac.csv` | 43,364 | 8,116 | 35,248 | 43,364 | 35,248 |

Tổng: **64,684 dòng** — 29,436 dịch thật + 35,248 dịch máy.

Tạo lại bằng:
```bash
python scripts/build_prompt_catalog_vi.py
```
