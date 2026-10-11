# File dịch tiếng Việt cho catalog prompt

Định dạng SAA-compatible: `tag,category,translation`, không header, UTF-8
không BOM, xuống dòng LF. Mọi tag trong 12 nhóm được chọn đều có nhãn.
Tag thiếu bản dịch thật nhận nhãn dịch máy dự phòng; dấu `_` được đổi thành
khoảng trắng, từ chưa biết được giữ nguyên để không bịa nghĩa.

| File | Tổng dòng | Dịch thật | Dịch máy | Tag nguồn | Chưa có bản dịch thật |
| --- | ---: | ---: | ---: | ---: | ---: |
| `01_chu_the.csv` | 938 | 472 | 466 | 938 | 466 |
| `04_loai_lore.csv` | 5,757 | 684 | 5,073 | 5,757 | 5,073 |
| `05_trang_phuc_quan_ao.csv` | 7,136 | 5,685 | 1,451 | 7,136 | 1,451 |
| `06_phu_kien.csv` | 3,366 | 2,573 | 793 | 3,366 | 793 |
| `07_ngoai_hinh.csv` | 11,540 | 7,719 | 3,821 | 11,540 | 3,821 |
| `08_tu_the_bieu_cam.csv` | 2,639 | 1,056 | 1,583 | 2,639 | 1,583 |
| `09_boi_canh.csv` | 1,198 | 441 | 757 | 1,198 | 757 |
| `10_phong_cach.csv` | 268 | 46 | 222 | 268 | 222 |
| `11_anh_sang_mau_sac.csv` | 3,722 | 1,665 | 2,057 | 3,722 | 2,057 |
| `12_bo_cuc_ky_thuat.csv` | 644 | 255 | 389 | 644 | 389 |
| `13_vat_the.csv` | 2,082 | 724 | 1,358 | 2,082 | 1,358 |
| `99_khac.csv` | 43,364 | 8,116 | 35,248 | 43,364 | 35,248 |

Tổng: **82,654 dòng** — 29,436 dịch thật + 53,218 dịch máy.

Tạo lại bằng:
```bash
python scripts/build_prompt_catalog_vi.py
```
