# File dịch tiếng Việt cho catalog prompt

Định dạng SAA-compatible: `tag,category,translation`, không header, UTF-8
không BOM, xuống dòng LF. Chỉ các tag có bản dịch thật được ghi; tag chưa
dịch bị bỏ qua, không bịa nhãn.

| File | Dòng đã dịch | Tag nguồn | Chưa dịch |
| --- | ---: | ---: | ---: |
| `01_chu_the.csv` | 472 | 938 | 466 |
| `04_loai_lore.csv` | 684 | 5,757 | 5,073 |
| `05_trang_phuc_quan_ao.csv` | 5,685 | 7,136 | 1,451 |
| `06_phu_kien.csv` | 2,573 | 3,366 | 793 |
| `07_ngoai_hinh.csv` | 7,719 | 11,540 | 3,821 |
| `08_tu_the_bieu_cam.csv` | 1,056 | 2,639 | 1,583 |
| `09_boi_canh.csv` | 441 | 1,198 | 757 |
| `10_phong_cach.csv` | 46 | 268 | 222 |
| `11_anh_sang_mau_sac.csv` | 1,665 | 3,722 | 2,057 |
| `12_bo_cuc_ky_thuat.csv` | 255 | 644 | 389 |
| `13_vat_the.csv` | 724 | 2,082 | 1,358 |
| `99_khac.csv` | 8,116 | 43,364 | 35,248 |

Tổng: **29,436 dòng dịch**.

Tạo lại bằng:
```bash
python scripts/build_prompt_catalog_vi.py
```
