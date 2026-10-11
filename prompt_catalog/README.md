# Catalog prompt đã chia nhóm

Nguồn: `danbooru_e621_merged_2026-10-01_pt20-ia-dd-ed-spc.csv` · **348,716 dòng**.
Các file con dùng cùng schema 5 cột, không có header. Phân nhóm là **độc quyền**:
mỗi tag xuất hiện đúng một lần; tag chưa khớp quy tắc nằm trong `99_khac.csv`.

| File | Nhóm | Số dòng | Kích thước |
| --- | --- | ---: | ---: |
| `01_chu_the.csv` | Chủ thể / số lượng nhân vật | 938 | 63,152 byte |
| `02_hoa_si.csv` | Họa sĩ | 137,145 | 4,211,058 byte |
| `03_tac_pham_nhan_vat.csv` | Tác phẩm / nhân vật | 128,917 | 5,266,545 byte |
| `04_loai_lore.csv` | Loài / lore | 5,757 | 242,719 byte |
| `05_trang_phuc_quan_ao.csv` | Trang phục / quần áo | 7,136 | 375,357 byte |
| `06_phu_kien.csv` | Phụ kiện | 3,366 | 165,893 byte |
| `07_ngoai_hinh.csv` | Ngoại hình / bộ phận cơ thể | 11,540 | 631,694 byte |
| `08_tu_the_bieu_cam.csv` | Tư thế / hành động / biểu cảm | 2,639 | 136,927 byte |
| `09_boi_canh.csv` | Bối cảnh / thiên nhiên | 1,198 | 68,456 byte |
| `10_phong_cach.csv` | Phong cách / chất liệu / loại hình | 268 | 13,447 byte |
| `11_anh_sang_mau_sac.csv` | Ánh sáng / màu sắc | 3,722 | 183,036 byte |
| `12_bo_cuc_ky_thuat.csv` | Bố cục / kỹ thuật / chất lượng | 644 | 32,065 byte |
| `13_vat_the.csv` | Vật thể / đồ vật | 2,082 | 106,296 byte |
| `99_khac.csv` | Khác | 43,364 | 2,141,249 byte |

Tạo lại bằng:
```bash
python scripts/split_prompt_catalog.py
```

Quy tắc ưu tiên: danh mục họa sĩ/tác phẩm/nhân vật/loài trước; sau đó là
trang phục → phụ kiện → ngoại hình → tư thế → bối cảnh → phong cách →
ánh sáng/màu → bố cục/kỹ thuật → vật thể → khác.
