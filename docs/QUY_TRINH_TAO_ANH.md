# Quy trình tạo ảnh chuẩn trong WAI Studio (ô 7 của notebook Colab)

Tài liệu này giải thích accordion **🧭 Quy trình chuẩn · khung prompt + negative tối ưu** và cách dùng nó cùng các tính năng khác của Studio. Mọi con số dưới đây lấy **trực tiếp từ mã** trong `colab/studio.py` (được nhúng nguyên văn vào ô 7 của `WAI_Illustrious_Studio_Colab.ipynb`).

> **Semi-auto tag complete (tham khảo Character Select SAA):** khi Studio khởi động, CSV tag prompt đã lọc (348.716 dòng, bỏ danh mục Metadata/không hợp lệ) được tải/đọc và xác minh SHA-256 một lần; autocomplete dùng chung catalog đó với tab Kho thẻ. Gõ một từ khóa ở cuối Prompt như `mắt`/`eyes`, `tóc`/`hair`, `nơ`/`bow` hoặc `váy`/`dress` để tìm các tên tag tiếng Anh trong CSV; mỗi kết quả kèm nhãn danh mục kiểu SAA — `[G] [A] [©] [C] [M]` cho Danbooru, `<G> <A> <©> <C> <S> <M> <L>` cho e621. Ba kiểu truy vấn giống SAA: `chữ đầu` tìm tiền tố, `*đuôi` tìm hậu tố, `*giữa*` tìm ở giữa tên thẻ, và `@tên` chỉ lọc thẻ họa sĩ (nhóm 1 và 8) mà **không** chèn dấu `@` vào prompt WAI-Illustrious. Chọn một dòng (chuột hoặc ↑↓ + Enter/Tab) để thay cụm từ khóa cuối; Esc đóng danh sách. `Ctrl+↑`/`Ctrl+↓` — hoặc hai nút `+0,1` / `−0,1` — chỉnh trọng số tag đang bôi đen, tag ở con trỏ, hoặc cụm cuối prompt, mỗi lần `0,1` trong dải `0,1–2,0`; về `1,0` thì dấu `(tag:…)` được gỡ bỏ. Nếu không tải được catalog thì thao tác tạo ảnh vẫn hoạt động và có thể thử tải lại từ tab Kho thẻ.

> **Thư viện prompt trên điện thoại:** danh sách prompt trong accordion 📚 là **danh sách chạm (radio) cuộn được** — chạm một dòng là nạp ngay, hoặc chọn dòng rồi bấm **⬇️ Nạp prompt đã chọn**.

> **Dịch prompt Việt → English:** nhập hoặc dán nội dung vào **Prompt** và/hoặc **Negative prompt**, sau đó bấm **🇻🇳 → 🇬🇧 Dịch prompt Việt sang English**. Translator chạy cục bộ, chỉ dùng nhãn Việt đã xác minh và tag English canonical trong catalog; không gọi API/model dịch ngoài. Cú pháp trọng số như `(tóc dài:1.2)` được giữ lại. Phần chưa có trong bộ từ vựng được giữ nguyên và báo trong status để bạn tự rà soát; đây là bộ dịch tag/cụm cho prompt ảnh, **không phải dịch câu tự do đầy đủ**. Các tên tag English canonical đã có sẵn cũng được giữ nguyên.

> **Catalog chia nhóm:** ngoài file tổng, thư mục `prompt_catalog/` có các file 5 cột không header theo nhóm độc quyền: `trang_phuc_quan_ao`, `phu_kien`, `ngoai_hinh`, `tu_the_bieu_cam`, `boi_canh`, `phong_cach`, `anh_sang_mau_sac`, `bo_cuc_ky_thuat`, cùng nhóm chủ thể/họa sĩ/tác phẩm/loài và `99_khac`. `manifest.json` xác nhận tổng số dòng bằng catalog chính; chạy `python scripts/split_prompt_catalog.py` để tạo lại. Các nhóm `01`, `04`–`13` và `99_khac` có file 3 cột trong `prompt_catalog_vi_vn/`, tạo bằng `python scripts/build_prompt_catalog_vi.py`. Toàn bộ tag còn thiếu trong 12 nhóm này nhận **53.218 nhãn dịch máy**; dấu `_` được đổi thành khoảng trắng, từ chưa biết giữ nguyên thay vì bịa nghĩa. Các nhóm `02` và `03` vẫn không nằm trong phạm vi file dịch.

> **Ảnh nguồn cho ◈ Biến đổi · ⤢ Phóng to · ✎ Sửa vùng:** không còn kiểu "luôn lấy ảnh mới nhất". Khung
> **02 · Kết quả** có ô **Ảnh sẽ nạp vào tab sửa / phóng** (mặc định = ảnh vừa tạo, bấm ảnh trong thư viện để
> đổi), nút **↪ Nạp ảnh đã chọn vào cả ba tab** để chuẩn bị cho cả sửa lẫn biến đổi trong một lần bấm, và ba nút
> `→` nếu chỉ cần một tab. `↻` nạp danh sách từ `/content/wai_outputs` nên ảnh của lượt tạo trước vẫn dùng được.

> **Giao diện phản hồi cả lúc đang tạo ảnh:** tìm tag và đọc danh sách ảnh nguồn chạy trong hàng đợi riêng, không chặn
> nhau với job GPU (job GPU vẫn một lượt một để giữ VRAM). Bấm ảnh trong thư viện để chọn ảnh nguồn là tức thời vì chỉ đọc
> bộ nhớ; chỉ `↻` mới quét `/content/wai_outputs`. Khi bạn gõ nhanh, lượt tìm catalog đang chạy sẽ tự hủy để theo kịp phím
> cuối cùng — vì vậy danh sách gợi ý có thể trễ một nhịp thay vì làm trang đứng lại.

> **Chuyển tab nhẹ:** tab **✎ Sửa vùng** chỉ nạp một bản ảnh (không còn bản sao `composite`) nên mở ngay sau khi
> "Nạp ảnh đã chọn vào cả ba tab" ít bị khựng hơn trên điện thoại; phím tắt `Ctrl+↑/↓` chỉ theo dõi vùng ô prompt
> rồi ngừng quan sát, không chạy theo mọi thay đổi của cả trang.

> **Công cụ tô vùng:** trong tab **✎ Sửa vùng ảnh**, thanh công cụ dọc của ImageEditor có **Brush** và **Eraser**; nét Brush màu trắng là vùng sửa, nền trong suốt/đen là vùng giữ. Nếu dùng điện thoại, xoay/cuộn nhẹ trong khung để thấy thanh công cụ bên trái; phiên bản Studio giữ sẵn một layer vẽ để cọ hoạt động ngay sau khi ảnh được nạp.

> **Gõ tiếng Việt mượt hơn từ đợt này:** bảng tra theo nhãn tiếng Việt và bảng tên thẻ đã chuẩn hoá được **dựng sẵn ở Ô 7/Ô 8**
> (bạn thấy dòng "Đã dựng index … từ khóa từ nhãn tiếng Việt"). Trước đây bảng này được dựng ngay trong cú gõ đầu tiên —
> mất ~1 s CPU và làm đứng cả trang, chỉ xuất hiện sau khi thêm tra nhãn Việt.

> **Khi gặp đơ, lấy số liệu thay vì mô tả:** Studio in vào output Ô 8 dòng `⏱ … hết X s` cho sự kiện chậm (ngưỡng 0,3 s) và
> `⏳ vẫn đang chạy sau 10 s`; ngay trên trang sẽ hiện dải riêng "Trang bị chặn X s khi chuyển tab „…" — dải này không phải mất kết nối và không có nút tải lại. Mở DevTools rồi gõ
> `window.__waiBlockLog` để xem 12 lần chặn gần nhất. Gửi các dòng đó là đủ để chỉ ra nguyên nhân.

> **Xác nhận bản đang chạy:** đầu trang có chip **"Bản dựng …"**. Mã giao diện nằm trong **ô 7**, nên sau khi lấy
> notebook mới phải chạy lại ô 7 rồi ô 8 và ô 9; chỉ chạy lại ô 9 thì trình duyệt vẫn dùng mã cũ và sẽ vẫn thấy đơ.

> **Nguyên tắc quan trọng:** cả sáu nút trong accordion **chỉ ghi nội dung hiển thị** vào ô *Prompt gửi model*, ô *Negative gửi model* hoặc ô báo cáo. Không có thẻ nào được ghép ngầm khi bạn bấm tạo ảnh — bạn xem, sửa hoặc xóa trước khi tạo.

---

## 1. Sáu nút trong **🧭 Quy trình chuẩn**

### 1.1. Sắp xếp prompt theo thứ tự chuẩn
Đọc prompt hiện tại, tách thành từng thẻ (phân tách bằng dấu phẩy), rồi:

- **Khử trùng lặp** theo phần lõi của thẻ (bỏ trọng số `(thẻ:1.1)` và ngoặc khi so sánh; ví dụ `(long hair:1.1)` và `long hair` bị coi là một).
- **Phân nhóm** mỗi thẻ vào 12 nhóm, xếp theo thứ tự ưu tiên CLIP — chủ thể và chi tiết
  nhân vật dẫn đầu, phần kỹ thuật xếp sau:

  `Chủ thể → Nhãn phân loại → Ngoại hình → Trang phục → Tư thế/hành động → Bố cục/góc máy → Bối cảnh → Ánh sáng/màu → Phong cách → Thẻ khác của bạn → Chất lượng → Độ nét (cuối prompt)`

  Việc khớp nhóm chạy trên bản chuẩn hoá: `blue_eyes` được đọc thành `blue eyes` trước khi
  so với luật (giữa `_` và chữ không có ranh giới từ nên `\beyes\b` không khớp), rồi mới thử
  lại trên bản gốc cho các luật có gạch dưới trong regex (`rating_\w+`). Cơ thể và bộ phận
  nhân vật (vú, đuôi, cánh, sừng, bộ phận sinh dục, hình xăm, khuyên xuyên) được xếp vào
  nhóm **Ngoại hình** để nằm trong khối dẫn đầu prompt; tay/chân/đầu vẫn thuộc **Tư thế**
  vì luật tư thế được xét trước.
  Trong một nhóm, thứ tự bạn đã viết được giữ nguyên. Thẻ không khớp nhóm nào nằm ở
  nhóm *Thẻ khác của bạn* (sau phong cách, ngay trước khối chất lượng). Khối chất lượng
  đặt **áp chót** thay vì mở đầu: nhà phát hành WAI v17 chỉ khuyến nghị *có* 2–3 thẻ
  `masterpiece/best quality` (không nhiều hơn), còn để cuối thì prompt không bị chúng
  chen mất chỗ của thẻ tả nhân vật; `absurdres` vẫn chốt cuối vì thẻ độ nét đặt sau cùng
  cho kết quả tốt nhất.
- **Thêm thẻ neo còn thiếu** của khung đã chọn — chỉ thêm khi nhóm đó **đang trống**, để không nhồi thừa thẻ chất lượng.
- Trả về chuỗi mới **kèm ghi chú**: số thẻ đã sắp xếp, phân nhóm, thẻ neo đã thêm, thẻ trùng đã bỏ.
- Nếu kết quả dài hơn 2200 ký tự, nút báo lỗi và **không** ghi gì (bỏ bớt thẻ rồi bấm lại).

### 1.2. Nạp negative đã chọn
Chọn một trong **8 bộ negative** ở dropdown, chọn cách áp dụng rồi bấm nút:

- **Ghi đè ô negative** — thay toàn bộ nội dung ô negative bằng bộ đã chọn.
- **Nối thêm thẻ còn thiếu** — chỉ thêm những thẻ chưa xuất hiện trong ô hiện tại.

Bảng bộ negative (số thẻ lấy từ `NEGATIVE_PRESETS`):

| Nhãn trong dropdown | Dùng khi | Số thẻ |
| --- | --- | ---: |
| Chuẩn nhà phát hành WAI v17 (ngắn nhất) | Mặc định cho hầu hết ảnh; đúng khuyến nghị nhà phát hành | 5 |
| Illustrious chuẩn · chất lượng + lỗi vẽ | Muốn negative đầy hơn cho lỗi vẽ/chất lượng | 15 |
| Sửa tay / chân / tỷ lệ cơ thể | Ảnh hay lỗi bàn tay, bàn chân, tỷ lệ | 17 |
| Giữ chất anime 2D (chống 3D/thực) | Ảnh bị ra chất 3D/nhựa/thực | 14 |
| Chân dung · mặt, mắt, răng | Chân dung cận mặt | 17 |
| Phong cảnh · không có nhân vật | Phong cảnh, tránh lọt người | 16 |
| An toàn nội dung (mọi nhân vật trưởng thành) | Muốn đẩy các thẻ nội dung người lớn/`loli`/`shota` sang negative | 12 |
| Inpaint / sửa vùng (rất ngắn) | Sửa vùng nhỏ, denoise thấp, cần negative ngắn | 8 |

Bộ **Chuẩn nhà phát hành WAI v17** = `bad quality, worst quality, worst detail, sketch, censor` — theo khuyến nghị “đừng thêm quá nhiều thẻ chất lượng/thẩm mỹ và đừng viết negative quá dài vì sẽ làm giảm chất lượng, ảnh dễ mờ” (xem [Nguồn](#nguồn)).

### 1.3. 🩺 Kiểm tra prompt & thông số
Chỉ đọc và báo cáo, **không sửa gì**. Các mục được kiểm:

| Nhóm kiểm tra | Ngưỡng trong mã | Ý nghĩa |
| --- | --- | --- |
| Độ dài prompt | ước lượng token theo khối **75 token** của SDXL (`<= 75` = ✅, `<= 150` = ⚠️, hơn nữa = ⚠️ nặng) | Phần vượt khối bị đẩy sang khối 75 token sau và mất ưu tiên; gợi ý dùng `BREAK` (viết hoa) để tách khối có chủ đích |
| Thẻ chất lượng | > 3 thẻ = ⚠️, 0 thẻ = gợi ý thêm `masterpiece, best quality, amazing quality` | Nhà phát hành cảnh báo thừa thẻ chất lượng/thẩm mỹ làm mờ ảnh |
| Thẻ trùng lặp | mọi thẻ trùng theo phần lõi | Lặp thẻ tốn token mà không tăng trọng số |
| Thẻ vừa dương vừa âm | thẻ xuất hiện ở cả prompt và negative | Hai lệnh ngược nhau làm model dao động |
| Cú pháp Pony | `score_…`, `source_…` | WAI-illustrious không dùng hệ Pony |
| Thẻ chữ/ký hiệu | `text`, `watermark`, `signature`, `logo`, `username`, `artist name` trong prompt dương | Nên đưa sang negative nếu không muốn chữ trong ảnh |
| `detailed eyes` | có trong prompt dương | Được cộng đồng báo gần như không tác dụng trên họ Illustrious; dùng LoRA mắt + `perfect eyes` hoặc mô tả cụ thể |
| Thẻ phong cách | thiếu `cel shading`, `anime illustration`… | Dễ ra chất 3D/nhựa khi thiếu thẻ chất lượng + phong cách |
| Số lượng chủ thể | thiếu `1girl`, `1boy`, `solo`, `no humans`… | Thiếu dễ thừa nhân vật |
| Trọng số | ngoài khoảng **0.5–1.2** (nhất là > 1.2) | Quá 1.2 dễ cháy nét/mất bố cục; ưu tiên mô tả bằng từ ngữ |
| Dấu gạch dưới | `long_hair`… | Illustrious đọc cả dạng cách (`long hair`) và tốn ít token hơn |
| Negative | trống = gợi ý tối thiểu; > 40 thẻ hoặc > 150 token = ⚠️ | Negative quá dài làm giảm chất lượng theo nhà phát hành |
| Thông số | steps **15–30**, CFG **5–7**, kích thước theo preset, hires strength **0.35–0.5** | Lệch khỏi khuyến nghị sẽ được nhắc |

Số token là **ước lượng heuristic** (mỗi từ ≈ 1 token, cộng thêm cho `_`, số, dấu câu, từ dài) — đủ để cảnh báo vượt khối 75 token, không phải số token chính xác của tokenizer CLIP.

### 1.4. 🧪 Kiểm tra thẻ với kho thẻ
Đối chiếu từng thẻ trong hai ô *Prompt gửi model* và *Negative gửi model* với kho CSV Danbooru + e621 (cùng catalog đã xác minh SHA-256 với tab 🏷️ Kho thẻ). Chỉ đọc và báo cáo, **không sửa gì**. Kết quả chia theo nhóm:

| Nhóm | Ý nghĩa |
| --- | --- |
| ✅ Đúng tên thẻ trong kho | Khớp tên thẻ chính trong CSV — chấp nhận khác dấu cách/viết hoa (`long hair` = `long_hair`); nếu khác tên chuẩn sẽ hiện `→ tên chuẩn` |
| 🔀 Alias của thẻ trong kho | Khớp tên phụ ở cột alias của CSV — hợp lệ, nhưng nên viết tên thẻ chính tiếng Anh |
| ℹ️ Nhãn tiếng Việt | Khớp nhãn/từ đồng nghĩa tiếng Việt của một thẻ — nên đổi sang tên thẻ chính (`tóc dài` → `long_hair`) |
| ℹ️ Thẻ chuẩn ngoài kho | Thẻ chất lượng WAI v17 (`masterpiece, best quality`…), thẻ của 8 bộ negative, thẻ gợi ý sửa vùng, thẻ nhóm Chi tiết mắt & móng, trigger LoRA `perfect eyes`, từ khóa `BREAK`/`AND` — chính Studio đề xuất nên không cần có trong CSV |
| ⚠️ Không có trong kho | Có thể là mô tả tự do (Illustrious vẫn đọc được) hoặc gõ sai chính tả — kèm gợi ý tên thẻ gần nhất (tìm trong kho trước, rồi so chuỗi trong các tên thẻ cùng ký tự đầu) |

Thẻ trùng chỉ được kiểm tra một lần (🩺 Kiểm tra prompt & thông số đã báo riêng). Lần đầu bấm, Studio dựng index tên/alias của ~350k thẻ nên chạy trong hàng đợi và có thể chậm một nhịp; các lần sau chỉ đọc bộ nhớ.

### 1.5. 🛠️ Sửa prompt thành thẻ chuẩn
Nút này biến ô *Prompt gửi model* thành một prompt dễ rà soát hơn bằng cách đối chiếu từng thẻ với cùng kho CSV đã xác minh SHA-256. Đây là công cụ **đề xuất**, không tự gửi ảnh và không thay đổi ô prompt khi bạn mới bấm nút phân tích.

1. Bấm **🛠️ Sửa prompt thành thẻ chuẩn**. Studio tách các thẻ theo dấu phẩy, giữ lại cú pháp trọng số như `(long_hari:1.2)`, rồi phân loại từng thẻ.
2. Alias (`longhair`), nhãn tiếng Việt (`tóc dài`) và lỗi gõ có gợi ý trong kho (`long_hari`) hiện thành dòng thay thế sang tên canonical; các dòng này **được chọn sẵn**.
3. Với thẻ đã đúng tên trong kho, Studio liệt kê tối đa vài thẻ canonical có chung từ (ví dụ `blue_eyes` → `light_blue_eyes`) dưới dạng **thay thế tùy chọn**; các dòng này không được chọn sẵn. Bạn có thể bỏ lựa chọn bắt buộc, chọn phương án khác hoặc đánh dấu thêm phương án tùy chọn.
4. Bấm **✅ Tạo prompt hoàn chỉnh**. Chỉ các dòng đang đánh dấu mới được áp dụng; trọng số/ngoặc được giữ nguyên, mô tả tự do và thẻ không có gợi ý giữ nguyên. Kết quả ghi vào lại ô *Prompt gửi model* để bạn đọc, sửa hoặc xóa trước khi tạo ảnh.

Nếu kho thẻ không tải được, Studio báo lỗi thân thiện và không sửa prompt. Nếu một thẻ mô tả tự do không có trong kho thì không nhất thiết là lỗi — hãy chỉ thay nó khi bạn thực sự muốn dùng tên tag canonical. Khi muốn bắt đầu lại danh sách đề xuất, sửa prompt rồi bấm nút phân tích lần nữa.

---

## 2. Khung prompt theo loại ảnh

Dropdown **Khung prompt theo loại ảnh** dùng cho nút *Sắp xếp prompt theo thứ tự chuẩn*. Bốn khung và thẻ neo (chỉ thêm khi nhóm đang trống):

| Khung | Thẻ neo thêm khi thiếu |
| --- | --- |
| Nhân vật · 1 nhân vật | `masterpiece, best quality, amazing quality` · `1girl, solo` · `anime illustration, cel shading` · `absurdres` |
| Chân dung cận mặt | như trên + `close-up, looking at viewer` · `soft lighting` |
| Phong cảnh · không nhân vật | như trên + `no humans` · `wide shot` · `detailed background, scenery` · `anime background` |
| Hành động / key visual | như trên + `dynamic pose` · `dynamic angle, depth of field` · `dramatic lighting` · `anime key visual, cel shading` |

---

## 3. Thông số khuyến nghị (nhà phát hành WAI v17)

| Thông số | Trong UI/notebook | Khuyến nghị v17 |
| --- | --- | --- |
| Sampler | Euler a (cố định trong mã) | Euler a |
| Steps | 10–45 (mặc định 25) | **15–30** |
| CFG | 1–12 (mặc định 6) | **5–7** |
| Kích thước gốc | preset 512² … 1024×1344 | ≥ 1024×1024, ví dụ 1024×1344 |
| Hires fix | `Tắt` / `1.25×` / `1.5×` / `1.75×` / `2×`, strength 0.2–0.7 (mặc định 0.4) | `1.5`, denoise **0.35–0.5** |
| Prompt dương | do bạn viết | `masterpiece, best quality, amazing quality` (không nhiều hơn) — Studio xếp áp chót, ngay trước `absurdres` |
| Negative | do bạn viết (nút nạp nhanh) | `bad quality, worst quality, worst detail, sketch, censor` |

Studio ghim sampler Euler a. Hires dùng weight chính thức **RealESRGAN_x4plus_anime_6B** (release `v0.2.2.4`) với RRDBNet 6 block, chạy theo tile rồi lấy mẫu xuống tỷ lệ bạn chọn (`1.25×`/`1.5×`/`1.75×`/`2×`); lượt sau WAI img2img tinh chỉnh với cùng prompt/seed/LoRA. Cách này tránh cài `realesrgan`/BasicSR và cây dependency cũ; model chỉ tải ở lần hires/upscale đầu tiên, vào `/content/wai_upscaler_cache`. Kích thước đầu ra được làm tròn xuống bội số 8 và giới hạn ≈4,2 MP. Weight dự kiến 17.938.799 byte; SHA-256 đang ghim `f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da` theo metadata mirror Hugging Face. Real-ESRGAN phát hành theo giấy phép [BSD-3-Clause](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE). GitHub Release API không công bố digest và sandbox không tải được asset GitHub vì lỗi TLS, nên **chưa xác minh độc lập SHA này với file chính thức**. Downloader từ chối mọi file sai kích thước/hash; cần xác nhận lần tải và chất lượng inference trên Colab/GPU thật trước khi coi pipeline đã được kiểm chứng.

---

## 4. Gợi ý phong cách (🎨) và auto-detailer

**🎨 Gợi ý phong cách** chỉ **thêm thẻ mô tả vào hai ô đang hiển thị** (không thay thế prompt, không phải preset phong cách):

- **Màu mắt** (17 lựa chọn) → `blue eyes`, `aqua eyes`, …
- **Kiểu dáng móng tay** (10) → `natural short nails`, `almond nails`, …
- **Màu sơn móng tay** (13) / **Màu sơn móng chân** (10) → `red nails`, `black toenails`, …

**Tự sửa mặt/tay (auto-detailer)** — tùy chọn, cần `ultralytics` (ô 2 cài, chỉ cảnh báo nếu lỗi):

- Mức: `Tắt` / `Mặt` / `Tay` / `Mặt + tay` (mặc định Tắt).
- Tham số: strength **0.2–0.7** (mặc định 0.4), ngưỡng phát hiện **0.1–0.9** (mặc định 0.3, thấp = dễ tìm hơn), số vùng tối đa **1–4** (mặc định 2).
- Weight YOLOv8 tải một lần vào `/content/wai_detailer_cache`: `face_yolov8n.pt` (6.230.011 byte, SHA-256 `70b640f8…`), `hand_yolov8n.pt` (6.237.883 byte, SHA-256 `3991202e…`) từ `Bingsu/adetailer` @ `c310c216`; sai hash thì file bị xóa và không nạp.
- Cách dùng: bật mức, tạo ảnh như bình thường. Studio phát hiện vùng, nới bounding box 25%, phóng vùng nhỏ lên tối thiểu 512 px, inpaint lại đúng vùng rồi dán về với mép mềm 12 px. Đây là gợi ý chỉnh sửa — **không đảm bảo** hết mọi lỗi ngón/mặt.

---

## 5. Semi-auto tag complete · mô phỏng theo Character Select SAA

Chức năng gợi ý tag dưới ô Prompt được xây theo mô hình **Semi-Auto Tag Complete** của
[Character Select SAA](https://github.com/mirabarukaso/character_select_stand_alone_app)
(vốn lấy danh mục từ `DraconicDragon/dbr-e621-lists-archive`, đúng file CSV mà repo này dùng).

| Hành vi trong SAA | Bản dịch trong Studio (`colab/studio.py`) |
| --- | --- |
| Gõ vài ký tự đầu → tag khớp tiền tố | `prompt.input` → `get_keyword_tag_suggestions` tìm trong catalog CSV đã nạp lúc khởi động |
| `*đuôi` / `*giữa*` | `*hair` chỉ trả thẻ kết thúc bằng `hair`; `*hair*` trả thẻ có `hair` ở giữa (khớp cả tên lẫn bí danh) |
| `@` bật chế độ tìm họa sĩ, lọc nhóm 1 và 8 | `@tên` lọc đúng `category` 1 (Danbooru artist) và 8 (e621 artist); **dấu `@` không được chèn vào prompt** vì WAI-Illustrious dùng tên họa sĩ trần, khác Anima |
| Bảng Mark/ID/Category/Group | Mỗi gợi ý có nhãn `[G] [A] [©] [C] [M]` (Danbooru) hoặc `<G> <A> <©> <C> <S> <M> <L>` (e621) từ `PROMPT_TAG_CATEGORY_MARKS` |
| Chuột hoặc ↑↓ + Enter/Tab chọn, Esc đóng | Dropdown Gradio: chọn bằng chuột/phiếm, Enter áp dụng, Esc đóng |
| `ctrl+↑` / `ctrl+↓` chỉnh trọng số tag hiện tại hoặc vùng bôi đen | `demo.load` gắn `PROMPT_TAG_WEIGHT_SHORTCUT_JS` bắt `Ctrl+↑/↓` trong ô prompt và kích hoạt hai nút `+0,1` / `−0,1`; `PROMPT_TAG_WEIGHT_SELECTION_JS` đánh dấu đoạn đang chọn bằng ký tự riêng `U+E000/U+E001` trước khi Python xử lý |
| Logic trọng số giống ComfyUI/WebUI nhưng chi tiết có thể khác | `adjust_prompt_tag_weight` bước `0,1`, kẹp trong `0,1–2,0`, `1,0` thì gỡ `(tag:1.0)`; nếu không có bôi đen thì áp dụng cho cụm cuối prompt. Dấu phẩy **trong** ngoặc không tách cụm (`_prompt_weight_boundary()` và JS cùng quét theo độ sâu ngoặc), nên `(long_hair, blue_eyes:1.1)` được coi là một nhóm và bấm lại sẽ chỉnh đúng trọng số của nhóm |
| File dịch để tìm theo ngôn ngữ khác (`data/danbooru_e621_merged_zh_cn.csv`) | `danbooru_e621_merged_vi_vn.csv` ở thư mục gốc, sinh bằng `scripts/build_vietnamese_translate_file.py`; bỏ qua nhóm họa sĩ/tác phẩm/nhân vật `1/3/4/8/10/11`, các nhóm còn lại có nhãn thật hoặc nhãn dịch máy, cùng ba trường `tag,category,translation`, không header, UTF-8 không BOM, LF, trường dịch không chứa dấu phẩy (SAA `split(',', 3)` sẽ cắt mất phần sau). Test dựng lại tệp và so byte |

Điểm khác biệt có chủ đích: Studio là Gradio trên Colab, không phải Electron, nên thao tác
bàn phím được cài bằng `js` tiền xử lý + một listener gắn lúc tải trang; khi `js` bị chặn
hoặc trình duyệt không hỗ trợ, hai nút `±` vẫn chỉnh được cụm cuối prompt (đường đi Python
thuần). Marker `U+E000/U+E001` chỉ tồn tại trong lần gọi đó và luôn bị loại trước khi ghi lại
vào ô prompt, nên không bao giờ lọt vào prompt gửi model.

File dịch là **lớp từ vựng, không phải dữ liệu sinh ảnh**: nhãn tiếng Việt không bao giờ được
chèn vào prompt. Script dùng `parse_tag_csv()` + `_is_translated_tag_label()` để ưu tiên nhãn thật;
tag chưa có nhãn thật trong các nhóm được phép nhận dịch máy dự phòng, còn họa sĩ/tác phẩm/nhân vật
bị bỏ qua. `--check` (được test gọi) sẽ báo lỗi nếu ai đó sửa từ điển hoặc quy tắc dịch mà quên tạo lại tệp.
Nhãn mới trong từ điển luôn thắng cột chú giải đông lạnh của CSV
(`_prefer_vietnamese_label`), và quy tắc ghép xếp danh từ bổ nghĩa theo trật tự tiếng Việt
(`rabbit_ear_hat` → **Mũ tai thỏ**, nối màu bằng `màu` với nhóm trang phục: `black_bra` →
**Áo ngực màu đen**). Bộ ghép có ba lớp từ vựng (2.964 mục cố định,
783 danh từ chính, 1.521 bổ ngữ, 36 màu) cộng mười bảy quy tắc cụm cho chi tiết nhân vật
(`<động từ>[ <phó từ hướng>]_<tân ngữ>`, `<A>_<giới từ>_<B>`, `<món đồ>_only`,
`<bộ phận>_<hướng>`, `<danh từ>_<trạng thái>`, `see_through_/floating_<x>`, `<bộ phận>less`,
danh từ chính là loài vật hay nội thất, `<món đồ|bộ phận>_<động từ>` (Kéo váy, Liếm đuôi),
nội động từ đứng trước (`melting_tail` → Đuôi đang tan chảy), lượng từ + loại từ
(`three_tails` → Ba cái đuôi, `multiple_arms` → Nhiều cánh tay) và tiền xử lý tên thẻ
(gạch nối `see-through_dress`, hậu tố `(giải nghĩa)` `pearl_(gem)`, số nhiều `curved_horns`
qua `_vi_singular` với danh sách `_TAG_VI_PLURAL_TRAPS` để không nhầm `shorts` → `short`,
khung đồng phục theo tên riêng (`_tag_vi_uniform_frame`: `tokiwadai_school_uniform` → Đồng phục trường
Tokiwadai — chỉ dịch khung, tên riêng được viết hoa qua `_vi_proper_name` và không bịa nghĩa), sở hữu cách
(`fool's_hat` → Mũ của chú hề) và động từ mặc/cởi (`undressing_another` → Đang cởi đồ người khác):
nhãn đã curate luôn được tra TRƯỚC khi chạy quy tắc ghép nên `high-waist_panties` vẫn giữ được bản dịch đã viết tay,
còn quy tắc ghép nhường cho quy tắc cụm khi token cuối là động từ có tân ngữ đứng ngay trước
(`pseudo_skirt_lift` → Nhấc chân váy giả) mà vẫn dịch được danh từ ở cùng vị trí (`nose_piercing` → Khuyên mũi).
Hai cơ chế giữ nhãn ghép tự nhiên: `_tag_vi_quantity` nhận chữ số 1..20 nên `13_hearts` → Mười ba trái tim mà
`69_position` vẫn là Vị trí 69; `_tidy_vietnamese_label` bỏ từ lặp liền nhau do ghép (`soccer_ball` → Quả bóng đá)
nhưng tra ngược chính từ điển để giữ từ láy (`chuồn chuồn`, `lùm lùm`, `chằm chằm`). Test `test_shipped_labels_are_clean`
quét tệp đã dựng và chặn cả hai loại lỗi (lặp từ, mở đầu bằng chữ thường). Đợt 11 thêm hai quy tắc
cấu trúc: `_TAG_VI_TRANSITIVE_RELATIONS` cho `<A>_<động từ>_<B>` (`male_penetrating_female` → Nam thâm nhập nữ,
chỉ chạy khi cả hai vế đã biết) và `_TAG_VI_GARMENT_ACTION_VERBS` để `<món đồ>_<động từ>` nhường cho quy tắc cụm
(`cloak_lift` → Nhấc áo choàng thay vì Nâng lên áo choàng) trong khi bộ phận cơ thể vẫn do compose xử lý
(`butt_grab` → Bóp mông); `_vi_modifier` còn nhận danh từ chính làm bổ ngữ (`penis_size_difference` →
Chênh lệch kích thước dương vật) nhưng WORDS phải tra trước bảng số lượng để `single_leg_armor` không mất chữ "chiếc":
thêm một danh từ chính mở khóa cả họ
thẻ, nên độ phủ tăng từ 1.707 lên **29.619** thẻ mà không phải dịch máy — và tăng có chủ đích theo
hướng ưu tiên thẻ tả nhân vật (mắt, tóc, mặt, tai/đuôi, trang phục, biểu cảm) trước bối cảnh hay metadata.
Từ đợt bố cục/bối cảnh, liên từ `to/at/with` trong cụm tương tác bị lược (`talking_to_viewer` → Nói chuyện
với người xem), và `_tag_vi_uniform_frame` chỉ nhường quy tắc ghép khi phần đầu là **màu** — thêm từ vào
`_TAG_VI_WORDS` (như `dream`, `paradise`) không được biến tên học viện thành mô tả chung. Chuỗi `<x>_shaped_<head>` được ưu tiên
thành “HEAD hình X” (`heart-shaped_pupils` → **Đồng tử hình trái tim**). Test chốt lại ba bất
biến: không trùng khóa trong các dict từ điển, mọi tính từ trong `_TAG_VI_ADJECTIVE_MODIFIERS`
phải có mặt trong `_TAG_VI_WORDS`, và không nhãn nào chứa dấu phẩy/xuống dòng. Tên họa sĩ/nhân vật/tác phẩm bị bỏ qua theo chủ trương của file dịch; các tag còn lại thiếu nhãn thật
được dịch máy dự phòng và dấu `_` trong nhãn được đổi thành khoảng trắng.

## 6. Nguồn

Các khuyến nghị ở mục 3 và bảng negative lấy từ hướng dẫn của nhà phát hành WAI-illustrious (đã đối chiếu công khai khi viết tài liệu này):

- Trang model tại commit được ghim: [LyliaEngine/waiIllustriousSDXL_v170 · README](https://huggingface.co/LyliaEngine/waiIllustriousSDXL_v170/blob/main/README.md) — Steps 15–30, CFG 5–7, Euler a, prompt dương `,masterpiece,best quality,amazing quality,`, negative `bad quality,worst quality,worst detail,sketch,censor,`, cảnh báo không thêm quá nhiều thẻ chất lượng/thẩm mỹ và không viết negative quá dài, 4 nhãn an toàn `general/sensitive/nsfw/explicit`.
- Bản sao nội dung trang model Civitai (version 2883731): [civarchive.com/models/827184](https://civarchive.com/models/827184?modelVersionId=2883731) — cùng nội dung khuyến nghị, kèm mục hires `1.5`, denoise 0.35–0.5.
- Bài tổng hợp thực hành v17: [lilting.ch — WAI-Illustrious v17 hands-on](https://lilting.ch/en/articles/wai-illustrious-v17-review) — so sánh thông số v15–v16 (25–40 steps) với v17 (15–30 steps) và cách dùng cùng negative ngắn.

Phần thứ tự thẻ, khối 75 token và các thẻ Illustrious bám tốt là **thực hành phổ biến của cộng đồng Illustrious** được ghi lại trong mã (`colab/studio.py`, khối *Căn cứ đã đối chiếu*), không phải tuyên bố chính thức của nhà phát hành.

## 7. Chưa được kiểm chứng

Mã và tài liệu này đã qua `python -m unittest discover -s tests -v` (**214 test: 168 đạt, 46 bỏ qua, 0 thất bại**) và `scripts/build_colab_studio.py` tái tạo notebook; các test CPU bao gồm pin/download giả lập, cache/hash, kích thước, output size và nhánh OOM → CPU giả lập, cùng nạp catalog CSV, autocomplete có toán tử `*`/`@`, định dạng file dịch tiếng Việt `danbooru_e621_merged_vi_vn.csv` (kèm vệ sinh nhãn: không lặp từ, viết hoa chữ đầu), trật tự từ trong nhãn ghép, tỷ lệ phủ nhóm chi tiết nhân vật, trật tự lượng từ/động từ, khung đồng phục theo tên riêng, sở hữu cách, cụm tương tác với người xem và vệ sinh từ điển (2.964 mục, không dấu phẩy/không ký tự ngoại lai). Sandbox thiếu PyTorch và `diffusers` nên các test kiến trúc checkpoint và inference bị bỏ qua; Gradio 6.15.2 và Pillow đã cài nên test dựng UI và chạy sự kiện `process_api` đã chạy thật. Chưa xác nhận SHA Anime6B với file chính thức do lỗi TLS khi tải, và **chưa** chạy hires/upscale hoặc đo chất lượng ảnh, tốc độ, VRAM trên GPU Colab. Xem `VERIFICATION.md` để biết đầy đủ giới hạn.
