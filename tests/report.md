# Báo cáo kiểm thử chatbot đặt bàn

- Thời điểm chạy: 2026-11-18 01:19
- Model LLM: `qwen2.5:3b` (Ollama, CPU) · Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Chế độ `llm` = Ollama + validator (dự phòng embedding nếu lỗi); `fallback` = chỉ NLU dự phòng.
- Tập **held-out** không được dùng để chỉnh prompt/few-shot/câu mẫu/luật; có test tự động kiểm tra không trùng câu (`test_fewshot_and_examples_disjoint_from_test_sets`).

## Tổng hợp

| Chế độ | Tập | Số ca | Intent | Slot | Ca đạt hoàn toàn | Bịa thông tin | Menu mặc định | Báo thành công sai | LLM lỗi | Trễ TB / p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| fallback | dev | 136 | **75.0%** | 97.3% (149) | 74.3% | 0 | 0 | 0 | 0 | 0.0s / 0.0s |
| fallback | heldout | 97 | **76.3%** | 96.2% (105) | 74.2% | 0 | 0 | 0 | 0 | 0.0s / 0.0s |
| llm | dev | 136 | **89.0%** | 98.0% (149) | 87.5% | 0 | 0 | 0 | 2 | 10.1s / 17.0s |
| llm | heldout | 97 | **93.8%** | 97.1% (105) | 90.7% | 0 | 0 | 0 | 3 | 10.3s / 16.0s |

## Kịch bản nhiều lượt (tỷ lệ gọi tool đúng)

| Chế độ | Kịch bản đạt | Lượt | Kỳ vọng tool đạt |
|---|---|---|---|
| fallback | 7/7 | 22 | 16/16 (100.0%) |
| llm | 7/7 | 22 | 16/16 (100.0%) |

## Mục tiêu

- [fallback] Intent held-out ≥ 90%: **CHƯA ĐẠT** (76.3%)
- [fallback] 0 ca báo đặt bàn thành công khi chưa commit: **ĐẠT** (0)
- [fallback] 0 ca trả menu mặc định cho câu có nghĩa: **ĐẠT** (0)
- [llm] Intent held-out ≥ 90%: **ĐẠT** (93.8%)
- [llm] 0 ca báo đặt bàn thành công khi chưa commit: **ĐẠT** (0)
- [llm] 0 ca trả menu mặc định cho câu có nghĩa: **ĐẠT** (0)

## Độ chính xác intent theo nhóm — fallback

| Nhóm | dev | heldout |
|---|---|---|
| bo_sung | 8/8 | 4/5 |
| chinh_sach | 4/8 | 2/5 |
| csv_goc | — | 11/12 |
| dat_ban | 5/8 | 4/5 |
| doi_chu_de_quay_lai | 4/8 | 5/5 |
| dong_y_xac_nhan | 7/8 | 4/5 |
| huy_don | 6/8 | 5/5 |
| khieu_nai | 5/8 | 3/5 |
| loi_app | 6/8 | 4/5 |
| ngoai_pham_vi | 5/8 | 3/5 |
| nhieu_dieu_kien | 7/8 | 5/5 |
| sua_don | 8/8 | 4/5 |
| thay_doi | 6/8 | 2/5 |
| thieu_thong_tin | 6/8 | 3/5 |
| thuc_don | 5/8 | 1/5 |
| tim_quan | 6/8 | 4/5 |
| tu_choi_xac_nhan | 7/8 | 5/5 |
| xoa_dieu_kien | 7/8 | 5/5 |

## Độ chính xác intent theo nhóm — llm

| Nhóm | dev | heldout |
|---|---|---|
| bo_sung | 8/8 | 5/5 |
| chinh_sach | 6/8 | 4/5 |
| csv_goc | — | 10/12 |
| dat_ban | 7/8 | 5/5 |
| doi_chu_de_quay_lai | 6/8 | 3/5 |
| dong_y_xac_nhan | 7/8 | 5/5 |
| huy_don | 8/8 | 5/5 |
| khieu_nai | 6/8 | 5/5 |
| loi_app | 7/8 | 4/5 |
| ngoai_pham_vi | 6/8 | 5/5 |
| nhieu_dieu_kien | 8/8 | 5/5 |
| sua_don | 8/8 | 5/5 |
| thay_doi | 8/8 | 5/5 |
| thieu_thong_tin | 7/8 | 5/5 |
| thuc_don | 5/8 | 5/5 |
| tim_quan | 8/8 | 5/5 |
| tu_choi_xac_nhan | 8/8 | 5/5 |
| xoa_dieu_kien | 8/8 | 5/5 |

## Danh sách ca sai — fallback

- **[dev/tim_quan]** `Tìm giúp mình quán lẩu ngon với` → intent ['create_reservation'] (nguồn fallback_embedding): intent ['create_reservation'] ≠ ['search_restaurant']
- **[dev/tim_quan]** `co quan sushi nao o quan 1 ko` → intent ['search_menu'] (nguồn fallback_embedding): intent ['search_menu'] ≠ ['search_restaurant']
- **[dev/thuc_don]** `mon nao duoi 100k` → intent ['search_restaurant'] (nguồn fallback_embedding): intent ['search_restaurant'] ≠ ['search_menu']
- **[dev/thuc_don]** `thực đơn` → intent ['confirm'] (nguồn fallback_embedding): intent ['confirm'] ≠ ['search_menu']
- **[dev/thuc_don]** `món tráng miệng có gì` → intent ['policy_question'] (nguồn fallback_embedding): intent ['policy_question'] ≠ ['search_menu']
- **[dev/nhieu_dieu_kien]** `kiếm chỗ ăn chay rẻ rẻ cho 3 người tối mai` → intent ['create_reservation'] (nguồn fallback_embedding): intent ['create_reservation'] ≠ ['search_restaurant', 'check_availability']
- **[dev/thieu_thong_tin]** `Còn bàn không em?` → intent ['create_reservation'] (nguồn fallback_embedding): intent ['create_reservation'] ≠ ['check_availability', 'search_restaurant']
- **[dev/thieu_thong_tin]** `hủy đơn giúp tôi` → intent ['chitchat'] (nguồn fallback_embedding): intent ['chitchat'] ≠ ['cancel_reservation']; block ['chitchat'] thiếu ['ask_slots']
- **[dev/thay_doi]** `đổi sang 8` → intent ['confirm'] (nguồn fallback_embedding): intent ['confirm'] ≠ ['search_restaurant', 'check_availability', 'update_reservation']
- **[dev/thay_doi]** `cho mình lùi lại 30 phút, 7 rưỡi` → intent ['update_reservation'] (nguồn fallback_embedding): intent ['update_reservation'] ≠ ['search_restaurant', 'check_availability']
- **[dev/xoa_dieu_kien]** `bo dieu kien gia di` → intent ['chitchat'] (nguồn fallback_embedding): intent ['chitchat'] ≠ ['search_restaurant']; slot budget_max chưa bị xóa (=200000)
- **[dev/xoa_dieu_kien]** `không cần rẻ đâu, quán ngon là được` → intent ['search_restaurant'] (nguồn fallback_embedding): slot budget_max chưa bị xóa (=150000)
- **[dev/doi_chu_de_quay_lai]** `À mà quán có chỗ đậu ô tô không em?` → intent ['check_availability'] (nguồn fallback_embedding): intent ['check_availability'] ≠ ['policy_question', 'restaurant_details']
- **[dev/doi_chu_de_quay_lai]** `be 5 tuoi di an co tinh tien khong` → intent ['confirm'] (nguồn fallback_embedding): intent ['confirm'] ≠ ['policy_question']
- **[dev/doi_chu_de_quay_lai]** `Khoan, cho mình hỏi đặt cọc bao nhiêu đã` → intent ['search_restaurant'] (nguồn fallback_embedding): intent ['search_restaurant'] ≠ ['policy_question']
- **[dev/doi_chu_de_quay_lai]** `uh quay lại chuyện đặt bàn lúc nãy` → intent ['cancel_reservation'] (nguồn fallback_embedding): intent ['cancel_reservation'] ≠ ['confirm', 'search_restaurant', 'create_reservation']
- **[dev/dat_ban]** `Chốt Sushi Sora giúp mình` → intent ['search_restaurant'] (nguồn fallback_embedding): intent ['search_restaurant'] ≠ ['create_reservation']
- **[dev/dat_ban]** `Mình chọn Chay An Nhiên, đặt luôn đi em` → intent ['search_restaurant'] (nguồn fallback_embedding): intent ['search_restaurant'] ≠ ['create_reservation']
- **[dev/dat_ban]** `giu cho e 1 ban o Hai San Bien Xanh 6h toi mai 2 nguoi` → intent ['search_restaurant'] (nguồn fallback_embedding): intent ['search_restaurant'] ≠ ['create_reservation']
- **[dev/huy_don]** `Hủy đơn DB005 giúp mình` → intent ['update_reservation'] (nguồn fallback_embedding): intent ['update_reservation'] ≠ ['cancel_reservation']; block ['booking_info', 'clarify'] thiếu ['confirm']
- **[dev/huy_don]** `bỏ đặt chỗ DB005` → intent ['update_reservation'] (nguồn fallback_embedding): intent ['update_reservation'] ≠ ['cancel_reservation']
- **[dev/tu_choi_xac_nhan]** `để mình bàn với gia đình đã, chưa đặt vội` → intent ['check_availability'] (nguồn fallback_embedding): intent ['check_availability'] ≠ ['deny']; confirmation=None ≠ 'no'
- **[dev/dong_y_xac_nhan]** `Ừ hủy đi` → intent ['deny'] (nguồn fallback_embedding): intent ['deny'] ≠ ['confirm', 'cancel_reservation']; confirmation='no' ≠ 'yes'; block ['pending_cancelled'] thiếu ['action_done']
- **[dev/chinh_sach]** `phòng VIP có phụ thu không` → intent ['check_availability'] (nguồn fallback_embedding): intent ['check_availability'] ≠ ['policy_question']
- **[dev/chinh_sach]** `co xuat hoa don VAT cho cong ty duoc ko` → intent ['complaint'] (nguồn fallback_embedding): intent ['complaint'] ≠ ['policy_question']
- **[dev/chinh_sach]** `giữ xe ô tô mất bao nhiêu tiền` → intent ['complaint'] (nguồn fallback_embedding): intent ['complaint'] ≠ ['policy_question']
- **[dev/chinh_sach]** `Đi đoàn 20 người thì đặt thế nào` → intent ['update_reservation'] (nguồn fallback_embedding): intent ['update_reservation'] ≠ ['policy_question']
- **[dev/khieu_nai]** `phuc vu cham qua, doi 40 phut chua co mon` → intent ['policy_question'] (nguồn fallback_embedding): intent ['policy_question'] ≠ ['complaint']
- **[dev/khieu_nai]** `toi muon gap quan ly de phan anh` → intent ['search_restaurant'] (nguồn fallback_embedding): intent ['search_restaurant'] ≠ ['complaint']
- **[dev/khieu_nai]** `Đặt bàn rồi mà tới nơi phải đứng chờ nửa tiếng không có chỗ` → intent ['policy_question'] (nguồn fallback_embedding): intent ['policy_question'] ≠ ['complaint']
- **[dev/loi_app]** `da tru tien coc ma app bao that bai` → intent ['greeting'] (nguồn fallback_embedding): intent ['greeting'] ≠ ['app_issue', 'complaint']
- **[dev/loi_app]** `Không thấy thông báo xác nhận đặt bàn trên điện thoại` → intent ['restaurant_details'] (nguồn fallback_embedding): intent ['restaurant_details'] ≠ ['app_issue']
- **[dev/ngoai_pham_vi]** `viet giup minh bai van ta me` → intent ['confirm'] (nguồn fallback_embedding): intent ['confirm'] ≠ ['out_of_scope']
- **[dev/ngoai_pham_vi]** `1 + 1 bằng mấy` → intent ['confirm'] (nguồn fallback_embedding): intent ['confirm'] ≠ ['out_of_scope']
- **[dev/ngoai_pham_vi]** `dạy mình nấu phở bò với` → intent ['search_restaurant'] (nguồn fallback_embedding): intent ['search_restaurant'] ≠ ['out_of_scope']
- **[heldout/csv_goc]** `Đặt 4 người giờ tăng lên 6 người được không?` → intent ['update_reservation'] (nguồn fallback_embedding): slot party_size=4 ≠ 6
- **[heldout/csv_goc]** `Nhà hàng này được đánh giá thế nào?` → intent ['search_restaurant'] (nguồn fallback_embedding): intent ['search_restaurant'] ≠ ['restaurant_details']
- **[heldout/tim_quan]** `Quận 3 có quán cơm Việt nào ngon không bạn` → intent ['search_restaurant'] (nguồn fallback_embedding): slot cuisine=None ≠ 'Món Việt'
- **[heldout/tim_quan]** `muon an do han quoc` → intent ['confirm'] (nguồn fallback_embedding): intent ['confirm'] ≠ ['search_restaurant']
- **[heldout/thuc_don]** `quán có bán mì ý không, giá sao` → intent ['policy_question'] (nguồn fallback_embedding): intent ['policy_question'] ≠ ['search_menu']
- **[heldout/thuc_don]** `Liệt kê giúp mình các món chay dưới 150k` → intent ['policy_question'] (nguồn fallback_embedding): intent ['policy_question'] ≠ ['search_menu']
- **[heldout/thuc_don]** `mon an kem cua quan co gi` → intent ['policy_question'] (nguồn fallback_embedding): intent ['policy_question'] ≠ ['search_menu']
- **[heldout/thuc_don]** `đồ uống có những gì` → intent ['policy_question'] (nguồn fallback_embedding): intent ['policy_question'] ≠ ['search_menu']
- **[heldout/thieu_thong_tin]** `kiem tra ban trong giup minh` → intent ['cancel_reservation'] (nguồn fallback_embedding): intent ['cancel_reservation'] ≠ ['check_availability', 'search_restaurant']
- **[heldout/thieu_thong_tin]** `Tra cứu đơn đặt bàn` → intent ['policy_question'] (nguồn fallback_embedding): intent ['policy_question'] ≠ ['get_reservation']; block ['policy_answer'] thiếu ['ask_slots']
- **[heldout/bo_sung]** `tầm 250 nghìn đổ lại` → intent ['search_menu'] (nguồn fallback_embedding): intent ['search_menu'] ≠ ['search_restaurant']
- **[heldout/thay_doi]** `doi qua 6h chieu` → intent ['update_reservation'] (nguồn fallback_embedding): intent ['update_reservation'] ≠ ['search_restaurant', 'check_availability']
- **[heldout/thay_doi]** `tăng thêm 3 người` → intent ['update_reservation'] (nguồn fallback_embedding): intent ['update_reservation'] ≠ ['search_restaurant', 'check_availability']
- **[heldout/thay_doi]** `đổi qua 6` → intent ['confirm'] (nguồn fallback_embedding): intent ['confirm'] ≠ ['search_restaurant', 'check_availability', 'update_reservation']
- **[heldout/dat_ban]** `dat Bien Xanh nhe` → intent ['chitchat'] (nguồn fallback_embedding): intent ['chitchat'] ≠ ['create_reservation']; slot restaurant_id=None ≠ 'NH007'
- **[heldout/sua_don]** `Đơn DB003 cho mình đổi qua 13 giờ` → intent ['get_reservation'] (nguồn fallback_embedding): intent ['get_reservation'] ≠ ['update_reservation']
- **[heldout/dong_y_xac_nhan]** `Đồng ý hủy` → intent ['deny'] (nguồn fallback_embedding): intent ['deny'] ≠ ['confirm', 'cancel_reservation']; confirmation='no' ≠ 'yes'
- **[heldout/chinh_sach]** `Đổi giờ đặt bàn có mất phí gì không?` → intent ['update_reservation'] (nguồn fallback_embedding): intent ['update_reservation'] ≠ ['policy_question']
- **[heldout/chinh_sach]** `Mình bị dị ứng hải sản thì quán có lưu ý gì không?` → intent ['complaint'] (nguồn fallback_embedding): intent ['complaint'] ≠ ['policy_question']
- **[heldout/chinh_sach]** `điểm thưởng thành viên dùng thế nào` → intent ['update_reservation'] (nguồn fallback_embedding): intent ['update_reservation'] ≠ ['policy_question']
- **[heldout/khieu_nai]** `Quán phục vụ quá tệ, tôi cần phản ánh` → intent ['restaurant_details'] (nguồn fallback_embedding): intent ['restaurant_details'] ≠ ['complaint']
- **[heldout/khieu_nai]** `Tới quán đúng giờ mà không có bàn, bực mình thật` → intent ['update_reservation'] (nguồn fallback_embedding): intent ['update_reservation'] ≠ ['complaint']
- **[heldout/loi_app]** `Trang đặt bàn bị trắng xóa` → intent ['cancel_reservation'] (nguồn fallback_embedding): intent ['cancel_reservation'] ≠ ['app_issue']
- **[heldout/ngoai_pham_vi]** `giai phuong trinh bac 2 giup minh` → intent ['greeting'] (nguồn fallback_embedding): intent ['greeting'] ≠ ['out_of_scope']
- **[heldout/ngoai_pham_vi]** `Mua iPhone ở đâu rẻ?` → intent ['search_menu'] (nguồn fallback_embedding): intent ['search_menu'] ≠ ['out_of_scope']

## Danh sách ca sai — llm

- **[dev/thuc_don]** `thực đơn` → intent ['search_restaurant'] (nguồn llm): intent ['search_restaurant'] ≠ ['search_menu']
- **[dev/thuc_don]** `Mẹ mình ăn chay trường, quán có món nào hợp với mẹ không, giá tầm dưới 80 nghìn thôi` → intent ['search_restaurant'] (nguồn llm_retry): intent ['search_restaurant'] ≠ ['search_menu']
- **[dev/thuc_don]** `món tráng miệng có gì` → intent ['policy_question'] (nguồn fallback_embedding): intent ['policy_question'] ≠ ['search_menu']
- **[dev/thieu_thong_tin]** `doi gio dat ban` → intent ['create_reservation'] (nguồn llm): intent ['create_reservation'] ≠ ['update_reservation']
- **[dev/xoa_dieu_kien]** `bo dieu kien gia di` → intent ['search_restaurant', 'policy_question'] (nguồn llm): slot budget_max chưa bị xóa (=200000)
- **[dev/xoa_dieu_kien]** `không cần rẻ đâu, quán ngon là được` → intent ['search_restaurant'] (nguồn llm): slot budget_max chưa bị xóa (=150000)
- **[dev/doi_chu_de_quay_lai]** `À mà quán có chỗ đậu ô tô không em?` → intent ['search_restaurant'] (nguồn llm): intent ['search_restaurant'] ≠ ['policy_question', 'restaurant_details']
- **[dev/doi_chu_de_quay_lai]** `Khoan, cho mình hỏi đặt cọc bao nhiêu đã` → intent ['check_availability'] (nguồn llm): intent ['check_availability'] ≠ ['policy_question']
- **[dev/dat_ban]** `Chốt Sushi Sora giúp mình` → intent ['confirm'] (nguồn llm): intent ['confirm'] ≠ ['create_reservation']
- **[dev/dong_y_xac_nhan]** `Ừ hủy đi` → intent ['deny'] (nguồn llm): intent ['deny'] ≠ ['confirm', 'cancel_reservation']; confirmation='no' ≠ 'yes'; block ['pending_cancelled'] thiếu ['action_done']
- **[dev/chinh_sach]** `Hủy bàn trước bao lâu thì không mất phí?` → intent ['cancel_reservation'] (nguồn llm): intent ['cancel_reservation'] ≠ ['policy_question']; block ['ask_slots'] thiếu ['policy_answer']
- **[dev/chinh_sach]** `Đi đoàn 20 người thì đặt thế nào` → intent ['create_reservation'] (nguồn llm): intent ['create_reservation'] ≠ ['policy_question']
- **[dev/khieu_nai]** `phuc vu cham qua, doi 40 phut chua co mon` → intent ['check_availability'] (nguồn llm): intent ['check_availability'] ≠ ['complaint']
- **[dev/khieu_nai]** `quán tính sai tiền` → intent ['app_issue'] (nguồn llm): intent ['app_issue'] ≠ ['complaint']
- **[dev/loi_app]** `Không thấy thông báo xác nhận đặt bàn trên điện thoại` → intent ['out_of_scope'] (nguồn llm): intent ['out_of_scope'] ≠ ['app_issue']
- **[dev/ngoai_pham_vi]** `viet giup minh bai van ta me` → intent ['chitchat'] (nguồn llm): intent ['chitchat'] ≠ ['out_of_scope']
- **[dev/ngoai_pham_vi]** `Đặt vé máy bay đi Hà Nội giúp mình` → intent ['app_issue'] (nguồn llm): intent ['app_issue'] ≠ ['out_of_scope']
- **[heldout/csv_goc]** `Tôi sẽ đến trễ 20 phút` → intent ['check_availability'] (nguồn llm_retry): intent ['check_availability'] ≠ ['policy_question', 'update_reservation']
- **[heldout/csv_goc]** `Đặt 4 người giờ tăng lên 6 người được không?` → intent ['create_reservation'] (nguồn llm): slot party_size=4 ≠ 6
- **[heldout/csv_goc]** `Tôi muốn nói chuyện với nhân viên` → intent ['chitchat'] (nguồn llm): intent ['chitchat'] ≠ ['complaint']
- **[heldout/doi_chu_de_quay_lai]** `Cho hỏi chút, quán có cho mang bánh sinh nhật vào không?` → intent ['search_restaurant'] (nguồn llm): intent ['search_restaurant'] ≠ ['policy_question']
- **[heldout/doi_chu_de_quay_lai]** `huy ban thi co mat coc khong` → intent ['cancel_reservation', 'policy_question'] (nguồn llm): intent ['cancel_reservation', 'policy_question'] ≠ ['policy_question']
- **[heldout/dat_ban]** `Lấy quán số 1 giúp mình` → intent ['create_reservation'] (nguồn llm): slot restaurant_id=None ≠ 'NH003'
- **[heldout/dat_ban]** `Mình đặt quán thứ hai` → intent ['create_reservation'] (nguồn llm): slot restaurant_id=None ≠ 'NH005'
- **[heldout/chinh_sach]** `điểm thưởng thành viên dùng thế nào` → intent ['out_of_scope'] (nguồn llm): intent ['out_of_scope'] ≠ ['policy_question']
- **[heldout/loi_app]** `Mình không nhận được tin nhắn xác nhận đặt bàn` → intent ['complaint'] (nguồn llm): intent ['complaint'] ≠ ['app_issue']
