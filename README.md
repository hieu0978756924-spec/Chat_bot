# Trợ lý đặt bàn nhà hàng (Local LLM qua Ollama)

Chatbot tiếng Việt: tìm quán, xem thực đơn, kiểm tra bàn trống, đặt / đổi / hủy bàn, trả lời chính sách (RAG) và tiếp nhận khiếu nại.
Chỉ dùng **LLM chạy cục bộ qua Ollama**, không gọi API LLM bên ngoài.

## 1. Kiến trúc

Mỗi tin nhắn đi qua **một pipeline duy nhất**:

```
Tin nhắn
 → Normalizer      core/normalizer.py   thêm dấu câu không dấu, viết tắt, "sáu người"→6, "7h tối"→19:00, "tối mai"→ngày
 → NLU             core/nlu.py          Ollama (format = JSON schema, temperature 0.1) + validator Pydantic + retry 1 lần
                   core/fallback_classifier.py   dự phòng k-NN embedding khi Ollama lỗi/timeout (log LLM_UNAVAILABLE)
 → State Manager   core/state.py        merge slot set/add/remove, đổi tiêu chí, "quán thứ 2", task_stack
 → Dialogue Policy core/policy.py       slot bắt buộc, tự kiểm tra bàn trống, xác nhận trước khi ghi DB
 → Tool Executor   agent/tools.py       10 tool có kiểm tra tham số, trả {ok, data, error_code, message}
                   actions/db.py        SQLite, truy vấn có tham số, transaction BEGIN IMMEDIATE
                   actions/rag.py       FAISS + Sentence Transformers cho chính sách
 → Response Gen    core/response_gen.py template Python, chỉ dùng dữ liệu tool trả về
```

```
chatbot/
├── run.bat                 khởi động 1 chạm (CRLF + ASCII — đừng lưu lại bằng LF/UTF-8)
├── main.py                 python main.py [--debug]
├── config.py               cấu hình tập trung (đọc .env)
├── .env / .env.example     model, timeout, đường dẫn, ngưỡng RAG
├── agent/   agent.py (điều phối pipeline, SessionStore), cli.py, tools.py
├── core/    normalizer, nlu, fallback_classifier, schema, state, policy, response_gen, entities, embeddings, text
├── actions/ db.py, rag.py, llm.py (trạng thái Ollama)
├── data/nlu/intent_examples.yaml    câu mẫu cho NLU dự phòng (không trùng câu test)
├── docs/policies/*.md      tài liệu chính sách cho RAG
├── dataset/*.csv           dữ liệu gốc (nhà hàng, món, đơn mẫu, chính sách)
├── data_store/             app.db (SQLite), faiss_policy.index (+ _meta.json)
├── scripts/init_db.py      tạo/reset DB (tự sao lưu), build FAISS index
├── tests/                  pytest: unit + kịch bản hội thoại YAML
└── logs/chatbot.log        log từng lượt (JSON trace)
```

## 2. Cài đặt

```bat
cd C:\Chat_bot\chatbot
python -m venv venv
venv\Scripts\pip install -r requirements.txt
ollama pull qwen2.5:3b
venv\Scripts\python scripts\init_db.py --index
```

### Chọn model (`OLLAMA_MODEL` trong `.env`)

| Máy | Model | Ghi chú đo thực tế |
|---|---|---|
| 8 GB RAM, chỉ CPU | `qwen2.5:3b` (mặc định) | ~13–20 s/lượt (i5-1155G7); lượt đầu thêm ~20 s nạp model |
| ≥16 GB RAM hoặc GPU | `qwen2.5:7b` | chính xác hơn; trên máy 8 GB bị swap, không dùng được |
| Không có Ollama | `LLM_PROVIDER=none` | chỉ NLU dự phòng (embedding), kém chính xác hơn |

## 3. Chạy

- Nhấp đúp `C:\Chat_bot\run.bat` (hoặc `chatbot\run.bat`). Script kiểm tra Python/venv, Ollama, model (hỏi trước khi `ollama pull`), DB và FAISS index.
- Hoặc: `venv\Scripts\python main.py`
- Lệnh trong chat: `/moi` hội thoại mới · `/state` xem trạng thái · `/debug` bật/tắt debug · `/thoat`.

### Chế độ debug
`run.bat --debug` hoặc `python main.py --debug`: mỗi lượt in câu đã chuẩn hóa, nguồn NLU (llm / llm_retry / fallback_embedding),
thời gian LLM, intents, slot_updates, slot bị loại (không có căn cứ), state sau merge, tool + tham số + kết quả, các block phản hồi.
Mọi lượt cũng được ghi vào `logs/chatbot.log`.

## 4. Kiểm thử

```bat
venv\Scripts\python -m pytest tests -q                       :: không cần Ollama (NLU dự phòng)
set RUN_OLLAMA_TESTS=1 && venv\Scripts\python -m pytest tests -q   :: thêm kịch bản chạy với Ollama thật
```
Cấu trúc:

| Thành phần | Nội dung |
|---|---|
| `tests/test_normalizer.py`, `test_state.py`, `test_policy.py`, `test_tools_db.py` | unit test (không cần Ollama): chuẩn hóa, merge state, policy, tool + transaction, RAG |
| `tests/test_failures.py` | lỗi DB (mock, 8+8 câu): không bao giờ báo thành công; lỗi Ollama (mock, 8 câu × kết nối/timeout + JSON sai): dùng dự phòng, báo người dùng, log `LLM_UNAVAILABLE` / `LLM_INVALID_OUTPUT`; kiểm tra tập test không trùng few-shot/câu mẫu |
| `tests/conversations/*.yaml` | kịch bản nhiều lượt; mỗi lượt kiểm tra intent, slot, tool + tham số, block và chống bịa. `regression_transcript.yaml` = transcript lỗi gốc |
| `tests/eval/nlu_dev.yaml` | 17 nhóm × 8 cách diễn đạt (có dấu, không dấu, sai chính tả, ngắn, dài) — dùng để soát lỗi |
| `tests/eval/nlu_heldout.yaml` | tập **held-out** (12 câu gốc trong `dataset/y_dinh_va_cau_hoi_mau.csv` + câu mới) — chỉ để đo, không dùng chỉnh prompt |
| `tests/eval/run_eval.py` | chạy đánh giá và ghi `tests/report.md` |

```bat
venv\Scripts\python tests\eval\run_eval.py --mode fallback            :: nhanh, không cần Ollama
venv\Scripts\python tests\eval\run_eval.py --mode both --report       :: Ollama + dự phòng, ghi tests\report.md (chậm trên CPU)
```
Báo cáo gồm: độ chính xác intent / slot, tỷ lệ gọi tool đúng (kịch bản), số ca bịa thông tin (menu mặc định,
báo thành công khi chưa commit, sai tình trạng bàn, quán không có trong DB), độ trễ, danh sách ca sai kèm lý do.
Test dùng DB tạm, không đụng `data_store/app.db`.

## 5. Dữ liệu nhà hàng

| Nguồn | Mã | Nội dung | Đặt bàn |
|---|---|---|---|
| `dataset/nha_hang.csv` | NH001–NH014 | Quán **demo** (hư cấu) ở TP.HCM và Quy Nhơn, đủ giá / giờ / thực đơn | Có (lịch bàn sinh tự động) |
| `dataset/nha_hang_quy_nhon.csv` | QN001–QN040 (mã gốc NH001–NH040 lưu ở cột `ma_nguon`) | 40 nhà hàng **thật** ở Quy Nhơn, trạng thái *chưa xác minh* | Không — chỉ tra cứu |

- Trường trống trong dataset Quy Nhơn (giá, giờ mở cửa, đánh giá, SĐT) được giữ trống; chatbot trả lời "chưa có thông tin" kèm nguồn tham khảo, không tự điền.
- Quán chưa liên kết đặt bàn: hỏi bàn trống / đặt bàn sẽ được trả lời thật là chưa liên kết, không sinh lịch bàn giả.
- Khi có thông tin xác minh (giờ mở cửa, sức chứa), thêm vào CSV và đặt `trang_thai_hoat_dong = Đã xác minh` để mở chức năng đặt bàn cho quán đó.
- Tìm theo phong cách ẩm thực: Âu, Á, Nhật, Hàn, Ý, Thái, Trung Hoa, Steak, Pizza, Lẩu, Nướng, Hải sản, Chay, Món Việt, Cà phê; tìm theo món tự do ("bún cá", "mì cay") trong tên / loại món của quán và thực đơn.
