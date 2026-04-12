# Kế hoạch: căn chỉnh toàn hệ Cooperative Goal Search (paper + Webots)

**Trạng thái:** bản kế hoạch để triển khai sau (chưa chỉnh code). Người dùng đang ở chế độ Plan để xây dựng kế hoạch, không thực thi trong phiên này.

## Mục tiêu

Một pipeline thống nhất: môi trường Python training = mô tả paper (Fig. 8) = world Webots; đường export không dao động; đường cong học phản ánh đúng tinh thần paper (proposed tốt nhất, R2 nhanh, R1/R3 hưởng lợi sharing).

---

## 1. Thiết kế môi trường (ưu tiên cao)

**Kích thước:** 300×300.

**Vật cản (hình chữ nhật, va chạm không xuyên tường):**

| Thành phần | Mô tả gần đúng |
|------------|----------------|
| Tường trái dọc | x ≈ 100, y từ ≈ 35 đến ≈ 265 |
| Tường phải dưới | x ≈ 200, y từ ≈ 0 đến ≈ 135 |
| Tường phải trên | x ≈ 200, y từ ≈ 205 đến ≈ 275 |
| Chướng ngại giữa | quanh (150, 175) |

**Đại diện trong code:** dùng hình chữ nhật axis-aligned với độ dày tường > 0 (ví dụ 5 đơn vị) để tránh lỗi số.

**Goal:** quanh (150, 275) — vùng đích dạng hình chữ nhật nhỏ (ví dụ 10×10).

**Xuất phát cố định (căn Webots):**

- R1: (25, 25)
- R2: (150, 25)
- R3: (275, 25)

**Chuyển động:** bốn hành động rời rạc (up / down / left / right), bước cố định 5 đơn vị; tùy chọn nhiễu Gauss (paper: σ=0.5) — **cần quyết định một nguồn:** hoặc nhiễu bật để khớp paper, hoặc nhiễu tắt để khớp replay Webots tuyệt đối.

**Episode:** va chạm **không** kết thúc episode; chỉ kết thúc khi đạt goal **hoặc** đủ `max_steps` (2000).

**File đích:** [`src/config.py`](d:\CCU_MasterResearch\cooperative_goal_search\src\config.py), [`src/env.py`](d:\CCU_MasterResearch\cooperative_goal_search\src\env.py).

---

## 2. Logic huấn luyện và baseline

- **Dyna-Q + cluster model:** giữ cấu trúc hiện có; kiểm tra cập nhật cụm incremental: `mean += (x - mean) / (n+1)` (đã đúng trong `Cluster.update`).

- **Bốn method:**
  - `dyna_no_sharing`: không chia sẻ.
  - `raw_sharing`: baseline Tan — **làm rõ trong code:** chỉ nhân bản Q-update từ kinh nghiệm người khác, **không** nhân đôi toàn bộ `model.update` (hoặc tương đương công bằng với paper) để baseline không quá mạnh so với proposed.
  - `request_sharing`: khi visit thấp, **append** cluster từ teammate (không fusion T).
  - `proposed`: khi visit thấp, merge cluster bằng **T-statistic** + weighted fusion — **không** chỉ copy danh sách cluster.

- **Planning:** giữ `planning_replay` + điều kiện `visit_threshold` cho request/proposed.

- **Siêu tham số:** `trials` mặc định 40 (paper); tinh chỉnh `share_t_threshold`, `visit_threshold`, `cluster_distance_threshold` sau khi có log.

**File đích:** [`src/agent.py`](d:\CCU_MasterResearch\cooperative_goal_search\src\agent.py), [`src/training.py`](d:\CCU_MasterResearch\cooperative_goal_search\src\training.py).

---

## 3. Export đường đi Webots (bug nghiêm trọng)

- **Không** export greedy rollout ngẫu nhiên sau train (dễ dao động 2-state).
- **Chỉ** export đường của episode **thành công** có **số bước tối thiểu** (best successful path), lưu trajectory theo thời gian.
- Nếu không có episode thành công: fallback có log cảnh báo (ví dụ greedy có kiểm tra chu kỳ ngắn).

**File đích:** [`src/training.py`](d:\CCU_MasterResearch\cooperative_goal_search\src\training.py), [`train_compare.py`](d:\CCU_MasterResearch\cooperative_goal_search\train_compare.py).

---

## 4. Căn chỉnh Webots

**Ánh xạ tọa độ (theo spec user):**

- `X_webots = x * 0.01 - 1.5`
- `Z_webots = y * 0.01 - 1.5`

**World (.wbt):** cùng layout vật cản / goal / vị trí robot với Python (quy đổi translation và kích thước box = độ dài paper × 0.01).

**Camera:** góc nhìn từ trên hoặc nghiêng rõ sàn; **DirectionalLight** (+ tùy chọn ambient/PointLight nếu còn tối).

**Controller:** dùng đúng công thức trên; waypoint lấy từ `webots_paths.json`; có thể tăng `TARGET_TOLERANCE` nhẹ nếu waypoint dày.

**File đích:** [`webots/worlds/cooperative_goal_search.wbt`](d:\CCU_MasterResearch\cooperative_goal_search\webots\worlds\cooperative_goal_search.wbt), [`webots/controllers/cooperative_goal_search/cooperative_goal_search.py`](d:\CCU_MasterResearch\cooperative_goal_search\webots\controllers\cooperative_goal_search\cooperative_goal_search.py).

---

## 5. Đầu ra mong đợi sau khi triển khai

1. `env.py` + `config.py` sạch, khớp Fig. 8 và điều kiện episode.
2. `agent.py` / `training.py` đúng Dyna-Q + sharing phân biệt proposed vs baseline.
3. Export path = best successful episode.
4. `.wbt` + controller khớp training.
5. Đồ thị: proposed tốt nhất; R2 nhanh; R1/R3 cải thiện nhờ sharing (có thể cần vài vòng chỉnh hyperparameter).

---

## 6. Logging (khi triển khai)

- In tỷ lệ đạt goal / mean steps (theo robot), tùy chọn `--verbose` / file log trong `save_dir`.

---

## Thứ tự thực hiện đề xuất (khi thoát Plan mode)

1. Sửa `config` + `env` (layout + starts + biên).
2. Sửa `raw_sharing` và đảm bảo `proposed` chỉ merge bằng T-test.
3. Ghi trajectory trong train demo → export best path.
4. Quy đổi sang Webots và cập nhật `.wbt`.
5. Chạy `train_compare.py`, kiểm tra curves và replay.

---

## Ghi chú

- Điều kiện paper “khi R1/R3 vào Area 2 thì nhận knowledge” có thể **chưa** mô hình hóa địa lý trong code hiện tại (chỉ dựa visit count); nếu cần sát paper hơn, có thể thêm phase sau.
- Triển khai mã cần **chế độ Agent** (Plan mode hiện chặn sửa file không phải markdown).
