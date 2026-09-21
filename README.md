# ĐỒ ÁN KHAI THÁC DỮ LIỆU (DATA MINING PROJECT)
## ĐỀ TÀI 4: SO SÁNH KHẢ NĂNG MỞ RỘNG CỦA APRIORI, FP-GROWTH VÀ IT-TREE TRÊN DỮ LIỆU BÁN LẺ LỚN

> **Học phần:** Thực hành Khai thác Dữ liệu  
> **Chuyên ngành:** Khoa học Dữ liệu - Khóa 3  
> **Bộ dữ liệu:** Market Basket Analysis (>500.000 dòng bán lẻ)

---

## 1. Thành viên Nhóm & Bảng Phân công Công việc

| STT | Họ và Tên | Mã Sinh Viên | Vai trò & Công việc chính | Công việc hỗ trợ | Tỷ lệ đóng góp (%) |
|:---:|---|:---:|---|---|:---:|
| 1 | *Nguyễn Văn A* (Nhóm trưởng) | *22120001* | **Data & Preprocessing Lead:** Khảo sát dữ liệu gốc, làm sạch >500k dòng, xử lý đơn hủy/mã kỹ thuật, mã hóa One-hot Sparse Matrix & TID-list. | Hỗ trợ chạy thực nghiệm, viết Mục 3, 4, 5 Báo cáo Word. | **33.3%** |
| 2 | *Trần Thị B* | *22120002* | **Algorithms & Benchmark Lead:** Cài đặt thuật toán IT-Tree (Eclat), tích hợp Apriori/FP-Growth, thiết kế thực nghiệm đo đạc (Runtime, Memory, Scalability). | Hỗ trợ lọc luật, vẽ biểu đồ kỹ thuật, viết Mục 6, 7 Báo cáo Word. | **33.3%** |
| 3 | *Lê Văn C* | *22120003* | **Rule Mining & Web App Lead:** Khai thác luật kết hợp, xây dựng bộ lọc luật ngẫu nhiên ($Lift \approx 1$), phát triển Web App Streamlit tương tác 3 Tab. | Hỗ trợ phân tích Insight kinh doanh, viết Mục 8, 9, Slide báo cáo. | **33.4%** |

---

## 2. Cấu trúc Thư mục Dự án

```text
THKTDL/
├── app/
│   └── streamlit_app.py         # Ứng dụng Web Demo tương tác (Benchmark & Recommender)
├── data/
│   ├── raw/                     # Chứa dữ liệu gốc Assignment-1_Data.csv (>500.000 dòng)
│   └── processed/               # Chứa cleaned_retail.csv và transactions.pkl
├── outputs/
│   ├── figures/                 # Biểu đồ EDA và biểu đồ so sánh Benchmark chuẩn học thuật
│   └── tables/                  # Bảng số liệu Benchmark CSV và Top 20 Luật Vàng CSV
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Pipeline làm sạch và chuẩn hóa dữ liệu
│   ├── algorithms.py            # Cài đặt IT-Tree và wrapper Apriori, FP-Growth
│   ├── benchmark.py             # Kịch bản thực nghiệm đo đạc độ nhạy & khả năng mở rộng
│   └── rule_mining.py           # Sinh luật, bộ lọc luật thông minh và hệ thống gợi ý
├── requirements.txt             # Danh mục các thư viện phụ thuộc
└── README.md                    # Tài liệu hướng dẫn cài đặt và vận hành
```

---

## 3. Hướng dẫn Cài đặt & Vận hành

### Bước 1: Cài đặt các thư viện cần thiết
Mở terminal tại thư mục gốc của dự án và chạy lệnh:
```bash
pip install -r requirements.txt
```

### Bước 2: Chạy Tiền xử lý dữ liệu (Data Preprocessing)
Thực hiện làm sạch hơn 520.000 dòng, khử đơn hủy, loại bỏ chi phí phi sản phẩm, tạo cấu trúc giao dịch:
```bash
python src/data_preprocessing.py
```
*Kết quả:* Sinh ra tệp dữ liệu sạch `data/processed/cleaned_retail.csv` và `data/processed/transactions.pkl`.

### Bước 3: Chạy Toàn bộ Pipeline Thực nghiệm & Benchmark
Thực hiện đo đạc tự động thời gian chạy (Runtime), mức tiêu thụ RAM (Memory Profiling) và vẽ các biểu đồ so sánh:
```bash
python src/benchmark.py
```
*Kết quả:*
- Biểu đồ lưu tại: `outputs/figures/` (Runtime vs Minsup, Frequent Itemsets count, Scalability curve).
- Bảng số liệu lưu tại: `outputs/tables/` (`benchmark_minsup.csv`, `benchmark_datasize.csv`, `gold_rules.csv`).

### Bước 4: Khởi chạy Ứng dụng Web Demo Streamlit
Chạy lệnh sau để mở giao diện demo tương tác trên trình duyệt:
```bash
python -m streamlit run app/streamlit_app.py
```
*(Hoặc `streamlit run app/streamlit_app.py` nếu biến môi trường PATH đã nhận lệnh streamlit)*.

Ứng dụng gồm 3 phân hệ chính:
1. **Thực nghiệm Benchmark:** Xem biểu đồ so sánh và có hộp cát (Sandbox) để chạy đo đạc trực tiếp giữa 3 thuật toán theo tham số tùy chỉnh.
2. **Hệ thống Gợi ý Mua kèm (Recommender):** Mô phỏng giỏ hàng thực tế, chọn sản phẩm $\to$ gợi ý các món đi kèm có Lift cao nhất kèm lời giải thích logic.
3. **Top 20 Luật Vàng & Đồ thị Mạng lưới:** Khám phá mạng lưới liên kết sản phẩm trực quan và các đề xuất chiến lược bán lẻ.

---

## 4. Tóm tắt Kết quả Thực nghiệm Chính

1. **Khả năng mở rộng:**
   - **FP-Growth:** Cho tốc độ vượt trội và ổn định nhất nhờ nén dữ liệu vào cây FP-Tree và chỉ quét cơ sở dữ liệu đúng 2 lần.
   - **IT-Tree (Eclat):** Tự triển khai bằng biểu diễn dọc (TID-list) kết hợp kỹ thuật duyệt DFS và sắp xếp thứ tự mục tăng dần cho tốc độ tiệm cận FP-Growth, nhanh hơn Apriori hàng chục lần.
   - **Apriori:** Bị thắt nút cổ chai (bottleneck) khi $minsup \le 1.5\%$ do bùng nổ tổ hợp ứng viên $C_k$ và phải quét lại ma trận nhiều lần.
2. **Giá trị Ứng dụng:**
   - Đã loại bỏ các luật ngẫu nhiên ($0.85 \le Lift \le 1.15$) và lọc luật dư thừa.
   - Trích xuất thành công các combo sản phẩm kinh điển như bộ tách trà Regency Vintage (`GREEN REGENCY TEACUP` + `PINK REGENCY TEACUP` $\to$ `ROSES REGENCY TEACUP` với $Lift > 15.0$), mang lại giá trị gia tăng doanh số trực tiếp cho doanh nghiệp bán lẻ.
