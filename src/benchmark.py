"""
Module Thực nghiệm Đánh giá & So sánh Khả năng mở rộng (Scalability Benchmark).
So sánh chi tiết 3 thuật toán: Apriori, FP-Growth, IT-Tree trên dữ liệu lớn (>520.000 dòng).
"""

# Thư viện hệ thống của Python
import sys
# Thư viện quản lý đường dẫn và tệp tin của hệ điều hành
import os

# Lấy đường dẫn tuyệt đối của thư mục gốc dự án
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Kiểm tra xem thư mục gốc đã có trong danh sách tìm kiếm module của Python hay chưa
if PROJECT_ROOT not in sys.path:
    # Chèn thư mục gốc vào đầu sys.path để import các module nội bộ an toàn
    sys.path.insert(0, PROJECT_ROOT)

# Thư viện đo lường thời gian thực thi hiệu năng cao
import time
# Thư viện đọc và ghi tệp dữ liệu JSON
import json
# Thư viện trích xuất thông tin phần cứng và hệ điều hành
import platform
# Thư viện theo dõi lượng RAM phân bổ đỉnh của chương trình
import tracemalloc
# Thư viện xử lý và phân tích dữ liệu dạng bảng
import pandas as pd
# Thư viện tính toán số học và mảng dữ liệu
import numpy as np
# Thư viện vẽ biểu đồ và đồ thị khoa học
import matplotlib.pyplot as plt
# Thư viện trực quan hóa dữ liệu thống kê nâng cao
import seaborn as sns

# Kiểm tra nếu console Windows chưa dùng bảng mã UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        # Cấu hình console sang UTF-8 để in tiếng Việt có dấu
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Import hàm nạp danh sách giao dịch giỏ hàng
from src.data_preprocessing import load_transactions
# Import hàm chuyển đổi danh sách giao dịch sang ma trận thưa One-hot
from src.data_preprocessing import get_onehot_sparse_df
# Import hàm bọc chạy thuật toán Apriori
from src.algorithms import run_apriori
# Import hàm bọc chạy thuật toán FP-Growth
from src.algorithms import run_fpgrowth
# Import hàm bọc chạy thuật toán IT-Tree
from src.algorithms import run_ittree
# Import lớp cài đặt thuật toán ITTreeMiner
from src.algorithms import ITTreeMiner

# Định nghĩa hàm tự động thu thập thông tin cấu hình phần cứng và hệ thống
def get_system_specs():
    # Khởi tạo từ điển lưu trữ thông số kỹ thuật của môi trường thực nghiệm
    specs = {
        # Tên và bản phát hành của hệ điều hành (ví dụ: Windows 11)
        'os': platform.system() + " " + platform.release(),
        # Số hiệu phiên bản chi tiết của hệ điều hành
        'os_version': platform.version(),
        # Kiểu kiến trúc máy tính (AMD64 hoặc x86_64)
        'machine': platform.machine(),
        # Thông tin chi tiết về chip vi xử lý (CPU)
        'processor': platform.processor(),
        # Phiên bản của trình thông dịch Python đang sử dụng
        'python_version': platform.python_version(),
    }
    # Trả về từ điển thông số cấu hình
    return specs

# Định nghĩa hàm thực thi Thí nghiệm 1: Khảo sát ảnh hưởng của minsup đến thời gian và bộ nhớ
def run_controlled_minsup_benchmark(transactions, minsup_list=[0.05, 0.04, 0.03, 0.02, 0.015, 0.012]):
    # In tiêu đề phân cách bắt đầu thí nghiệm 1
    print("\n" + "="*70)
    # In tên thí nghiệm 1
    print(" BẮT ĐẦU THÍ NGHIỆM 1: ĐỘ NHẠY VỚI NGƯỠNG MINSUP (RUNTIME & MEMORY)")
    # In thanh phân cách kết thúc tiêu đề
    print("="*70)

    # In thông báo chuẩn bị ma trận thưa One-hot cho Apriori và FP-Growth
    print("[*] Đang chuẩn bị ma trận thưa cho Apriori & FP-Growth...")
    
    # Chuyển đổi danh sách giỏ hàng thành ma trận thưa Scipy CSR Matrix
    df_sparse, _ = get_onehot_sparse_df(transactions)
    
    # In thông tin kích thước ma trận thưa (Số giao dịch x Số mặt hàng)
    print(f"    - Kích thước ma trận thưa: {df_sparse.shape[0]:,} giao dịch x {df_sparse.shape[1]:,} mặt hàng")

    # Khởi tạo danh sách lưu trữ kết quả đo đạc từng ngưỡng
    results = []

    # Lặp qua từng giá trị ngưỡng minsup trong danh sách thử nghiệm
    for minsup in minsup_list:
        # In thông báo ngưỡng minsup hiện tại đang được thử nghiệm
        print(f"\n---> Đang thực nghiệm với minsup = {minsup*100:.1f}% ({minsup})...")
        
        # Khởi tạo dòng từ điển lưu kết quả của ngưỡng hiện tại
        row = {'minsup': minsup, 'minsup_pct': f"{minsup*100:.1f}%"}

        # Bắt đầu theo dõi mức phân bổ bộ nhớ RAM của Python bằng tracemalloc
        tracemalloc.start()
        
        # Bắt đầu bấm giờ đo thời gian thực thi của FP-Growth
        t0 = time.perf_counter()
        
        # Chạy thuật toán FP-Growth trên ma trận thưa
        fp_res, fp_time = run_fpgrowth(df_sparse, min_support=minsup)
        
        # Lấy lượng RAM đỉnh phân bổ trong quá trình chạy và chuyển đổi sang đơn vị Megabytes
        fp_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        
        # Dừng theo dõi bộ nhớ của FP-Growth
        tracemalloc.stop()
        
        # Lưu thời gian thực thi của FP-Growth làm tròn 4 chữ số thập phân
        row['fpgrowth_time'] = round(fp_time, 4)
        
        # Lưu lượng RAM đỉnh của FP-Growth làm tròn 2 chữ số thập phân
        row['fpgrowth_mem_mb'] = round(fp_mem, 2)
        
        # Lưu số lượng tập phổ biến tìm được
        row['itemsets_count'] = len(fp_res)
        
        # In kết quả đo đạc của FP-Growth ra màn hình console
        print(f"    [FP-Growth] Time: {fp_time:.3f}s | Peak Mem: {fp_mem:.2f}MB | Itemsets: {len(fp_res):,}")

        # Bắt đầu theo dõi phân bổ bộ nhớ RAM cho IT-Tree
        tracemalloc.start()
        
        # Bắt đầu bấm giờ đo thời gian thực thi của IT-Tree
        t0 = time.perf_counter()
        
        # Chạy thuật toán IT-Tree tự cài đặt trên danh sách transactions trực tiếp
        it_res, it_time, it_nodes = run_ittree(transactions, min_support=minsup)
        
        # Lấy lượng RAM đỉnh phân bổ của IT-Tree và chuyển sang MB
        it_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        
        # Dừng theo dõi bộ nhớ của IT-Tree
        tracemalloc.stop()
        
        # Lưu thời gian thực thi của IT-Tree
        row['ittree_time'] = round(it_time, 4)
        
        # Lưu lượng RAM đỉnh của IT-Tree
        row['ittree_mem_mb'] = round(it_mem, 2)
        
        # Lưu số lượng nút cây mà IT-Tree đã duyệt qua
        row['ittree_nodes'] = it_nodes
        
        # In kết quả đo đạc của IT-Tree ra màn hình console
        print(f"    [IT-Tree]   Time: {it_time:.3f}s | Peak Mem: {it_mem:.2f}MB | Nodes: {it_nodes:,}")

        # Kiểm tra ngưỡng an toàn cho Apriori: Chỉ chạy khi minsup >= 0.015 để tránh bùng nổ tổ hợp làm treo máy
        if minsup >= 0.015:
            # Bắt đầu theo dõi bộ nhớ cho Apriori
            tracemalloc.start()
            
            # Bắt đầu bấm giờ đo thời gian thực thi của Apriori
            t0 = time.perf_counter()
            
            # Chạy thuật toán Apriori trên ma trận thưa
            ap_res, ap_time = run_apriori(df_sparse, min_support=minsup)
            
            # Lấy lượng RAM đỉnh phân bổ của Apriori và chuyển sang MB
            ap_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
            
            # Dừng theo dõi bộ nhớ của Apriori
            tracemalloc.stop()
            
            # Lưu thời gian chạy của Apriori
            row['apriori_time'] = round(ap_time, 4)
            
            # Lưu lượng RAM đỉnh của Apriori
            row['apriori_mem_mb'] = round(ap_mem, 2)
            
            # In kết quả đo đạc của Apriori ra console
            print(f"    [Apriori]   Time: {ap_time:.3f}s | Peak Mem: {ap_mem:.2f}MB")
        else:
            # In thông báo lý do khoa học bỏ qua Apriori ở các ngưỡng nhỏ
            print("    [Apriori]   Bỏ qua ngưỡng <= 1.0% do chi phí tính toán bùng nổ tổ hợp (OOM/Timeout)")
            
            # Gán giá trị NaN cho thời gian của Apriori
            row['apriori_time'] = np.nan
            
            # Gán giá trị NaN cho bộ nhớ của Apriori
            row['apriori_mem_mb'] = np.nan

        # Thêm kết quả của ngưỡng hiện tại vào danh sách kết quả chung
        results.append(row)

    # Chuyển đổi toàn bộ danh sách kết quả thành DataFrame
    df_minsup_benchmark = pd.DataFrame(results)
    
    # Trả về bảng số liệu thực nghiệm đo đạc độ nhạy minsup
    return df_minsup_benchmark

# Định nghĩa hàm thực thi Thí nghiệm 2: Khảo sát khả năng mở rộng theo kích thước dữ liệu
def run_datasize_scalability_benchmark(transactions, minsup=0.02, fractions=[0.2, 0.4, 0.6, 0.8, 1.0]):
    # In tiêu đề bắt đầu thí nghiệm 2
    print("\n" + "="*70)
    # In thông báo ngưỡng minsup cố định dùng trong thí nghiệm 2
    print(f" BẮT ĐẦU THÍ NGHIỆM 2: KHẢ NĂNG MỞ RỘNG THEO KÍCH THƯỚC (minsup={minsup*100}%)")
    # In thanh phân cách kết thúc tiêu đề
    print("="*70)

    # Đếm tổng số lượng giao dịch đầy đủ có trong kho dữ liệu
    n_total = len(transactions)
    
    # Khởi tạo danh sách lưu trữ kết quả đo đạc theo từng tỷ lệ kích thước
    results = []

    # Duyệt qua từng tỷ lệ phần trăm kích thước dữ liệu (20%, 40%, 60%, 80%, 100%)
    for frac in fractions:
        # Tính số lượng giao dịch mẫu tương ứng với tỷ lệ phần trăm frac
        n_sample = int(n_total * frac)
        
        # In thông tin tỷ lệ và số lượng giao dịch đang thực nghiệm
        print(f"\n---> Đang thử nghiệm với {frac*100:.0f}% dữ liệu ({n_sample:,} transactions)...")
        
        # Cắt tập con danh sách giao dịch từ đầu đến n_sample
        sub_tx = transactions[:n_sample]
        
        # Biến đổi tập con giao dịch sang ma trận thưa One-hot
        df_sparse_sub, _ = get_onehot_sparse_df(sub_tx)

        # Khởi tạo dòng kết quả cho kích thước hiện tại
        row = {
            # Tỷ lệ số thập phân
            'fraction': frac,
            # Tỷ lệ chuỗi hiển thị phần trăm
            'fraction_pct': f"{int(frac*100)}%",
            # Số lượng giao dịch thực tế
            'n_transactions': n_sample
        }

        # Đo thời gian thực thi của thuật toán FP-Growth
        _, fp_time = run_fpgrowth(df_sparse_sub, min_support=minsup)
        # Lưu thời gian của FP-Growth
        row['fpgrowth_time'] = round(fp_time, 4)

        # Đo thời gian thực thi của thuật toán IT-Tree
        _, it_time, _ = run_ittree(sub_tx, min_support=minsup)
        # Lưu thời gian của IT-Tree
        row['ittree_time'] = round(it_time, 4)

        # Đo thời gian thực thi của thuật toán Apriori
        _, ap_time = run_apriori(df_sparse_sub, min_support=minsup)
        # Lưu thời gian của Apriori
        row['apriori_time'] = round(ap_time, 4)

        # In tóm tắt kết quả đo đạc thời gian của cả 3 thuật toán ra console
        print(f"    Kết quả: Apriori={ap_time:.2f}s | FP-Growth={fp_time:.2f}s | IT-Tree={it_time:.2f}s")
        
        # Thêm kết quả của kích thước hiện tại vào danh sách chung
        results.append(row)

    # Chuyển đổi danh sách kết quả thành DataFrame
    df_datasize_benchmark = pd.DataFrame(results)
    
    # Trả về bảng kết quả khả năng mở rộng
    return df_datasize_benchmark

# Định nghĩa hàm vẽ các biểu đồ trực quan hóa kết quả benchmark đạt chuẩn 300 DPI
def plot_benchmark_charts(df_minsup, df_datasize, output_dir='outputs/figures'):
    # Tạo thư mục lưu trữ ảnh biểu đồ nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    # Cấu hình phong cách biểu đồ Seaborn với lưới trắng và cỡ chữ 1.1
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. VẼ BIỂU ĐỒ RUNTIME THEO MINSUP (LOG SCALE)
    # Khởi tạo khung hình kích thước 10 x 6 inches
    plt.figure(figsize=(10, 6))
    
    # Lấy danh sách các nhãn phần trăm minsup cho trục hoành X
    x_labels = df_minsup['minsup_pct'].tolist()
    
    # Vẽ đường thời gian của FP-Growth màu xanh dương
    plt.plot(x_labels, df_minsup['fpgrowth_time'], marker='o', linewidth=2.5, color='#1f77b4', label='FP-Growth (Cây tiền tố)')
    
    # Vẽ đường thời gian của IT-Tree màu xanh lá cây
    plt.plot(x_labels, df_minsup['ittree_time'], marker='s', linewidth=2.5, color='#2ca02c', label='IT-Tree (Duyệt dọc TID)')
    
    # Lọc các dòng hợp lệ của Apriori để vẽ đường nét đứt màu đỏ
    valid_ap = df_minsup.dropna(subset=['apriori_time'])
    plt.plot(valid_ap['minsup_pct'], valid_ap['apriori_time'], marker='^', linewidth=2.5, color='#d62728', linestyle='--', label='Apriori (BFS Horizontal)')

    # Bật thang đo Logarithmic cho trục tung Y để nhìn rõ độ chênh lệch cấp số nhân
    plt.yscale('log')
    
    # Đặt tiêu đề biểu đồ với cỡ chữ 14 và in đậm
    plt.title('So sánh Thời gian thực thi (Runtime) theo Ngưỡng Minsup (Log Scale)', fontsize=14, fontweight='bold', pad=15)
    
    # Đặt nhãn trục hoành X
    plt.xlabel('Ngưỡng Hỗ trợ Tối thiểu (minsup)', fontsize=12, labelpad=10)
    
    # Đặt nhãn trục tung Y
    plt.ylabel('Thời gian thực thi (giây - Log Scale)', fontsize=12, labelpad=10)
    
    # Hiển thị bảng chú giải (legend) với nền trắng mờ
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    # Tự động căn chỉnh lề biểu đồ không bị cắt chữ
    plt.tight_layout()
    
    # Tạo đường dẫn tệp ảnh biểu đồ runtime
    chart1_path = os.path.join(output_dir, 'benchmark_runtime_minsup.png')
    
    # Xuất ảnh ra đĩa cứng với độ phân giải cao 300 DPI
    plt.savefig(chart1_path, dpi=300)
    
    # Đóng khung hình hiện tại để giải phóng bộ nhớ đồ họa
    plt.close()
    
    # In thông báo đã lưu ảnh thành công
    print(f"    -> Đã lưu biểu đồ: {chart1_path}")

    # 2. VẼ BIỂU ĐỒ BÙNG NỔ SỐ LƯỢNG ITEMSET THEO MINSUP
    # Khởi tạo khung hình kích thước 10 x 6 inches
    plt.figure(figsize=(10, 6))
    
    # Vẽ đường biểu diễn số lượng itemsets màu cam
    plt.plot(x_labels, df_minsup['itemsets_count'], marker='D', linewidth=2.5, color='#ff7f0e')
    
    # Đặt tiêu đề biểu đồ số lượng itemsets
    plt.title('Sự bùng nổ số lượng Frequent Itemsets khi giảm Minsup', fontsize=14, fontweight='bold', pad=15)
    
    # Đặt nhãn trục X
    plt.xlabel('Ngưỡng Hỗ trợ Tối thiểu (minsup)', fontsize=12, labelpad=10)
    
    # Đặt nhãn trục Y
    plt.ylabel('Số lượng Tập mục phổ biến sinh ra', fontsize=12, labelpad=10)
    
    # Lặp qua từng điểm dữ liệu để gắn nhãn số liệu cụ thể lên biểu đồ
    for i, txt in enumerate(df_minsup['itemsets_count']):
        # Gắn nhãn số lượng itemset có dấu phẩy phân cách hàng nghìn
        plt.annotate(f"{txt:,}", (x_labels[i], txt), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')
        
    # Căn chỉnh lề biểu đồ
    plt.tight_layout()
    
    # Tạo đường dẫn tệp ảnh itemsets
    chart2_path = os.path.join(output_dir, 'benchmark_itemsets_minsup.png')
    
    # Lưu ảnh với độ phân giải 300 DPI
    plt.savefig(chart2_path, dpi=300)
    
    # Đóng khung hình
    plt.close()
    
    # In thông báo lưu ảnh
    print(f"    -> Đã lưu biểu đồ: {chart2_path}")

    # 3. VẼ BIỂU ĐỒ ĐƯỜNG CONG KHẢ NĂNG MỞ RỘNG (SCALABILITY CURVE)
    # Khởi tạo khung hình kích thước 10 x 6 inches
    plt.figure(figsize=(10, 6))
    
    # Lấy danh sách tỷ lệ kích thước phần trăm cho trục X
    x_data = df_datasize['fraction_pct'].tolist()
    
    # Vẽ đường mở rộng của FP-Growth
    plt.plot(x_data, df_datasize['fpgrowth_time'], marker='o', linewidth=2.5, color='#1f77b4', label='FP-Growth')
    
    # Vẽ đường mở rộng của IT-Tree
    plt.plot(x_data, df_datasize['ittree_time'], marker='s', linewidth=2.5, color='#2ca02c', label='IT-Tree')
    
    # Vẽ đường mở rộng của Apriori
    plt.plot(x_data, df_datasize['apriori_time'], marker='^', linewidth=2.5, color='#d62728', linestyle='--', label='Apriori')
    
    # Đặt tiêu đề biểu đồ khả năng mở rộng
    plt.title('Khả năng mở rộng theo Kích thước Dữ liệu (Scalability Curve)', fontsize=14, fontweight='bold', pad=15)
    
    # Đặt nhãn trục X
    plt.xlabel('Tỷ lệ Kích thước Dữ liệu (% Số lượng Transactions)', fontsize=12, labelpad=10)
    
    # Đặt nhãn trục Y
    plt.ylabel('Thời gian thực thi (giây)', fontsize=12, labelpad=10)
    
    # Hiển thị bảng chú giải
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    # Căn chỉnh lề
    plt.tight_layout()
    
    # Tạo đường dẫn tệp ảnh scalability
    chart3_path = os.path.join(output_dir, 'benchmark_datasize_scalability.png')
    
    # Lưu ảnh 300 DPI
    plt.savefig(chart3_path, dpi=300)
    
    # Đóng khung hình
    plt.close()
    
    # In thông báo lưu ảnh
    print(f"    -> Đã lưu biểu đồ: {chart3_path}")

# Định nghĩa hàm vẽ biểu đồ khám phá dữ liệu ban đầu EDA
def plot_eda_charts(df_clean, transactions, output_dir='outputs/figures'):
    # Tạo thư mục outputs/figures nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Thiết lập giao diện biểu đồ Seaborn
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. BIỂU ĐỒ PHÂN PHỐI KÍCH THƯỚC GIỎ HÀNG
    # Tính độ dài số lượng món hàng cho từng giao dịch
    basket_sizes = [len(tx) for tx in transactions]
    
    # Khởi tạo khung hình kích thước 10 x 5 inches
    plt.figure(figsize=(10, 5))
    
    # Tính giá trị phân vị thứ 99% để loại bỏ các đơn hàng bán buôn ngoại lệ quá lớn
    p99 = int(np.percentile(basket_sizes, 99))
    
    # Lọc chỉ lấy các giỏ hàng có số món <= p99
    filtered_sizes = [s for s in basket_sizes if s <= p99]
    
    # Vẽ biểu đồ phân phối tần số (Histogram) kết hợp đường ước lượng mật độ KDE
    sns.histplot(filtered_sizes, bins=35, kde=True, color='#2b5c8f')
    
    # Đặt tiêu đề cho biểu đồ phân phối kích thước giỏ hàng
    plt.title(f'Phân phối Số lượng Mặt hàng trên mỗi Giỏ hàng (Cắt ngưỡng 99% = {p99} món)', fontsize=13, fontweight='bold', pad=15)
    
    # Đặt nhãn trục X
    plt.xlabel('Kích thước Giỏ hàng (Số món / Hóa đơn)', fontsize=11)
    
    # Đặt nhãn trục Y
    plt.ylabel('Tần số (Số hóa đơn)', fontsize=11)
    
    # Căn chỉnh lề
    plt.tight_layout()
    
    # Tạo đường dẫn tệp ảnh
    chart_path = os.path.join(output_dir, 'eda_basket_distribution.png')
    
    # Lưu ảnh 300 DPI
    plt.savefig(chart_path, dpi=300)
    
    # Đóng khung hình
    plt.close()
    
    # In thông báo
    print(f"    -> Đã lưu biểu đồ: {chart_path}")

    # 2. BIỂU ĐỒ TOP 15 MẶT HÀNG BÁN CHẠY NHẤT
    # Đếm tần số xuất hiện của từng mặt hàng và lấy 15 mặt hàng đứng đầu
    top_15 = df_clean['Itemname'].value_counts().head(15)
    
    # Khởi tạo khung hình kích thước 12 x 6 inches
    plt.figure(figsize=(12, 6))
    
    # Vẽ biểu đồ thanh ngang màu xanh dương thể hiện số lần xuất hiện của 15 mặt hàng
    sns.barplot(x=top_15.values, y=top_15.index, color='#2563EB')
    
    # Đặt tiêu đề biểu đồ top 15 mặt hàng
    plt.title('Top 15 Mặt hàng Xuất hiện Nhiều nhất trong các Giao dịch', fontsize=13, fontweight='bold', pad=15)
    
    # Đặt nhãn trục X
    plt.xlabel('Số lần xuất hiện (Giao dịch)', fontsize=11)
    
    # Đặt nhãn trục Y
    plt.ylabel('Tên mặt hàng', fontsize=11)
    
    # Căn chỉnh lề
    plt.tight_layout()
    
    # Tạo đường dẫn tệp ảnh
    chart_path = os.path.join(output_dir, 'eda_top_items.png')
    
    # Lưu ảnh 300 DPI
    plt.savefig(chart_path, dpi=300)
    
    # Đóng khung hình
    plt.close()
    
    # In thông báo
    print(f"    -> Đã lưu biểu đồ: {chart_path}")

# Định nghĩa hàm điều phối chạy toàn bộ quy trình thực nghiệm và xuất bảng biểu
def run_full_benchmark_pipeline():
    # Bước 1: Thu thập thông số phần cứng của máy tính thực nghiệm
    specs = get_system_specs()
    
    # Ghi thông số phần cứng ra tệp JSON
    with open('outputs/tables/system_specs.json', 'w', encoding='utf-8') as f:
        json.dump(specs, f, indent=4, ensure_ascii=False)
        
    # In thông tin cấu hình máy lên màn hình
    print(f"[*] Cấu hình thực nghiệm: {specs['os']} | CPU: {specs['processor']} | Python {specs['python_version']}")

    # Bước 2: Nạp danh sách giỏ hàng và bảng dữ liệu sạch
    transactions = load_transactions()
    df_clean = pd.read_csv('data/processed/cleaned_retail.csv', low_memory=False)

    # Bước 3: Tạo và xuất các biểu đồ khám phá dữ liệu EDA
    print("\n[*] Đang tạo biểu đồ phân tích EDA...")
    plot_eda_charts(df_clean, transactions)

    # Bước 4: Chạy Thí nghiệm 1 khảo sát độ nhạy với ngưỡng minsup
    minsup_list = [0.05, 0.04, 0.03, 0.02, 0.015, 0.012]
    df_minsup = run_controlled_minsup_benchmark(transactions, minsup_list=minsup_list)
    
    # Xuất bảng kết quả Thí nghiệm 1 ra file CSV
    df_minsup.to_csv('outputs/tables/benchmark_minsup.csv', index=False)
    print("    -> Đã lưu bảng kết quả: outputs/tables/benchmark_minsup.csv")

    # Bước 5: Chạy Thí nghiệm 2 khảo sát khả năng mở rộng theo kích thước dữ liệu
    df_datasize = run_datasize_scalability_benchmark(transactions, minsup=0.02)
    
    # Xuất bảng kết quả Thí nghiệm 2 ra file CSV
    df_datasize.to_csv('outputs/tables/benchmark_datasize.csv', index=False)
    print("    -> Đã lưu bảng kết quả: outputs/tables/benchmark_datasize.csv")

    # Bước 6: Vẽ các biểu đồ kết quả benchmark đạt chuẩn báo cáo
    print("\n[*] Đang vẽ biểu đồ thực nghiệm chuẩn báo cáo...")
    plot_benchmark_charts(df_minsup, df_datasize)

    # Bước 7: Khai thác luật kết hợp tại minsup = 0.015 và xuất Top 20 Luật Vàng
    print("\n[*] Đang khai thác và lọc Top 20 Luật Vàng (Gold Rules)...")
    
    # Import các hàm lọc và trích xuất luật
    from src.rule_mining import mine_association_rules, filter_valuable_rules, extract_gold_rules
    
    # Khởi tạo đối tượng khai thác ITTreeMiner
    miner = ITTreeMiner()
    
    # Khai thác tập phổ biến tại ngưỡng 1.5%
    freq_df = miner.mine(transactions, min_support=0.015)
    
    # Sinh luật kết hợp với min_confidence = 25% và min_lift = 1.5
    all_rules = mine_association_rules(freq_df, min_confidence=0.25, min_lift=1.5)
    
    # Lọc bỏ các luật độc lập ngẫu nhiên và khử luật dư thừa
    filtered_rules = filter_valuable_rules(all_rules, min_lift=1.5, max_redundant=True)
    
    # Trích xuất 20 luật vàng tiêu biểu nhất
    gold_rules = extract_gold_rules(filtered_rules, top_n=20)
    
    # Xuất bảng toàn bộ luật sạch ra file CSV
    filtered_rules.to_csv('outputs/tables/filtered_rules.csv', index=False)
    
    # Xuất bảng top 20 luật vàng ra file CSV
    gold_rules.to_csv('outputs/tables/gold_rules.csv', index=False)
    print("    -> Đã lưu danh sách luật kết hợp vào: outputs/tables/gold_rules.csv")
    
    # In thông báo hoàn tất thành công toàn bộ pipeline
    print(f"\n[+] HOÀN TẤT TOÀN BỘ PIPELINE THỰC NGHIỆM THÀNH CÔNG!")

# Điểm khởi chạy chương trình khi thực thi từ dòng lệnh
if __name__ == '__main__':
    # Gọi hàm chạy toàn bộ pipeline thực nghiệm
    run_full_benchmark_pipeline()
