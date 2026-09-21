"""
Streamlit Web App: Hệ thống Khai thác Luật kết hợp & So sánh Khả năng mở rộng Thuật toán
Đề tài 4: So sánh Apriori, FP-Growth và IT-Tree trên dữ liệu bán lẻ lớn (>500.000 dòng).
python -m streamlit run app/streamlit_app.py
Chạy lại pipeline thực nghiệm nếu muốn đo đạc lại:python src/benchmark.py
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# Thêm đường dẫn gốc vào sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# Gọi module
from src.data_preprocessing import load_transactions, get_onehot_sparse_df
from src.algorithms import run_apriori, run_fpgrowth, run_ittree
from src.rule_mining import mine_association_rules, filter_valuable_rules, extract_gold_rules, recommend_products

st.set_page_config(
    page_title="Data Mining Lab - Đề tài 4",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2563EB;
    }
    .gold-rule-card {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_cached_data():
    summary_path = os.path.join(PROJECT_ROOT, 'outputs', 'tables', 'data_summary.json')
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)

    # Đọc benchmark kết quả có sẵn
    minsup_bm_path = os.path.join(PROJECT_ROOT, 'outputs', 'tables', 'benchmark_minsup.csv')
    df_minsup = pd.read_csv(minsup_bm_path) if os.path.exists(minsup_bm_path) else pd.DataFrame()

    datasize_bm_path = os.path.join(PROJECT_ROOT, 'outputs', 'tables', 'benchmark_datasize.csv')
    df_datasize = pd.read_csv(datasize_bm_path) if os.path.exists(datasize_bm_path) else pd.DataFrame()

    gold_rules_path = os.path.join(PROJECT_ROOT, 'outputs', 'tables', 'gold_rules.csv')
    df_gold = pd.read_csv(gold_rules_path) if os.path.exists(gold_rules_path) else pd.DataFrame()

    filtered_rules_path = os.path.join(PROJECT_ROOT, 'outputs', 'tables', 'filtered_rules.csv')
    df_filtered_rules = pd.read_csv(filtered_rules_path) if os.path.exists(filtered_rules_path) else pd.DataFrame()

    # Chuyển đổi chuỗi frozenset sang set nếu cần
    if not df_filtered_rules.empty and isinstance(df_filtered_rules['antecedents'].iloc[0], str):
        import ast
        def parse_frozenset(val):
            val = str(val).strip()
            if val.startswith('frozenset(') and val.endswith(')'):
                val = val[len('frozenset('):-1]
            try:
                return ast.literal_eval(val)
            except Exception:
                return set()
        df_filtered_rules['antecedents'] = df_filtered_rules['antecedents'].apply(parse_frozenset)
        df_filtered_rules['consequents'] = df_filtered_rules['consequents'].apply(parse_frozenset)

    transactions = load_transactions(os.path.join(PROJECT_ROOT, 'data', 'processed', 'transactions.pkl'))
    return summary, df_minsup, df_datasize, df_gold, df_filtered_rules, transactions

try:
    summary, df_minsup, df_datasize, df_gold, df_filtered_rules, transactions = load_cached_data()
except Exception as e:
    st.error(f"Vui lòng chạy file `src/benchmark.py` trước để tạo dữ liệu thực nghiệm! Lỗi: {e}")
    st.stop()

# SIDEBAR
with st.sidebar:
    st.image("https://img.icons8.com/color/96/data-configuration.png", width=70)
    st.title("ĐỒ ÁN KHAI THÁC DỮ LIỆU")
    st.markdown("**Đề tài 4:** So sánh khả năng mở rộng của *Apriori, FP-Growth và IT-Tree* trên dữ liệu bán lẻ lớn.")
    st.markdown("---")
    
    st.markdown("### 📌 Thống kê Dữ liệu")
    if summary:
        st.write(f"• **Tổng số bản ghi gốc:** {summary.get('initial_rows', 0):,}")
        st.write(f"• **Bản ghi sau làm sạch:** {summary.get('cleaned_rows', 0):,}")
        st.write(f"• **Số hóa đơn hợp lệ:** {summary.get('cleaned_bills', 0):,}")
        st.write(f"• **Số giao dịch (≥2 món):** {summary.get('total_transactions_usable', 0):,}")
        st.write(f"• **Số mặt hàng duy nhất:** {summary.get('cleaned_items', 0):,}")
        st.write(f"• **Độ dài giỏ hàng TB:** {summary.get('avg_basket_size', 0):.1f} món")
    
    st.markdown("---")
    st.markdown("👨‍💻 **Nhóm 4**")
    st.caption("Ứng dụng phục vụ Báo cáo & Trình diễn Demo tại lớp.")

# HEADER
st.markdown('<div class="main-header">Hệ thống So sánh Thuật toán & Khai thác Giỏ hàng Bán lẻ</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Phân tích chuyên sâu khả năng mở rộng (Scalability Benchmark) giữa Apriori, FP-Growth & IT-Tree trên bộ dữ liệu >500k dòng</div>', unsafe_allow_html=True)

# TABS
tab1, tab2, tab3 = st.tabs([
    "📊 1. Thực nghiệm Benchmark & Mở rộng",
    "🛒 2. Hệ thống Gợi ý Mua kèm (Recommender)",
    "🏆 3. Top 20 Luật Vàng & Đồ thị Mạng lưới"
])

# ==========================================
# TAB 1: BENCHMARK
# ==========================================
with tab1:
    st.subheader("1. Kết quả Thực nghiệm Đo đạc (Controlled Benchmark)")
    
    col1, col2 = st.columns(2)
    with col1:
        if not df_minsup.empty:
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(x=df_minsup['minsup_pct'], y=df_minsup['fpgrowth_time'], mode='lines+markers', name='FP-Growth (Cây tiền tố)', line=dict(color='#2563EB', width=3)))
            fig_time.add_trace(go.Scatter(x=df_minsup['minsup_pct'], y=df_minsup['ittree_time'], mode='lines+markers', name='IT-Tree (Duyệt dọc TID)', line=dict(color='#059669', width=3)))
            
            valid_ap = df_minsup.dropna(subset=['apriori_time'])
            fig_time.add_trace(go.Scatter(x=valid_ap['minsup_pct'], y=valid_ap['apriori_time'], mode='lines+markers', name='Apriori (BFS Horizontal)', line=dict(color='#DC2626', width=2, dash='dash')))
            
            fig_time.update_layout(
                title="Thời gian thực thi (Runtime) theo Minsup (Log Scale)",
                xaxis_title="Ngưỡng hỗ trợ tối thiểu (minsup)",
                yaxis_title="Thời gian (giây)",
                yaxis_type="log",
                hovermode="x unified",
                template="plotly_white"
            )
            st.plotly_chart(fig_time, use_container_width=True)
            
    with col2:
        if not df_minsup.empty:
            fig_items = px.bar(
                df_minsup, 
                x='minsup_pct', 
                y='itemsets_count',
                title="Sự bùng nổ số lượng Frequent Itemsets khi giảm Minsup",
                labels={'minsup_pct': 'Ngưỡng Minsup', 'itemsets_count': 'Số tập phổ biến'},
                text_auto=',',
                color_discrete_sequence=['#EA580C']
            )
            fig_items.update_layout(template="plotly_white")
            st.plotly_chart(fig_items, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if not df_datasize.empty:
            fig_scale = go.Figure()
            fig_scale.add_trace(go.Scatter(x=df_datasize['fraction_pct'], y=df_datasize['fpgrowth_time'], mode='lines+markers', name='FP-Growth', line=dict(color='#2563EB', width=3)))
            fig_scale.add_trace(go.Scatter(x=df_datasize['fraction_pct'], y=df_datasize['ittree_time'], mode='lines+markers', name='IT-Tree', line=dict(color='#059669', width=3)))
            fig_scale.add_trace(go.Scatter(x=df_datasize['fraction_pct'], y=df_datasize['apriori_time'], mode='lines+markers', name='Apriori', line=dict(color='#DC2626', width=2, dash='dash')))
            fig_scale.update_layout(
                title="Khả năng mở rộng theo Kích thước Dữ liệu (Scalability Curve)",
                xaxis_title="Tỷ lệ tập dữ liệu (% Transactions)",
                yaxis_title="Thời gian thực thi (giây)",
                template="plotly_white",
                hovermode="x unified"
            )
            st.plotly_chart(fig_scale, use_container_width=True)
            
    with col4:
        st.markdown("#### 💡 Đánh giá & Rút ra kết luận thuật toán:")
        st.markdown("""
        - **FP-Growth:** Nén dữ liệu vào cây FP-Tree gọn gàng, quét dữ liệu đúng 2 lần. Không sinh tập ứng viên nên duy trì tốc độ cực nhanh và ổn định kể cả khi minsup giảm sâu.
        - **IT-Tree (Eclat):** Tận dụng cấu trúc TID-list dạng dọc và phép giao tập hợp `Set Intersection`. Tốc độ cực nhanh trên tập dữ liệu này nhờ việc sắp xếp thứ tự các mục theo tần số tăng dần để tỉa nhánh sớm.
        - **Apriori:** Khởi tạo $C_k$ theo cấp số nhân và phải quét lại cơ sở dữ liệu ở mỗi độ dài $k$. Khi minsup $\le 1.0\%$, Apriori bị nghẽn I/O và bùng nổ thời gian chạy hoặc tràn RAM.
        """)

    st.markdown("---")
    st.subheader("2. Chạy Thử nghiệm Tương tác Trực tiếp (Live Benchmark Sandbox)")
    st.caption("Bạn có thể điều chỉnh tham số dưới đây để đo đạc trực tiếp trên máy hiện tại:")
    
    scol1, scol2, scol3 = st.columns(3)
    with scol1:
        sel_algos = st.multiselect("Chọn thuật toán chạy thử:", ["FP-Growth", "IT-Tree", "Apriori"], default=["FP-Growth", "IT-Tree"])
    with scol2:
        test_minsup = st.slider("Chọn ngưỡng minsup (%):", min_value=1.0, max_value=5.0, value=2.0, step=0.5) / 100.0
    with scol3:
        test_pct = st.slider("Tỷ lệ mẫu giao dịch (%):", min_value=10, max_value=50, value=20, step=10) / 100.0

    if st.button("🚀 Chạy Thực Nghiệm Đo Đạc Trực Tiếp", type="primary"):
        n_sample = int(len(transactions) * test_pct)
        sub_tx = transactions[:n_sample]
        df_sp_sub, _ = get_onehot_sparse_df(sub_tx)
        
        st.info(f"Đang thực hiện đo đạc trên **{n_sample:,} giao dịch** với **minsup = {test_minsup*100:.1f}%**...")
        bench_live = []

        if "FP-Growth" in sel_algos:
            res_fp, t_fp = run_fpgrowth(df_sp_sub, min_support=test_minsup)
            bench_live.append({'Thuật toán': 'FP-Growth', 'Thời gian (giây)': round(t_fp, 4), 'Số Itemsets tìm được': len(res_fp)})

        if "IT-Tree" in sel_algos:
            res_it, t_it, nodes = run_ittree(sub_tx, min_support=test_minsup)
            bench_live.append({'Thuật toán': 'IT-Tree', 'Thời gian (giây)': round(t_it, 4), 'Số Itemsets tìm được': len(res_it)})

        if "Apriori" in sel_algos:
            res_ap, t_ap = run_apriori(df_sp_sub, min_support=test_minsup)
            bench_live.append({'Thuật toán': 'Apriori', 'Thời gian (giây)': round(t_ap, 4), 'Số Itemsets tìm được': len(res_ap)})

        df_live = pd.DataFrame(bench_live)
        st.dataframe(df_live, use_container_width=True)
        st.success("✅ Hoàn tất thực nghiệm trực tiếp!")

# ==========================================
# TAB 2: SMART RECOMMENDER
# ==========================================
with tab2:
    st.subheader("🛒 Mô phỏng Giỏ hàng & Đề xuất Mua kèm (Cross-Selling System)")
    st.markdown("Chọn các sản phẩm khách hàng đang có trong giỏ hàng để hệ thống truy xuất các luật kết hợp có **Lift cao nhất** và gợi ý sản phẩm đi kèm:")

    # Lấy danh sách các mặt hàng phổ biến nhất để người dùng dễ chọn
    all_items = sorted(list({item for tx in transactions[:2000] for item in tx}))
    default_items = [
        "GREEN REGENCY TEACUP AND SAUCER",
        "ROSES REGENCY TEACUP AND SAUCER"
    ]
    valid_default = [x for x in default_items if x in all_items]

    selected_items = st.multiselect(
        "Chọn sản phẩm trong giỏ hàng:",
        options=all_items,
        default=valid_default
    )

    top_k = st.slider("Số lượng sản phẩm đề xuất tối đa:", min_value=1, max_value=8, value=4)

    if st.button("💡 Gợi ý Sản phẩm Mua kèm", type="primary"):
        if not selected_items:
            st.warning("Vui lòng chọn ít nhất 1 sản phẩm vào giỏ hàng.")
        else:
            recs = recommend_products(selected_items, df_filtered_rules, top_k=top_k)
            if not recs:
                st.info("Không tìm thấy sản phẩm gợi ý nào có Lift thỏa mãn điều kiện với giỏ hàng này. Hãy thử chọn các sản phẩm gia dụng/decor phổ biến!")
            else:
                st.success(f"Tìm thấy **{len(recs)}** sản phẩm phù hợp nhất để kích cầu mua kèm:")
                for r in recs:
                    st.markdown(f"""
                    <div class="gold-rule-card">
                        <h4 style="color:#1D4ED8; margin:0;">🎁 {r['item']}</h4>
                        <p style="margin:5px 0 0 0; color:#374151;">
                            • <b>Độ kích cầu (Lift):</b> <span style="color:#DC2626; font-weight:bold;">{r['lift']:.2f}x</span> (Khách mua giỏ này có khả năng mua thêm món này gấp {r['lift']:.1f} lần người bình thường)<br>
                            • <b>Độ tin cậy (Confidence):</b> <b>{r['confidence']*100:.1f}%</b> | <b>Độ phổ biến (Support):</b> {r['support']*100:.2f}%<br>
                            • <i>Dựa trên luật liên kết với:</i> <code>{' + '.join(r['based_on'])}</code>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

# ==========================================
# TAB 3: GOLD RULES & NETWORK GRAPH
# ==========================================
with tab3:
    st.subheader("🏆 Danh sách Top 20 Luật Vàng (Gold Rules) Giá trị nhất")
    st.markdown("Các luật đã được lọc bỏ luật ngẫu nhiên ($Lift \\approx 1$), khử luật dư thừa và sắp xếp theo độ kích cầu:")

    if not df_gold.empty:
        search_kw = st.text_input("🔍 Tìm kiếm theo tên sản phẩm trong luật:", "")
        df_show = df_gold.copy()
        if search_kw:
            df_show = df_show[df_show['Rule_Display'].str.contains(search_kw.upper())]
        st.dataframe(df_show, use_container_width=True)

    st.markdown("---")
    st.subheader("🕸️ Đồ thị Mạng lưới Liên kết Sản phẩm (Association Network)")
    st.caption("Biểu diễn mối tương quan giữa các sản phẩm có chỉ số Lift cao nhất:")

    if not df_gold.empty:
        # Xây dựng đồ thị NetworkX
        G = nx.Graph()
        top_rules_graph = df_gold.head(12)
        for _, row in top_rules_graph.iterrows():
            ant = row['Antecedent_Text']
            con = row['Consequent_Text']
            lift_val = float(row['Lift'])
            G.add_edge(ant, con, weight=lift_val)

        pos = nx.spring_layout(G, k=0.8, seed=42)
        
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#9CA3AF'),
            hoverinfo='none',
            mode='lines'
        )

        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                color='#3B82F6',
                size=16,
                line=dict(width=2, color='#1E3A8A')
            )
        )

        fig_net = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=20),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        template="plotly_white"
                    ))
        st.plotly_chart(fig_net, use_container_width=True)
