# ============================================================
# PURE IDENTITY & USER-LEVEL FEATURE ENGINEERING
# ============================================================

import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# 1. LOAD SESSION-LEVEL DATA
# ============================================================
print("1. Đang tải dữ liệu Session...")
df = pd.read_parquet("/kaggle/input/datasets/phanthanhhoang/cert-r42-session/session_r4.2.parquet")

# ============================================================
# 2. PRE-COMPUTE STATISTICS (TÍNH TRƯỚC BÊN NGOÀI VÒNG LẶP)
# ============================================================
print("2. Đang tính toán các chỉ số thống kê (Pre-computing)...")

# A. Thống kê đồng cấp theo Role
peer_metrics = ['n_usb', 'n_http', 'n_file', 'n_email']
role_stats = df.groupby('role')[peer_metrics].agg(['mean', 'std']).fillna(1e-8)

def get_role_stat(role, metric, stat_type):
    return role_stats.loc[role, (metric, stat_type)]

# B. Tính trước Quy mô Team và Dept (Tránh quét df trong vòng lặp)
team_sizes = df.groupby('team')['user'].nunique().to_dict()
dept_sizes = df.groupby('dept')['user'].nunique().to_dict()

# ============================================================
# 3. TRÍCH XUẤT ĐẶC TRƯNG MỨC USER
# ============================================================
print("3. Bắt đầu trích xuất đặc trưng User-Level...")

grouped = df.groupby('user')
manager_keywords = ['manager', 'director', 'lead', 'chief', 'supervisor', 'head']

user_features = []

for user, g in tqdm(grouped, total=len(grouped)):
    feat = {'user': user}
    
    # Lấy dòng đầu tiên để trích xuất các ĐẶC TRƯNG TĨNH (Không đổi)
    first_row = g.iloc[0]

    # --------------------------------------------------------
    # A. ORGANIZATIONAL & PSYCHOMETRIC IDENTITY (Đặc trưng tĩnh)
    # --------------------------------------------------------
    STATIC_COLS = [
        'role', 'b_unit', 'f_unit', 'dept', 'team', 
        'O', 'C', 'E', 'A', 'N'
    ]
    for col in STATIC_COLS:
        feat[col] = first_row[col]

    # --------------------------------------------------------
    # B. PRIVILEGE & POSITION
    # --------------------------------------------------------
    feat['is_admin'] = int(first_row['ITAdmin'])
    
    role_text = str(first_row['role']).lower()
    feat['is_manager'] = int(any(k in role_text for k in manager_keywords))

    # Tối ưu O(1) tra cứu Dictionary cho Team/Dept size
    feat['team_size'] = team_sizes.get(first_row['team'], 1)
    feat['dept_size'] = dept_sizes.get(first_row['dept'], 1)

    # --------------------------------------------------------
    # C. PEER-RELATIVE (Đặc trưng tương đối so với Đồng cấp)
    # --------------------------------------------------------
    user_role = first_row['role']
    for metric in peer_metrics:
        user_mean = g[metric].mean()
        role_mean = get_role_stat(user_role, metric, 'mean')
        role_std = get_role_stat(user_role, metric, 'std')
        
        # Z-score (Độ lệch pha so với đồng nghiệp)
        feat[f'peer_{metric.replace("n_", "")}_z'] = (user_mean - role_mean) / (role_std + 1e-8)

    # --------------------------------------------------------
    # D. LONG-TERM RISK TENDENCY (Xu hướng rủi ro dài hạn)
    # --------------------------------------------------------
    feat['afterhour_ratio'] = g['isafterhour'].mean()
    feat['weekend_ratio'] = g['isweekend'].mean()
    feat['weekend_afterhour_ratio'] = g['isweekendafterhour'].mean()

    # --------------------------------------------------------
    # E. LONG-TERM RISK CATEGORY TENDENCY (Tỷ lệ web rủi ro)
    # --------------------------------------------------------
    total_http = g['n_http'].sum() + 1e-8
    feat['job_site_ratio'] = g['http_n_jobf'].sum() / total_http
    feat['leak_site_ratio'] = g['http_n_leakf'].sum() / total_http
    feat['hack_site_ratio'] = g['http_n_hackf'].sum() / total_http

    # --------------------------------------------------------
    # F. LABEL
    # --------------------------------------------------------
    feat['insider'] = g['insider'].max()
    
    user_features.append(feat)

# ============================================================
# 4. POST-PROCESSING & SAVE
# ============================================================
print("\n4. Đang dọn dẹp và lưu file...")
df_user = pd.DataFrame(user_features)

print(f"Kích thước ban đầu: {df_user.shape}")

# Xóa các cột hằng số (chỉ có 1 giá trị duy nhất cho mọi user, không có ích cho ML)
constant_cols = [c for c in df_user.columns if df_user[c].nunique() <= 1]
if constant_cols:
    print(f"Bỏ các cột không mang thông tin: {constant_cols}")
    df_user.drop(columns=constant_cols, inplace=True)

print(f"Kích thước cuối cùng: {df_user.shape}")

# Lưu file Parquet
save_path = "/kaggle/working/user-level-pure-identity-r4.2.parquet"
df_user.to_parquet(save_path, index=False)

print(f"\n=> HOÀN THÀNH! File đã lưu tại: {save_path}")