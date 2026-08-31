# ============================================================
# PURE IDENTITY & USER-LEVEL FEATURE ENGINEERING (CHO R5.2)
# ============================================================

import numpy as np
import pandas as pd
from tqdm import tqdm

# 🚀 SỬA 1: Đổi đường dẫn thành file Session của r5.2
print("1. Đang tải dữ liệu Session r5.2...")
# ĐIỀN ĐƯỜNG DẪN R5.2 CỦA BẠN VÀO ĐÂY:
df = pd.read_csv("/kaggle/input/datasets/phanthanhhoang/r5-2-session/sessionr5.2.csv")

print("2. Đang tính toán các chỉ số thống kê (Pre-computing)...")
peer_metrics = ['n_usb', 'n_http', 'n_file', 'n_email']
role_stats = df.groupby('role')[peer_metrics].agg(['mean', 'std']).fillna(1e-8)

def get_role_stat(role, metric, stat_type):
    # Dùng try-except để đề phòng r5.2 có role lạ chưa được tính toán
    try:
        return role_stats.loc[role, (metric, stat_type)]
    except KeyError:
        return 0.0 if stat_type == 'mean' else 1.0

team_sizes = df.groupby('team')['user'].nunique().to_dict()
dept_sizes = df.groupby('dept')['user'].nunique().to_dict()

print("3. Bắt đầu trích xuất đặc trưng User-Level...")
grouped = df.groupby('user')
user_features = []

for user, g in tqdm(grouped, total=len(grouped)):
    feat = {'user': user}
    first_row = g.iloc[0]

    # --- A. ĐẶC TRƯNG TĨNH ---
    # Bỏ b_unit ra khỏi đây để khớp với schema bạn yêu cầu
    STATIC_COLS = ['role', 'f_unit', 'dept', 'team', 'O', 'C', 'E', 'A', 'N']
    for col in STATIC_COLS:
        feat[col] = first_row[col]

    # --- B. PRIVILEGE & POSITION ---
    feat['is_admin'] = int(first_row['ITAdmin'])
    # Đã bỏ feat['is_manager'] 
    feat['team_size'] = team_sizes.get(first_row['team'], 1)
    feat['dept_size'] = dept_sizes.get(first_row['dept'], 1)

    # --- C. PEER-RELATIVE ---
    user_role = first_row['role']
    for metric in peer_metrics:
        user_mean = g[metric].mean()
        role_mean = get_role_stat(user_role, metric, 'mean')
        role_std = get_role_stat(user_role, metric, 'std')
        feat[f'peer_{metric.replace("n_", "")}_z'] = (user_mean - role_mean) / (role_std + 1e-8)

    # --- D. LONG-TERM RISK TENDENCY ---
    feat['afterhour_ratio'] = g['isafterhour'].mean()
    feat['weekend_ratio'] = g['isweekend'].mean()
    feat['weekend_afterhour_ratio'] = g['isweekendafterhour'].mean()

    # --- E. RISK CATEGORY ---
    total_http = g['n_http'].sum() + 1e-8
    feat['job_site_ratio'] = g['http_n_jobf'].sum() / total_http
    feat['leak_site_ratio'] = g['http_n_leakf'].sum() / total_http
    feat['hack_site_ratio'] = g['http_n_hackf'].sum() / total_http

    # --- F. LABEL ---
    feat['insider'] = g['insider'].max()
    
    user_features.append(feat)

print("\n4. Đang dọn dẹp và chốt Schema...")
df_user = pd.DataFrame(user_features)

# 🚀 SỬA 2: Ép cứng đúng 24 cột này, không được thêm bớt
EXPECTED_COLUMNS = [
    'user', 'role', 'f_unit', 'dept', 'team', 
    'O', 'C', 'E', 'A', 'N', 'is_admin', 'team_size', 'dept_size', 
    'peer_usb_z', 'peer_http_z', 'peer_file_z', 'peer_email_z', 
    'afterhour_ratio', 'weekend_ratio', 'weekend_afterhour_ratio', 
    'job_site_ratio', 'leak_site_ratio', 'hack_site_ratio', 'insider'
]

# Lấy đúng danh sách cột (Tránh tình trạng code thừa sinh ra cột lạ)
df_user = df_user[EXPECTED_COLUMNS]

# 🚀 SỬA 3: TUYỆT ĐỐI KHÔNG DÙNG LỆNH df_user.drop(columns=constant_cols) NỮA
# Kệ các cột có giá trị không đổi, Neural Network tự biết cách đánh trọng số = 0 cho nó.

save_path = "/kaggle/working/user_static_r52.csv"
df_user.to_csv(save_path, index=False)

print(f"Kích thước cuối cùng: {df_user.shape} (Phải đúng 24 cột)")
print(f"\n=> HOÀN THÀNH! File đã lưu tại: {save_path}")