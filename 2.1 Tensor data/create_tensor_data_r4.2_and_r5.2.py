# ==============================================================================
# 🚀 TRI-VIEW MASTER PIPELINE: ĐỒNG BỘ HÓA SESSION, EVENT VÀ USER-DAY 
# ==============================================================================
import os
import gc
import random
import warnings
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

def set_all_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"🔒 [HỆ THỐNG] Đã khóa chặt toàn bộ tính ngẫu nhiên với SEED = {seed}")

set_all_seeds(42)
warnings.filterwarnings('ignore')

# ĐƯỜNG DẪN DỮ LIỆU
PATH_EVENT = '/kaggle/input/datasets/phanthanhhoang/event-level-r4-2/event_r4.2.parquet' 
PATH_SESSION = '/kaggle/input/datasets/phanthanhhoang/cert-r42-session/session_r4.2.parquet'
PATH_USERDAY = '/kaggle/input/datasets/doanthimo80/user-level/user-level-r4.2.parquet' 
SAVE_DIR = "./tensor_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==============================================================================
# 🧩 1. ĐỌC VÀ TIỀN XỬ LÝ TẬP EVENT (LAZY POLARS)
# ==============================================================================
print("\n[1/5] Đang scan tập EVENT (Lazy Loading)...")
df_ev_lazy = pl.scan_parquet(PATH_EVENT).with_columns(pl.col("insider").cast(pl.Int8))

# Ép kiểu nhẹ gọn ngay trong Lazy Graph
time_cast_exprs = []
for c in ["time_diff_sec", "time_diff_capped"]:
    # Kiểm tra cột có tồn tại không (Phải load 1 dòng để lấy schema)
    if c in df_ev_lazy.columns: time_cast_exprs.append(pl.col(c).cast(pl.Int32))
for c in ["time_diff_log", "hour_sin", "hour_cos", "event_rate"]:
    if c in df_ev_lazy.columns: time_cast_exprs.append(pl.col(c).cast(pl.Float32))
if time_cast_exprs:
    df_ev_lazy = df_ev_lazy.with_columns(time_cast_exprs)

# --- Category Encoding (NLP Token Style) ---
print(" -> Đang tạo Category Encoding (Mã hóa Integer) cho Event...")
categorical_cols_ev = ['act', 'file_type', 'http_type', 'time']
cat_cols_to_keep_ev = [c for c in categorical_cols_ev if c in df_ev_lazy.columns]

vocab_sizes = {} 
if cat_cols_to_keep_ev:
    for col in cat_cols_to_keep_ev:
        uniques = (
            df_ev_lazy.select(pl.col(col).unique())
            .collect(streaming=True)[col]
            .drop_nulls()
            .to_list()
        )
        mapping = {v: i + 1 for i, v in enumerate(uniques)}
        vocab_sizes[col] = len(mapping) + 1 
        df_ev_lazy = df_ev_lazy.with_columns(
            pl.col(col).replace(mapping, default=0).cast(pl.Int16)
        )
print("📌 Kích thước từ điển (Vocab Size) truyền cho Model:", vocab_sizes)

df_ev_lazy = df_ev_lazy.with_columns(
    (pl.col("time_stamp").cast(pl.Int64) / 1e9).cast(pl.Float64)
)
gc.collect()

exclude_cols_ev = ['actid', 'pcid', 'insider', 'time_stamp', 'user', 'day', 'week']
feature_cols_ev = [col for col in df_ev_lazy.columns if col not in exclude_cols_ev]

# ==============================================================================
# 🧩 2. ĐỌC VÀ TIỀN XỬ LÝ TẬP SESSION (PANDAS)
# ==============================================================================
print("\n[2/5] Đang xử lý tập SESSION...")
df_sess = pd.read_parquet(PATH_SESSION)
df_sess['insider'] = df_sess['insider'].astype(int)
df_sess = df_sess.sort_values(by=['user', 'starttime']).reset_index(drop=True)

eps = 1e-5
if 'duration' in df_sess.columns:
    df_sess['duration_quantile'] = pd.qcut(df_sess['duration'], q=[0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1], labels=False, duplicates='drop')
if 'n_allact' in df_sess.columns:
    if 'n_http' in df_sess.columns: df_sess['http_ratio'] = df_sess['n_http'] / (df_sess['n_allact'] + eps)
    if 'n_email' in df_sess.columns: df_sess['email_ratio'] = df_sess['n_email'] / (df_sess['n_allact'] + eps)
    if 'n_file' in df_sess.columns: df_sess['file_ratio'] = df_sess['n_file'] / (df_sess['n_allact'] + eps)
    if 'n_usb' in df_sess.columns: df_sess['usb_ratio'] = df_sess['n_usb'] / (df_sess['n_allact'] + eps)

new_feature_cols = ['duration_quantile', 'http_ratio', 'email_ratio', 'file_ratio', 'usb_ratio']
existing_new_features = [col for col in new_feature_cols if col in df_sess.columns]
df_sess[existing_new_features] = df_sess[existing_new_features].fillna(0)

print(" -> Đang tính Z-Score nhân quả cho Session...")
activity_cols = ['n_http', 'n_email', 'n_file', 'n_usb']
split_index = int(len(df_sess) * 0.80)

for act_col in activity_cols:
    if act_col in df_sess.columns:
        mean_col = df_sess.groupby('user')[act_col].transform(lambda x: x.expanding().mean().shift(1))
        std_col = df_sess.groupby('user')[act_col].transform(lambda x: x.expanding().std().shift(1))
        
        safe_global_mean = df_sess[act_col].iloc[:split_index].mean()
        safe_global_std = df_sess[act_col].iloc[:split_index].std()
        
        mean_col = mean_col.fillna(safe_global_mean)
        std_col = std_col.fillna(safe_global_std).replace(0, 1.0)
        df_sess[f'{act_col}_zscore'] = (df_sess[act_col] - mean_col) / std_col

if 'isafterhour' in df_sess.columns:
    df_sess['user_afterhour_ratio'] = df_sess.groupby('user')['isafterhour'].transform(lambda x: x.expanding().mean().shift(1)).fillna(0)
    df_sess['afterhour_anomaly_score'] = df_sess['isafterhour'] - df_sess['user_afterhour_ratio']

exclude_cols_sess = ['insider', 'starttime', 'endtime', 'sessionid', 'user', 'day', 'week']
feature_cols_sess = [col for col in df_sess.columns if col not in exclude_cols_sess]

# ==============================================================================
# ⚔️ 3. ĐỒNG BỘ CHIA TẬP DEV / TEST (TỶ LỆ 80/20)
# ==============================================================================
print("\n[3/5] Đang phân bổ dữ liệu chuỗi thời gian: Dev (80%) và Test (20%)...")

df_sess = df_sess.sort_values(by=['starttime']).reset_index(drop=True)
test_idx = int(len(df_sess) * 0.80)

df_sess_dev = df_sess.iloc[:test_idx].copy()
df_sess_test_raw = df_sess.iloc[test_idx:].copy()

known_insiders_dev = set(df_sess_dev[df_sess_dev['insider'] != 0]['user'])
df_sess_test = df_sess_test_raw[~df_sess_test_raw['user'].isin(known_insiders_dev)].copy()

time_split_test = df_sess['starttime'].iloc[test_idx]
valid_test_users = df_sess_test['user'].unique().tolist()

# Đổi tên thành df_ev_dev và df_ev_test cho gọn (Bản chất vẫn là Polars Dataframe)
df_ev_dev = (
    df_ev_lazy
    .filter(pl.col('time_stamp') < time_split_test)
    .sort(["user", "time_stamp"])
    .collect(streaming=True)
)
df_ev_test = (
    df_ev_lazy
    .filter(pl.col('time_stamp') >= time_split_test)
    .sort(["user", "time_stamp"])
    .filter(pl.col('user').is_in(valid_test_users))
    .collect(streaming=True)
)
del df_ev_lazy 
gc.collect()
print(f" -> Đã chia Session: Dev ({len(df_sess_dev)}) | Test ({len(df_sess_test)})")

# ==============================================================================
# 🧠 4. ĐỌC VÀ TIỀN XỬ LÝ TẬP USER-DAY (100% POLARS - BẤT TỬ RAM)
# ==============================================================================
print("\n[4/5] Đang xử lý tập dữ liệu vĩ mô USER-DAY (Chế độ tối ưu RAM)...")
df_ud_pl = pl.read_parquet(PATH_USERDAY)

users_in_dev = list(set(df_sess_dev['user']))
users_in_test = list(set(df_sess_test['user']))
df_ud_dev = df_ud_pl.filter(pl.col("user").is_in(users_in_dev))
df_ud_test = df_ud_pl.filter(pl.col("user").is_in(users_in_test))

del df_ud_pl; gc.collect()

# --- 4.1 Zero Variance User-Day ---
metadata_cols_ud = ['starttime', 'endtime', 'user', 'day_id', 'day', 'week', 'insider']
feature_cols_ud = [c for c in df_ud_dev.columns if c not in metadata_cols_ud]
dead_cols_ud = []
for c in feature_cols_ud:
    if df_ud_dev.select(pl.col(c).n_unique()).item() <= 1:
        dead_cols_ud.append(c)

df_ud_dev = df_ud_dev.drop(dead_cols_ud)
df_ud_test = df_ud_test.drop(dead_cols_ud)
feature_cols_ud = [c for c in feature_cols_ud if c not in dead_cols_ud]

# --- 4.2 Frequency Smoothing User-Day ---
print(" -> Đang làm mịn (Frequency Smoothing) User-Day...")
categorical_cols_ud = ['role', 'b_unit', 'f_unit', 'dept', 'team', 'ITAdmin']
cat_cols_to_keep_ud = [c for c in feature_cols_ud if c in categorical_cols_ud]
total_dev_rows = df_ud_dev.height

for col in cat_cols_to_keep_ud:
    freq_df = (
        df_ud_dev.group_by(col)
        .agg(pl.col(col).count().alias("count"))
        .with_columns((pl.col("count") / total_dev_rows).alias("freq"))
        .with_columns(
            pl.when(pl.col("freq") < 0.01).then(0.01).otherwise(pl.col("freq")).cast(pl.Float32)
        )
        .select([col, "freq"])
    )
    
    df_ud_dev = (
        df_ud_dev.join(freq_df, on=col, how="left")
        .with_columns(pl.col("freq").fill_null(0.01))
        .drop(col).rename({"freq": col})
    )
    
    df_ud_test = (
        df_ud_test.join(freq_df, on=col, how="left")
        .with_columns(pl.col("freq").fill_null(0.01))
        .drop(col).rename({"freq": col})
    )
    del freq_df; gc.collect()

# --- 4.3 Lưu File ---
df_ud_dev.write_parquet(f"{SAVE_DIR}/userday_clean_dev.parquet")
df_ud_test.write_parquet(f"{SAVE_DIR}/userday_clean_test.parquet")
del df_ud_dev, df_ud_test; gc.collect()

# ==============================================================================
# 🧹 4.4 DỌN DẸP ZERO-VARIANCE CHO SESSION VÀ EVENT
# ==============================================================================
print("\n -> Dọn dẹp Zero-variance cho Session và Event...")

# Session (Pandas)
dead_cols_sess = [c for c in feature_cols_sess if df_sess_dev[c].nunique() <= 1]
df_sess_dev.drop(columns=dead_cols_sess, inplace=True, errors='ignore')
df_sess_test.drop(columns=dead_cols_sess, inplace=True, errors='ignore')
feature_cols_sess = [c for c in feature_cols_sess if c not in dead_cols_sess]

# Event (Polars) -> PHẢI DÙNG CÚ PHÁP POLARS
dead_cols_ev = []
for c in feature_cols_ev:
    if df_ev_dev.select(pl.col(c).n_unique()).item() <= 1:
        dead_cols_ev.append(c)

df_ev_dev = df_ev_dev.drop(dead_cols_ev)
df_ev_test = df_ev_test.drop(dead_cols_ev)
feature_cols_ev = [c for c in feature_cols_ev if c not in dead_cols_ev]
gc.collect()

# ==============================================================================
# 📊 4.5 FREQUENCY SMOOTHING SESSION (POLARS JOIN)
# ==============================================================================
print(" -> Đang làm mịn (Frequency Smoothing) Session bằng Polars Join...")
df_sess_dev_pl = pl.from_pandas(df_sess_dev)
df_sess_test_pl = pl.from_pandas(df_sess_test)

categorical_cols_sess = ['pc', 'start_with', 'end_with', 'role', 'b_unit', 'f_unit', 'dept', 'team', 'ITAdmin']
cat_cols_to_keep_sess = [c for c in categorical_cols_sess if c in df_sess_dev_pl.columns]
total_dev_rows = df_sess_dev_pl.height

for col in cat_cols_to_keep_sess:
    freq_df = (
        df_sess_dev_pl
        .group_by(col)
        .agg(pl.len().alias("count"))
        .with_columns(
            (pl.col("count") / total_dev_rows).clip(lower_bound=0.01).cast(pl.Float32).alias("freq")
        )
        .select([col, "freq"])
    )

    df_sess_dev_pl = df_sess_dev_pl.join(freq_df, on=col, how="left").drop(col).rename({"freq": col})
    df_sess_test_pl = df_sess_test_pl.join(freq_df, on=col, how="left").with_columns(pl.col("freq").fill_null(0.01)).drop(col).rename({"freq": col})
    del freq_df; gc.collect()

df_sess_dev = df_sess_dev_pl.to_pandas()
df_sess_test = df_sess_test_pl.to_pandas()
del df_sess_dev_pl, df_sess_test_pl; gc.collect()

# --- 4.6 MAPPING ID NGƯỜI DÙNG ---
print(" -> Đang tạo Mapping ID người dùng...")
unique_users_dev = sorted(df_sess_dev['user'].unique().tolist())
user_mapping_dev = {u: i for i, u in enumerate(unique_users_dev)}

unique_users_test = sorted(df_sess_test['user'].unique().tolist())
user_mapping_test = {u: i for i, u in enumerate(unique_users_test)}

try: del df_sess_test_raw 
except NameError: pass
gc.collect()

# ==============================================================================
# 🧩 5. HÀM TẠO TENSOR ĐỒNG BỘ ĐA TẦNG
# ==============================================================================
def create_and_save_polars_windows(df_sess_pd, df_ev_pl_obj, save_dir, prefix, user_mapping,
                                    session_window=5, event_max_len=895, 
                                    stride_benign=5, stride_attack=1, users_per_chunk=5):
    
    print(f"\n-> Đóng gói mô hình bằng thuật toán SearchSorted ({prefix})...")
    df_sess = pl.from_pandas(df_sess_pd)
    df_ev = df_ev_pl_obj
    
    exclude_sess = ['insider', 'starttime', 'endtime', 'sessionid', 'user', 'day', 'week']
    f_cols_sess = [c for c in df_sess.columns if c not in exclude_sess]
    f_cols_ev = [c for c in df_ev.columns if c not in exclude_cols_ev]
    
    X_sess_list, X_ev_list, y_list, w_list, uid_list, day_list = [], [], [], [], [], []
    chunk_idx, user_count = 0, 0
    unique_users = df_sess["user"].unique().to_list()
    
    for user in tqdm(unique_users):
        u_sess = df_sess.filter(pl.col("user") == user)
        u_ev = df_ev.filter(pl.col("user") == user)
        
        if u_sess.height < session_window or u_ev.height == 0: continue
            
        current_user_id = user_mapping.get(user, -1)
        if current_user_id == -1: continue 
            
        sess_feats = u_sess.select(f_cols_sess).to_numpy()
        sess_times_start = u_sess["starttime"].to_numpy()
        sess_times_end = u_sess["endtime"].to_numpy()
        sess_labels = u_sess["insider"].to_numpy()
        sess_weeks = u_sess["week"].to_numpy()
        sess_days = u_sess["day"].to_numpy() 
        
        ev_feats = u_ev.select(f_cols_ev).to_numpy()
        ev_times = u_ev["time_stamp"].to_numpy() 
        
        for i in range(len(u_sess) - session_window + 1):
            window_labels = sess_labels[i : i + session_window]
            max_label = np.max(window_labels)
            
            if max_label > 0:
                if i % stride_attack != 0: continue
            else:
                if i % stride_benign != 0: continue
            
            start_t = sess_times_start[i]
            end_t = sess_times_end[i + session_window - 1]
            
            left_idx = np.searchsorted(ev_times, start_t, side='left')
            right_idx = np.searchsorted(ev_times, end_t, side='right')
            sub_ev = ev_feats[left_idx:right_idx]
            
            if len(sub_ev) > event_max_len:
                sub_ev = sub_ev[-event_max_len:]
            elif len(sub_ev) < event_max_len:
                padding = np.zeros((event_max_len - len(sub_ev), len(f_cols_ev)))
                sub_ev = np.vstack((padding, sub_ev)) if len(sub_ev) > 0 else padding
                
            X_ev_list.append(sub_ev)
            X_sess_list.append(sess_feats[i : i + session_window])
            y_list.append(max_label)
            w_list.append(sess_weeks[i + session_window - 1])
            uid_list.append(current_user_id) 
            day_list.append(sess_days[i + session_window - 1]) 
            
        user_count += 1
        if user_count >= users_per_chunk:
            if len(X_sess_list) > 0:
                np.savez_compressed(f"{save_dir}/{prefix}_chunk_{chunk_idx}.npz",
                                    X_sess=np.array(X_sess_list, dtype=np.float16),
                                    X_ev=np.array(X_ev_list, dtype=np.float16),
                                    y=np.array(y_list, dtype=np.int8),
                                    w=np.array(w_list, dtype=np.int16),
                                    user_id=np.array(uid_list, dtype=np.int16),
                                    day=np.array(day_list, dtype=np.int16))
            X_sess_list, X_ev_list, y_list, w_list, uid_list, day_list = [], [], [], [], [], []
            user_count = 0
            chunk_idx += 1
            gc.collect()
            
    if len(X_sess_list) > 0:
        np.savez_compressed(f"{save_dir}/{prefix}_chunk_{chunk_idx}.npz",
                            X_sess=np.array(X_sess_list, dtype=np.float16),
                            X_ev=np.array(X_ev_list, dtype=np.float16),
                            y=np.array(y_list, dtype=np.int8),
                            w=np.array(w_list, dtype=np.int16),
                            user_id=np.array(uid_list, dtype=np.int16),
                            day=np.array(day_list, dtype=np.int16))
        
    print(f"✅ Hoàn tất tập {prefix.upper()}! Đã xuất {chunk_idx + 1} file chunks.")

# # ==============================================================================
# # 🚀 6. THỰC THI CUỘN DỮ LIỆU ĐA TẦNG VÀ GIẢI PHÓNG RAM
# # ==============================================================================
print("\n[5/5] TIẾN HÀNH CUỘN WINDOW CHO TẬP DEV VÀ TEST ĐỘC LẬP...")
SESSION_W = 5
EVENT_MAX_LEN = 895

print(" -> Đang sắp xếp lại Session Dev và Test theo User -> Time...")
df_sess_dev = df_sess_dev.sort_values(by=['user', 'starttime']).reset_index(drop=True)
df_sess_test = df_sess_test.sort_values(by=['user', 'starttime']).reset_index(drop=True)

df_ev_dev = df_ev_dev.sort(by=['user', 'time_stamp'])
df_ev_test = df_ev_test.sort(by=['user', 'time_stamp'])

create_and_save_polars_windows(
    df_sess_dev, df_ev_dev, save_dir=SAVE_DIR, prefix="dev",
    user_mapping=user_mapping_dev,
    session_window=SESSION_W, event_max_len=EVENT_MAX_LEN, 
    stride_benign=1, stride_attack=1, users_per_chunk=50 # GIẢM CHUNK ĐỂ BẢO VỆ RAM
)

del df_sess_dev, df_ev_dev
gc.collect()

create_and_save_polars_windows(
    df_sess_test, df_ev_test, save_dir=SAVE_DIR, prefix="test",
    user_mapping=user_mapping_test,
    session_window=SESSION_W, event_max_len=EVENT_MAX_LEN, 
    stride_benign=1, stride_attack=1, users_per_chunk=50 # GIẢM CHUNK ĐỂ BẢO VỆ RAM
)

print("\n🧹 HỆ THỐNG: Đang quét sạch hoàn toàn các biến thô ra khỏi RAM...")
del  df_sess_test, df_ev_test
gc.collect()




# ==============================================================================
# 🌟 KÈM THÊM: CHUẨN BỊ R5.2 (CHIA SẴN DEV 80% / TEST 20% DỰ PHÒNG TƯƠNG LAI)
# ==============================================================================
print("\n" + "="*75)
print("🚀 [EXTRA] ĐANG CHUẨN BỊ TẬP R5.2 (TIME-AWARE SPLIT 80/20 + Z-SCORE + ALIGNMENT)")
print("="*75)

PATH_SESSION_R52 = '/kaggle/input/datasets/phanthanhhoang/r5-2-session/sessionr5.2.csv'
PATH_USERDAY_R52 = '/kaggle/input/datasets/doanthimo80/user-r52/user_static_r52.csv'
SAVE_DIR_R52 = "./tensor_data_r52"
os.makedirs(SAVE_DIR_R52, exist_ok=True)

# ------------------------------------------------------------------------------
# 1. ĐỌC VÀ TIỀN XỬ LÝ SESSION R5.2 (BAO GỒM Z-SCORE)
# ------------------------------------------------------------------------------
print(" -> Đang đọc và tính Z-Score Session r5.2...")
df_sess_r52 = pd.read_csv(PATH_SESSION_R52)
df_sess_r52['insider'] = df_sess_r52['insider'].astype(int)

# Sắp xếp chặt chẽ theo user và thời gian để tính Z-Score cho đúng
df_sess_r52 = df_sess_r52.sort_values(by=['user', 'starttime']).reset_index(drop=True)

# 🔥 1.1 TÍNH Z-SCORE VÀ TỶ LỆ NGOÀI GIỜ (GIỐNG HỆT R4.2)
activity_cols = ['n_http', 'n_email', 'n_file', 'n_usb']
split_index_r52 = int(len(df_sess_r52) * 0.80)

for act_col in activity_cols:
    if act_col in df_sess_r52.columns:
        mean_col = df_sess_r52.groupby('user')[act_col].transform(lambda x: x.expanding().mean().shift(1))
        std_col = df_sess_r52.groupby('user')[act_col].transform(lambda x: x.expanding().std().shift(1))
        
        safe_global_mean = df_sess_r52[act_col].iloc[:split_index_r52].mean()
        safe_global_std = df_sess_r52[act_col].iloc[:split_index_r52].std()
        
        mean_col = mean_col.fillna(safe_global_mean)
        std_col = std_col.fillna(safe_global_std).replace(0, 1.0)
        df_sess_r52[f'{act_col}_zscore'] = (df_sess_r52[act_col] - mean_col) / std_col

if 'isafterhour' in df_sess_r52.columns:
    df_sess_r52['user_afterhour_ratio'] = df_sess_r52.groupby('user')['isafterhour'].transform(lambda x: x.expanding().mean().shift(1)).fillna(0)
    df_sess_r52['afterhour_anomaly_score'] = df_sess_r52['isafterhour'] - df_sess_r52['user_afterhour_ratio']

# Trả lại thứ tự sắp xếp chuẩn Time-aware để cắt Dev/Test
df_sess_r52 = df_sess_r52.sort_values(by=['starttime']).reset_index(drop=True)

# Cắt Dev/Test
df_sess_r52_dev = df_sess_r52.iloc[:split_index_r52].copy()
df_sess_r52_test_raw = df_sess_r52.iloc[split_index_r52:].copy()

# Loại bỏ Hacker đã lộ mặt ở tập Dev ra khỏi tập Test
known_insiders_dev_r52 = set(df_sess_r52_dev[df_sess_r52_dev['insider'] != 0]['user'])
df_sess_r52_test = df_sess_r52_test_raw[~df_sess_r52_test_raw['user'].isin(known_insiders_dev_r52)].copy()

print(f" -> Đã chia r5.2: Dev ({len(df_sess_r52_dev)} dòng) | Test ({len(df_sess_r52_test)} dòng)")

# ------------------------------------------------------------------------------
# 2. FEATURE ENGINEERING VÀ ÉP KHUÔN CỘT (FEATURE ALIGNMENT)
# ------------------------------------------------------------------------------
print(" -> Đang Feature Alignment (Ép khuôn 100% giống r4.2)...")

# Tập hợp các cột metadata cần thiết để giữ lại cho quá trình cuộn Window
metadata_cols = ['insider', 'starttime', 'endtime', 'sessionid', 'user', 'day', 'week']
# Tạo danh sách đích danh các cột cần giữ (đảm bảo không bị trùng)
target_columns = list(dict.fromkeys(feature_cols_sess + metadata_cols))

for df_temp in [df_sess_r52_dev, df_sess_r52_test]:
    # 2.1 Tính Quantile và Ratio
    if 'duration' in df_temp.columns:
        df_temp['duration_quantile'] = pd.qcut(df_temp['duration'].rank(method='first'), q=[0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1], labels=False)
    if 'n_allact' in df_temp.columns:
        if 'n_http' in df_temp.columns: df_temp['http_ratio'] = df_temp['n_http'] / (df_temp['n_allact'] + eps)
        if 'n_email' in df_temp.columns: df_temp['email_ratio'] = df_temp['n_email'] / (df_temp['n_allact'] + eps)
        if 'n_file' in df_temp.columns: df_temp['file_ratio'] = df_temp['n_file'] / (df_temp['n_allact'] + eps)
        if 'n_usb' in df_temp.columns: df_temp['usb_ratio'] = df_temp['n_usb'] / (df_temp['n_allact'] + eps)
    df_temp[existing_new_features] = df_temp[existing_new_features].fillna(0)
    
    # 🔥 2.2 ÉP KHUÔN CỘT (ALIGMENT)
    # Bù cột thiếu bằng số 0
    missing_cols = [c for c in target_columns if c not in df_temp.columns]
    for c in missing_cols:
        df_temp[c] = 0
        
    # Loại bỏ các cột thừa (cột có ở r5.2 nhưng r4.2 không có)
    extra_cols = [c for c in df_temp.columns if c not in target_columns]
    df_temp.drop(columns=extra_cols, inplace=True, errors='ignore')

# Mapping User
unique_users_dev_r52 = sorted(df_sess_r52_dev['user'].unique().tolist())
user_mapping_dev_r52 = {u: i for i, u in enumerate(unique_users_dev_r52)}

unique_users_test_r52 = sorted(df_sess_r52_test['user'].unique().tolist())
user_mapping_test_r52 = {u: i for i, u in enumerate(unique_users_test_r52)}

# ------------------------------------------------------------------------------
# 3. ĐỌC VÀ LƯU USER-DAY R5.2 (ÉP KHUÔN CỘT TĨNH)
# ------------------------------------------------------------------------------
print(" -> Đang lọc và ép khuôn User-Day r5.2...")
df_ud_r52_pl = pl.read_csv(PATH_USERDAY_R52)

df_ud_r52_dev = df_ud_r52_pl.filter(pl.col("user").is_in(unique_users_dev_r52))
df_ud_r52_test = df_ud_r52_pl.filter(pl.col("user").is_in(unique_users_test_r52))

# Cắt cột chết và ép khuôn cho User y hệt r4.2 (Dùng feature_cols_ud của r4.2)
target_ud_cols = list(dict.fromkeys(feature_cols_ud + metadata_cols_ud))

def align_user_cols(df_pl, target_cols):
    current_cols = df_pl.columns
    # Thêm cột thiếu (mặc định = 0.0)
    missing = [c for c in target_cols if c not in current_cols]
    if missing:
        df_pl = df_pl.with_columns([pl.lit(0.0).alias(c) for c in missing])
    # Bỏ cột thừa
    return df_pl.select([c for c in target_cols if c in df_pl.columns])

df_ud_r52_dev = align_user_cols(df_ud_r52_dev, target_ud_cols)
df_ud_r52_test = align_user_cols(df_ud_r52_test, target_ud_cols)

df_ud_r52_dev.write_parquet(f"{SAVE_DIR_R52}/userday_clean_dev_r52.parquet")
df_ud_r52_test.write_parquet(f"{SAVE_DIR_R52}/userday_clean_test_r52.parquet")
del df_ud_r52_pl, df_ud_r52_dev, df_ud_r52_test; gc.collect()

# ------------------------------------------------------------------------------
# 4. HÀM CUỘN WINDOW SESSION (BỎ EVENT)
# ------------------------------------------------------------------------------
def create_session_only_windows(df_sess_pd, save_dir, prefix, user_mapping, session_window=5, stride_benign=1, stride_attack=1):
    print(f" -> Đang cuộn Window Session ({prefix})...")
    df_sess = pl.from_pandas(df_sess_pd)
    # Lúc này df_sess_pd đã được ép khuôn, nên f_cols_sess sẽ khớp 100% với r4.2
    exclude_sess = ['insider', 'starttime', 'endtime', 'sessionid', 'user', 'day', 'week']
    f_cols_sess = [c for c in df_sess.columns if c not in exclude_sess]
    
    X_sess_list, y_list, w_list, uid_list, day_list = [], [], [], [], []
    chunk_idx, user_count = 0, 0
    unique_users = df_sess["user"].unique().to_list()
    
    for user in tqdm(unique_users, desc=f"Rolling {prefix}"):
        u_sess = df_sess.filter(pl.col("user") == user).sort("starttime")
        if u_sess.height < session_window: continue
            
        current_user_id = user_mapping.get(user, -1)
        if current_user_id == -1: continue 
            
        sess_feats = u_sess.select(f_cols_sess).to_numpy()
        sess_labels = u_sess["insider"].to_numpy()
        sess_weeks = u_sess["week"].to_numpy()
        sess_days = u_sess["day"].to_numpy() 
        
        for i in range(len(u_sess) - session_window + 1):
            window_labels = sess_labels[i : i + session_window]
            max_label = np.max(window_labels)
            
            if max_label > 0 and i % stride_attack != 0: continue
            if max_label == 0 and i % stride_benign != 0: continue
            
            X_sess_list.append(sess_feats[i : i + session_window])
            y_list.append(max_label)
            w_list.append(sess_weeks[i + session_window - 1])
            uid_list.append(current_user_id) 
            day_list.append(sess_days[i + session_window - 1]) 
            
        user_count += 1
        if user_count >= 100: 
            if len(X_sess_list) > 0:
                np.savez_compressed(f"{save_dir}/{prefix}_chunk_{chunk_idx}.npz",
                                    X_sess=np.array(X_sess_list, dtype=np.float16),
                                    y=np.array(y_list, dtype=np.int8),
                                    w=np.array(w_list, dtype=np.int16),
                                    user_id=np.array(uid_list, dtype=np.int16),
                                    day=np.array(day_list, dtype=np.int16))
            X_sess_list, y_list, w_list, uid_list, day_list = [], [], [], [], []
            user_count, chunk_idx = 0, chunk_idx + 1
            gc.collect()
            
    if len(X_sess_list) > 0:
        np.savez_compressed(f"{save_dir}/{prefix}_chunk_{chunk_idx}.npz",
                            X_sess=np.array(X_sess_list, dtype=np.float16),
                            y=np.array(y_list, dtype=np.int8),
                            w=np.array(w_list, dtype=np.int16),
                            user_id=np.array(uid_list, dtype=np.int16),
                            day=np.array(day_list, dtype=np.int16))
    print(f"✅ Hoàn tất {prefix}! Đã xuất {chunk_idx + 1} file chunks ra {save_dir}.")

# --- THỰC THI CUỘN CHO DEV VÀ TEST CỦA R5.2 ---
create_session_only_windows(
    df_sess_r52_dev, save_dir=SAVE_DIR_R52, prefix="dev_r52", 
    user_mapping=user_mapping_dev_r52, session_window=SESSION_W, 
    stride_benign=1, stride_attack=1 
)

create_session_only_windows(
    df_sess_r52_test, save_dir=SAVE_DIR_R52, prefix="test_r52", 
    user_mapping=user_mapping_test_r52, session_window=SESSION_W, 
    stride_benign=1, stride_attack=1 
)

print("\n🎉 HOÀN TẤT TOÀN BỘ MASTER PIPELINE! MỌI DỮ LIỆU ĐÃ SẴN SÀNG CHO MỌI KỊCH BẢN!")