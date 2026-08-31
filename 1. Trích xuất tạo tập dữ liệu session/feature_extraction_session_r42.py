import os, sys
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
import csv
import gc
import re
import time
import shutil
from joblib import Parallel, delayed

BASE_PATH = "/kaggle/input/cert-r4-2/archive" 
LDAP_PATH = os.path.join(BASE_PATH, "LDAP")
ANSWERS_PATH = "/kaggle/input/cert-r4-2-answer/answers" # Hoặc đường dẫn chứa file insiders.csv

def time_convert(inp, mode, real_sd='2010-01-02', sd_monday="2009-12-28"):
    r42_fmt = '%m/%d/%Y %H:%M:%S'
    if mode == 't2dt':
        return datetime.strptime(inp, r42_fmt)
    elif mode == 't2wn': 
        startdate = datetime.strptime(real_sd, '%Y-%m-%d')
        # Tính số ngày chênh lệch và chia 7 để ra số tuần
        return (datetime.strptime(inp, r42_fmt) - startdate).days // 7
    elif mode == 'dt2dn': 
        startdate = datetime.strptime(sd_monday, '%Y-%m-%d')
        return (inp - startdate).days
    elif mode == 'dt2date':
        return inp.strftime("%Y-%m-%d")
    elif mode == 't2date':
        return datetime.strptime(inp, r42_fmt).strftime("%Y-%m-%d")
    return None

def vectorized_is_after_whour(dates):
    # Workhours: 7:30 - 17:30
    # Đổi hết ra phút trong ngày để so sánh số nguyên cho nhanh
    minutes = dates.dt.hour * 60 + dates.dt.minute
    start_min = 7 * 60 + 30  # 450
    end_min = 17 * 60 + 30   # 1050
    return (minutes < start_min) | (minutes > end_min)

def vectorized_is_weekend(dates):
    # Trả về True nếu là T7 (5) hoặc CN (6)
    return dates.dt.dayofweek.isin([5, 6])

# --- DATA PREPROCESSING ---
def combine_by_timerange_pandas(dname = 'r4.2', chunk_size=300000):
    all_columns = ['id', 'date', 'user', 'pc', 'type', 
                   'activity', 'url', 'filename', 'content', 
                   'to', 'cc', 'bcc', 'from', 'size', '#att']

    pa_schema = pa.schema([
        ('id', pa.string()),
        ('date', pa.timestamp('ns')), # Timestamp nanosecond (mặc định của pandas)
        ('user', pa.string()),
        ('pc', pa.string()),
        ('type', pa.string()),
        ('activity', pa.string()),
        ('url', pa.string()),
        ('filename', pa.string()),
        ('content', pa.string()),
        ('to', pa.string()),
        ('cc', pa.string()),
        ('bcc', pa.string()),
        ('from', pa.string()),
        ('size', pa.string()),
        ('#att', pa.string())
    ])
    
    allacts = ['device','email','file', 'http','logon']
    http_path = os.path.join(BASE_PATH, 'http.csv')
    with open(http_path, 'r') as f:
        next(f) # Bỏ qua header
        firstline = f.readline().strip()
    
    firstdate_dt = time_convert(firstline.split(',')[1],'t2dt')
    firstdate_dt = firstdate_dt - timedelta(int(firstdate_dt.strftime("%w")))
    firstdate = time_convert(firstdate_dt, 'dt2date')
    
    act_handles = {act: open(os.path.join(BASE_PATH, act+'.csv'), 'r') for act in allacts}
    for h in act_handles.values(): next(h, None) # skip header
    
    lines = {act: act_handles[act].readline() for act in allacts}
    stop = {act: 0 for act in allacts}
    week_index = 0

    print(f"Start processing from date: {firstdate}")
    
    while sum(stop.values()) < 5:
        thisweek_list = []
        week_file_name = f"DataByWeek/{week_index}.parquet"
        writer = None
        for act in allacts:
            while lines[act]:
                tmp = next(csv.reader([lines[act]]))
                if time_convert(tmp[1], 't2wn', real_sd=firstdate) == week_index:
                    # Map columns based on r4.2 format
                    if act == 'email': cols = ['id', 'date', 'user', 'pc', 'to', 'cc', 'bcc', 'from', 'size', '#att', 'content']
                    elif act in ['logon', 'device']: cols = ['id', 'date', 'user', 'pc', 'activity']
                    elif act in ['http']: cols = ['id', 'date', 'user', 'pc', 'url', 'content']
                    elif act in ['file']: cols = ['id', 'date', 'user', 'pc', 'filename', 'content']
                    
                    entry = dict(zip(cols, tmp))
                    entry['type'] = act
                    thisweek_list.append(entry)
                    lines[act] = act_handles[act].readline()

                    if len(thisweek_list) >= chunk_size:
                        df_chunk = pd.DataFrame(thisweek_list)
                        
                        # Chuẩn hóa cột (Schema Enforcement)
                        for c in all_columns:
                            if c not in df_chunk.columns:
                                df_chunk[c] = None
                        df_chunk = df_chunk[all_columns] # Sắp xếp đúng thứ tự
                        
                        # Xử lý datetime
                        df_chunk['date'] = pd.to_datetime(df_chunk['date'], format="%m/%d/%Y %H:%M:%S")
                        
                        # Chuyển sang PyArrow Table
                        try:
                            table = pa.Table.from_pandas(df_chunk, schema=pa_schema)
                        except Exception as e:
                            print(f"Schema Error at week {week_index}: {e}")
                            # Fallback nếu schema lỗi (hiếm gặp)
                            table = pa.Table.from_pandas(df_chunk)
                        
                        # Khởi tạo writer nếu chưa có (dùng schema của chunk đầu tiên)
                        if writer is None:
                            writer = pq.ParquetWriter(week_file_name, table.schema, compression='snappy')
                        
                        # Ghi chunk và xóa RAM
                        try: writer.write_table(table)
                        except Exception as e:
                            print(f"Error writing chunk at week {week_index}: {e}")

                        del df_chunk, table
                        thisweek_list = [] # Reset buffer
                        gc.collect() # Ép giải phóng RAM
                        
                else: break
            if not lines[act]: stop[act] = 1
        
        if thisweek_list:
            df_chunk = pd.DataFrame(thisweek_list)
            for c in all_columns:
                if c not in df_chunk.columns:
                    df_chunk[c] = None
            df_chunk = df_chunk[all_columns]
            
            df_chunk['date'] = pd.to_datetime(df_chunk['date'], format="%m/%d/%Y %H:%M:%S")
            
            try: table = pa.Table.from_pandas(df_chunk, schema=pa_schema)
            except:
                table = pa.Table.from_pandas(df_chunk)
            
            if writer is None:
                writer = pq.ParquetWriter(week_file_name, table.schema, compression='snappy')
            
            writer.write_table(table)
            del df_chunk, table
            gc.collect()

        # Đóng writer để hoàn tất file tuần
        if writer: writer.close()
            
        print(f"Week {week_index} processed.")
        week_index += 1

def process_user_pc(upd, roles): 
    # Xác định PC nào thuộc về người dùng nào
    upd['sharedpc'] = None
    upd['npc'] = upd['pcs'].apply(lambda x: len(x))
    
    # Trường hợp user chỉ dùng 1 PC duy nhất
    upd.at[upd['npc']==1,'pc'] = upd[upd['npc']==1]['pcs'].apply(lambda x: x[0])
    
    # Xử lý các máy tính dùng chung (multi-user)
    multiuser_pcs = np.concatenate(upd[upd['npc']>1]['pcs'].values).tolist()
    set_multiuser_pc = list(set(multiuser_pcs))
    count = {}
    for pc in set_multiuser_pc:
        count[pc] = multiuser_pcs.count(pc)
        
    for u in upd[upd['npc']>1].index:
        sharedpc = upd.loc[u]['pcs']
        count_u_pc = [count[pc] for pc in upd.loc[u]['pcs']]
        # Chọn PC có ít người dùng nhất làm máy chính
        the_pc = count_u_pc.index(min(count_u_pc))
        upd.at[u,'pc'] = sharedpc[the_pc]
        
        # Nếu không phải ITAdmin, các máy còn lại trong danh sách được coi là sharedpc
        if roles.loc[u] != 'ITAdmin':
            sharedpc.remove(sharedpc[the_pc])
            upd.at[u,'sharedpc']= sharedpc
    return upd

def getuserlist(dname = 'r4.2', psycho = True):
    # Đọc dữ liệu nhân sự từ các file LDAP
    allfiles = [os.path.join(LDAP_PATH, f1) for f1 in os.listdir(LDAP_PATH) if os.path.isfile(os.path.join(LDAP_PATH, f1))]
    alluser = {}
    alreadyFired = []
    
    for file in allfiles:
        filename_only = os.path.basename(file) # Lấy tên file (ví dụ: 2010-01.csv)
        af = (pd.read_csv(file, delimiter=',')).values
        employeesThisMonth = []    
        for i in range(len(af)):
            employeesThisMonth.append(af[i][1])
            if af[i][1] not in alluser:
                # Dùng filename_only để split lấy ngày tháng
                alluser[af[i][1]] = af[i][0:1].tolist() + af[i][2:].tolist() + [filename_only.split('.')[0] , np.nan]
        firedEmployees = list(set(alluser.keys()) - set(alreadyFired) - set(employeesThisMonth))
        alreadyFired = alreadyFired + firedEmployees
        for e in firedEmployees:
            alluser[e][-1] = filename_only.split('.')[0]
    
    # Thêm dữ liệu tâm lý học (O-C-E-A-N)
    psycho_path = os.path.join(BASE_PATH, "psychometric.csv")
    if psycho and os.path.isfile(psycho_path):
        p_score = pd.read_csv(psycho_path, delimiter = ',').values
        for id in range(len(p_score)):
            alluser[p_score[id,1]] = alluser[p_score[id,1]] + list(p_score[id,2:])
        df = pd.DataFrame.from_dict(alluser, orient='index')
        df.columns = ['uname', 'email', 'role', 'b_unit', 'f_unit', 'dept', 'team', 'sup','wstart', 'wend', 'O', 'C', 'E', 'A', 'N']
    else:
        df = pd.DataFrame.from_dict(alluser, orient='index')
        df.columns = ['uname', 'email', 'role', 'b_unit', 'f_unit', 'dept', 'team', 'sup', 'wstart', 'wend']

    # Chuyển đổi tên người quản lý (supervisor) thành index
    for i in df.index:
        if type(df.loc[i]['sup']) == str:
            sup_matches = df[df['uname'] == df.loc[i]['sup']].index
            sup = sup_matches[0] if len(sup_matches) > 0 else None
        else:
            sup = None
        df.at[i, 'sup'] = sup
        
    # Xác định PC dựa trên log 2 tuần đầu tiên
    w1 = pd.read_parquet("DataByWeek/0.parquet")
    w2 = pd.read_parquet("DataByWeek/1.parquet")
    user_pc_dict = pd.DataFrame(index=df.index)
    user_pc_dict['pcs'] = None  
  
    for u in df.index:
        pc = list(set(w1[w1['user']==u]['pc']) & set(w2[w2['user']==u]['pc']))
        user_pc_dict.at[u,'pcs'] = pc
        
    upd = process_user_pc(user_pc_dict, df['role'])
    df['pc'] = upd['pc']
    df['sharedpc'] = upd['sharedpc']
    return df

def get_mal_userdata(data = 'r4.2', usersdf = None):
    # Lọc danh sách kẻ nội gián cho tập r4.2
    insider_path = os.path.join(ANSWERS_PATH, "insiders.csv")
    listmaluser = pd.read_csv(insider_path)
    listmaluser['dataset'] = listmaluser['dataset'].apply(lambda x: str(x))
    listmaluser = listmaluser[listmaluser['dataset'] == "4.2"]
    
    # Chuyển đổi thời gian bắt đầu/kết thúc sang định dạng datetime
    listmaluser['start'] = pd.to_datetime(listmaluser['start'], format="%m/%d/%Y %H:%M:%S")
    listmaluser['end'] = pd.to_datetime(listmaluser['end'], format="%m/%d/%Y %H:%M:%S")
    
    if usersdf is None:
        usersdf = getuserlist(data)
        
    usersdf['malscene'] = 0
    usersdf['mstart'] = None
    usersdf['mend'] = None
    usersdf['malacts'] = None
    
    for i in listmaluser.index:
        u_id = listmaluser['user'][i]
        usersdf.loc[u_id, 'mstart'] = listmaluser['start'][i]
        usersdf.loc[u_id, 'mend'] = listmaluser['end'][i]
        usersdf.loc[u_id, 'malscene'] = listmaluser['scenario'][i]
        
        # Đọc chi tiết các hành động độc hại từ folder đáp án r4.2
        scenario_num = str(listmaluser['scenario'][i]) 
        folder_name = f"r4.2-{scenario_num}" 
        file_name = listmaluser['details'][i]
        mal_file_path = os.path.join(ANSWERS_PATH, folder_name, file_name)
        try:
            with open(mal_file_path, 'r') as f:
                malacts = f.read().strip().split("\n")

            if malacts and 'id' in malacts[0].lower(): malacts = malacts[1:]
            malacts = [x.split(',') for x in malacts]
            mal_users = np.array([x[3].strip('"') for x in malacts])
            mal_act_ids = np.array([x[1].strip('"') for x in malacts])
        
            usersdf.at[u_id, 'malacts'] = mal_act_ids[mal_users == u_id]
        
        except FileNotFoundError:
            print(f"LỖI: Không tìm thấy file tại {mal_file_path}")
            usersdf.at[u_id, 'malacts'] = []
                    
    return usersdf

# --- FEATURE EXTRACTION (Focus: r4.2) ---
def vectorized_email_process(df_email):
    # Nếu DataFrame rỗng thì trả về rỗng ngay
    if len(df_email) == 0: return pd.DataFrame()
    
    # 1. Chuẩn bị dữ liệu: Ép kiểu string và xử lý NaN
    # Lưu ý: Cần copy để không ảnh hưởng dữ liệu gốc
    to_col = df_email['to'].fillna('').astype(str)
    cc_col = df_email['cc'].fillna('').astype(str)
    bcc_col = df_email['bcc'].fillna('').astype(str)
    
    # Hàm phụ: Đếm số lượng email trong chuỗi (dựa vào số dấu chấm phẩy ';')
    # Logic: Nếu chuỗi rỗng -> 0. Nếu không -> đếm ';' + 1
    def count_recipients(series_str):
        # Trả về 0 nếu chuỗi rỗng hoặc chỉ có khoảng trắng
        # Ngược lại đếm số dấu ';' và cộng 1
        return np.where(series_str.str.strip() == '', 0, series_str.str.count(';') + 1)
    
    # Hàm phụ: Đếm số lượng email NỘI BỘ (@dtaa.com)
    def count_internal(series_str):
        return series_str.str.count('dtaa.com')

    # 2. Tính toán số lượng người nhận (n_des, n_bccdes)
    n_to = count_recipients(to_col)
    n_cc = count_recipients(cc_col)
    n_bcc = count_recipients(bcc_col)
    
    n_des = n_to + n_cc + n_bcc
    
    # 3. Tính toán email GỬI RA NGOÀI (External)
    # Logic Vectorized: Tổng số email - Số email chứa "dtaa.com" = Số email ngoại
    # (Nhanh hơn nhiều so với việc split list rồi check từng cái)
    n_ext_to = n_to - count_internal(to_col)
    n_ext_cc = n_cc - count_internal(cc_col)
    n_ext_bcc = n_bcc - count_internal(bcc_col)
    
    n_exdes = n_ext_to + n_ext_cc + n_ext_bcc
    
    # Cờ đánh dấu (Flags)
    Xemail = (n_exdes > 0).astype(int)      # Có ít nhất 1 email ngoại
    exbccmail = (n_ext_bcc > 0).astype(int) # Có email ngoại trong BCC

    # 4. Xử lý Content (Nội dung)
    content = df_email['content'].fillna('').astype(str)
    email_text_len = content.str.len()
    # Đếm số từ: đếm khoảng trắng + 1 (nếu không rỗng)
    email_text_nwords = np.where(content.str.strip() == '', 0, content.str.count(' ') + 1)

    # 5. Trả về DataFrame kết quả (đúng thứ tự cột như logic cũ)
    # Các cột: n_des, #att, Xemail, n_exdes, n_bccdes, exbccmail, size, slen, nwords
    return pd.DataFrame({
        'n_des': n_des,
        'n_atts': df_email['#att'].fillna(0).astype(int),
        'Xemail': Xemail,
        'n_exdes': n_exdes,
        'n_bccdes': n_bcc,
        'exbccmail': exbccmail,
        'email_size': df_email['size'].fillna(0).astype(int),
        'email_text_slen': email_text_len,
        'email_text_nwords': email_text_nwords
    }, index=df_email.index)

def vectorized_http_process(df_http):
    # Nếu DataFrame rỗng thì trả về rỗng ngay
    if len(df_http) == 0: return pd.DataFrame()

    # 1. Chuẩn bị dữ liệu (Handle NaNs)
    url = df_http['url'].fillna('').astype(str)
    content = df_http['content'].fillna('').astype(str)

    # 2. Tính toán các đặc trưng cơ bản (Basic Features)
    url_len = url.str.len()
    
    # url_depth: đếm số '/' trừ 2, chặn dưới tại 0
    url_depth = np.maximum(0, url.str.count('/') - 2)
    
    content_len = content.str.len()
    
    # content_nwords: đếm khoảng trắng + 1
    content_nwords = content.str.count(' ') + 1

    # 3. Xử lý tên miền (Domain Extraction & Normalization)
    # Lấy phần giữa // và / đầu tiên. Regex: //(.*?)/
    domains = url.str.extract(r"//(.*?)/")[0].fillna('')
    domains = domains.str.replace("www.", "", regex=False)

    # Logic rút gọn subdomain:
    # Điều kiện 1: len(dn) > 2 -> Tức là có từ 2 dấu chấm trở lên (ví dụ: a.b.c)
    cond_parts = domains.str.count(r'\.') >= 2
    
    # Điều kiện 2: Không nằm trong whitelist
    whitelist_pattern = r"google\.com|\.co\.uk|\.co\.nz|live\.com"
    cond_not_whitelist = ~domains.str.contains(whitelist_pattern, regex=True)
    
    # Điều kiện kết hợp để rút gọn
    mask_normalize = cond_parts & cond_not_whitelist
    
    # Thực hiện rút gọn: Lấy 2 phần cuối cùng (tương đương ".".join(dn[-2:]))
    # Regex: lấy tất cả ký tự không phải dấu chấm, theo sau bởi 1 dấu chấm, và nhóm ký tự cuối
    # Ví dụ: sub.example.com -> example.com
    domains_normalized = np.where(
        mask_normalize, 
        domains.str.extract(r'([^.]+\.[^.]+)$')[0].fillna(domains), 
        domains
    )
    
    # Chuyển về Series để dùng các hàm str tiếp theo
    domains_final = pd.Series(domains_normalized, index=df_http.index)

    # 4. Phân loại Website (Categorization) - Dùng np.select
    
    # Định nghĩa danh sách
    l_cloud = ['dropbox.com', 'drive.google.com', 'mega.co.nz', 'account.live.com']
    l_leak = ['wikileaks.org', 'freedom.press', 'theintercept.com']
    l_social = ['facebook.com', 'twitter.com', 'plus.google.com', 'instagr.am', 'instagram.com',
                'flickr.com', 'linkedin.com', 'reddit.com', 'about.com', 'youtube.com', 'pinterest.com',
                'tumblr.com', 'quora.com', 'vine.co', 'match.com', 't.co']
    l_job = ['indeed.com', 'monster.com', 'careerbuilder.com', 'simplyhired.com']
    l_hack = ['webwatchernow.com', 'actionalert.com', 'relytec.com', 'refog.com', 'wellresearchedreviews.com',
              'softactivity.com', 'spectorsoft.com', 'best-spy-soft.com']

    # Tạo các điều kiện (Conditions) - Thứ tự quan trọng tương ứng if-elif
    
    # C1: Cloud (Nhóm 3)
    c1 = domains_final.isin(l_cloud)
    
    # C2: Leak (Nhóm 5)
    c2 = domains_final.isin(l_leak)
    
    # C3: Social (Nhóm 2)
    c3 = domains_final.isin(l_social)
    
    # C4: Job (Nhóm 4)
    # Logic: (list) OR ('job' in d AND ('hunt' in d OR 'search' in d)) OR ('aol.com' in d AND ('recruit' in u OR 'job' in u))
    has_job_kw = domains_final.str.contains('job', regex=False)
    has_hunt_search = domains_final.str.contains(r'hunt|search', regex=True)
    is_aol = domains_final.str.contains('aol.com', regex=False)
    url_has_rec_job = url.str.contains(r'recruit|job', regex=True)
    
    c4 = (domains_final.isin(l_job)) | \
         (has_job_kw & has_hunt_search) | \
         (is_aol & url_has_rec_job)
         
    # C5: Hack (Nhóm 6)
    # Logic: (list) OR ('keylog' in d)
    c5 = (domains_final.isin(l_hack)) | (domains_final.str.contains('keylog', regex=False))
    
    # Tổng hợp kết quả (Priority: Cloud > Leak > Social > Job > Hack > Other)
    conditions = [c1, c2, c3, c4, c5]
    choices = [3, 5, 2, 4, 6]
    
    # Default là 1 (Other)
    r = np.select(conditions, choices, default=1)

    # 5. Trả về DataFrame kết quả
    return pd.DataFrame({
        'http_type': r,
        'url_len': url_len,
        'url_depth': url_depth,
        'http_c_len': content_len,
        'http_c_nwords': content_nwords
    }, index=df_http.index)

def vectorized_file_process(df_file):
    # Nếu DataFrame rỗng thì trả về rỗng ngay
    if len(df_file) == 0: return pd.DataFrame()

    # 1. Chuẩn bị dữ liệu (Handle NaNs)
    filename = df_file['filename'].fillna('').astype(str)
    content = df_file['content'].fillna('').astype(str)

    # 2. Trích xuất đuôi file (Extension)
    # Logic cũ: if "." in filename -> split(".")[-1] else "unknown"
    # Logic mới: Dùng Regex lấy phần sau dấu chấm cuối cùng. Nếu không có -> NaN -> fillna('unknown')
    ftype = filename.str.extract(r'\.([^.]+)$')[0].str.lower().fillna('unknown')

    # 3. Xác định loại ổ đĩa (Disk)
    # Logic cũ: C -> 1, R -> 2, Else -> 0
    cond_c = filename.str.startswith('C')
    cond_r = filename.str.startswith('R')
    
    # Dùng np.select nhanh hơn if-else
    disk = np.select([cond_c, cond_r], [1, 2], default=0)

    # 4. Độ sâu thư mục (Depth)
    # Đếm số lượng dấu gạch chéo ngược '\'
    # Lưu ý: Cần escape string trong regex (\\\\ để match \)
    file_depth = filename.str.count(r'\\')

    # 5. Đặc trưng nội dung (Content Features)
    fsize = content.str.len()
    # Đếm số từ: đếm khoảng trắng + 1
    f_nwords = content.str.count(' ') + 1

    # 6. Phân loại nhóm file (Category r)
    # Định nghĩa bảng ánh xạ (nhanh hơn if-elif)
    ext_map = {
        'zip': 2, 'rar': 2, '7z': 2,          # Nén
        'jpg': 3, 'png': 3, 'bmp': 3,         # Ảnh
        'doc': 4, 'docx': 4, 'pdf': 4,        # Văn bản
        'txt': 5, 'cfg': 5, 'rtf': 5,         # Text/Config
        'exe': 6, 'sh': 6                     # Thực thi
    }
    
    # Map extension sang số. Những cái không có trong map (NaN) -> fillna(1) (Other)
    r = ftype.map(ext_map).fillna(1).astype(int)

    # 7. Trả về DataFrame kết quả
    return pd.DataFrame({
        'file_type': r,
        'file_len': fsize,
        'file_nwords': f_nwords,
        'disk': disk,
        'file_depth': file_depth
    }, index=df_file.index)

def vectorized_from_pc(df_acts, users_df):
    # map_own: User -> PC chính chủ
    map_own = users_df['pc'].to_dict()
    # map_sup: User -> PC của Sếp
    s_sup_pc = users_df['sup'].map(map_own) # map_own đóng vai trò lookup PC
    map_sup = s_sup_pc.to_dict()
    shared_exploded = users_df[['sharedpc']].dropna().explode('sharedpc')
    # Tạo set các tuple: {(user1, pcA), (user1, pcB), (user2, pcC)...}
    shared_set = set(zip(shared_exploded.index, shared_exploded['sharedpc']))

    # --- BƯỚC 2: CHUẨN BỊ DỮ LIỆU ---
    act_users = df_acts['user']
    act_pcs = df_acts['pc']
    # Tìm PC mong đợi (Target PC) cho mỗi dòng
    target_own_pcs = act_users.map(map_own)
    target_sup_pcs = act_users.map(map_sup)
    # --- BƯỚC 3: SO SÁNH VECTOR (LOGIC CHÍNH) ---
    # 1. Check Own PC (Code 0)
    cond_own = (act_pcs == target_own_pcs)
    # 2. Check Shared PC (Code 1)
    cond_shared = [(u, p) in shared_set for u, p in zip(act_users, act_pcs)]
    # 3. Check Supervisor PC (Code 3)
    cond_sup = (act_pcs == target_sup_pcs)
    # --- BƯỚC 4: TỔNG HỢP THEO THỨ TỰ ƯU TIÊN ---
    # Logic cũ: if Own -> 0, elif Shared -> 1, elif Sup -> 3, else -> 2
    conditions = [
        cond_own,       # Ưu tiên 1
        cond_shared,    # Ưu tiên 2
        cond_sup        # Ưu tiên 3
    ]
    choices = [0, 1, 3]
    # np.select hoạt động như if-elif-else vector
    # Mặc định là 2 (Other's PC)
    result = np.select(conditions, choices, default=2)
    return result

def process_week_num(week, users, userlist='all', data='r4.2', chunk_size=300000):
    # 1. Chuẩn bị dữ liệu đầu vào
    file_path = f"DataByWeek/{week}.parquet"
    if not os.path.exists(file_path): return        
    
    # Đọc dữ liệu (Load toàn bộ tuần vào RAM, nhanh hơn chunking nhỏ lẻ)
    acts_week = pd.read_parquet(file_path)
    
    # Ép kiểu datetime và sort
    acts_week['date'] = pd.to_datetime(acts_week['date'])
    acts_week.sort_values('date', inplace=True)
    
    # Map User ID sang số nguyên (để tiết kiệm bộ nhớ cho model sau này)
    user_dict = {idx: i for (i, idx) in enumerate(users.index)}
    acts_week['user_int'] = acts_week['user'].map(user_dict).fillna(-1).astype(int)
    
    # ---------------------------------------------------------
    # 2. VECTORIZED FEATURE ENGINEERING (Xử lý hàng loạt)
    # ---------------------------------------------------------
    
    # A. Tính toán Thời gian (Time, Day)
    is_we = vectorized_is_weekend(acts_week['date'])
    is_after = vectorized_is_after_whour(acts_week['date'])
    
    # Logic: 1:HC, 2:Ngoài giờ, 3:Cuối tuần, 4:Đêm cuối tuần
    acts_week['time'] = np.where(is_we, 
                                 np.where(is_after, 4, 3), 
                                 np.where(is_after, 2, 1))
    
    # Tính 'day' (số ngày từ mốc 2009-12-28)
    sd_monday = datetime.strptime("2009-12-28", '%Y-%m-%d')
    acts_week['day'] = (acts_week['date'] - sd_monday).dt.days

    # B. Tính toán PC (Own/Shared/Supervisor)
    # ---------------------------------------
    # Gọi hàm vectorized_from_pc đã tối ưu ở bước trước
    acts_week['pc_code'] = vectorized_from_pc(acts_week, users)

    # C. Map Activity Type (Logon, Http...)
    # -------------------------------------
    uacts_mapping = {'logon':1, 'logoff':2, 'connect':3, 'disconnect':4, 'http':5, 'email':6, 'file':7}
    
    # Chuẩn hóa cột activity (vì log r4.2 có lúc viết hoa/thường)
    # Chỉ xử lý dòng nào là Logon/Logoff/Connect để tiết kiệm time
    mask_log = acts_week['activity'].isin(['Logon', 'Logoff', 'Connect', 'Disconnect'])
    # Tạo cột tạm type_clean
    acts_week['type_clean'] = acts_week['type']
    acts_week.loc[mask_log, 'type_clean'] = acts_week.loc[mask_log, 'activity'].str.strip().str.lower()
    
    acts_week['act_num'] = acts_week['type_clean'].map(uacts_mapping).fillna(0).astype(int)

    # D. Trích xuất đặc trưng chi tiết (Sub-features)
    # ----------------------------------------------
    # Khởi tạo tất cả cột feature bằng 0
    feature_cols = [
        'usb_dur', 
        'file_type', 'file_len', 'file_nwords', 'disk', 'file_depth',
        'http_type', 'url_len', 'url_depth', 'http_c_len', 'http_c_nwords',
        'n_des', 'n_atts', 'Xemail', 'n_exdes', 'n_bccdes', 'exbccmail', 'email_size', 'email_text_slen', 'email_text_nwords'
    ]
    for col in feature_cols:
        acts_week[col] = 0

    # 1. Xử lý FILE (Gọi hàm vectorized_file_process)
    mask_file = acts_week['type'] == 'file'
    if mask_file.any():
        df_file_feats = vectorized_file_process(acts_week[mask_file])
        for c in df_file_feats.columns: acts_week.loc[mask_file, c] = df_file_feats[c]

    # 2. Xử lý EMAIL (Gọi hàm vectorized_email_process)
    mask_email = acts_week['type'] == 'email'
    if mask_email.any():
        df_email_feats = vectorized_email_process(acts_week[mask_email])
        for c in df_email_feats.columns: acts_week.loc[mask_email, c] = df_email_feats[c]

    # 3. Xử lý HTTP (Gọi hàm vectorized_http_process)
    mask_http = acts_week['type'] == 'http'
    if mask_http.any():
        df_http_feats = vectorized_http_process(acts_week[mask_http])
        for c in df_http_feats.columns: acts_week.loc[mask_http, c] = df_http_feats[c]

    # 4. Xử lý USB Duration (Vectorized Logic - Group & Shift)
    # Lọc ra các hành động Connect/Disconnect
    usb_mask = acts_week['activity'].isin(['Connect', 'Disconnect'])
    df_usb = acts_week.loc[usb_mask, ['user', 'pc', 'date', 'activity']].copy()
    
    # Sort để Connect và Disconnect nằm cạnh nhau
    df_usb.sort_values(['user', 'pc', 'date'], inplace=True)
    
    # Shift (-1) để lấy thời gian của dòng tiếp theo đưa lên dòng hiện tại
    df_usb['next_activity'] = df_usb.groupby(['user', 'pc'])['activity'].shift(-1)
    df_usb['next_date'] = df_usb.groupby(['user', 'pc'])['date'].shift(-1)
    
    # Tính duration: Chỉ khi dòng này là Connect VÀ dòng sau là Disconnect
    valid_pair = (df_usb['activity'] == 'Connect') & (df_usb['next_activity'] == 'Disconnect')
    df_usb['duration'] = 0
    df_usb.loc[valid_pair, 'duration'] = (df_usb.loc[valid_pair, 'next_date'] - df_usb.loc[valid_pair, 'date']).dt.total_seconds()
    
    # Map ngược lại vào bảng gốc (Mặc định là 0)
    acts_week['usb_dur'] = 0
    # Chỉ update những dòng Connect có cặp Disconnect hợp lệ
    acts_week.loc[df_usb[valid_pair].index, 'usb_dur'] = df_usb.loc[valid_pair, 'duration']

    # E. Gán nhãn Insider / Malicious Act
    # -----------------------------------
    # Logic: 
    # 1. Insider: Nếu user có malscene > 0 VÀ hành động nằm trong khoảng thời gian [mstart, mend]
    # 2. Mal_Act: Nếu ID hành động nằm trong danh sách 'malacts' của user đó
    
    # Merge thông tin user (malscene, mstart, mend) vào bảng acts_week
    user_info = users[['malscene', 'mstart', 'mend']].copy()
    # Chuyển mstart/mend về datetime để so sánh
    user_info['mstart'] = pd.to_datetime(user_info['mstart'])
    user_info['mend'] = pd.to_datetime(user_info['mend'])
    
    # Merge trái (Left Join)
    acts_week = acts_week.merge(user_info, left_on='user', right_index=True, how='left')
    
    # Tính cột Insider (User có phải là insider tại thời điểm này không)
    # Điều kiện: Có malscene > 0 VÀ date >= mstart VÀ date <= mend
    cond_time = (acts_week['date'] >= acts_week['mstart']) & (acts_week['date'] <= acts_week['mend'])
    acts_week['insider'] = np.where((acts_week['malscene'] > 0) & cond_time, 
                                    acts_week['malscene'], 
                                    0).astype(int)

    # Tính cột Mal_Act (Hành động cụ thể này có độc hại không)
    # Tạo một Set chứa tất cả các cặp (User, ActID) độc hại để tra cứu O(1)
    mal_pairs = set()
    for u_idx, row in users.iterrows():
        if isinstance(row['malacts'], (list, np.ndarray)):
            for mid in row['malacts']:
                mal_pairs.add((u_idx, str(mid))) # Lưu cặp (User, ActionID)
    
    # Hàm check nhanh
    def check_mal_pair(row):
        return 1 if (row['user'], str(row['id'])) in mal_pairs else 0
    
    # Apply logic check (vẫn dùng apply dòng nhưng nhanh hơn vì check set O(1))
    # Nếu dataset quá lớn, có thể dùng merge, nhưng apply check set vẫn khá nhanh
    acts_week['mal_act'] = acts_week.apply(check_mal_pair, axis=1)

    # ---------------------------------------------------------
    # 3. LƯU FILE (FORMATTING & SAVING)
    # ---------------------------------------------------------
    # Tạo các cột meta
    acts_week['actid'] = acts_week.index
    acts_week['pcid'] = acts_week['pc'] # Giữ nguyên PC string (PC-xxx) cho Session Logic
    
    # Đổi tên các cột đã tính toán về tên chuẩn
    rename_map = {
        'user_int': 'user', # Output user phải là số int (từ user_dict)
        'date': 'time_stamp',
        'act_num': 'act',
        'pc_code': 'pc'     # Output pc phải là số int (0,1,2,3)
    }
    acts_week.rename(columns=rename_map, inplace=True)
    
    # List các cột cần lưu
    final_cols = ['actid', 'pcid', 'time_stamp', 'user', 'day', 'act', 'pc', 'time'] + \
                 feature_cols + ['mal_act', 'insider']
    
    df_final = acts_week[final_cols].copy()
    
    # Ép kiểu int cho các cột số liệu (trừ pcid và time_stamp)
    cols_to_int = ['user', 'day', 'act', 'pc', 'time'] + feature_cols + ['mal_act', 'insider']
    df_final[cols_to_int] = df_final[cols_to_int].astype(int)
    
    # Lưu file Parquet (Ghi 1 lần, không cần chunking vì đã xử lý xong hết)
    save_path = f"NumDataByWeek/{week}_num.parquet"
    
    # Dùng PyArrow để ghi
    table = pa.Table.from_pandas(df_final)
    pq.write_table(table, save_path, compression='snappy')
    
    # Dọn dẹp RAM
    del acts_week, df_final, table, user_info
    gc.collect()
    
# --- SESSION LOGIC ---
def get_sessions(uw, first_sid=0):
    sessions = {}
    open_sessions = {}
    sid = 0
    
    for i in uw.index:
        current_pc = uw.loc[i]['pcid']
        act_type = uw.loc[i]['act'] # 1: Logon, 2: Logoff
        
        if current_pc in open_sessions:
            if act_type == 2: # Logoff
                open_sessions[current_pc][2] = 1 # end_with logoff
                open_sessions[current_pc][4] = uw.loc[i]['time_stamp']
                open_sessions[current_pc][6].append(i)
                sessions[sid] = [first_sid + sid] + open_sessions.pop(current_pc)
                sid += 1
            elif act_type == 1: # New Logon on same PC without logoff
                open_sessions[current_pc][2] = 2
                sessions[sid] = [first_sid + sid] + open_sessions.pop(current_pc)
                sid += 1
                open_sessions[current_pc] = [current_pc, 1, 0, uw.loc[i]['time_stamp'], uw.loc[i]['time_stamp'], 1, [i]]
            else:
                open_sessions[current_pc][4] = uw.loc[i]['time_stamp']
                open_sessions[current_pc][6].append(i)
        else:
            start_status = 1 if act_type == 1 else 2
            open_sessions[current_pc] = [current_pc, start_status, 0, uw.loc[i]['time_stamp'], uw.loc[i]['time_stamp'], 1, [i]]

    for pc in open_sessions:
        sessions[sid] = [first_sid + sid] + open_sessions[pc]
        sid += 1

    return sessions

def get_u_features_dicts(ul, data = 'r4.2'):
    """Tạo từ điển ánh xạ thông tin người dùng cho r4.2"""
    ufdict = {}
    # r4.2 không có cột 'project'
    list_uf = ['role', 'b_unit', 'f_unit', 'dept', 'team']
    
    for f in list_uf:
        ul[f] = ul[f].astype(str)
        tmp = list(set(ul[f]))
        tmp.sort()
        # Ánh xạ mỗi giá trị chữ sang một số nguyên
        ufdict[f] = {idx: i for i, idx in enumerate(tmp)}
    return (ul, ufdict, list_uf)

def proc_u_features(uf, ufdict, list_f = None, data = 'r4.2'):
    """Chuyển đổi đặc trưng của một user cụ thể sang dạng số"""
    if type(list_f) != list:
        # Mặc định danh sách cột cho r4.2
        list_f = ['role', 'b_unit', 'f_unit', 'dept', 'team']

    out = []
    for f in list_f:
        # Lấy giá trị số từ từ điển ánh xạ đã tạo
        out.append(ufdict[f][uf[f]])
    return out

def f_stats_calc(ud, fn, stats_f, countonly_f = {}, get_stats = False):
    """Tính toán các chỉ số thống kê cơ bản (Mean, Min, Max,...)"""
    f_count = len(ud)
    r = []
    f_names = []
    
    # Tính thống kê cho các trường dữ liệu số (ví dụ: độ dài nội dung, kích thước file)
    for f in stats_f:
        inp = ud[f].values
        if get_stats:
            if f_count > 0:
                r += [np.min(inp), np.max(inp), np.median(inp), np.mean(inp), np.std(inp)]
            else: 
                r += [0.0, 0.0, 0.0, 0.0, 0.0]
            f_names += [fn+'_min_'+f, fn+'_max_'+f, fn+'_med_'+f, fn+'_mean_'+f, fn+'_std_'+f]
        else:
            # Mặc định chỉ lấy trung bình nếu get_stats = False
            r += [np.mean(inp)] if f_count > 0 else [0.0]
            f_names += [fn+'_mean_'+f]
        
    # Đếm số lượng xuất hiện của các giá trị cụ thể (ví dụ: số lần dùng PC lạ)
    for f in countonly_f:
        for v in countonly_f[f]:
            r += [sum(ud[f].values == v)]
            f_names += [fn+'_n-'+f+str(v)]
            
    return (f_count, r, f_names)

def f_calc_subfeatures(ud, fname, filter_col, filter_vals, filter_names, sub_features, countonly_subfeatures):
    """Tính toán đặc trưng cho các nhóm con (ví dụ: chỉ tính riêng cho email gửi ra ngoài)"""
    # 1. Tính tổng quát cho toàn bộ hành động
    [n, stats, fnames] = f_stats_calc(ud, fname, sub_features, countonly_subfeatures)
    allf = [n] + stats
    allf_names = ['n_' + fname] + fnames
    
    # 2. Lọc và tính riêng cho từng phân loại con (filter)
    for i in range(len(filter_vals)):
        filtered_data = ud[ud[filter_col] == filter_vals[i]]
        [n_sf, sf_stats, sf_fnames] = f_stats_calc(filtered_data, filter_names[i], sub_features, countonly_subfeatures)
        allf += [n_sf] + sf_stats
        allf_names += [fname + '_n_' + filter_names[i]] + [fname + '_' + x for x in sf_fnames]
        
    return (allf, allf_names)

def f_calc(ud, mode = 'session', data = 'r4.2'):
    # Khởi tạo các biến cơ bản
    n_weekendact = (ud['time'] == 3).sum()
    is_weekend = 1 if n_weekendact > 0 else 0
    
    # Đối với mode 'session', không đếm số lượng theo PC vì mỗi session thường gắn với 1 PC
    all_countonlyf = {}
    
    # 1. Thống kê chung cho tất cả hành động trong session
    [all_f, all_f_names] = f_calc_subfeatures(ud, 'allact', None, [], [], [], all_countonlyf)
    
    # 2. Thống kê hành động Logon (act == 1)
    logon_countonlyf = {}
    logon_statf = []
    [all_logonf, all_logonf_names] = f_calc_subfeatures(ud[ud['act']==1], 'logon', None, [], [], logon_statf, logon_countonlyf)
    
    # 3. Thống kê thiết bị USB (act == 3) - r4.2 chỉ có usb_dur
    device_countonlyf = {}
    device_statf = ['usb_dur']
    [all_devicef, all_devicef_names] = f_calc_subfeatures(ud[ud['act']==3], 'usb', None, [], [], device_statf, device_countonlyf)
    
    # 4. Thống kê File (act == 7) - r4.2 lược bỏ to_usb, from_usb, file_act
    # r4.2: disk 0: unknown, 1: C, 2: R
    file_countonlyf = {'disk':[0, 1, 2]}
    (all_filef, all_filef_names) = f_calc_subfeatures(
        ud[ud['act']==7], 'file', 'file_type', [1,2,3,4,5,6], 
        ['otherf','compf','phof','docf','txtf','exef'], 
        ['file_len', 'file_depth', 'file_nwords'], 
        file_countonlyf
    )
    
    # 5. Thống kê Email (act == 6)
    email_stats_f = ['n_des', 'n_atts', 'n_exdes', 'n_bccdes', 'email_size', 'email_text_slen', 'email_text_nwords']
    mail_countonlyf = {'Xemail':[1], 'exbccmail':[1]}
    # r4.2 không có bộ lọc send/receive trong file email.csv thô
    (all_emailf, all_emailf_names) = f_calc_subfeatures(ud[ud['act']==6], 'email', None, [], [], email_stats_f, mail_countonlyf)
    
    # 6. Thống kê HTTP (act == 5)
    http_count_subf = {} # Lược bỏ pc và http_act (chỉ có ở r6)
    (all_httpf, all_httpf_names) = f_calc_subfeatures(
        ud[ud['act']==5], 'http', 'http_type', [1,2,3,4,5,6], 
        ['otherf','socnetf','cloudf','jobf','leakf','hackf'], 
        ['url_len', 'url_depth', 'http_c_len', 'http_c_nwords'], 
        http_count_subf
    )
        
    # Xác định thông tin Insider (mal_act)
    numActs = all_f[0]
    mal_u = 0
    if (ud['mal_act']).sum() > 0:
        tmp = list(set(ud['insider']))
        if len(tmp) > 1 and 0.0 in tmp:
            tmp.remove(0.0)
        mal_u = tmp[0]
        
    # Gộp tất cả vector đặc trưng của session
    features_tmp = all_f + all_logonf + all_devicef + all_filef + all_emailf + all_httpf
    fnames_tmp = all_f_names + all_logonf_names + all_devicef_names + all_filef_names + all_emailf_names + all_httpf_names
    
    return [numActs, is_weekend, features_tmp, fnames_tmp, mal_u]

def session_instance_calc(ud, sinfo, week, mode, data, uw, v, list_uf):
    # Lấy thông tin ngày thực hiện hành động đầu tiên trong session
    d = ud.iloc[0]['day']
    
    # Tính toán tỷ lệ thời gian hoạt động trong session
    # 1: Giờ hành chính, 2: Ngoài giờ, 3: Cuối tuần, 4: Đêm cuối tuần
    total_acts = len(ud)
    perworkhour = sum(ud['time'] == 1) / total_acts
    perafterhour = sum(ud['time'] == 2) / total_acts
    perweekend = sum(ud['time'] == 3) / total_acts
    perweekendafterhour = sum(ud['time'] == 4) / total_acts
    
    # Tính toán thời lượng và mốc thời gian của session
    st_timestamp = min(ud['time_stamp'])
    end_timestamp = max(ud['time_stamp'])
    
    # Thời lượng session tính bằng phút
    s_dur = (end_timestamp - st_timestamp).total_seconds() / 60
    
    # Giờ bắt đầu và kết thúc (quy đổi sang số thực, ví dụ 7:30 = 7.5)
    s_start = st_timestamp.hour + st_timestamp.minute / 60
    s_end = end_timestamp.hour + end_timestamp.minute / 60
    
    # Timestamp chuẩn (Epoch time)
    starttime = st_timestamp.timestamp()
    endtime = end_timestamp.timestamp()
    
    # Đếm số ngày session kéo dài (thường là 1)
    n_days = len(set(ud['day']))
    
    # Gọi hàm f_calc đã lọc cho r4.2 để lấy các đặc trưng thống kê hoạt động
    tmp = f_calc(ud, mode, data)
    
    # Tạo vector instance hoàn chỉnh cho session
    # Kết hợp: Thông tin thời gian + Đặc trưng Session + Thông tin User + Đặc trưng Hoạt động + Nhãn Insider
    session_instance = [
        starttime, endtime, v, sinfo[0], d, week, ud.iloc[0]['pc'], 
        perworkhour, perafterhour, perweekend, perweekendafterhour, 
        n_days, s_dur, 
        sinfo[6], # n_concurrent_login (số lượng login đồng thời)
        sinfo[2], # start_with (bắt đầu bằng logon hay không)
        sinfo[3], # end_with (kết thúc bằng logoff hay không)
        s_start, s_end
    ] + \
    (uw.loc[v, list_uf + ['ITAdmin', 'O', 'C', 'E', 'A', 'N']]).tolist() + \
    tmp[2] + [tmp[4]] # tmp[2] là features, tmp[4] là nhãn insider
    
    return (session_instance, tmp[3]) # Trả về instance và danh sách tên cột (tmp[3])

def to_csv(week, mode, data, ul, uf_dict, list_uf, chunk_size=300000):
    # Khởi tạo từ điển ánh xạ user ID
    user_dict = {i : idx for (i, idx) in enumerate(ul.index)} 
    
    # Thiết lập ID session duy nhất dựa trên tuần
    # Ví dụ: tuần 1 sẽ bắt đầu từ 100000, tuần 2 từ 200000
    first_sid = week * 100000 
    
    # Định nghĩa các cột thông tin cơ bản cho Session r4.2
    cols2a = [
        'starttime', 'endtime', 'user', 'sessionid', 'day', 'week', 'pc', 
        'isworkhour', 'isafterhour', 'isweekend', 'isweekendafterhour', 
        'n_days', 'duration', 'n_concurrent_sessions', 'start_with', 'end_with', 
        'ses_start', 'ses_end'
    ] + list_uf + ['ITAdmin', 'O', 'C', 'E', 'A', 'N']
    
    cols2b = ['insider']        

    # Đọc dữ liệu số đã xử lý của tuần hiện tại
    w = pd.read_parquet(f"NumDataByWeek/{week}_num.parquet")
    usnlist = list(set(w['user'].astype('int').values))
    
    # Tạo bảng thông tin User tĩnh cho tuần này
    cols_u = ['week'] + list_uf + ['ITAdmin', 'O', 'C', 'E', 'A', 'N', 'insider'] 
    uwdict = {}
    for v in user_dict:
        if v in usnlist:
            is_ITAdmin = 1 if ul.loc[user_dict[v], 'role'] == 'ITAdmin' else 0
            # Mã hóa các thông tin category (role, dept...) thành số
            u_feats = proc_u_features(ul.loc[user_dict[v]], uf_dict, list_uf, data=data)
            # Lấy chỉ số tâm lý OCEAN
            ocean = (ul.loc[user_dict[v], ['O', 'C', 'E', 'A', 'N']]).tolist()
            # Lấy nhãn insider
            insider_label = int(list(set(w[w['user'] == v]['insider']))[0])
            
            uwdict[v] = [week] + u_feats + [is_ITAdmin] + ocean + [insider_label]
            
    uw = pd.DataFrame.from_dict(uwdict, orient='index', columns=cols_u)    
    
    output_file = f"tmp/{week}{mode}.parquet"
    writer = None
    towrite_buffer = []
    full_columns = None
    
    # Duyệt qua từng User
    for v in user_dict:
        if v in usnlist:
            uactw = w[w['user'] == v]
            
            # Get sessions
            sessions = get_sessions(uactw, first_sid)
            first_sid += len(sessions)

            all_sess_info = list(sessions.values())
            # Lưu ý: sinfo[4] là start_time, sinfo[5] là end_time (dạng timestamp hoặc object)
            starts = np.array([s[4] for s in all_sess_info])
            ends = np.array([s[5] for s in all_sess_info])
            
            # Tính concurrency cho từng session
            # Logic: Session A bị coi là concurrent nếu nó trùng thời gian với bất kỳ session B nào khác (PC khác)
            concurrent_counts = []
            for k in range(len(all_sess_info)):
                cur_start = starts[k]
                cur_end = ends[k]
                # Đếm số lượng session có khoảng thời gian giao nhau (Overlap)
                # Điều kiện overlap: (Start_B < End_A) AND (End_B > Start_A)
                overlaps = np.sum((starts < cur_end) & (ends > cur_start))
                # overlaps sẽ luôn >= 1 (trùng với chính nó). 
                concurrent_counts.append(overlaps)

            # Cập nhật lại giá trị vào dict sessions
            for idx, key in enumerate(sessions):
                sessions[key][6] = concurrent_counts[idx] # Ghi đè vào vị trí dummy số 1 cũ
            
            for s in sessions:
                sinfo = sessions[s]
                ud = uactw.loc[sinfo[7]]
                
                if len(ud) > 0:                     
                    # Tính feature
                    session_instance, i_fnames = session_instance_calc(
                        ud, sinfo, week, mode, data, uw, v, list_uf
                    )
                    
                    # Nếu là lần đầu tiên, xác định danh sách cột đầy đủ
                    if writer is None:
                         full_columns = cols2a + i_fnames + cols2b
                    
                    towrite_buffer.append(session_instance)
                    
                    # --- FLUSH BUFFER NẾU ĐẦY ---
                    if len(towrite_buffer) >= chunk_size:
                        df_chunk = pd.DataFrame(towrite_buffer, columns=full_columns)
                        
                        # Convert sang Table và ghi
                        table = pa.Table.from_pandas(df_chunk)
                        
                        if writer is None:
                            writer = pq.ParquetWriter(output_file, table.schema, compression='snappy')
                        
                        writer.write_table(table)
                        
                        # Dọn dẹp RAM
                        del df_chunk, table
                        towrite_buffer = [] # Reset buffer
                        gc.collect()

    # --- GHI PHẦN CÒN DƯ (Buffer còn lại) ---
    if len(towrite_buffer) > 0:
        try:
            df_chunk = pd.DataFrame(towrite_buffer, columns=full_columns)
            table = pa.Table.from_pandas(df_chunk)
            
            if writer is None:
                writer = pq.ParquetWriter(output_file, table.schema, compression='snappy')
            
            writer.write_table(table)
        except UnboundLocalError:
            # Trường hợp không có session nào được tạo ra
            pass

    if writer:
        writer.close()
    
    # Xóa biến lớn để giải phóng RAM cho joblib process khác
    del w, uw
    gc.collect()

if __name__ == "__main__":
    # 1. Kiểm tra thư mục hiện tại có phải là r4.2 không
    dname = 'r4.2'
    # 2. Tạo các thư mục tạm và thư mục chứa kết quả
    for folder in ["tmp", "ExtractedData", "DataByWeek", "NumDataByWeek"]:
        if not os.path.exists(folder):
            os.mkdir(folder)
    
    # Cấu hình số luồng xử lý song song (mặc định 8)
    numCores = 4
    if len(sys.argv) > 1:
        try:
            # Cố gắng đọc tham số nếu người dùng truyền số vào
            numCores = int(sys.argv[1])
        except ValueError:
            # Nếu gặp lỗi (do sys.argv[1] là '-f' của Jupyter), bỏ qua và dùng mặc định
            print(f"Running in Notebook mode. Using default numCores = {numCores}")
        
    # r4.2 có 73 tuần dữ liệu
    numWeek = 73 
    st = time.time()
    
    #### Bước 1: Phân tách dữ liệu nguồn theo từng tuần
    combine_by_timerange_pandas(dname)
    print(f"Step 1 - Separate data by week - done. Time (mins): {(time.time()-st)/60:.2f}")
    st = time.time()
    
    #### Bước 2: Lấy danh sách nhân sự và dán nhãn Insider
    users = get_mal_userdata(dname)
    print(f"Step 2 - Get user list & Insider labels - done. Time (mins): {(time.time()-st)/60:.2f}")
    st = time.time()
    
    #### Bước 3: Chuyển đổi log thô sang dạng số (Numerical)
    Parallel(n_jobs=numCores)(delayed(process_week_num)(i, users, data=dname) for i in range(numWeek))
    print(f"Step 3 - Numerical conversion - done. Time (mins): {(time.time()-st)/60:.2f}")
    st = time.time()
    
    #### Bước 4: Trích xuất đặc trưng theo SESSION và xuất ra CSV
    mode = 'session'
    (ul, uf_dict, list_uf) = get_u_features_dicts(users, data=dname)
    
    # Chạy song song việc gom nhóm session và tính toán đặc trưng thống kê
    Parallel(n_jobs=numCores)(delayed(to_csv)(i, mode, dname, ul, uf_dict, list_uf) for i in range(numWeek))

    # Gộp tất cả các file pickle tạm thời trong 'tmp/' thành một file CSV duy nhất
    output_file = f'ExtractedData/{mode}_{dname}.parquet'
    print(f"Starting to merge files into {output_file}...")
    writer = None
    ref_schema = None
    for w in range(numWeek):
        week_file = f"tmp/{w}{mode}.parquet"
        if os.path.exists(week_file):
            # Đọc file pickle tuần hiện tại
            df_chunk = pd.read_parquet(week_file)
            
            if writer is None:
                # Đây là chunk đầu tiên (thường là tuần 0 hoặc 1) -> Làm chuẩn
                table = pa.Table.from_pandas(df_chunk)
                writer = pq.ParquetWriter(output_file, table.schema, compression='snappy')
                writer.write_table(table)
                # Lưu lại kiểu dữ liệu của pandas để ép các chunk sau
                ref_dtypes = df_chunk.dtypes
            else:
                try:
                    df_chunk = df_chunk.astype(ref_dtypes)
                except Exception as e:
                    print(f"Warning: Could not cast types for week {w}. Reason: {e}")
                
                table = pa.Table.from_pandas(df_chunk)
                try: 
                    writer.write_table(table)
                except Exception as e:
                    print(f"Error writing week {w}: {e}")
        else: pass
    # Đóng writer để hoàn tất file
    if writer:
        writer.close()
    print(f'Step 4 - Extracted {mode} data to {output_file}. Time (mins): {(time.time()-st)/60:.2f}')

    #### Bước 5: Dọn dẹp thư mục tạm
    print("Cleaning up temporary files...")
    for x in ["tmp", "DataByWeek", "NumDataByWeek"]:
        if os.path.exists(x): shutil.rmtree(x)