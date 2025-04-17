import datetime
import dask.dataframe as dd
import os
import re

# 讀取原始檔案
file_path = "training.csv"
df = dd.read_csv(file_path, sample=1000000)

# 確保 database/training 資料夾存在
os.makedirs("database/training", exist_ok=True)

# 獲取第0欄和最後一欄名稱
# 其實不用，因為他們不是買賣超開頭，會自然保留
first_column = df.columns[0]
last_column = df.columns[-1]

# 過濾不是以「買超」或「賣超」開頭的欄位（排除這些欄位）
df1_columns = [col for col in df.columns if not re.match(r"^(買超|賣超)", col)]

# 組合要保留的欄位：第一欄 + 篩選後欄位 + 最後一欄
selected_columns = df1_columns

# 選取欄位
filtered_df = df[selected_columns]

# 儲存結果為 CSV
output_path = "database/training/filtered_number_16.csv"
filtered_df.compute().to_csv(output_path, index=False, encoding="utf-8")
