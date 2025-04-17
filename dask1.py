import datetime
import dask.dataframe as dd
import os
import re

# 讀取原始檔案
file_path = "training.csv"
df = dd.read_csv(file_path, sample=1000000)

# 確保database資料夾存在
os.makedirs("database/training", exist_ok=True)

# 獲取第0欄和最後一欄的名稱
first_column = df.columns[0]
last_column = df.columns[-1]

# 為每個數字分別處理，從1到15
for i in range(1, 16):
    pattern = f"第{i}名"
    filtered_columns = [col for col in df.columns if pattern in col]
    
    if filtered_columns:
        # 將第0欄和最後一欄添加到篩選後的欄位清單中
        # 使用集合確保不會重複添加
        all_columns = [first_column] + filtered_columns
        if last_column not in all_columns:  # 如果最後一欄不在列表中，才添加
            all_columns.append(last_column)
        
        # 選取這些欄位並存檔
        df_filtered_number = df[all_columns]
        df_filtered_number.to_csv(f"database/filtered_number_{i}.csv", index=False, single_file=True)