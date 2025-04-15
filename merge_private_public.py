import pandas as pd

# 只讀取欄位名稱，不載入全部資料
columns_to_keep = pd.read_csv("0413LR_final/filter_training.csv", nrows=0).columns.tolist()

# 讀取 public 與 private 資料
df1 = pd.read_csv('../public_x.csv')
df2 = pd.read_csv('../private_x.csv')

# 取得與 columns_to_keep 的交集欄位（避免目標欄位造成錯誤）
available_columns = [col for col in columns_to_keep if col in df1.columns and col in df2.columns]

# 選擇交集欄位
df1 = df1[available_columns]
df2 = df2[available_columns]

# 縱向合併
merged_df = pd.concat([df1, df2], axis=0, ignore_index=True)

# 儲存合併後的檔案
merged_df.to_csv('../merged_test.csv', index=False)
