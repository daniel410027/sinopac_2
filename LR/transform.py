import dask.dataframe as dd
import os

# 讀取原始檔案
file_path = "../LR_database/training.csv"
df = dd.read_csv(file_path, sample=1000000)

# 獲取第0欄和最後一欄的名稱
first_column = df.columns[0]
last_column = df.columns[-1]

# 建立集合記錄已使用欄位
used_columns = set([first_column, last_column])

# 收集所有差超欄位
all_diff_columns = []

# 處理第1名到第15名
for i in range(1, 16):
    print(f"🔍 處理第{i}名欄位中...")

    buy_pattern = f"買超第{i}名"
    sell_pattern = f"賣超第{i}名"
    
    buy_columns = [col for col in df.columns if col.startswith(buy_pattern)]
    sell_columns = [col for col in df.columns if col.startswith(sell_pattern)]
    
    print(f"  ▶ 找到買超欄位：{buy_columns}")
    print(f"  ▶ 找到賣超欄位：{sell_columns}")
    
    for buy_col in buy_columns:
        suffix = buy_col.replace(buy_pattern, "")
        matching_sell_col = f"{sell_pattern}{suffix}"
        if matching_sell_col in sell_columns:
            diff_col_name = f"差超第{i}名{suffix}"
            diff_col = df[buy_col] - df[matching_sell_col]
            all_diff_columns.append(diff_col.rename(diff_col_name))
            used_columns.update([buy_col, matching_sell_col])
            print(f"    ✅ 成功建立：{diff_col_name}")
        else:
            print(f"    ⚠️ 無對應賣超欄位：{buy_col}")

# 建立差超欄位 DataFrame
df_diff = dd.concat(all_diff_columns, axis=1)

# 處理剩下未用到的欄位
other_columns = [col for col in df.columns if col not in used_columns]
df_else = df[other_columns]

# 合併所有欄位：第一欄、差超欄位、剩下欄位、最後一欄
df_result = dd.concat([df[[first_column]], df_diff, df_else, df[[last_column]]], axis=1)

# 確保資料夾存在
os.makedirs("../LR_database", exist_ok=True)

# 儲存結果
output_path = "../LR_database/transform_training.csv"
df_result.to_csv(output_path, index=False, single_file=True)
print(f"\n✅ 已儲存轉換後檔案至：{output_path}")
