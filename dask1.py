import dask.dataframe as dd
import os

# 讀取原始檔案
file_path = "database/training.csv"
df = dd.read_csv(file_path, sample=1000000)

# 確保資料夾存在
os.makedirs("database/training", exist_ok=True)

# 獲取第0欄和最後一欄的名稱
first_column = df.columns[0]
last_column = df.columns[-1]

# 建立集合記錄已使用欄位
used_columns = set([first_column, last_column])

# 處理第1名到第15名
for i in range(1, 16):
    print(f"🔍 處理第{i}名欄位中...")

    buy_pattern = f"買超第{i}名"
    sell_pattern = f"賣超第{i}名"
    
    buy_columns = [col for col in df.columns if col.startswith(buy_pattern)]
    sell_columns = [col for col in df.columns if col.startswith(sell_pattern)]
    
    print(f"  ▶ 找到買超欄位：{buy_columns}")
    print(f"  ▶ 找到賣超欄位：{sell_columns}")
    
    diff_columns = {}
    for buy_col in buy_columns:
        suffix = buy_col.replace(buy_pattern, "")
        matching_sell_col = f"{sell_pattern}{suffix}"
        if matching_sell_col in sell_columns:
            diff_col_name = f"差超第{i}名{suffix}"
            diff_columns[diff_col_name] = df[buy_col] - df[matching_sell_col]
            used_columns.update([buy_col, matching_sell_col])
            print(f"    ✅ 成功建立：{diff_col_name}")
        else:
            print(f"    ⚠️ 無對應賣超欄位：{buy_col}")

    if diff_columns:
        df_diff_values = dd.concat(list(diff_columns.values()), axis=1)
        df_diff_values.columns = list(diff_columns.keys())
        df_output = dd.concat([df[[first_column]], df_diff_values, df[[last_column]]], axis=1)
        output_path = f"database/filtered_number_{i}.csv"
        df_output.to_csv(output_path, index=False, single_file=True)
        print(f"  💾 已儲存至：{output_path}")
    else:
        print(f"  ⛔ 沒有任何配對成功的欄位，跳過存檔")

# 處理剩下未用到的欄位
other_columns = [col for col in df.columns if col not in used_columns]
print(f"\n📦 剩下未用欄位共 {len(other_columns)} 欄，將儲存為 filtered_number_16.csv")
df_else = df[other_columns]
df_else.to_csv("database/filtered_number_16.csv", index=False, single_file=True)
print("✅ 全部處理完畢！")
