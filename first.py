import pandas as pd
import dask.dataframe as dd
import os

file_path = "../../training.csv"
df = dd.read_csv(file_path, sample=1000000)


# 合併後的篩選特徵列表檔案路徑
merged_features_file_path = "../0413LR_5/sort_shap.csv"

# 輸出篩選後的 training 資料檔案路徑
output_filtered_train_path = "filter_training.csv"

# 確保database資料夾存在
# os.makedirs("database/public_x", exist_ok=True)

# 使用 pandas 讀取合併後的篩選特徵列表
try:
    # 第一輪先使用shap中重要性>0.003的資料
    df_merged_features = pd.read_csv(merged_features_file_path)
    print(len(df_merged_features)) # 2275
    df_merged_features = df_merged_features[df_merged_features['mean_abs_shap'] > 0.015]
    print(len(df_merged_features)) # 1000
except FileNotFoundError:
    print(f"錯誤：找不到合併後的特徵列表檔案 {merged_features_file_path}。請確認檔案是否存在於指定路徑。")
    exit()
except Exception as e:
    print(f"讀取合併後的特徵列表檔案 {merged_features_file_path} 時發生錯誤：{e}")
    exit()

# 從合併後的特徵列表中提取所有 'feature' 欄位的值
if 'feature' in df_merged_features.columns:
    filtered_feature_names = df_merged_features['feature'].tolist()

    # 獲取第0欄和最後一欄的名稱
    first_column = df.columns[0]
    last_column = df.columns[-1]

    # 創建要選擇的欄位列表
    columns_to_select = [first_column] + filtered_feature_names
    if last_column not in columns_to_select and last_column in df.columns:
        columns_to_select.append(last_column)

    # 確保要選擇的欄位實際存在於 training.csv 中
    existing_columns_to_select = [col for col in columns_to_select if col in df.columns]

    if existing_columns_to_select:
        # 使用 Dask DataFrame 選取這些欄位
        df_filtered_train = df[existing_columns_to_select]

        # 將篩選後的 Dask DataFrame 儲存為一個 CSV 檔案
        try:
            df_filtered_train.to_csv(output_filtered_train_path, index=False, single_file=True)
            print(f"已從 training.csv 篩選欄位並儲存至 {output_filtered_train_path}")
        except Exception as e:
            print(f"儲存篩選後的 training 資料時發生錯誤：{e}")
    else:
        print("警告：在 training.csv 中找不到任何與合併特徵列表中的欄位相符的欄位，加上第0欄和最後一欄。")

else:
    print(f"錯誤：合併後的特徵列表檔案 {merged_features_file_path} 中找不到 'feature' 欄位。")
    print("請檢查檔案結構。")