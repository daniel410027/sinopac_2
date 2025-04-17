import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
import joblib
import json

# ========== Step 1: 載入模型與 Imputer ==========
model = joblib.load('optuna_best_lgbm.pkl')
imputer = joblib.load("optuna_imputer.pkl")

# 載入模型資訊以取得閾值
with open('optuna_model_info.json', 'r') as f:
    model_info = json.load(f)
    threshold = model_info.get('best_threshold', 0.5)
    
print(f"模型特徵數量: {len(model.feature_names_in_)}")
print(f"使用預測閾值: {threshold}")

# ========== Step 2: 讀取測試資料 ==========
test_df = pd.read_csv("../../database/lgbm_merged_test.csv", encoding="utf-8")
test_ids = test_df["ID"].values

# 取得欄位列表
columns = test_df.columns.tolist()

# 切出欄位
first_column = columns[:1]  # 第一欄 (ID)
middle_column = columns[1:]  # 中間欄位（不含首欄）

# 選取 middle 中的前 120 欄
middle_column_120 = middle_column[:120]

# 重新組合欄位順序
selected_columns = first_column + middle_column_120

# 只保留需要的欄位
test_df = test_df[selected_columns]

X_test_raw = test_df.drop(["ID"], axis=1, errors='ignore')
X_test_numeric = X_test_raw.select_dtypes(include=["number"])

# ========== Step 3: Imputer 處理 ==========
X_test_numeric = X_test_numeric.reindex(columns=model.feature_names_in_, fill_value=0)
X_test_imputed = imputer.transform(X_test_numeric)

# 轉回 DataFrame
X_test_df = pd.DataFrame(X_test_imputed, columns=X_test_numeric.columns)

# ========== Step 4: 對齊特徵 ==========
# 進行對齊特徵
aligned_X = pd.DataFrame()
model_features = model.feature_names_in_

# 收集對齊好的特徵
aligned_features = []
for feat in model_features:
    if feat in X_test_df.columns:
        aligned_features.append(X_test_df[feat])
    else:
        print(f"為缺少的特徵 '{feat}' 補 0")
        aligned_features.append(pd.Series(0, index=X_test_df.index))

# 一次性合併所有特徵
aligned_X = pd.concat(aligned_features, axis=1)
aligned_X.columns = model_features  # 設置正確的欄位名稱

print(f"最終特徵數: {aligned_X.shape[1]}")

# ========== Step 5: 預測 ==========
# 使用 XGBClassifier 的 predict_proba 方法獲取預測概率
y_pred_proba = model.predict_proba(aligned_X)[:, 1]
y_pred = (y_pred_proba > threshold).astype(int)

# ========== Step 6: 輸出結果 ==========
result_df = pd.DataFrame({
    "ID": test_ids,
    "飆股": y_pred
})
result_df.to_csv("result.csv", index=False)
print("預測完成，已儲存至 result.csv")
print(f"預測分布：0: {(y_pred == 0).sum()} 筆，1: {(y_pred == 1).sum()} 筆")