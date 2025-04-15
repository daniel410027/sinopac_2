import pandas as pd
import numpy as np
import catboost as cb
import optuna
import joblib
import json
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ======== 圖表中文字體設定 ========
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'Heiti TC', 'STFangsong']
plt.rcParams['axes.unicode_minus'] = False

# ======== 讀取資料 ========
df = pd.read_csv("../../database/filter_training.csv", encoding="utf-8")
# 取得欄位列表
columns = df.columns.tolist()

# 切出欄位
first_column = columns[:1]  # 第一欄
middle_column = columns[1:-1]  # 中間欄位（不含首尾）
last_column = columns[-1:]  # 最後一欄

# 選取 middle 中的前 120 欄
middle_column_120 = middle_column[:120]

# 重新組合欄位順序
selected_columns = first_column + middle_column_120 + last_column

# 建立新的 DataFrame
df = df[selected_columns]
target = df["飆股"]
features_raw = df.drop(["ID", "飆股"], axis=1)
features_numeric = features_raw.select_dtypes(include=["number"])

# ======== 標準化處理（可選） ========
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_numeric)

# ======== 缺失值處理 ========
# 移除全為 NaN 的欄位
cols_all_nan = features_numeric.columns[features_numeric.isna().all()].tolist()
print(f"移除全部為缺失值的欄位: {cols_all_nan}")
features_numeric = features_numeric.drop(columns=cols_all_nan)

# 使用中位數填補缺失值
imputer = SimpleImputer(strategy='median')
features_imputed = imputer.fit_transform(features_numeric)
features_clean = pd.DataFrame(features_imputed, columns=features_numeric.columns)

# ======== 類別不平衡比例計算 ========
positive = (target == 1).sum()
negative = (target == 0).sum()
scale_pos_weight = negative / positive
print(f"✅ scale_pos_weight 計算完成：{scale_pos_weight:.2f}")

# ======== 切分訓練與驗證集 ========
X_train, X_val, y_train, y_val = train_test_split(features_clean, target, test_size=0.2, random_state=62)

# ======== 準備CatBoost數據集 ========
train_data = cb.Pool(X_train, label=y_train)
eval_data = cb.Pool(X_val, label=y_val)

# ======== Optuna 目標函數 ========
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0),
        "scale_pos_weight": scale_pos_weight,
        "loss_function": "Logloss",
        "eval_metric": "F1",
        "random_seed": 62,
        "verbose": False
    }

    model = cb.CatBoost(params)
    model.fit(train_data, eval_set=eval_data, early_stopping_rounds=50, verbose=False)

    y_val_proba = model.predict(X_val, prediction_type="Probability")[:, 1]

    # 最佳閾值搜尋
    thresholds = np.linspace(0.1, 0.9, 9)
    best_f1, best_threshold = 0, 0.5
    for threshold in thresholds:
        y_pred = (y_val_proba > threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold

    trial.set_user_attr('best_threshold', best_threshold)
    trial.set_user_attr('best_iteration', model.get_best_iteration())
    return best_f1

# ======== 執行 Optuna 超參數搜尋 ========
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

best_params = study.best_params
best_threshold = study.best_trial.user_attrs['best_threshold']
best_iteration = study.best_trial.user_attrs['best_iteration']
print("🥇 最佳參數：", best_params)
print(f"🎯 最佳閾值：{best_threshold:.2f}")
print(f"🌲 最佳迭代：{best_iteration}")

# ======== 最佳模型訓練與預測 ========
best_params['scale_pos_weight'] = scale_pos_weight
best_params['loss_function'] = 'Logloss'
best_params['eval_metric'] = "F1"
best_params['random_seed'] = 62
best_params['iterations'] = best_iteration  # 使用最佳迭代次數

best_model = cb.CatBoost(best_params)
best_model.fit(train_data, verbose=False)

y_val_proba = best_model.predict(X_val, prediction_type="Probability")[:, 1]
y_val_pred = (y_val_proba > best_threshold).astype(int)

# ======== 儲存模型與預處理器 ========
best_model.save_model("optuna_best_catboost.cbm")
joblib.dump(imputer, "optuna_imputer.pkl")
with open("optuna_model_info.json", "w") as f:
    json.dump({
        'best_params': {k: float(v) if isinstance(v, (int, float)) else v for k, v in best_params.items()},
        'best_threshold': float(best_threshold),
        'features': list(features_clean.columns),
        'scale_pos_weight': float(scale_pos_weight)
    }, f, indent=4)

print("✅ 模型與資訊已儲存完成")

# ======== 評估報告與混淆矩陣 ========
print("\n📊 分類報告：")
print(classification_report(y_val, y_val_pred))

print("\n🔍 混淆矩陣：")
print(confusion_matrix(y_val, y_val_pred))

# ======== 特徵重要性視覺化 ========
feature_importance = pd.DataFrame({
    'feature': features_clean.columns,
    'importance': best_model.get_feature_importance()
}).sort_values('importance', ascending=False)

feature_importance.to_csv("feature_importance.csv", index=False)

plt.figure(figsize=(12, 8))
plt.barh(feature_importance.head(20)['feature'], feature_importance.head(20)['importance'])
plt.xlabel('重要性')
plt.ylabel('特徵')
plt.title('Top 20 特徵重要性')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("optuna_feature_importance.png")
plt.close()
print("✅ 特徵重要性圖已儲存為 optuna_feature_importance.png")

# ======== 預測機率分布圖 ========
plt.figure(figsize=(10, 6))
plt.hist(y_val_proba, bins=50, alpha=0.7, color='blue')
plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'最佳閾值 ({best_threshold:.2f})')
plt.xlabel('預測機率')
plt.ylabel('頻率')
plt.title('驗證集預測機率分布')
plt.legend()
plt.grid(True)
plt.savefig("optuna_prediction_distribution.png")
plt.close()
print("✅ 預測概率分布圖已儲存為 optuna_prediction_distribution.png")