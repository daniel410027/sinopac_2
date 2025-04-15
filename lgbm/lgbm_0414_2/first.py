import pandas as pd
import numpy as np
import lightgbm as lgb
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
target = df["飆股"]
features_raw = df.drop(["ID", "飆股"], axis=1)
features_numeric = features_raw.select_dtypes(include=["number"])

# ======== 標準化處理（可選） ========
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_numeric)

# ======== 缺失值處理（Winsorize略過，這裡保留原始） ========
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

# ======== 自定義 F1 評估函數（可用於 CV 評估） ========
def lgbm_f1_score(y_true, y_pred):
    y_pred_binary = np.round(y_pred)
    return 'f1', f1_score(y_true, y_pred_binary), True

# ======== Optuna 目標函數 ========
def objective(trial):
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "scale_pos_weight": scale_pos_weight,
        "max_bin": trial.suggest_int("max_bin", 255, 1024),
        "random_state": 62,
        "verbose": -1
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    y_val_proba = model.predict_proba(X_val)[:, 1]

    # 最佳閾值搜尋
    thresholds = np.linspace(0.1, 0.9, 9)
    best_f1, best_threshold = 0, 0.5
    for threshold in thresholds:
        y_pred = (y_val_proba > threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold

    trial.set_user_attr('best_threshold', best_threshold)
    return best_f1

# ======== 執行 Optuna 超參數搜尋 ========
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

best_params = study.best_params
best_threshold = study.best_trial.user_attrs['best_threshold']
print("🥇 最佳參數：", best_params)
print(f"🎯 最佳閾值：{best_threshold:.2f}")

# ======== 最佳模型訓練與預測 ========
best_params['scale_pos_weight'] = scale_pos_weight
best_model = lgb.LGBMClassifier(**best_params, random_state=62)
best_model.fit(X_train, y_train)

y_val_proba = best_model.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_proba > best_threshold).astype(int)

# ======== 儲存模型與預處理器 ========
joblib.dump(best_model, "optuna_best_lgbm.pkl")
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
    'importance': best_model.feature_importances_
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


### 下一份不用這個
# ======== 測試不同特徵數量對 F1 的影響（每次都用 Optuna 重新調參）+ 中間儲存 ========


save_path = "f1_scores_by_feature_count.json"
f1_scores_by_feature_count = {}

# 嘗試讀取已存在的中間結果
if os.path.exists(save_path):
    with open(save_path, "r") as f:
        f1_scores_by_feature_count = json.load(f)
    print(f"📂 已讀取中間結果，共有 {len(f1_scores_by_feature_count)} 筆")
else:
    print("📁 尚無中間結果，從頭開始")

feature_counts = list(range(100, min(251, len(feature_importance)+1), 1))
print("\n🚀 開始測試不同特徵數量對 F1 分數的影響（支援中途存檔與續跑）...")

for top_n in feature_counts:
    key = str(top_n)
    if key in f1_scores_by_feature_count:
        print(f"⏩ 特徵數: {top_n} 已計算過，略過")
        continue

    selected_features = feature_importance['feature'].head(top_n).tolist()
    X_train_sub = X_train[selected_features]
    X_val_sub = X_val[selected_features]

    def objective_sub(trial):
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "scale_pos_weight": scale_pos_weight,
            "max_bin": trial.suggest_int("max_bin", 255, 1024),
            "random_state": 62,
            "verbose": -1
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_sub, y_train)

        y_pred_proba = model.predict_proba(X_val_sub)[:, 1]
        thresholds = np.linspace(0.1, 0.9, 9)
        best_f1 = 0

        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            if f1 > best_f1:
                best_f1 = f1

        return best_f1

    study_sub = optuna.create_study(direction="maximize")
    study_sub.optimize(objective_sub, n_trials=10, show_progress_bar=False)

    best_f1 = study_sub.best_value
    f1_scores_by_feature_count[key] = best_f1
    print(f"✅ 特徵數: {top_n}，最佳 F1 Score: {best_f1:.4f}")

    # 每次跑完就儲存一次
    with open(save_path, "w") as f:
        json.dump(f1_scores_by_feature_count, f, indent=4)

# ======== 繪圖：特徵數 vs F1 Score ========
feature_counts_done = [int(k) for k in f1_scores_by_feature_count.keys()]
f1_scores_done = [f1_scores_by_feature_count[k] for k in f1_scores_by_feature_count]

plt.figure(figsize=(10, 6))
plt.plot(feature_counts_done, f1_scores_done, marker='o')
plt.xlabel("使用的特徵數量")
plt.ylabel("F1 分數")
plt.title("不同特徵數量對 F1 Score 的影響")
plt.grid(True)
plt.savefig("f1_score_by_feature_count.png")
plt.close()
print("📈 特徵數量 vs F1 score 圖已儲存為 f1_score_by_feature_count.png")
