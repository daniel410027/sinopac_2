import pandas as pd
import numpy as np
import optuna
import joblib
import json
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ======== 圖表中文字體設定 ========
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'Heiti TC', 'STFangsong']
plt.rcParams['axes.unicode_minus'] = False

# ======== estimator 名稱設定 ========
ada_param_name = 'estimator' if sklearn.__version__ >= '1.2' else 'base_estimator'

# ======== 讀取與前處理資料 ========
df = pd.read_csv("../database/filter_training.csv", encoding="utf-8")
columns = df.columns.tolist()
selected_columns = columns[:1] + columns[1:-1][:120] + columns[-1:]
df = df[selected_columns]

target = df["飆股"]
features_numeric = df.drop(["ID", "飆股"], axis=1).select_dtypes(include=["number"])
features_numeric = features_numeric.drop(columns=features_numeric.columns[features_numeric.isna().all()])

imputer = SimpleImputer(strategy='median')
features_imputed = imputer.fit_transform(features_numeric)
features_clean = pd.DataFrame(features_imputed, columns=features_numeric.columns)

positive, negative = (target == 1).sum(), (target == 0).sum()
scale_pos_weight = negative / positive
print(f"✅ scale_pos_weight 計算完成：{scale_pos_weight:.2f}")

X_train, X_val, y_train, y_val = train_test_split(features_clean, target, test_size=0.2, random_state=62)

# ======== Optuna 搜尋函數 ========
def objective(trial):
    dt_params = {
        "max_depth": trial.suggest_int("max_depth", 1, 5),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20)
    }

    ada_params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
        ada_param_name: DecisionTreeClassifier(**dt_params),
        "random_state": 62
    }

    model = AdaBoostClassifier(**ada_params)
    model.fit(X_train, y_train)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    thresholds = np.linspace(0.1, 0.9, 9)
    best_f1, best_threshold = 0, 0.5
    for threshold in thresholds:
        y_pred = (y_val_proba > threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold

    trial.set_user_attr("best_threshold", best_threshold)
    return best_f1

# ======== 執行 Optuna ========
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

best_params = study.best_trial.params
best_threshold = study.best_trial.user_attrs["best_threshold"]
print("🥇 最佳參數：", best_params)
print(f"🎯 最佳閾值：{best_threshold:.2f}")

# ======== 重建與訓練最佳模型 ========
base_estimator = DecisionTreeClassifier(
    max_depth=best_params.pop("max_depth"),
    min_samples_split=best_params.pop("min_samples_split")
)

ada_init_args = {
    **best_params,
    ada_param_name: base_estimator,
    "random_state": 62
}

best_model = AdaBoostClassifier(**ada_init_args)
best_model.fit(X_train, y_train)

# ======== 儲存模型與資訊 ========
joblib.dump(best_model, "optuna_best_adaboost.pkl")
joblib.dump(imputer, "optuna_imputer.pkl")
with open("optuna_adaboost_info.json", "w") as f:
    json.dump({
        'best_params': best_params,
        'base_estimator': {'max_depth': base_estimator.max_depth, 'min_samples_split': base_estimator.min_samples_split},
        'best_threshold': float(best_threshold),
        'features': list(features_clean.columns),
        'scale_pos_weight': float(scale_pos_weight),
        'sklearn_version': sklearn.__version__
    }, f, indent=4)
print("✅ 模型與資訊已儲存完成")

# ======== 預測與報告 ========
y_val_proba = best_model.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_proba > best_threshold).astype(int)

print("\n📊 分類報告：")
print(classification_report(y_val, y_val_pred))

print("\n🔍 混淆矩陣：")
print(confusion_matrix(y_val, y_val_pred))

# ======== 特徵重要性圖 ========
feature_importance = pd.DataFrame({
    'feature': features_clean.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

feature_importance.to_csv("adaboost_feature_importance.csv", index=False)

plt.figure(figsize=(12, 8))
plt.barh(feature_importance.head(20)['feature'], feature_importance.head(20)['importance'])
plt.xlabel('重要性')
plt.ylabel('特徵')
plt.title('Top 20 特徵重要性')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("adaboost_feature_importance.png")
plt.close()
print("✅ 特徵重要性圖已儲存")

# ======== 預測分布圖 ========
plt.figure(figsize=(10, 6))
plt.hist(y_val_proba, bins=50, alpha=0.7, color='blue')
plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'最佳閾值 ({best_threshold:.2f})')
plt.xlabel('預測機率')
plt.ylabel('頻率')
plt.title('驗證集預測機率分布')
plt.legend()
plt.grid(True)
plt.savefig("adaboost_prediction_distribution.png")
plt.close()
print("✅ 預測機率分布圖已儲存")
