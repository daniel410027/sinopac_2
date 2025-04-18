import pandas as pd
import os
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import json
import shap

# 📁 資料夾路徑
data_dir = "../LR_database"
output_dir = "../LR_database/lr_results"
imputed_dir = "../LR_database/imputed_data"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(imputed_dir, exist_ok=True)

try:
    for i in range(1, 17):
        output_path = os.path.join(output_dir, f"feature_importance_{i}.json")
        
        # 檢查是否已完成
        if os.path.exists(output_path):
            print(f"⏭ 已存在，跳過：feature_importance_{i}.json")
            continue

        file_path = os.path.normpath(os.path.join(data_dir, f"filtered_number_{i}.csv"))
        print(f"🚀 正在處理：{file_path}")
        
        # 讀取資料
        df = pd.read_csv(file_path)

        # 拆分欄位
        X_raw = df.iloc[:, 1:-1]  # 去掉 ID 與 飆股
        y = df.iloc[:, -1]
        feature_names = X_raw.columns.tolist()

        # 缺值填補（中位數）
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X_raw)

        # 儲存填補缺值但尚未標準化的資料
        df_imputed = pd.DataFrame(X_imputed, columns=feature_names)
        df_imputed.insert(0, 'ID', df.iloc[:, 0])         # 還原 ID
        df_imputed['飆股'] = y                            # 還原 target
        df_imputed.to_csv(os.path.join(imputed_dir, f"imputed_{i}.csv"), index=False)
        print(f"📄 已儲存填補缺值資料：imputed_{i}.csv")

        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Optuna 調參
        def objective(trial):
            C = trial.suggest_loguniform('C', 1e-3, 1e2)
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
            solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
            model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000)
            score = cross_val_score(model, X_scaled, y, cv=3, scoring='f1').mean()
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)
        
        best_params = study.best_params
        print(f"✅ 最佳參數：{best_params}")
        
        # 訓練最佳模型
        final_model = LogisticRegression(**best_params, solver='liblinear' if best_params['penalty'] == 'l1' else 'lbfgs', max_iter=1000)
        final_model.fit(X_scaled, y)

        # 使用 SHAP 計算特徵重要性
        explainer = shap.LinearExplainer(final_model, X_scaled, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_scaled)
        shap_importance = dict(zip(
            feature_names,
            np.abs(shap_values).mean(axis=0).tolist()
        ))

        # 儲存 SHAP 特徵重要性
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(shap_importance, f, ensure_ascii=False, indent=2)

        print(f"📦 已儲存 SHAP feature_importance_{i}.json\n")

except KeyboardInterrupt:
    print("⛔️ 使用者中斷程序，已安全停止")
except Exception as e:
    print(f"⚠️ 發生錯誤：{e}")

print("🎉 處理完成（或已中斷）")
