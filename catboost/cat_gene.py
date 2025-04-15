import pandas as pd
import numpy as np
import catboost as cb
import joblib
import json
import matplotlib.pyplot as plt
import os
import random
from deap import base, creator, tools, algorithms

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

# ======== 設置遺傳演算法參數範圍 ========
param_ranges = {
    'iterations': (100, 2000),
    'learning_rate': (0.01, 0.3),
    'depth': (4, 10),
    'l2_leaf_reg': (0.1, 10.0),
    'border_count': (32, 255),
    'bagging_temperature': (0.0, 1.0),
    'random_strength': (0.1, 10.0)
}

# ======== 建立遺傳演算法工具 ========
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# 定義基因編碼（每個參數編碼為一個基因）
toolbox.register("attr_iterations", random.randint, param_ranges['iterations'][0], param_ranges['iterations'][1])
toolbox.register("attr_learning_rate", random.uniform, param_ranges['learning_rate'][0], param_ranges['learning_rate'][1])
toolbox.register("attr_depth", random.randint, param_ranges['depth'][0], param_ranges['depth'][1])
toolbox.register("attr_l2_leaf_reg", random.uniform, param_ranges['l2_leaf_reg'][0], param_ranges['l2_leaf_reg'][1])
toolbox.register("attr_border_count", random.randint, param_ranges['border_count'][0], param_ranges['border_count'][1])
toolbox.register("attr_bagging_temp", random.uniform, param_ranges['bagging_temperature'][0], param_ranges['bagging_temperature'][1])
toolbox.register("attr_random_strength", random.uniform, param_ranges['random_strength'][0], param_ranges['random_strength'][1])

# 創建個體和族群
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_iterations, toolbox.attr_learning_rate, toolbox.attr_depth, 
                  toolbox.attr_l2_leaf_reg, toolbox.attr_border_count, toolbox.attr_bagging_temp,
                  toolbox.attr_random_strength), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ======== 定義評估函數 ========
def evaluate_model(individual):
    iterations, learning_rate, depth, l2_leaf_reg, border_count, bagging_temp, random_strength = individual
    
    params = {
        "iterations": int(iterations),
        "learning_rate": learning_rate,
        "depth": int(depth),
        "l2_leaf_reg": l2_leaf_reg,
        "border_count": int(border_count),
        "bagging_temperature": bagging_temp,
        "random_strength": random_strength,
        "scale_pos_weight": scale_pos_weight,
        "loss_function": "Logloss",
        "eval_metric": "F1",
        "random_seed": 62,
        "verbose": False
    }
    
    try:
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
                
        # 保存該個體的閾值和最佳迭代次數
        individual.best_threshold = best_threshold
        individual.best_iteration = model.get_best_iteration()
        individual.f1_score = best_f1
        
        return (best_f1,)
    except Exception as e:
        print(f"評估錯誤: {e}")
        return (0.0,)

toolbox.register("evaluate", evaluate_model)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[param_ranges[k][0] for k in param_ranges], 
                 up=[param_ranges[k][1] for k in param_ranges], eta=20.0, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# ======== 執行遺傳演算法 ========
print("🧬 開始遺傳演算法搜尋...")
population = toolbox.population(n=30)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)
stats.register("std", np.std)

population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=15, 
                                          stats=stats, halloffame=hof, verbose=True)

# 取得最佳個體
best_ind = hof[0]
best_params = {
    "iterations": int(best_ind.best_iteration),  # 使用早停找到的最佳迭代次數
    "learning_rate": best_ind[1],
    "depth": int(best_ind[2]),
    "l2_leaf_reg": best_ind[3],
    "border_count": int(best_ind[4]),
    "bagging_temperature": best_ind[5],
    "random_strength": best_ind[6],
    "scale_pos_weight": scale_pos_weight,
    "loss_function": "Logloss",
    "eval_metric": "F1",
    "random_seed": 62,
    "verbose": False
}

best_threshold = best_ind.best_threshold
best_f1 = best_ind.f1_score

print("\n🥇 最佳參數：", best_params)
print(f"🎯 最佳閾值：{best_threshold:.2f}")
print(f"📊 最佳 F1 分數：{best_f1:.4f}")

# ======== 最佳模型訓練與預測 ========
best_model = cb.CatBoost(best_params)
best_model.fit(train_data, verbose=False)

y_val_proba = best_model.predict(X_val, prediction_type="Probability")[:, 1]
y_val_pred = (y_val_proba > best_threshold).astype(int)

# ======== 儲存模型與預處理器 ========
best_model.save_model("genetic_best_catboost.cbm")
joblib.dump(imputer, "genetic_imputer.pkl")
with open("genetic_model_info.json", "w") as f:
    json.dump({
        'best_params': {k: float(v) if isinstance(v, (int, float)) else v for k, v in best_params.items()},
        'best_threshold': float(best_threshold),
        'features': list(features_clean.columns),
        'scale_pos_weight': float(scale_pos_weight),
        'best_f1': float(best_f1)
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
plt.savefig("genetic_feature_importance.png")
plt.close()
print("✅ 特徵重要性圖已儲存為 genetic_feature_importance.png")

# ======== 預測機率分布圖 ========
plt.figure(figsize=(10, 6))
plt.hist(y_val_proba, bins=50, alpha=0.7, color='blue')
plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'最佳閾值 ({best_threshold:.2f})')
plt.xlabel('預測機率')
plt.ylabel('頻率')
plt.title('驗證集預測機率分布')
plt.legend()
plt.grid(True)
plt.savefig("genetic_prediction_distribution.png")
plt.close()
print("✅ 預測概率分布圖已儲存為 genetic_prediction_distribution.png")

# ======== 學習曲線 ========
generations = range(len(logbook))
avg_fitness = [d['avg'] for d in logbook]
max_fitness = [d['max'] for d in logbook]

plt.figure(figsize=(10, 6))
plt.plot(generations, avg_fitness, 'b-', label='平均適應度')
plt.plot(generations, max_fitness, 'r-', label='最佳適應度')
plt.xlabel('世代')
plt.ylabel('F1 分數')
plt.title('遺傳演算法學習曲線')
plt.legend()
plt.grid(True)
plt.savefig("genetic_learning_curve.png")
plt.close()
print("✅ 遺傳演算法學習曲線已儲存為 genetic_learning_curve.png")