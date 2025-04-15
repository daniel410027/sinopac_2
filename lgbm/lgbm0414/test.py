import joblib

# 載入 imputer
imputer = joblib.load('optuna_imputer.pkl')

# 查看 imputer 的類型
print(f"物件類型: {type(imputer)}")

# 查看 imputer 的基本資訊
print(f"Imputer 策略: {imputer.strategy}")
print(f"填補值: {imputer.statistics_}")

# 檢查 imputer 處理的特徵名稱
if hasattr(imputer, 'feature_names_in_'):
    print(f"特徵數量: {len(imputer.feature_names_in_)}")
    print(f"前 10 個特徵名稱: {imputer.feature_names_in_[:10]}")

# 查看其他重要屬性
print("\n所有公開屬性:")
for attr in dir(imputer):
    if not attr.startswith('_'):  # 忽略私有屬性
        try:
            value = getattr(imputer, attr)
            # 如果是方法，就跳過
            if callable(value):
                continue
            # 如果是陣列，只顯示形狀或前幾個元素
            if hasattr(value, 'shape'):
                print(f"{attr}: 形狀={value.shape}")
            elif isinstance(value, list) and len(value) > 10:
                print(f"{attr}: [前 5 個元素] {value[:5]}")
            else:
                print(f"{attr}: {value}")
        except Exception as e:
            print(f"{attr}: <無法顯示> - {str(e)}")

# 如果想看每個特徵的中位數值 (假設使用中位數策略)
if hasattr(imputer, 'statistics_') and hasattr(imputer, 'feature_names_in_'):
    print("\n前 20 個特徵的中位數值:")
    for i, (feature, stat) in enumerate(zip(imputer.feature_names_in_, imputer.statistics_)):
        if i < 20:  # 只顯示前 20 個
            print(f"{feature}: {stat}")