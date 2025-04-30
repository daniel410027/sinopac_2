import pdblp
import blpapi
from xbbg import blp
import pandas as pd

con = pdblp.BCon(port = 8194, timeout = 5000)
con.start()


index_name = blp.bds('RAY Index', "INDX_MWEIGHT", END_DATE_OVERRIDE="20000101")
index_name.to_csv('RAY_Index_MWEIGHT.csv', index=False)
print(len(index_name))
# 2814
print(index_name)

con.stop()

# 提取 member_ticker_and_exchange_code 欄位的最後兩個字符（結尾部分）
index_name['suffix'] = index_name['member_ticker_and_exchange_code'].str[-2:]
# 統計不同結尾的數量
unique_suffixes = index_name['suffix'].value_counts()
# 顯示不同結尾的統計數量
print(unique_suffixes)
'''
suffix
UN    1552
UQ    1163
UA      73
US      26

其中，前722筆資料數字前綴，根據chatgpt，推測可能是
1. 內部識別碼 (Internal Identifiers)
2. 特殊交易產品（如衍生品、債券等）
3. 債券識別碼 (Bond Identifiers)
4. 合約標識符（如期貨、選擇權等）
'''

# 去除資料中數字前綴的資料
index_name_cleaned = index_name[~index_name['member_ticker_and_exchange_code'].str.match(r'^\d')]
print(index_name_cleaned)

# 提取 member_ticker_and_exchange_code 欄位並加上 " Equity"
code_list = index_name_cleaned[['member_ticker_and_exchange_code']].copy()
code_list['member_ticker_and_exchange_code'] = code_list['member_ticker_and_exchange_code'].apply(lambda x: x + " Equity")

# 儲存為 CSV 檔案
code_list.to_csv('code_list.csv', index=False)

# 顯示資料
print(code_list)
