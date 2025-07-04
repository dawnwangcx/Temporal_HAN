import os
from WindPy import w
import pandas as pd

# 初始化Wind API
w.start()

# 定义合约代码和时间范围
start_date = "2012-01-01"
end_date = "2024-09-01"  # 可以根据需要调整
contract_code = pd.read_excel('Futures.xlsx')

for contract in contract_code.codes:
    print(contract)
    data = w.wsd(contract, "open,high,low,close,volume", start_date, end_date,usedf=True)[1]
    name = w.wsd(contract,"sec_name").Data[0][0]
    data.columns = [f'{contract}_open',f'{contract}_high',f'{contract}_low',f'{contract}_close',f'{contract}_volume']
    data.to_csv(name+'.csv')

w.close()