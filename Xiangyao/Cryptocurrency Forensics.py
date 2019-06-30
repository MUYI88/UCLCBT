import json
import pandas as pd
import pickle
import numpy as np
from sklearn.externals import joblib
from matplotlib.backends.backend_pdf import PdfPages
from pyecharts import options as opts


with open('rippleledger-2.json','r') as f:
    rippledata = json.load(f)

for name in rippledata['result'].keys():
    print(name) #ledger, ledger_hash, ledger_index, status, validated


ledger=rippledata['result']['ledger'].keys() #leger下的健
joblib.dump(ledger, 'ledger.pkl')
ledger=joblib.load('ledger.pkl').values

############################################################################
################# Data ##############################

accountState=pd.DataFrame(rippledata['result']['ledger']['accountState'])
transactions=pd.DataFrame(rippledata['result']['ledger']['transactions'])

joblib.dump(accountState, 'accountState.pkl') # 储存accountState数据
joblib.dump(transactions, 'transactions.pkl') # 储存transactions数据

accountState=joblib.load('accountState.pkl')  #load accountState数据
transactions=joblib.load('transactions.pkl')  #load transactions数据

#close_time_human=pd.DataFrame(rippledata['result']['ledger']['close_time_human'])
#：'2019-Jun-03 11:07:42.000000000'

accountState.shape[1] #列数：62
accountState.shape[0] #行数：2997971

transactions.shape[1] #列数：17
transactions.shape[0] #行数：23

transaction_name=transactions.columns.values.tolist()
accountState_name=accountState.columns.values.tolist()

#amount,balance,index重复，taker gets，taker pays
#在aS中有些会缺少account和balance前两项
#LedgerEntryType** >>> ledger object type



#joblib.dump(Balance,'Balance.pkl')
Balance=joblib.load('Balance.pkl')  #load accountState中Balance的数据

#joblib.dump(LedgerEntryType,'LedgerEntryType.pkl')
LedgerEntryType=joblib.load('LedgerEntryType.pkl')
LedgerEntryType=list(LedgerEntryType)

from collections import Counter
import time

def get_count_by_counter(l):
    t1 = time.time()
    count = Counter(l)   #类型： <class 'collections.Counter'>
    t2 = time.time()
    print (t2-t1)
    count_dict = dict(count)   #类型： <type 'dict'>
    return count_dict

ledger_type=get_count_by_counter(LedgerEntryType) #输出ledger object type的元素及其次数
ledger_type=list(ledger_type)

len(LedgerEntryType)

lt_number=ledger_type.values()
lt_number=list(lt_number)

#lt_number=np.array(lt_number) #list转numpy

lt_key=ledger_type.keys()
lt_key=list(lt_key)

import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
plt.bar(lt_key, lt_number)
plt.xticks(rotation=45)
font = {'weight' : 'bold',
        'size'   : 12,
        }

plt.xlabel('Ledger Object Type',fontdict=font)
plt.ylabel('The number of Ledger Object Type',fontdict=font)
plt.show()

########  acoountState: 'Account' columns ##########
Account=accountState['Account']
joblib.dump(Account,'Account.pkl')
Account=joblib.load('Account.pkl')

account_no_nan=Account.dropna()
account=account_no_nan.drop_duplicates(keep='first', inplace=False)
account.shape[0]    #不重复的account数量  1681149
account_no_nan.shape[0]   #不含nan的所有account数量  1731068


merge=sum(np.array(lt_number[5:10]))
merge=merge+np.array(lt_number[3])
merge=['other',merge]
offer=['offer',lt_number[4]]
pie_list=[ [a,b] for a,b in zip(lt_key[0:3],lt_number[0:3])]+[offer]+[merge]



runfile('/Users/apple/PycharmProjects/Forensics/pyecharts/example/pie_example.py', wdir='/Users/apple/PycharmProjects/Forensics/pyecharts/example')
#在此程序下才能正确执行饼图

#pie_list = [['offer', 46190],['DirectoryNode', 467108],['AccountRoot', 1681149], ['RippleState', 781751], ['other', 21773]]
#顺序改变
pie_1 = Pie().add("", pie_list,is_clockwise=False,
                  label_opts=opts.LabelOpts(
                      position="outside",
                      formatter=" {b|{b}: }{c}  {per|{d}%}  ",

                      background_color="#eee",
                      border_color="#aaa",
                      border_width=1,
                      border_radius=2,
                      rich={
                          "a": {"color": "#999", "lineHeight": 22, "align": "center"},
                          "abg": {
                              "backgroundColor": "#e3e3e3",
                              "width": "100%",
                              "align": "right",
                              "height": 0,
                              "borderRadius": [4, 4, 0, 0],
                          },
                          "hr": {
                              "borderColor": "#aaa",
                              "width": "100%",
                              "borderWidth": 0.5,
                              "height": 0,
                          },
                          "b": {"fontSize": 12, "lineHeight": 33},
                          "per": {
                              "color": "#eee",
                              "backgroundColor": "#334455",
                              "padding": [2, 4],
                              "borderRadius": 2,
                          },
                      },
                  ),
                  ).set_global_opts(title_opts=opts.TitleOpts(title="Ledger Object Type Occupation"))
pie_1.render()

sum(lt_number) #长度一样，不含NaN值
print(pd.isnull(c)) # 判断是否series里有NaN值
print(c.isnull())

####### DATA ######
url='https://data.ripple.com/v2/transactions/?start=2019-04-26T00:00:00z&end=2019-04-26T00:00:12&result=tesSUCCESS&limit=100&marker=+mark'
r=requests.get(url)
response_dict=r.json()
mark=response_dict['marker']