import json
import pandas as pd
import pickle
import numpy as np
from sklearn.externals import joblib
from matplotlib.backends.backend_pdf import PdfPages
from pyecharts import options as opts
from pyecharts.charts import Pie
import requests
import time
import  multiprocessing as mp


with open('rippleledger-2.json','r') as f:
    rippledata = json.load(f)

for name in rippledata['result'].keys():
    print(name) #ledger, ledger_hash, ledger_index, status, validated


ledger=rippledata['result']['ledger'].keys() #leger's key
joblib.dump(ledger, 'ledger.pkl')
ledger=joblib.load('ledger.pkl').values

############################################################################
################# Data ##############################

accountState=pd.DataFrame(rippledata['result']['ledger']['accountState'])
transactions=pd.DataFrame(rippledata['result']['ledger']['transactions'])

joblib.dump(accountState, 'accountState.pkl') # store accountState data
joblib.dump(transactions, 'transactions.pkl') # store transactions data

accountState=joblib.load('accountState.pkl')  #load accountState data
transactions=joblib.load('transactions.pkl')  #load transactions data

#close_time_human=pd.DataFrame(rippledata['result']['ledger']['close_time_human'])
#：'2019-Jun-03 11:07:42.000000000'

accountState.shape[1] #columns：62
accountState.shape[0] #rows：2997971

transactions.shape[1] #columns：17
transactions.shape[0] #rows：23

transaction_name=transactions.columns.values.tolist()
accountState_name=accountState.columns.values.tolist()



#joblib.dump(Balance,'Balance.pkl')
Balance=joblib.load('Balance.pkl')  #load accountState中Balance的数据

#joblib.dump(LedgerEntryType,'LedgerEntryType.pkl')
LedgerEntryType=joblib.load('LedgerEntryType.pkl')
LedgerEntryType=list(LedgerEntryType)

from collections import Counter
import time

def get_count_by_counter(l):
    t1 = time.time()
    count = Counter(l)
    t2 = time.time()
    print (t2-t1)
    count_dict = dict(count)
    return count_dict

ledger_type=get_count_by_counter(LedgerEntryType) #output ledger object type's element and frequent
ledger_type=list(ledger_type)

len(LedgerEntryType)

ledger_type=get_count_by_counter(LedgerEntryType)
lt_number=ledger_type.values()
lt_number=list(lt_number)



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
account.shape[0]
account_no_nan.shape[0]


merge=sum(np.array(lt_number[5:10]))
merge=merge+np.array(lt_number[3])
merge=['other',merge]
offer=['offer',lt_number[4]]
pie_list=[ [a,b] for a,b in zip(lt_key[0:3],lt_number[0:3])]+[offer]+[merge]




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





########## Data Collection ##########
marker_exists = True
first = True

time_start=time.time()

while marker_exists == True:
    if first == True:
        url_0 = 'https://data.ripple.com/v2/transactions/?start=2019-05-02T00:00:01z&end=2019-05-03T00:00:00&result=tesSUCCESS'
        url_0
        r = requests.get(url_0)
        response_dict = r.json()
        mark = response_dict['marker']
        try:
            response_dict['marker']
        except NameError:
            marker_exists = False
        else:
            marker_exists = True  # 存在

        transaction_update = response_dict['transactions']  # Transactions generated by the first response

        first=False

    else:

        mark = response_dict['marker']
        url = url_0 + '&limit=100&marker=' + mark
        url
        r = requests.get(url)
        response_dict = r.json()

        try:
            response_dict['transactions']
        except KeyError:
            transaction_exists = False
        else:
            transaction_exists = True  # exists

        if transaction_exists == True:

            new_transaction = response_dict['transactions']
            transaction_update = transaction_update + new_transaction
            try:
                response_dict['marker']
            except KeyError:
                marker_exists = False
            else:
                marker_exists = True  # exists

        else:
            while transaction_exists == False:
                time.sleep(20)
                mark = mark
                url = url_0 + '&limit=100&marker=' + mark
                url
                r = requests.get(url)
                response_dict = r.json()

                try:
                    response_dict['transactions']
                except KeyError:
                    transaction_exists = False
                else:
                    transaction_exists = True  # exists

            new_transaction = response_dict['transactions']
            transaction_update = transaction_update + new_transaction

            try:
                response_dict['marker']
            except KeyError:
                marker_exists = False
            else:
                marker_exists = True  # exists

time_end=time.time()
print('totally cost',time_end-time_start)


#################################################################################
###################### May data ############

time_start = time.time()

Data={}
for i in range(7, 9):
    marker_exists = True
    first = True

    a = str(i)
    b = str(i + 1)
    while marker_exists == True:
        if first == True:
            #url_0 = 'https://data.ripple.com/v2/transactions/?start=2019-05-0' + a + 'T00:00:01z&end=2019-05-0' + b + 'T00:00:00&result=tesSUCCESS'
            url_0 = 'https://data.ripple.com/v2/transactions/?start=2019-05-' + a + 'T00:00:01z&end=2019-05-' + b + 'T00:00:00&result=tesSUCCESS'
            url_0
            r = requests.get(url_0)
            response_dict = r.json()
            mark = response_dict['marker']
            try:
                response_dict['marker']
            except NameError:
                marker_exists = False
            else:
                marker_exists = True

            transaction_update = response_dict['transactions']
            first = False

        else:

            mark = response_dict['marker']
            url = url_0 + '&limit=100&marker=' + mark
            url

            try:
                requests.get(url)
            except OSError:
                process_continue=False
            else:
                process_continue=True

            while process_continue==False:
                time.sleep(60)

                try:
                    requests.get(url)
                except OSError:
                    process_continue = False
                else:
                    process_continue = True

            r = requests.get(url) #r.status_code 503

            while r.status_code!=200:
                time.sleep(60)
                r = requests.get(url)

            response_dict = r.json()

            try:
                response_dict['transactions']
            except KeyError:
                transaction_exists = False
            else:
                transaction_exists = True

            if transaction_exists == True:

                new_transaction = response_dict['transactions']
                transaction_update = transaction_update + new_transaction
                try:
                    response_dict['marker']
                except KeyError:
                    marker_exists = False
                else:
                    marker_exists = True  # 存在

            else:
                while transaction_exists == False:
                    time.sleep(20)
                    mark = mark
                    url = url_0 + '&limit=100&marker=' + mark
                    url
                    r = requests.get(url)
                    response_dict = r.json()

                    try:
                        response_dict['transactions']
                    except KeyError:
                        transaction_exists = False
                    else:
                        transaction_exists = True  # 存在

                new_transaction = response_dict['transactions']
                transaction_update = transaction_update + new_transaction

                try:
                    response_dict['marker']
                except KeyError:
                    marker_exists = False
                else:
                    marker_exists = True

    #Data["Data_050" + str(i)] = transaction_update
    Data["Data_05" + str(i)] = transaction_update

    #with open("Data_050"+str(i)+'.pickle', 'wb') as f:
     #       pickle.dump(Data["Data_050" + str(i)], f)

    with open("Data_05"+str(i)+'.pickle', 'wb') as f:  #store
            pickle.dump(Data["Data_05" + str(i)], f)

time_end = time.time()
print('totally cost', time_end - time_start)


############# April data ########################################

time_start = time.time()

Data={}
for i in range(17, 24):
    marker_exists = True
    first = True

    a = str(i)
    b = str(i + 1)
    while marker_exists == True:
        if first == True:
            #url_0 = 'https://data.ripple.com/v2/transactions/?start=2019-05-0' + a + 'T00:00:01z&end=2019-05-0' + b + 'T00:00:00&result=tesSUCCESS'
            url_0 = 'https://data.ripple.com/v2/transactions/?start=2019-04-' + a + 'T00:00:01z&end=2019-04-' + b + 'T00:00:00&result=tesSUCCESS'
            url_0
            r = requests.get(url_0)
            response_dict = r.json()
            mark = response_dict['marker']
            try:
                response_dict['marker']
            except NameError:
                marker_exists = False
            else:
                marker_exists = True

            transaction_update = response_dict['transactions']
            first = False

        else:

            mark = response_dict['marker']
            url = url_0 + '&limit=100&marker=' + mark
            url

            try:
                requests.get(url)
            except OSError:
                process_continue=False
            else:
                process_continue=True

            while process_continue==False:
                time.sleep(60)

                try:
                    requests.get(url)
                except OSError:
                    process_continue = False
                else:
                    process_continue = True

            r = requests.get(url) #r.status_code 503

            while r.status_code!=200:
                time.sleep(60)
                r = requests.get(url)

            response_dict = r.json()

            try:
                response_dict['transactions']
            except KeyError:
                transaction_exists = False
            else:
                transaction_exists = True

            if transaction_exists == True:

                new_transaction = response_dict['transactions']
                transaction_update = transaction_update + new_transaction
                try:
                    response_dict['marker']
                except KeyError:
                    marker_exists = False
                else:
                    marker_exists = True

            else:
                while transaction_exists == False:
                    time.sleep(20)
                    mark = mark
                    url = url_0 + '&limit=100&marker=' + mark
                    url
                    r = requests.get(url)
                    response_dict = r.json()

                    try:
                        response_dict['transactions']
                    except KeyError:
                        transaction_exists = False
                    else:
                        transaction_exists = True

                new_transaction = response_dict['transactions']
                transaction_update = transaction_update + new_transaction

                try:
                    response_dict['marker']
                except KeyError:
                    marker_exists = False
                else:
                    marker_exists = True

    #Data["Data_040" + str(i)] = transaction_update
    Data["Data_04" + str(i)] = transaction_update

    #with open("Data_040"+str(i)+'.pickle', 'wb') as f:
     #       pickle.dump(Data["Data_040" + str(i)], f)

    with open("Data_04"+str(i)+'.pickle', 'wb') as f:
            pickle.dump(Data["Data_04" + str(i)], f)

time_end = time.time()
print('totally cost', time_end - time_start)

