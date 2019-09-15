import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pyecharts import options as opts
import time
from pyecharts.charts import Bar, Page
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot
import networkx as nx
import requests
from sklearn.externals import joblib
import csv
from collections import Counter
from scipy import optimize
from sklearn import linear_model
import seaborn as sns
import statistics
import datetime



time_start = time.time()
with open('total_data.pickle', 'rb') as f:
    total_data = pickle.load(f)
time_end = time.time()
print('totally cost', time_end - time_start)


transaction=total_data['tx']
hash=total_data['hash']

tx_type = []
for item in transaction:
     type1=item['TransactionType']
     tx_type.append(type1)
time_end = time.time()


count_type=pd.value_counts(tx_type)
merge=sum(count_type[3:13])

index_type=count_type.index
draw_name=['OfferCreate', 'OfferCancel', 'Payment', 'Other']
count_type=list(count_type)
draw_count=count_type[0:3]
draw_count.append(merge)

account_list = []
for item in transaction:
     account=item['TransactionType']
     account_list.append(account)



find_offercreate = 'OfferCreate'
offercreate_index=[i for i,v in enumerate(tx_type) if v==find_offercreate]

find_offercancel = 'OfferCancel'
offercancel_index=[i for i,v in enumerate(tx_type) if v==find_offercancel]

find_payment = 'Payment'
payment_index=[i for i,v in enumerate(tx_type) if v==find_payment]

offercreate=total_data.loc[offercreate_index]
offercancel=total_data.loc[offercancel_index]
payment=total_data.loc[payment_index]

offercreate.reset_index(drop=True, inplace=True)
offercancel.reset_index(drop=True, inplace=True)
payment.reset_index(drop=True, inplace=True)

offercreate_date=[]
for item in offercreate['date']:
    offercreate_date.append(item[0:10])

offercreate_date_count=dict(Counter(offercreate_date)) #9664575
offercreate_date_count_values=np.array(list(offercreate_date_count.values()))

date1=['04-15','04-16','04-17','04-18','04-19','04-20','04-21','04-22','04-23','04-24','04-25','04-26','04-27','04-28',
       '04-29','04-30','05-01','05-02','05-03','05-04','05-05','05-06','05-07','05-08','05-09','05-10','05-11','05-12',
       '05-13','05-14','05-15','05-16','05-17','05-18','05-19']

offercreate_date_hash={} #Separate the daily hash and store it day by day
offercreate_date_tx={}
s=0
i=-1
d=0
for item0 in date1 :
    i=i+1
    offercreate_date_hash[str(item0)] = []
    offercreate_date_tx[str(item0)] = []
    d = d + offercreate_date_count_values[i]

    offercreate_date_hash[str(item0)] = offercreate['hash'][s:d]
    offercreate_date_tx[str(item0)] = offercreate['tx'][s:d]
    s = s + offercreate_date_count_values[i]


################ filt out data ##########################################

offercancel_tx=offercancel['tx']
offercancel_account=[]
for item in offercancel_tx:
    offercancel_account.append(item['Account'])

offercancel_account_count=pd.value_counts(offercancel_account)
offercancel_account_duplicate=list(set(offercancel_account))  #Account that appeared in OfferCancel


offercreate_tx=offercreate['tx']
offercreate_offersequence=[]
for item in offercreate_tx:
    if item.__contains__('OfferSequence'):

        a=item['Account']+'-'+str(item['OfferSequence'])
        offercreate_offersequence.append(a)

len(offercreate_offersequence)  #8236127
offercreate_offersequence=list(set(offercreate_offersequence))
len(offercreate_offersequence)  #total cancel 8234142


offercancel_offersequence=[]
for item in offercancel_tx:
    a = item['Account'] + '-' + str(item['OfferSequence'])
    offercancel_offersequence.append(a)

len(offercancel_offersequence) #1351132
offercancel_offersequence=list(set(offercancel_offersequence))
len(offercancel_offersequence)  #1337072, Someone repeats cancelling the same offer


total_offersequence=offercreate_offersequence+offercancel_offersequence
len(total_offersequence) #9571214
total_offersequence=list(set(total_offersequence))
len(total_offersequence)  #Total number of cancelled orders：9556820
                          # Some people use offercancel to cancel and also use offercreate to cancel.
account_sequence=[]
for item in offercreate_tx:
     t=item['Account']+'-'+str(item['Sequence'])
     account_sequence.append(t)

len(list(set(account_sequence)))==len(account_sequence) #TRUE, Represent no duplication.

set1=set(account_sequence)
set2=set(total_offersequence)  #9556820
set3=set1&set2
len(set3) #9553501, 3000 cancelled offercreation offers were not created during this period（offercreate）


sequence_index=[i for i, item in enumerate(account_sequence) if item in set3]
len(sequence_index)  #9553501
offercancel_list=offercreate.loc[sequence_index]
offercancel_list.reset_index(drop=True, inplace=True) #extract offercreate transactions that canceled in this period

#### Table 8.1, how many times the currency occurs in the takergets of offercancel_list. #####
offercancel_list_tx=offercancel_list['tx']
gets_currency=[]
for item in offercancel_list_tx:
    if type(item['TakerGets']) != dict:
        # gets_amount=float(item['TakerGets'])/1000000
        gets_currency.append('XRP')
    else:
        gets_currency.append(item['TakerGets']['currency'])

gets_currency_duplicates=list(set(gets_currency))

gets_currency_count=pd.value_counts((gets_currency_duplicates))


gets_amount=[]  # len=9553434，67 transactions are for other currencies
gets_amount_1000=[]
for item in offercancel_list_tx:
    l=len(gets_amount)
    if type(item['TakerGets']) != dict:
        gets_amount.append(float(item['TakerGets'])/1000000*0.3)
    else:
        if item['TakerGets']['currency'] == 'CNY':
            gets_amount.append(float(item['TakerGets']['value'])*0.14)
        elif item['TakerGets']['currency'] == 'USD':
            gets_amount.append(float(item['TakerGets']['value']))
        elif item['TakerGets']['currency'] == 'EUR':
            gets_amount.append(float(item['TakerGets']['value'])*1.12)
        elif item['TakerGets']['currency'] == 'BTC':
            gets_amount.append(float(item['TakerGets']['value']) * 11888)
        elif item['TakerGets']['currency'] == 'ETH':
            gets_amount.append(float(item['TakerGets']['value']) * 225.13)
        elif item['TakerGets']['currency'] == 'ETC':
            gets_amount.append(float(item['TakerGets']['value']) * 5.91)
        elif item['TakerGets']['currency'] == 'XLM':
            gets_amount.append(float(item['TakerGets']['value']) * 0.078)
        elif item['TakerGets']['currency'] == 'BRL':
            gets_amount.append(float(item['TakerGets']['value']) * 0.25)
        elif item['TakerGets']['currency'] == 'DSH':
            gets_amount.append(float(item['TakerGets']['value']) * 0.028)
        elif item['TakerGets']['currency'] == 'BCH':
            gets_amount.append(float(item['TakerGets']['value']) * 336.57)
        elif item['TakerGets']['currency'] == 'ULT':
            gets_amount.append(float(item['TakerGets']['value']) * 0.03)
        elif item['TakerGets']['currency'] == 'UST':
            gets_amount.append(float(item['TakerGets']['value']) * 0.0009)
        elif item['TakerGets']['currency'] == 'ALV':
            gets_amount.append(float(item['TakerGets']['value']) * 0.0032)
        elif item['TakerGets']['currency'] == 'REP':
            gets_amount.append(float(item['TakerGets']['value']) * 11.05)

    if len(gets_amount)-l==1:
        if gets_amount[-1]>1000:
            gets_amount_1000.append(item)



len(gets_amount_1000) #2330940

gets_1000_account=[]
for item in gets_amount_1000:
    gets_1000_account.append(item['Account'])

gets_1000_account_count=pd.value_counts(gets_1000_account)
get_1000_account_list=list(gets_1000_account_count.index)


offercreate_account=[]
for item in offercreate_tx:
    offercreate_account.append(item['Account'])

offercreate_account_count=pd.value_counts(offercreate_account)
offercreate_account_index_50=list(offercreate_account_count.index)[0:234] #These accounts have more than 50 offercreations
offercreate_account_50=list(offercreate_account_count)[0:234]

account_50_index=[i for i, item in enumerate(offercreate_account_index_50) if item in get_1000_account_list]

############### manipulate account #########################
manipulate_account=[]
ratio_list=[]
i=-1
for item in get_1000_account_list:
    i=i+1
    if offercreate_account_index_50.__contains__(item):
        idx=offercreate_account_index_50.index(item)

        ratio=list(gets_1000_account_count)[i]/offercreate_account_50[idx]
        if ratio>0.7:
            manipulate_account.append(item)
            ratio_list.append(ratio)

manipulate_account_exchange={}
for item in manipulate_account:
    manipulate_account_exchange[str(item)]=[]



gets_1000_sequence=[]
for item in manipulate_account:

    for item1 in gets_amount_1000:
        if item==item1['Account']:
            if type(item1['TakerGets'])!=dict:
                gets_currency='XRP'
            else:
                gets_currency=item1['TakerGets']['currency']

            if type(item1['TakerPays'])!=dict:
                pays_currency='XRP'
            else:
                pays_currency=item1['TakerPays']['currency']

            exchange_currency=gets_currency+'/'+pays_currency

            manipulate_account_exchange[str(item)].append(exchange_currency)
            gets_1000_sequence.append(item1)  # cancelled offers related to manipulate address(691527)


##### Table 8.2: Currencies exchanged by the suspicious addresses in the gets_amount_1000
manipulate_account_exchange_count={}
for item2 in manipulate_account:
        manipulate_account_exchange_count[str(item2)] = []


for item in manipulate_account_exchange:
    item1=manipulate_account_exchange[str(item)]
    count_exchange=pd.value_counts(item1)
    #list1=list(count_exchange.index)
    #list2=list(count_exchange)

    manipulate_account_exchange_count[str(item)] = count_exchange

####### filt out data related to suspicious address ##############################3
account_1000_sequence=[] #account+sequence
for item in gets_1000_sequence:
    f = item['Account'] + '-' + str(item['Sequence'])
    account_1000_sequence.append(f)

len(account_1000_sequence)==len(list(set(account_1000_sequence))) #True：691527
set_account_1000_sequence=set(account_1000_sequence)


offercreate_offercancel_index=offercreate_index+offercancel_index
offercreate_offercancel_index=sorted(offercreate_offercancel_index)
offercreate_offercancel=total_data.loc[offercreate_offercancel_index] #extract offercreate+offercancel
offercreate_offercancel.reset_index(drop=True, inplace=True)


f1=[] #index of offers that include offersequence
i=-1
for item in offercreate_offercancel['tx']:
    i=i+1
    if item.__contains__('OfferSequence'):
        f1.append(i)

f2=offercreate_offercancel.loc[f1]
f2.reset_index(drop=True, inplace=True)


f3=[]
for item in f2['tx']:
    f4=item['Account'] + '-' + str(item['OfferSequence'])
    f3.append(f4)


find_index_1000=[(i,item) for i, item in enumerate(f3) if item in set_account_1000_sequence]
find_index_1000=pd.DataFrame(find_index_1000)
find_index_1000.shape[0] #699804

find_index_1000_duplicates=find_index_1000.drop_duplicates(1,keep='first',inplace=False)
find_index_1000_duplicates.shape[0] #691527
find_index_1000_duplicates.reset_index(drop=True, inplace=True)

final_index=find_index_1000_duplicates[0]
final_index=list(final_index)
len(final_index)  #691527

offercancel_1000=f2.loc[final_index]  #
offercancel_1000.reset_index(drop=True, inplace=True)  #691527

offercancel_1000_date=list(offercancel_1000['date'])

offercancel_1000_time=[]
for item in offercancel_1000_date:
    offercancel_1000_time.append(item[0:10])

cancel_count=Counter(offercancel_1000_time)
cancel_count=dict(cancel_count)
cancel_count_keys=list(cancel_count.keys())
cancel_count_values=list(cancel_count.values())


manipulate_account_count_index=[i for i, item in enumerate(get_1000_account_list) if item in set(manipulate_account)]

manipulate_account_count=pd.DataFrame(list(gets_1000_account_count)).loc[manipulate_account_count_index]
manipulate_account_count_address=pd.DataFrame(get_1000_account_list).loc[manipulate_account_count_index]
manipulate_account_count=list(manipulate_account_count[0])
manipulate_account_count_address=list(manipulate_account_count_address[0])

len(manipulate_account_count_address)  # 17

with open('manipulate_account_count_address.pickle', 'wb') as f:
    pickle.dump(manipulate_account_count_address, f)

with open('manipulate_account.pickle', 'wb') as f:
    pickle.dump(manipulate_account, f)

with open('offercancel_1000.pickle', 'wb') as f:
    pickle.dump(offercancel_1000, f)

with open('manipulate_account_exchange_count.pickle', 'wb') as f:
    pickle.dump(manipulate_account_exchange_count, f)




#################### Graph #######################################

with open('manipulate_account_count_address.pickle', 'rb') as f:
    manipulate_account_count_address = pickle.load(f)

with open('manipulate_account.pickle', 'rb') as f:
    manipulate_account = pickle.load(f)

with open('offercancel_1000.pickle', 'rb') as f:
    offercancel_1000 = pickle.load(f)

with open('manipulate_account_exchange_count.pickle', 'rb') as f:
    manipulate_account_exchange_count= pickle.load(f)




with open('Total_OfferCreate_Fulfill.pickle', 'rb') as f:
    Total_OfferCreate_Fulfill = pickle.load(f)

offercreate_account_full=[]
for item in offercreate['tx']:
    offercreate_account_full.append(item['Account'])

c={'04-15':0,'04-16':0,'04-17':0,'04-18':0, '04-19':0, '04-20':0, '04-21':0, '04-22':0,'04-23':0,'04-24':0,
   '04-25':0,'04-26':0,'04-27':0,'04-28':0,'04-29':0,'04-30':0,'05-01':0,'05-02':0,'05-03':0,'05-04':0,'05-05':0,
'05-06':0,'05-07':0,'05-08':0,'05-09':0,'05-10':0,'05-11':0,'05-12':0,'05-13':0,'05-14':0,'05-15':0,'05-16':0,
'05-17':0,'05-18':0,'05-19':0}

offercreate_manipulate_index={}
offercreate_manipulate={}
offercreate_Date_list={}
offercreate_time_list={}
offercreate_Counter_list={}
draw_list={}
for item in manipulate_account: # manipulate_account=manipulate_account_count_address
    offercreate_manipulate_index[str(item)] = []
    offercreate_manipulate_index[str(item)]=[i for i, v in enumerate(offercreate_account_full) if v== item]

    offercreate_manipulate[str(item)]=[]
    offercreate_manipulate[str(item)]=offercreate.loc[offercreate_manipulate_index[str(item)]]
    offercreate_manipulate[str(item)].reset_index(drop=True, inplace=True)

    offercreate_Date_list[str(item)] = []
    offercreate_Date_list[str(item)] = list(offercreate_manipulate[str(item)]['date'])

    offercreate_time_list[str(item)] = []
    for item1 in offercreate_Date_list[str(item)]:
        offercreate_time_list[str(item)].append(item1[5:10])

    offercreate_Counter_list[str(item)] = Counter(offercreate_time_list[str(item)])

    draw_list[str(item)]=[]
    draw_list[str(item)]={'04-15':0,'04-16':0,'04-17':0,'04-18':0, '04-19':0, '04-20':0, '04-21':0, '04-22':0,'04-23':0,'04-24':0,
   '04-25':0,'04-26':0,'04-27':0,'04-28':0,'04-29':0,'04-30':0,'05-01':0,'05-02':0,'05-03':0,'05-04':0,'05-05':0,
'05-06':0,'05-07':0,'05-08':0,'05-09':0,'05-10':0,'05-11':0,'05-12':0,'05-13':0,'05-14':0,'05-15':0,'05-16':0,
'05-17':0,'05-18':0,'05-19':0}
    offercreate_count=dict(offercreate_Counter_list[str(item)])

    l1=[i for i,v in enumerate(list(draw_list[str(item)].keys())) if v in list(offercreate_count.keys())]
    i = -1
    for item2 in l1:
        i = i + 1
        draw_list[str(item)][list(draw_list[str(item)].keys())[item2]] = list(offercreate_count.values())[i]

########## offercreate_fulfill ##########

offercreate_fulfill_account=[]
for item in Total_OfferCreate_Fulfill['tx']:
    offercreate_fulfill_account.append(item['Account'])

len1=0 #1311
for item in manipulate_account:
    l2 = [i for i, v in enumerate(offercreate_fulfill_account) if v == item]
    len1=len1+len(l2)



offercreate_fulfill_manipulate_index={}
offercreate_fulfill_manipulate={}
offercreate_fulfill_Date_list={}
offercreate_fulfill_time_list={}
offercreate_fulfill_Counter_list={}
draw_fulfill_list={}
for item in manipulate_account: # manipulate_account=manipulate_account_count_address
    offercreate_fulfill_manipulate_index[str(item)] = []
    offercreate_fulfill_manipulate_index[str(item)]=[i for i, v in enumerate(offercreate_fulfill_account) if v== item]
    offercreate_fulfill_manipulate[str(item)]=[]
    offercreate_fulfill_manipulate[str(item)]=Total_OfferCreate_Fulfill.loc[offercreate_fulfill_manipulate_index[str(item)]]
    offercreate_fulfill_manipulate[str(item)].reset_index(drop=True, inplace=True)
    offercreate_fulfill_Date_list[str(item)] = []
    offercreate_fulfill_Date_list[str(item)] = list(offercreate_fulfill_manipulate[str(item)]['date'])
    offercreate_fulfill_time_list[str(item)] = []
    for item1 in offercreate_fulfill_Date_list[str(item)]:
        offercreate_fulfill_time_list[str(item)].append(item1[5:10])
    offercreate_fulfill_Counter_list[str(item)] = Counter(offercreate_fulfill_time_list[str(item)])

    draw_fulfill_list[str(item)] = []
    draw_fulfill_list[str(item)] = {'04-15': 0, '04-16': 0, '04-17': 0, '04-18': 0, '04-19': 0, '04-20': 0, '04-21': 0,
                            '04-22': 0, '04-23': 0, '04-24': 0,
                            '04-25': 0, '04-26': 0, '04-27': 0, '04-28': 0, '04-29': 0, '04-30': 0, '05-01': 0,
                            '05-02': 0, '05-03': 0, '05-04': 0, '05-05': 0,
                            '05-06': 0, '05-07': 0, '05-08': 0, '05-09': 0, '05-10': 0, '05-11': 0, '05-12': 0,
                            '05-13': 0, '05-14': 0, '05-15': 0, '05-16': 0,
                            '05-17': 0, '05-18': 0, '05-19': 0}
    offercreate_fulfill_count = dict(offercreate_fulfill_Counter_list[str(item)])

    l1 = [i for i, v in enumerate(list(draw_fulfill_list[str(item)].keys())) if v in list(offercreate_fulfill_count.keys())]
    i = -1
    for item2 in l1:
        i = i + 1
        draw_fulfill_list[str(item)][list(draw_fulfill_list[str(item)].keys())[item2]] = list(offercreate_fulfill_count.values())[i]


###### offercancel ####################
offercancel_1000_account=[] #691527
for item in offercancel_1000['tx']:
    offercancel_1000_account.append(item['Account'])

A_list={}
for item in manipulate_account:
    A_list[str(item)] = []

for item in manipulate_account_count_address:
    A_list[str(item)] = [i for i, v in enumerate(offercancel_1000_account) if v== item]

D_list={}
for item in manipulate_account:
    D_list[str(item)] = []
    D_list[str(item)]=offercancel_1000.loc[A_list[str(item)]]
    D_list[str(item)].reset_index(drop=True, inplace=True)



Date_list={}
for item in manipulate_account:
    Date_list[str(item)] = []
    Date_list[str(item)] = list(D_list[str(item)]['date'])


time_list={}
for item in manipulate_account:
    time_list[str(item)] = []
    for item1 in Date_list[str(item)]:
        time_list[str(item)].append(item1[5:10])


Counter_list={}
draw_cancel_list={}
for item in manipulate_account:
    Counter_list[str(item)] = Counter(time_list[str(item)])
    draw_cancel_list[str(item)] = []
    draw_cancel_list[str(item)] = {'04-15': 0, '04-16': 0, '04-17': 0, '04-18': 0, '04-19': 0, '04-20': 0, '04-21': 0,
                                    '04-22': 0, '04-23': 0, '04-24': 0,
                                    '04-25': 0, '04-26': 0, '04-27': 0, '04-28': 0, '04-29': 0, '04-30': 0, '05-01': 0,
                                    '05-02': 0, '05-03': 0, '05-04': 0, '05-05': 0,
                                    '05-06': 0, '05-07': 0, '05-08': 0, '05-09': 0, '05-10': 0, '05-11': 0, '05-12': 0,
                                    '05-13': 0, '05-14': 0, '05-15': 0, '05-16': 0,
                                    '05-17': 0, '05-18': 0, '05-19': 0}
    offercancel_count = dict(Counter_list[str(item)])

    l1 = [i for i, v in enumerate(list(draw_cancel_list[str(item)].keys())) if v in list(offercancel_count.keys())]
    i = -1
    for item2 in l1:
        i = i + 1
        draw_cancel_list[str(item)][list(draw_cancel_list[str(item)].keys())[item2]] =  list(offercancel_count.values())[i]

############## offercreate - offercancel (offers canceled by offercreate） ##################

###### start
draw_full_cancel_list={}
for item0 in manipulate_account:
    draw_full_cancel_list[str(item0)] = []


for item in manipulate_account:
    s = 0
    d = 0
    i = -1
    for item1 in date1:
        #h = offercreate_date_data[str(item1)]
        h=offercreate['hash']

        i = i + 1
        d = d + list(draw_cancel_list[str(item)].values())[i]
        a22 = D_list[str(item)]['hash'][s:d]
        set1 = set(a22)
        tt = [i for i, v in enumerate(h) if v in set1]

        l = len(tt)
        draw_full_cancel_list[str(item)].append(l)


        s = s + list(draw_cancel_list[str(item)].values())[i]

#with open('draw_full_cancel_list.pickle', 'wb') as f:
#   pickle.dump(draw_full_cancel_list, f)

with open('draw_full_cancel_list.pickle', 'rb') as f:
    draw_full_cancel_list = pickle.load(f)

################### OfferCancel- offercancel (offers canceled by OfferCancel)

offercancel_cancel_date=[]
for item in offercancel['date']:
    offercancel_cancel_date.append(item[0:10])

offercancel_cancel_date_count=dict(Counter(offercancel_cancel_date)) #9664575
offercancel_cancel_date_count_values=np.array(list(offercancel_cancel_date_count.values()))

##### start
draw_offercancel_cancel_list={}
for item0 in manipulate_account:
    draw_offercancel_cancel_list[str(item0)] = []


for item in manipulate_account:
    s = 0
    d = 0
    i = -1
    for item1 in date1:
        #h = offercreate_date_data[str(item1)]
        h=offercancel['hash']

        i = i + 1
        d = d + list(draw_cancel_list[str(item)].values())[i]
        a22 = D_list[str(item)]['hash'][s:d]
        set1 = set(a22)
        tt = [i for i, v in enumerate(h) if v in set1]

        l = len(tt)
        draw_offercancel_cancel_list[str(item)].append(l)


        s = s + list(draw_cancel_list[str(item)].values())[i]

#with open('draw_offercancel_cancel_list.pickle', 'wb') as f:
#   pickle.dump(draw_offercancel_cancel_list, f)

with open('draw_offercancel_cancel_list.pickle', 'rb') as f:
    draw_offercancel_cancel_list = pickle.load(f)



########### Garph- The evolution of the number of OfferCreate, Fulfilled OfferCreate, Cancella- tions of Suspicious Address over time #######
plt.figure(11, figsize=(30, 28))
i=0
for item in manipulate_account[0:6]:
    i=i+1
    plt.subplot(6,3,i)
    plt.xticks([])
    #plt.xticks(rotation=90)
    plt.plot(list(draw_list[str(item)].keys()),list(draw_list[str(item)].values()))
    plt.plot(list(draw_fulfill_list[str(item)].keys()), list(draw_fulfill_list[str(item)].values()))
    plt.plot(list(draw_cancel_list[str(item)].keys()), list(draw_cancel_list[str(item)].values()))
    plt.plot(list(draw_list[str(item)].keys()),draw_full_cancel_list[str(item)],linestyle='--')
    plt.plot(list(draw_list[str(item)].keys()),draw_offercancel_cancel_list[str(item)],linestyle='-.')
    plt.text(24, 1000, 'Address'+' '+str(i), fontdict={'size': '22', 'color': 'k'})
    plt.subplots_adjust(wspace=0.3)
    #plt.legend('OfferCreate', 'Fulfilled OfferCreate', 'OfferCancel')

    font = {'weight': 'normal',
            'size': 22}
    plt.rc('font', **font)

plt.subplot(6,3,7)
plt.plot(list(draw_list[str(manipulate_account[6])].keys()), list(draw_list[str(manipulate_account[6])].values()))
plt.plot(list(draw_fulfill_list[str(manipulate_account[6])].keys()), list(draw_fulfill_list[str(manipulate_account[6])].values()))
plt.plot(list(draw_cancel_list[str(manipulate_account[6])].keys()), list(draw_cancel_list[str(manipulate_account[6])].values()))
plt.plot(list(draw_list[str(manipulate_account[6])].keys()), draw_full_cancel_list[str(manipulate_account[6])], linestyle='--')
plt.plot(list(draw_list[str(manipulate_account[6])].keys()), draw_offercancel_cancel_list[str(manipulate_account[6])], linestyle='-.')
plt.xticks([])
plt.text(25, 500, 'Address'+' '+str(7), fontdict={'size': '22', 'color': 'k'})
plt.subplots_adjust(wspace=0.3)
font = {'weight': 'normal',
            'size': 22
            }
plt.rc('font', **font)

plt.ylabel('Frequency',fontdict={'weight': 'normal', 'size': 32})

i=7
for item in manipulate_account[7:13]:
    i=i+1
    plt.subplot(6,3,i)
    plt.xticks([])
    #plt.xticks(rotation=90)
    plt.plot(list(draw_list[str(item)].keys()),list(draw_list[str(item)].values()))
    plt.plot(list(draw_fulfill_list[str(item)].keys()), list(draw_fulfill_list[str(item)].values()))
    plt.plot(list(draw_cancel_list[str(item)].keys()), list(draw_cancel_list[str(item)].values()))
    plt.plot(list(draw_list[str(item)].keys()),draw_full_cancel_list[str(item)],linestyle='--')
    plt.plot(list(draw_list[str(item)].keys()),draw_offercancel_cancel_list[str(item)],linestyle='-.')
    plt.text(16, 75, 'Address' +' '+ str(i), fontdict={'size': '22', 'color': 'k'})
    font = {'weight': 'normal','size': 22}
    plt.rc('font', **font)
    plt.subplots_adjust(wspace=0.3)

plt.subplot(6,3,14)
plt.plot(list(draw_list[str(manipulate_account[13])].keys()), list(draw_list[str(manipulate_account[13])].values()))
plt.plot(list(draw_fulfill_list[str(manipulate_account[13])].keys()), list(draw_fulfill_list[str(manipulate_account[13])].values()))
plt.plot(list(draw_cancel_list[str(manipulate_account[13])].keys()), list(draw_cancel_list[str(manipulate_account[13])].values()))
plt.plot(list(draw_list[str(manipulate_account[13])].keys()), draw_full_cancel_list[str(manipulate_account[13])], linestyle='--')
plt.plot(list(draw_list[str(manipulate_account[13])].keys()), draw_offercancel_cancel_list[str(manipulate_account[13])], linestyle='-.')
plt.xticks([])
plt.text(5, 18, 'Address'+' '+str(14), fontdict={'size': '22', 'color': 'k'})
plt.subplots_adjust(wspace=0.3)
font = {'weight': 'normal',
            'size': 22
            }
plt.rc('font', **font)

i=14
for item in manipulate_account[14:17]:
    i=i+1
    ax=plt.subplot(6,3,i)
    plt.xticks([])
    #plt.xticks(rotation=90)
    plt.plot(list(draw_list[str(item)].keys()),list(draw_list[str(item)].values()),label='OfferCreate')
    plt.plot(list(draw_fulfill_list[str(item)].keys()), list(draw_fulfill_list[str(item)].values()),label='Fulfilled OfferCreate')
    plt.plot(list(draw_cancel_list[str(item)].keys()), list(draw_cancel_list[str(item)].values()), label='Cancellations')
    plt.plot(list(draw_list[str(item)].keys()),draw_full_cancel_list[str(item)],linestyle='--',label='Cancellations by OfferCreate')
    plt.plot(list(draw_list[str(item)].keys()),draw_offercancel_cancel_list[str(item)],linestyle='-.',label='Cancellations by OfferCancel')
    plt.text(0, 17, 'Address' + ' '+str(i), fontdict={'size': '22', 'color': 'k'})

    xticks = list(range(0, len(list(draw_list[str(item)].keys())), 7))
    xlabels = [list(draw_list[str(item)].keys())[x] for x in xticks]
    xticks.append(len(list(draw_list[str(item)].keys())) - 1)
    xlabels.append(list(draw_list[str(item)].keys())[-1])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=15,fontsize=22)



    font = {'weight': 'normal',
            'size': 22
            }
    plt.rc('font', **font)
    plt.subplots_adjust(wspace=0.3)
ax.set_xlabel("Date",fontdict={'weight': 'normal', 'size': 32})
plt.savefig('figure12.pdf')
plt.show()

###### analysis of address 5 ###############
address5_0426=D_list[manipulate_account[4]][3814:11183]
address5_0426.reset_index(drop=True, inplace=True)
address5_tx=address5_0426['tx']

address5=[]
for item in address5_tx:
    address5.append(item['Account']+ '-'+ str(item['OfferSequence']))
address5=set(address5)

account_sequence=[]
for item in offercreate['tx']:
    account_sequence.append(item['Account']+ '-'+ str(item['Sequence']))

index2=[i for i, item in enumerate(account_sequence) if item in address5]

address5_offercreate=offercreate.loc[index2]
address5_offercreate.reset_index(drop=True, inplace=True)

exchange1=[]
for item in address5_offercreate['tx']:
    if type(item['TakerGets'])==dict:
        g=item['TakerGets']['currency']
    else:
        g='XRP'
    if type(item['TakerPays'])==dict:
        p=item['TakerPays']['currency']
    else:
        p='XRP'
    exchange1.append(g+'/'+p)

address5_USD_XRP_index=[i for i, item in enumerate(exchange1) if item=='USD/XRP'] #only trade USD/XRP和XRP/USD on that day
address5_XRP_USD_index=[i for i, item in enumerate(exchange1) if item=='XRP/USD']

address5_USD_XRP=address5_offercreate.loc[address5_USD_XRP_index]
address5_XRP_USD=address5_offercreate.loc[address5_XRP_USD_index]
address5_USD_XRP.reset_index(drop=True, inplace=True)
address5_XRP_USD.reset_index(drop=True, inplace=True)

USD_XRP_RATE=[]
for item in address5_USD_XRP['tx']:
    if type(item['TakerGets'])==dict:
        g_v=float(item['TakerGets']['value'])
    else:
        g_v=float(item['TakerGets'])/1000000
    if type(item['TakerPays'])==dict:
        p_v=float(item['TakerPays']['value'])
    else:
        p_v=float(item['TakerPays'])/1000000

    USD_XRP_RATE.append(p_v/g_v)



XRP_USD_RATE=[]
for item in address5_XRP_USD['tx']:
    if type(item['TakerGets'])==dict:
        g_v1=float(item['TakerGets']['value'])
    else:
        g_v1=float(item['TakerGets'])/1000000
    if type(item['TakerPays'])==dict:
        p_v1=float(item['TakerPays']['value'])
    else:
        p_v1=float(item['TakerPays'])/1000000

    XRP_USD_RATE.append(p_v1/g_v1)

plt.figure(13)
plt.plot(USD_XRP_RATE)
plt.show()

plt.figure(14)
plt.plot(XRP_USD_RATE)
plt.show()





########################################
url="https://data.ripple.com/v2/accounts/rENDnFwR3CPvrsPjD9XXeqVoXeVt2CpPWX/balance_changes?currency=USD&start=2019-04-26T00:00:00Z&limit=100"
r = requests.get(url)
response_dict = r.json()

balance_change=response_dict['balance_changes'][0:28]
balance_change=pd.DataFrame(balance_change)
balance_change_hash=list(balance_change['tx_hash'])
#h_0426=list(offercreate_date_hash['04-25'])+list(offercreate_date_hash['04-26'])

set_balance_change_hash=set(balance_change_hash)
index3=[i for i, item in enumerate(total_data['hash']) if item in set_balance_change_hash]
len(index3)==len(balance_change_hash)
with open('Data_0426.pickle', 'rb') as f:
    Data_0426 = pickle.load(f)

Data_0426=pd.DataFrame(Data_0426)
index4=[i for i, item in enumerate(Data_0426['hash']) if item in set_balance_change_hash]
data_tx=Data_0426['tx'].loc[index4]
data_tx.reset_index(drop=True, inplace=True)
Data_0426_meta=Data_0426['meta'][index4]
Data_0426_meta.reset_index(drop=True, inplace=True)

sequence_list=[]
c=[]
i=-1
for item in Data_0426_meta:
    i=i+1
    affected_node=item['AffectedNodes']
    for item1 in affected_node:
        if item1.__contains__('DeletedNode'):
            dn=item1['DeletedNode']['LedgerEntryType']=='Offer'
            if dn:
                j2=item1['DeletedNode']['FinalFields']['Account']=='rENDnFwR3CPvrsPjD9XXeqVoXeVt2CpPWX'
                if j2:
                    s2=item1['DeletedNode']['FinalFields']['Sequence']
                    sequence_list.append(s1)
                    c.append(i)

    affected_node=item['AffectedNodes']
    for item1 in affected_node:
        if item1.__contains__('CreatedNode'):
            cn=item1['CreatedNode']['LedgerEntryType']=='Offer'
            if cn:
                j1=item1['CreatedNode']['NewFields']['Account']=='rENDnFwR3CPvrsPjD9XXeqVoXeVt2CpPWX'
                if j1:
                    s1=item1['CreatedNode']['NewFields']['Sequence']
                    sequence_list.append(s1)
                    c.append(i)




    for item1 in affected_node:
        if item1.__contains__('ModifiedNode'):
            a=item1['ModifiedNode']['LedgerEntryType']=='Offer'
            if a:
                b=item1['ModifiedNode']['FinalFields']['Account']=='rENDnFwR3CPvrsPjD9XXeqVoXeVt2CpPWX'
                if b:
                    s=item1['ModifiedNode']['FinalFields']['Sequence']
                    sequence_list.append(s)
                    c.append(i)

partially_fulfill_index=[14,22,23]
s14=24189922
s22=24198607
s23=24198615

sequence_list[22]=s22  #manual modification
address5_account= 'rENDnFwR3CPvrsPjD9XXeqVoXeVt2CpPWX'
ac=[]
for item in sequence_list:
    ac.append(address5_account + '-' + str(item))

tx_0426= offercreate_date_tx['04-26']
tx_0426.reset_index(drop=True, inplace=True)
account_s=[]
for item in tx_0426:
    account_s.append(item['Account']+'-'+str(item['Sequence']))




index5=[i for i, item in enumerate(account_s) if item in ac]
offer=tx_0426.loc[index5]
offer.reset_index(drop=True, inplace=True)
len(index5)==len(ac)

offer_rate=[]
offer_sequence=[] #offersequenc and offer_rate correspond one to one
for item in offer:
    if type(item['TakerGets'])==dict:
        g_v1=float(item['TakerGets']['value'])
    else:
        g_v1=float(item['TakerGets'])/1000000
    if type(item['TakerPays'])==dict:
        p_v1=float(item['TakerPays']['value'])
    else:
        p_v1=float(item['TakerPays'])/1000000

    offer_rate.append(p_v1/g_v1)
    offer_sequence.append(item['Sequence'])

i=-1
xrp_amount=[]
for item in balance_change['amount_change']:
    item=float(item)
    i=i+1
    sl=sequence_list[i]
    index6 =offer_sequence.index(sl)
    rate=offer_rate[index6]
    if item>0:
        xrp_amount.append(-item/rate)
    else:
        xrp_amount.append(-item*rate)

pay_xrp=[]
for item in xrp_amount:
    if item<0:
        pay_xrp.append(item)

sum_pay_xrp=sum(pay_xrp)

receive_xrp=[]
for item in xrp_amount:
    if item>0:
        receive_xrp.append(item)

sum_receive_xrp=sum(receive_xrp)


amountchange=balance_change['amount_change']
change=[]
for item in amountchange:
    change.append(float(item))
total_change=sum(change) #2187.882931461231

negative=[]
for item in change:
    if item<0:
        negative.append(item)
sum1=sum(negative) #-5180.13421778671
positive_xrp=-sum1*3.4731


positive=[]
for item in change:
    if item>0:
        positive.append(item)

sum2=sum(positive) #7368.01714924794
negative_xrp=sum2*3.4731



account_0426=[]
for item in offercreate_date_tx['04-26']:
    account_0426.append(item['Account'])

index7=[i for i, item in enumerate(account_0426) if item=='rENDnFwR3CPvrsPjD9XXeqVoXeVt2CpPWX']

offer5=offercreate_date_tx['04-26'].loc[index7]
offer5.reset_index(drop=True, inplace=True)

cost=[]
for item in offer5:
    cost.append(float(item['Fee'])/1000000)

sum_cost=sum(cost)  #78.6554999999929 XRP


xrp_usd=[]
c1=[]
address5_xrp_rate=[]
for item in offercreate_date_tx['04-26']:
    if type(item['TakerGets'])!=dict:
        if type(item['TakerPays'])==dict:
            if item['TakerPays']['currency']=='USD':
                xrp_usd.append(item)
                if item['Account']=='rENDnFwR3CPvrsPjD9XXeqVoXeVt2CpPWX':
                    c1.append(1)
                    rg=float(item['TakerGets'])/1000000
                    rp=float(item['TakerPays']['value'])
                    address5_xrp_rate.append(rp/rg)
len(xrp_usd) #27887
sum(c1) #2066

usd_xrp=[]
c2=[]
c3=[]
address5_usd_rate=[]
for item in offercreate_date_tx['04-26']:
    if type(item['TakerPays'])!=dict:
        if type(item['TakerGets'])==dict:
            if item['TakerGets']['currency']=='USD':
                usd_xrp.append(item)
                if item['Account']=='rENDnFwR3CPvrsPjD9XXeqVoXeVt2CpPWX':
                    c2.append(1)
                    rg1=float(item['TakerGets']['value'])
                    rp1=float(item['TakerPays'])/1000000
                    address5_usd_rate.append(rp1/rg1)
                    if float(item['TakerGets']['value'])>1000:
                        c3.append(1)
len(usd_xrp) #10654
sum(c2)  #5425
sum(c3)  #5322


usd_xrp_t=[]
for item in offercreate['tx']:
    if type(item['TakerPays'])!=dict:
        if type(item['TakerGets'])==dict:
            if item['TakerGets']['currency']=='USD':
                usd_xrp_t.append(item)
len(usd_xrp_t)
len(usd_xrp_t)/35 #3821

xrp_usd_t=[]
for item in offercreate['tx']:
    if type(item['TakerGets'])!=dict:
        if type(item['TakerPays'])==dict:
            if item['TakerPays']['currency']=='USD':
                xrp_usd_t.append(item)
len(xrp_usd_t)
len(xrp_usd_t)/35 #18833


########
url1 ="https://min-api.cryptocompare.com/data/histohour?fsym=USD&tsym=XRP&limit=23&toTs=1556319600"
r = requests.get(url1)
ipdata = r.json()

url2 ="https://min-api.cryptocompare.com/data/histohour?fsym=XRP&tsym=USD&limit=23&toTs=1556319600"
r2 = requests.get(url2)
ipdata1 = r2.json()

market_USD_XRP=[]
t=[]
for item in ipdata['Data']:
    market_USD_XRP.append(float(item['close']))
    time_number=item['time']
    dateArray = datetime.datetime.utcfromtimestamp(time_number)
    otherStyleTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
    t.append(otherStyleTime)


market_XRP_USD=[]
for item in ipdata1['Data']:
    market_XRP_USD.append(float(item['close']))

##### Graph- Hourly close price of XRP/USD (blue line) and USD/XRP (red line) on April 26 #####
plt.figure(14)
#plt.subplot(121)
#plt.plot(t,market_USD_XRP)
ax=plt.subplot(111)
ax.plot(market_XRP_USD)
plt.xlabel('Time')
plt.ylabel('Close Price (XRP/USD)')
font = {'weight' : 'normal',
        'size'   : 18}
plt.rc('font', **font)


ax2=ax.twinx()
ax2.plot(market_USD_XRP,c='r')

plt.ylabel('Close Price (USD/XRP)')
font = {'weight' : 'normal',
        'size'   : 18}
plt.rc('font', **font)
plt.savefig('figure14.pdf')
plt.show()


##### Graph- The exchange rates of OfferCreate Transactions of Address5 on April 26 ###
plt.figure(15,figsize=(15,6))
plt.subplot(121)
plt.plot(address5_usd_rate)
plt.xlabel('OfferCreate Transactions')
plt.ylabel('Exchange Rates (USD/XRP)')

plt.subplot(122)
plt.plot(address5_xrp_rate)
plt.xlabel('OfferCreate Transactions')
plt.ylabel('Exchange Rates (XRP/USD)')
plt.subplots_adjust(wspace =0.6)
font = {'weight' : 'normal',
        'size'   : 18}
plt.rc('font', **font)
plt.savefig('figure15.pdf')
plt.show()


###### Graph- Daily close price of USD/XRP between 28/03 and 26/04 in year 2019 #########
url3 ="https://min-api.cryptocompare.com/data/histoday?fsym=USD&tsym=XRP&limit=29&toTs=1556236800"
r = requests.get(url3)
data_month = r.json()
y=['03-28','03-29','03-30','03-31','04-01','04-02','04-03','04-04','04-05','04-06','04-07',
   '04-08','04-09','04-10','04-11','04-12','04-13','04-14','04-15','04-16','04-17','04-18',
   '04-19','04-20','04-21','04-22','04-23','04-24','04-25','04-26']

market_USD_trend=[]
t=[]
for item in data_month['Data']:
    market_USD_trend.append(float(item['close']))


plt.figure(16,figsize=(8,5.5))
ax=plt.subplot(111)
plt.plot(market_USD_trend)
xticks=list(range(0,len(y),5))
xlabels=[y[x] for x in xticks]
xticks.append(len(y)-1)
xlabels.append(y[-1])
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)
plt.xlabel('Date',fontdict={'weight': 'normal', 'size': 18})
plt.ylabel('Close Price (USD/XRP)',fontdict={'weight': 'normal', 'size': 18})

plt.savefig('figure16.pdf')
plt.show()


#############  Big offer analysis ######################

####### USD/XRP ########
fulfill_0426=Total_OfferCreate_Fulfill.loc[17856:19523]
fulfill_0426.reset_index(drop=True, inplace=True)

fulfill5_USD=[]
for item in fulfill_0426['tx']:
    if type(item['TakerGets']) == dict:
        g5 = item['TakerGets']['currency']=='USD'
        if g5:
            if type(item['TakerPays']) != dict:
                fulfill5_USD.append(item)

v5=[]
fulfill_as=[]
for item in fulfill5_USD:
    v5.append(item['TakerGets']['value'])
    fulfill_as.append(item['Account'] + '-' + str(item['Sequence']))

usd_offer=[]
for item in offercreate['tx']:
    if type(item['TakerGets']) == dict:
        g5 = item['TakerGets']['currency']=='USD'
        if g5:
            if type(item['TakerPays']) != dict:
                usd_offer.append(float(item['TakerGets']['value']))

q1=np.percentile(usd_offer, 99) #8640.0

usd_offer_0426=[]
big_offer=[] #89
for item in offercreate_date_tx['04-26']:
    if type(item['TakerGets']) == dict:
        g5 = item['TakerGets']['currency']=='USD'
        if g5:
            if type(item['TakerPays']) != dict:
                v=float(item['TakerGets']['value'])
                usd_offer_0426.append(v)
                if v>=7226:
                    big_offer.append(item)

big_offer_sequence=[] #79
big_offer_fulfill=[]
for item in big_offer:
    acc_s=item['Account'] + '-' + str(item['Sequence'])
    if acc_s not in fulfill_as:
        big_offer_sequence.append(acc_s)
    else:
        big_offer_fulfill.append(item)

data1=[]
value_usd=[]
for item in big_offer_fulfill:
    data1.append(item['Account']+'-'+item['TakerGets']['value']+'-'+str((float(item['TakerPays'])/1000000)/float(item['TakerGets']['value'])))
    value_usd.append(float(item['TakerGets']['value']))

sum(value_usd) #97520.78768553087


#Data_0426=pd.DataFrame(Data_0426)
big_offer_value =[] #0 (3)
seq0=[]
i=-1
index0=[]
for item in Data_0426['meta']:
    i=i+1
    affected_nodes=item['AffectedNodes']
    for item1 in affected_nodes:
        if item1.__contains__('ModifiedNode'):
            a=item1['ModifiedNode']['LedgerEntryType']=='Offer'
            if a:
                b=item1['ModifiedNode']['FinalFields']['Account']+'-'+str(item1['ModifiedNode']['FinalFields']['Sequence'])
                if b in big_offer_sequence:
                    s=float(item1['ModifiedNode']['PreviousFields']['TakerGets']['value'])-float(item1['ModifiedNode']['FinalFields']['TakerGets']['value'])
                    big_offer_value.append(s)
                    seq0.append(b)
                    index0.append(i)
l_usd=list(set(seq0))  #'r9udPU9C9ktV1UTk2igVHA3uvGC8CvJQkq-6626' big_offer的第53项,10000 USD
sum(big_offer_value) #9999.75 USD

#################### XRP/USD #######################3
fulfill5_XRP=[] #59
for item in fulfill_0426['tx']:
    if type(item['TakerPays']) == dict:
        g5 = item['TakerPays']['currency']=='USD'
        if g5:
            if type(item['TakerGets']) != dict:
                fulfill5_XRP.append(item)
v5_XRP=[]
fulfill_as_XRP=[] #59
for item in fulfill5_XRP:
    v5_XRP.append(float(item['TakerGets'])/1000000)
    fulfill_as_XRP.append(item['Account'] + '-' + str(item['Sequence']))

xrp_offer=[]
for item in offercreate['tx']:
    if type(item['TakerPays']) == dict:
        g5 = item['TakerPays']['currency']=='USD'
        if g5:
            if type(item['TakerGets']) != dict:
                xrp_offer.append(float(item['TakerGets'])/1000000)

q1=np.percentile(xrp_offer, 90) #20000

xrp_offer_0426=[] #27887
big_offer_xrp=[] #3929
for item in offercreate_date_tx['04-26']:
    if type(item['TakerPays']) == dict:
        g5 = item['TakerPays']['currency']=='USD'
        if g5:
            if type(item['TakerGets']) != dict:
                v=float(item['TakerGets'])/1000000
                xrp_offer_0426.append(v)
                if v>=20000:
                    big_offer_xrp.append(item)

big_offer_sequence_xrp=[] #3928
big_offer_xrp_fulfill=[]
for item in big_offer_xrp:
    acc_s_xrp=item['Account'] + '-' + str(item['Sequence'])
    if acc_s_xrp not in fulfill_as_XRP:
        big_offer_sequence_xrp.append(acc_s_xrp)
    else:
        big_offer_xrp_fulfill.append(item)


big_offer_value_xrp =[]
seq=[] #82
for item in Data_0426['meta']:
    affected_nodes=item['AffectedNodes']
    for item1 in affected_nodes:
        if item1.__contains__('ModifiedNode'):
            a=item1['ModifiedNode']['LedgerEntryType']=='Offer'
            if a:
                b=item1['ModifiedNode']['FinalFields']['Account']+'-'+str(item1['ModifiedNode']['FinalFields']['Sequence'])
                if b in big_offer_sequence_xrp:
                    s=(float(item1['ModifiedNode']['PreviousFields']['TakerGets'])-float(item1['ModifiedNode']['FinalFields']['TakerGets']))/1000000
                    big_offer_value_xrp.append(s)
                    seq.append(b)

seq_set=list(set(seq)) #45
a=pd.DataFrame(big_offer_value_xrp)

offer_value={}
big_offer_v=[]
for item in seq_set:
    offer_value[str(item)] = []
    index=[i for i, v in enumerate(seq) if v==item]
    s1=sum(np.array(a.loc[index]))
    offer_value[str(item)].append(s1)
    if s1>20000:
        big_offer_v.append(offer_value[str(item)])


index1=[i for i, v in enumerate(big_offer_sequence_xrp) if v=='rGgebEDzQLYgn7Hh429rDHck6VmsnP5rm2-12033']
data=big_offer_xrp[1317] #index1=1317


######## Are there any transactions between address 5 and address X in these five weeks?
list1=['rGgebEDzQLYgn7Hh429rDHck6VmsnP5rm2-rENDnFwR3CPvrsPjD9XXeqVoXeVt2CpPWX','rENDnFwR3CPvrsPjD9XXeqVoXeVt2CpPWX-rGgebEDzQLYgn7Hh429rDHck6VmsnP5rm2']

p_tx = payment['tx']
account2destination=[]
for item in p_tx:
    account = item['Account']
    destination=item['Destination']
    account_destination=account+"-"+destination
    account2destination.append(account_destination)

find2=[i for i, v in enumerate(list1) if v in account2destination]  #[], no payment transaction between the two accounts