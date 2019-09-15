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

time_start = time.time()
with open('total_data.pickle', 'rb') as f:
    total_data = pickle.load(f)  # 12251547 rows in total
time_end = time.time()
print('totally cost', time_end - time_start)

transaction = total_data['tx']
hash = total_data['hash']


################### The Proportion of Different Transaction Types #####################
tx_type = []
for item in transaction:
    type1 = item['TransactionType']
    tx_type.append(type1)
time_end = time.time()

count_type = pd.value_counts(tx_type)  # Number of times for each type

merge = sum(count_type[3:13])

index_type = count_type.index
draw_name = ['OfferCreate', 'OfferCancel', 'Payment', 'Other']
count_type = list(count_type)
draw_count = count_type[0:3]
draw_count.append(merge)


account_list = []
for item in transaction:
    account = item['TransactionType']
    account_list.append(account)



find_offercreate = 'OfferCreate'
offercreate_index = [i for i, v in enumerate(tx_type) if v == find_offercreate]

find_offercancel = 'OfferCancel'
offercancel_index = [i for i, v in enumerate(tx_type) if v == find_offercancel]

find_payment = 'Payment'
payment_index = [i for i, v in enumerate(tx_type) if v == find_payment]

offercreate = total_data.loc[offercreate_index]
offercancel = total_data.loc[offercancel_index]
payment = total_data.loc[payment_index]

offercreate.reset_index(drop=True, inplace=True)
offercancel.reset_index(drop=True, inplace=True)
payment.reset_index(drop=True, inplace=True)

############### Gephi ###############


p_tx = payment['tx']
payment_account = []
payment_destination = []
exchange = []
normal_payment = []
account2destination = []
normal_node = []
normal_edges = []
for item in p_tx:
    account = item['Account']
    destination = item['Destination']
    account_destination = account + "-" + destination
    payment_account.append(account)
    payment_destination.append(destination)
    account2destination.append(account_destination)

    if account == destination:
        exchange.append(item)
    else:
        normal_payment.append(item)
        normal_node.append(account)
        normal_edges.append([account, destination])

account_number = pd.value_counts(payment_account)
destination_number = pd.value_counts(payment_destination)




account_2500 = account_number[:2500]  # 2500 accounts with the highest frequency of occurrence
destination_2500 = destination_number[:2500]

# account_2500=account_number[:1500] #1500 accounts with the highest frequency of occurrence
# destination_2500=destination_number[:1500]
# account_2500=pd.DataFrame(account_2500)
# destination_2500=pd.DataFrame(destination_2500)

account_index_2500 = account_2500.index.to_list()
destination_index_2500 = destination_2500.index.to_list()

# gephi_nodes=pd.concat([account_index_2500,destination_index_2500],ignore_index=True) #5000
gephi_nodes = account_index_2500 + destination_index_2500
gephi_nodes = list(set(gephi_nodes))  # A total of 3,495 after deduplicate

# gephi_nodes=pd.DataFrame(gephi_nodes)
# gephi_nodes_duplicate=gephi_nodes.drop_duplicates(keep='first', inplace=False)
# gephi_nodes_duplicate.shape[0]

# gephi_nodes=np.array(gephi_nodes)
# gephi_nodes=gephi_nodes.tolist()

# gephi_nodes=account_index_2500+destination_index_2500


gephi_edges = []
for item in p_tx:
    account = item['Account']
    destination = item['Destination']
    if account_index_2500.__contains__(account):
        # if gephi_nodes.__contains__(account):
        r = True
    else:
        r = False
    if destination_index_2500.__contains__(destination):
        # if gephi_nodes.__contains__(destination):
        r1 = True
    else:
        r1 = False
    if r & r1:
        gephi_edges.append([account, destination])  # len(gephi_edges)=846965

G = nx.DiGraph()
G.add_nodes_from(gephi_nodes)  # node-number=3490
G.add_edges_from(gephi_edges)  # edges-number=27973
#nx.write_gexf(G, 'payment-relationship-2500-edit.gexf')

nx.write_gexf(G, 'payment-relationship-2500.gexf')
# nx.write_gexf(G,'payment-relationship-1500.gexf')

gephi_degree = pd.read_excel('gephi degree.xlsx')

gephi_degree = np.array(gephi_degree)
gephi_degree = list(gephi_degree)
degree_count = np.unique(gephi_degree, return_counts=True)

######### suspicious node ###########

node1 = 'r3YCmH6DNHQH1cUGu7y6yWP8bfN71DEVds'  # purple
node2 = 'r8miCrhYD65gDsXARgjdPvKZGdohgbg4V'  # green
node3 = 'rUA8WmMB6f2SLBtwcKj1dpCXEyjMNs4Yyn'  # orange

account2destination_suspicious = [node1 + '-' + node2, node1 + '-' + node3, node2 + '-' + node1, node2 + '-' + node3,
                                  node3 + '-' + node1, node3 + '-' + node2]

set_account2destination = set(account2destination)
find_suspicious_payment = [i for i, v in enumerate(account2destination_suspicious) if
                           v in set_account2destination]  # 0，2，4

node_list = [node1, node2, node3]

suspicious_edges = []
suspicious_nodes = []
check_xrp = []
for item in p_tx:
    account = item['Account']
    destination = item['Destination']
    if node_list.__contains__(account):
        r = True

    else:
        r = False

    if node_list.__contains__(destination):
        r1 = True

    else:
        r1 = False
    if r | r1:
        suspicious_edges.append([account, destination])
        suspicious_nodes.append(account)
        suspicious_nodes.append(destination)
        if type(item['Amount']) != dict:
            check_xrp.append(1)
        else:
            check_xrp.append(0)
sum(check_xrp) == len(check_xrp)  # True

suspicious_nodes = list(set(suspicious_nodes))

G = nx.DiGraph()
G.add_nodes_from(suspicious_nodes)  # node-number=3490
G.add_edges_from(suspicious_edges)  # edges-number=27973
nx.write_gexf(G, 'suspicious relationship.gexf')

modularity = pd.read_csv("suspicious_edges.csv")

node1_list = []
node2_list = []
node3_list = []
for item in p_tx:
    account = item['Account']
    destination = item['Destination']
    if item['Account'] == node1:
        node1_list.append(destination)
    elif item['Destination'] == node1:
        node1_list.append(account)
    elif item['Account'] == node2:
        node2_list.append(destination)
    elif item['Destination'] == node2:
        node2_list.append(account)
    elif item['Destination'] == node3:
        node3_list.append(account)
    if item['Account'] == node3:
        node3_list.append(destination)

set_node1_list = set(node1_list)
set_node2_list = set(node2_list)
set_node3_list = set(node3_list)

class1 = modularity.loc[0:2902]  # 2903
node1_gephi = set(class1['Id'])  # 2903
class2 = modularity.loc[2903:5540]
node2_gephi = set(class2['Id'])  # 2638
class3 = modularity.loc[5541:7626]
node3_gephi = set(class3['Id'])  # 2086

find_1 = [i for i, v in enumerate(set_node1_list) if v in node1_gephi]  # 2902, the one that's missing is node1 itself
find_2 = [i for i, v in enumerate(set_node3_list) if v in node2_gephi]  # 2637
find_3 = [i for i, v in enumerate(set_node2_list) if v in node3_gephi]  # 2085

receive_1 = []
pay_1 = []
receive_2 = []
pay_2 = []
receive_3 = []
pay_3 = []

node1_amount = []
node2_amount = []
node3_amount = []
node2_send = []
for item in p_tx:
    account = item['Account']
    destination = item['Destination']
    if item['Account'] == node1:
        pay_1.append(destination)
    elif item['Destination'] == node1:
        receive_1.append(account)
        node1_amount.append(float(item['Amount']) / 1000000)
    elif item['Account'] == node2:
        pay_2.append(destination)
        node2_send.append(float(item['Amount']) / 1000000)
    elif item['Destination'] == node2:
        receive_2.append(account)
        node2_amount.append(float(item['Amount']) / 1000000)
    elif item['Destination'] == node3:
        receive_3.append(account)
        node3_amount.append(float(item['Amount']) / 1000000)
    if item['Account'] == node3:
        pay_3.append(destination)

print(len(set(pay_1)))  # 6712
print(len(set(pay_2)))  # 3684
print(len(set(pay_3)))  # 29 accounts，412 expenditure
print(len(set(receive_1)))  # 6
print(len(set(receive_2)))  # 1
print(len(set(receive_3)))  # 4966 accounts，1020 income

find_31 = [i for i, item in enumerate(account2destination) if item == node3 + '-' + node1]  # Index of Node3 Transfer Money to Node1
h = list(payment['hash'].loc[find_31])
payment_31 = p_tx.loc[find_31]
amount_31 = []
for item in payment_31:
    amount_31.append(float(item['Amount']) / 1000000)

node3_amount_median = np.percentile(node3_amount, 50)  # 1464.2003 XRP

plt.figure()
plt.hist(node2_send)
plt.show()

receive_1 = []
pay_1 = []
receive_2 = []
pay_2 = []
receive_3 = []
pay_3 = []

node1_send = []
node1_receive = []
send_amount1 = []
receive_amount1 = []
for item in p_tx:
    account = item['Account']
    destination = item['Destination']
    a = item['Account'] == node1
    d = item['Destination'] == node1
    if a:
        pay_1.append(destination)
        send_amount1.append(float(item['Amount']) / 1000000)
    if d:
        receive_1.append(account)
        receive_amount1.append(float(item['Amount']) / 1000000)

sum(send_amount1)  # Total expenditure 10960574.230000097
sum(receive_amount1)  # Total income 10949305.8

receive_12 = []
send_amount2 = []
receive_amount2 = []
for item in p_tx:
    account = item['Account']
    destination = item['Destination']
    a = item['Account'] == node2
    d = item['Destination'] == node2
    if a:
        pay_2.append(destination)
        send_amount2.append(float(item['Amount']) / 1000000)
    if d:
        receive_2.append(account)
        receive_amount2.append(float(item['Amount']) / 1000000)
        if account == node1:
            receive_12.append(float(item['Amount']) / 1000000)

sum(send_amount2)  # Total expenditure 1334927.55
sum(receive_amount2)  # receive1340000
sum(receive_12)  # Purple dot to green dot 1290000

node3_send = []
node3_receive = []
node3_1 = []
for item in p_tx:
    account = item['Account']
    destination = item['Destination']
    a = item['Account'] == node3
    d = item['Destination'] == node3
    if a:
        pay_3.append(destination)
        node3_send.append(float(item['Amount']) / 1000000)
        if destination == node1:
            node3_1.append(float(item['Amount']) / 1000000)
    if d:
        receive_3.append(account)
        node3_receive.append(float(item['Amount']) / 1000000)

total_pay3 = sum(node3_send)  # 17190469.0
total_receive3 = sum(node3_receive)  # 17398002.405139968
# index1=[i for i, item in enumerate(pay_3) if item==node1]
node3_node1 = sum(node3_1)  # 9050000

print(len(set(pay_1)))  # 6712
print(len(set(pay_2)))  # 3685
print(len(set(pay_3)))  # 29 accounts ，412 expenditure
print(len(set(receive_1)))  # 6
print(len(set(receive_2)))  # 2
print(len(set(receive_3)))  # 4966个账户，1020 income



######## Extract Fulfilled OfferCreation #################

total_offercreate_fulfill = []
offercreate_fulfill_number = []

for i in range(15, 31):
    with open("Data_04" + str(i) + '.pickle', 'rb') as f:
        Data_0415 = pickle.load(f)

    Data_0415 = pd.DataFrame(Data_0415)
    transaction = Data_0415['tx']

    tx_type = []
    for item in transaction:
        type1 = item['TransactionType']
        tx_type.append(type1)
    time_end = time.time()
    find_offercreate = 'OfferCreate'
    offercreate_index = [i for i, v in enumerate(tx_type) if v == find_offercreate]

    offercreate = Data_0415.loc[offercreate_index]
    offercreate.reset_index(drop=True, inplace=True)
    offercreate_meta = offercreate['meta']

    offercreate_fulfill = []
    for item in offercreate_meta:
        affectednode = item['AffectedNodes']
        modi_account = []
        for item1 in affectednode:
            m = item1.__contains__('CreatedNode')
            if m == True:
                v = item1['CreatedNode']['LedgerEntryType'] == 'Offer'
                if v:
                    modi_account.append(item1)

        a = len(modi_account) == 0
        if a == True:
            offercreate_fulfill.append(1)
        else:
            offercreate_fulfill.append(0)

    find_offercreate_fulfill = 1
    offercreate_fulfill_index = [i for i, v in enumerate(offercreate_fulfill) if v == find_offercreate_fulfill]
    OfferCreate_fulfill = offercreate.loc[offercreate_fulfill_index]
    OfferCreate_fulfill.reset_index(drop=True, inplace=True)

    total_offercreate_fulfill.append(OfferCreate_fulfill)
    offercreate_fulfill_number.append(len(OfferCreate_fulfill))

Total_OfferCreate_Fulfill = []
Total_OfferCreate_Fulfill = pd.DataFrame(Total_OfferCreate_Fulfill)

for item in total_offercreate_fulfill:
    a = item
    Total_OfferCreate_Fulfill = pd.concat([Total_OfferCreate_Fulfill, a], ignore_index=True)

# with open('Total_OfferCreate_Fulfill.pickle', 'wb') as f:
#   pickle.dump(Total_OfferCreate_Fulfill, f)

oc_date = Total_OfferCreate_Fulfill['date']
oc_d = []
for item in oc_date:
    a = item[0:10]
    oc_d.append(a)

offercreate_number = Counter(oc_d)
offercreate_number = dict(offercreate_number)
offercreate_number_keys = np.array(list(offercreate_number.keys()))
offercreate_number_values = offercreate_number.values()
offercreate_number_values = np.array(list(offercreate_number_values))  # fulfill的订单数

with open('Total_OfferCreate_Fulfill.pickle', 'rb') as f:
    Total_OfferCreate_Fulfill = pickle.load(f)

##################################################

total_data_date = total_data['date']
total_data_d = []
for item in total_data_date:
    a = item[0:10]
    total_data_d.append(a)

total_data_number = Counter(total_data_d)  # Total number of transactions per day
total_data_number = dict(total_data_number)
total_data_number_keys = list(total_data_number.keys())
total_data_number_values = total_data_number.values()
total_data_number_values = np.array(list(total_data_number_values))

ocreate_date = offercreate['date']
ocreate_d = []
for item in ocreate_date:
    a = item[0:10]
    ocreate_d.append(a)

offercreate_number_t = Counter(ocreate_d)  # Number of offercreation orders submitted per day
offercreate_number_t = dict(offercreate_number_t)
offercreate_number_t_keys = offercreate_number_t.keys()
offercreate_number_t_values = offercreate_number_t.values()
offercreate_number_t_values = np.array(list(offercreate_number_t_values))

ocancel_date = offercancel['date']
ocancel_d = []
for item in ocancel_date:
    a = item[0:10]
    ocancel_d.append(a)

offercancel_number = Counter(ocancel_d)  # Number of offer cancel orders submitted per day
offercancel_number = dict(offercancel_number)
offercancel_number_keys = offercancel_number.keys()
offercancel_number_values = offercancel_number.values()
offercancel_number_values = np.array(list(offercancel_number_values))

payment_date = payment['date']
payment_d = []
for item in payment_date:
    a = item[0:10]
    payment_d.append(a)

payment_number = Counter(payment_d)  # Payment orders per day
payment_number = dict(payment_number)
payment_number_keys = list(payment_number.keys())
payment_number_values = payment_number.values()
payment_number_values = np.array(list(payment_number_values))

############# PAYMENT META #################################
#p_meta = []
#for i in range(15, 29):
 #   with open("Data_04" + str(i) + '.pickle', 'rb') as f:
  #      Data_0415 = pickle.load(f)

   # Data_0415 = pd.DataFrame(Data_0415)
    #Data_tx = Data_0415['tx']
    #Data_meta = Data_0415['meta']

    #tx_type = []
    #for item in Data_tx:
     #   type1 = item['TransactionType']
      #  tx_type.append(type1)

    #find_payment = 'Payment'
    #payment_index = [i for i, v in enumerate(tx_type) if v == find_payment]
    #payment_meta = Data_meta.loc[payment_index]

    #payment_meta = list(payment_meta)
    #p_meta = p_meta + payment_meta

#p_meta1 = 1  # 0501-0509 data
#p_meta2 = 2  # should be 05-10 to 05-19 data

# del p_meta1[0:2] There are four duplicate data, three at 00:00 on May 1, and one at 00:00 on May 20.
# del p_meta1[0]
# del p_meta2[461075]
#p_meta_total = p_meta + p_meta1 + p_meta2

#p_deliver_amount = []  # Each meta contains 'delivered_amount'
#for item in p_meta_total:
 #   deliver_amount = item['delivered_amount']
  #  p_deliver_amount.append(deliver_amount)

# with open('p_deliver_amount.pickle', 'wb') as f:
#   pickle.dump(p_deliver_amount, f)

with open('p_deliver_amount.pickle', 'rb') as f:
    p_deliver_amount = pickle.load(f)

payment_currency = []
payment_amount = []
payment_date = []
Payment_Top1_Currency = []
Payment_Top1_Currency_Number = []
Payment_Top2_Currency = []
Payment_Top2_Currency_Number = []
Payment_Top3_Currency = []
Payment_Top3_Currency_Number = []

s = 0
i = 0
d = 0
for item in payment_number_values:
    i = i + 1
    d = d + item
    a = p_deliver_amount[s:d]
    payment_today_amount = []
    payment_today_currency = []
    for item1 in a:

        if type(item1) != dict:
            amount = float(item1) / 1000000
            payment_amount.append(amount)
            payment_currency.append('XRP')
            payment_date.append(i)
            payment_today_amount.append(amount)
            payment_today_currency.append('XRP')
        else:
            value1 = item1['value']
            currency1 = item1['currency']
            payment_amount.append(value1)
            payment_currency.append(currency1)
            payment_date.append(i)
            payment_today_amount.append(value1)
            payment_today_currency.append(currency1)

    currency_count = pd.value_counts(payment_today_currency)
    payment_top1_currency = list(currency_count.index)[0]
    payment_top1_currency_number = list(currency_count)[0]
    Payment_Top1_Currency.append(payment_top1_currency)
    Payment_Top1_Currency_Number.append(payment_top1_currency_number)

    payment_top2_currency = list(currency_count.index)[1]
    payment_top2_currency_number = list(currency_count)[1]
    Payment_Top2_Currency.append(payment_top2_currency)
    Payment_Top2_Currency_Number.append(payment_top2_currency_number)

    payment_top3_currency = list(currency_count.index)[2]
    payment_top3_currency_number = list(currency_count)[2]
    Payment_Top3_Currency.append(payment_top3_currency)
    Payment_Top3_Currency_Number.append(payment_top3_currency_number)

    s = s + item

######## Binance Analysis ###################################
Binance_address = ['rJb5KsHsDHF1YS5B5DU6QCkH5NsPaKQTcy', 'rEb8TK3gBgk5auZkwc6sHnwrGVJH8DuaLh']

BA = []
BD = []

for item in payment_number_values:
    i = i + 1
    d = d + item
    a = p_tx[s:d]
    Binance_daily_transactions_account = []
    Binance_daily_transactions_destination = []

    for item1 in a:
        acc = Binance_address.__contains__(item1['Account'])
        des = Binance_address.__contains__(item1['Destination'])
        if acc & des != True:
            if acc:
                Binance_daily_transactions_account.append(1)
            else:
                Binance_daily_transactions_account.append(0)

            if des:
                Binance_daily_transactions_destination.append(1)
            else:
                Binance_daily_transactions_destination.append(0)

    sum_Binance_account = sum(np.array(Binance_daily_transactions_account))
    sum_Binance_destination = sum(np.array(Binance_daily_transactions_destination))

    BA.append(sum_Binance_account)
    BD.append(sum_Binance_destination)

    s = s + item

fig = plt.figure(8, figsize=(18, 8))
ax = plt.subplot(121)
plt.plot(BA)
xticks = list(range(0, len(payment_number_keys), 7))
xlabels = [payment_number_keys[x] for x in xticks]
xticks.append(len(payment_number_keys) - 1)
xlabels.append(payment_number_keys[-1])
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=15)
plt.xlabel("Date")
plt.ylabel("Number of Pay-out Transactions")
font = {'weight': 'normal',
        'size': 16}
plt.rc('font', **font)
ax = plt.subplot(122)
plt.plot(BD)
xticks = list(range(0, len(payment_number_keys), 7))
xlabels = [payment_number_keys[x] for x in xticks]
xticks.append(len(payment_number_keys) - 1)
xlabels.append(payment_number_keys[-1])
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=15)
plt.xlabel("Date")
plt.ylabel("Number of Pay-in Transactions")
font = {'weight': 'normal',
        'size': 16}
plt.rc('font', **font)
fig.subplots_adjust(wspace=0.4)
plt.savefig('figure8.pdf')
plt.show()

########################################################################################


A = list(set(Payment_Top2_Currency))
B = list(set(Payment_Top3_Currency))
top_2_3_currency = list(set(Payment_Top2_Currency + Payment_Top3_Currency))

payment_currency = []
payment_amount = []
payment_date = []

currency_list = {}
for item2 in top_2_3_currency:
    currency_list[str(item2)] = []

currency_list_amount = {}
for item2 in top_2_3_currency:
    currency_list_amount[str(item2)] = []

Value3 = []

s = 0
i = 0
d = 0
for item in payment_number_values:
    i = i + 1
    d = d + item
    a = p_deliver_amount[s:d]
    payment_today_amount = []
    payment_today_currency = []
    for item1 in a:

        if type(item1) != dict:
            amount = float(item1) / 1000000
            payment_amount.append(amount)
            payment_currency.append('XRP')
            payment_date.append(i)
            payment_today_amount.append(amount)
            payment_today_currency.append('XRP')
        else:
            value1 = item1['value']
            currency1 = item1['currency']
            payment_amount.append(value1)
            payment_currency.append(currency1)
            payment_date.append(i)
            payment_today_amount.append(value1)
            payment_today_currency.append(currency1)

    currency_count = pd.value_counts(payment_today_currency)
    currency_count_index = list(currency_count.index)
    currency_count_number = list(currency_count)

    for item2 in top_2_3_currency:
        if currency_count_index.__contains__(item2):
            find_item2 = currency_count_index.index(item2)
            item2_number = currency_count_number[find_item2]
            currency_list[str(item2)].append(item2_number)
        else:
            currency_list[str(item2)].append(0)

        value2 = 0

        for item1 in a:

            if type(item1) == dict:
                if item2 == item1['currency']:
                    v = float(item1['value'])
                    value2 = value2 + v

        currency_list_amount[str(item2)].append(value2)

    value3 = 0
    for item1 in a:

        if type(item1) != dict:
            value3 = value3 + float(item1) / 1000000  # XRP amount

    Value3.append(value3)

    s = s + item

############## Graph- Evolution of the number of transactions over time  ###########################

fig = plt.figure(5, figsize=(12, 5.7))  # Evolution of the number of transactions over time
ax = plt.subplot(121)
plt.plot(total_data_number_values, c='r', marker='D', label='Total Transactions')
plt.plot(offercreate_number_t_values, c='b', marker='o', label='OfferCreate')
plt.plot(offercreate_number_values, c='orange', marker='.', label='Fulfilled OfferCreate')
plt.plot(offercancel_number_values, c='g', marker='+', label='OfferCancel')
plt.plot(payment_number_values, c='k', marker='*', label='Payment')
xticks = list(range(0, len(payment_number_keys), 7))
xlabels = [payment_number_keys[x] for x in xticks]
xticks.append(len(payment_number_keys) - 1)
xlabels.append(payment_number_keys[-1])
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=15)
plt.xlabel("Date", fontdict={'weight': 'normal', 'size': 12})
plt.ylabel("Number of Transactions", fontdict={'weight': 'normal', 'size': 12})
legend = ax.legend(fontsize='large')
font = {'weight': 'normal',
        'size': 10}
plt.rc('font', **font)

ax = plt.subplot(122)
plt.plot(offercreate_number_values, c='orange', marker='.', label='Fulfilled OfferCreate')
plt.plot(offercancel_number_values, c='g', marker='+', label='OfferCancel')
plt.plot(payment_number_values, c='k', marker='*', label='Payment')
xticks = list(range(0, len(payment_number_keys), 7))
xlabels = [payment_number_keys[x] for x in xticks]
xticks.append(len(payment_number_keys) - 1)
xlabels.append(payment_number_keys[-1])
plt.subplots_adjust(wspace=2, hspace=0.2)
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=15)
plt.xlabel("Date", fontdict={'weight': 'normal', 'size': 12})
plt.ylabel("Number of Transactions", fontdict={'weight': 'normal', 'size': 12})
legend = ax.legend(fontsize='large')
font = {'weight': 'normal',
        'size': 10}

fig.subplots_adjust(wspace=0.4)
plt.savefig('figure7.pdf')
plt.show()



################ Graph- Evolution of the number and amount of various currencies payments over time ########

fig = plt.figure(5, figsize=(19, 20.5))
ax = plt.subplot(421)
ax.plot(Payment_Top1_Currency_Number, marker='.')
ax.set_ylabel('Payments', fontdict={'weight': 'normal', 'size': 20})
ax.set_title("XRP")

ax2 = ax.twinx()
ax2.plot(Value3, c='r', marker='.')
xticks = list(range(0, len(payment_number_keys), 7))
xlabels = [payment_number_keys[x] for x in xticks]
xticks.append(len(payment_number_keys) - 1)
xlabels.append(payment_number_keys[-1])
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=15)
ax.set_xlabel("Date", fontdict={'weight': 'normal', 'size': 20})
ax2.set_ylabel("Payments amounts", fontdict={'weight': 'normal', 'size': 20})

ax = plt.subplot(422)
ax.plot(currency_list['UST'], marker='.')
ax.set_ylabel('Payments', fontdict={'weight': 'normal', 'size': 20})
ax.set_title("UST")

ax2 = ax.twinx()
ax2.plot(currency_list_amount['UST'], c='r', marker='.')
xticks = list(range(0, len(payment_number_keys), 7))
xlabels = [payment_number_keys[x] for x in xticks]
xticks.append(len(payment_number_keys) - 1)
xlabels.append(payment_number_keys[-1])
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=15)
ax.set_xlabel("Date", fontdict={'weight': 'normal', 'size': 20})
ax2.set_ylabel("Payments amounts", fontdict={'weight': 'normal', 'size': 20})

ax = plt.subplot(423)
ax.plot(currency_list['ALV'], marker='.')
ax.set_ylabel('Payments', fontdict={'weight': 'normal', 'size': 20})
ax.set_title("ALV")

ax2 = ax.twinx()
ax2.plot(currency_list_amount['ALV'], c='r', marker='.')
xticks = list(range(0, len(payment_number_keys), 7))
xlabels = [payment_number_keys[x] for x in xticks]
xticks.append(len(payment_number_keys) - 1)
xlabels.append(payment_number_keys[-1])
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=15)
ax.set_xlabel("Date", fontdict={'weight': 'normal', 'size': 20})
ax2.set_ylabel("Payments amounts", fontdict={'weight': 'normal', 'size': 20})

ax = plt.subplot(424)
ax.plot(currency_list['CNY'], marker='.')
ax.set_ylabel('Payments', fontdict={'weight': 'normal', 'size': 20})
ax.set_title("CNY")

ax2 = ax.twinx()
ax2.plot(currency_list_amount['CNY'], c='r', marker='.')
xticks = list(range(0, len(payment_number_keys), 7))
xlabels = [payment_number_keys[x] for x in xticks]
xticks.append(len(payment_number_keys) - 1)
xlabels.append(payment_number_keys[-1])
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=15)

ax.set_xlabel("Date", fontdict={'weight': 'normal', 'size': 20})
ax2.set_ylabel("Payments amounts", fontdict={'weight': 'normal', 'size': 20})

ax = plt.subplot(425)
ax.plot(currency_list['BTC'], marker='.')
ax.set_ylabel('Payments', fontdict={'weight': 'normal', 'size': 20})
ax.set_title("BTC")

ax2 = ax.twinx()
ax2.plot(currency_list_amount['BTC'], c='r', marker='.')
xticks = list(range(0, len(payment_number_keys), 7))
xlabels = [payment_number_keys[x] for x in xticks]
xticks.append(len(payment_number_keys) - 1)
xlabels.append(payment_number_keys[-1])
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=15)
ax.set_xlabel("Date", fontdict={'weight': 'normal', 'size': 20})
ax2.set_ylabel("Payments amounts", fontdict={'weight': 'normal', 'size': 20})

ax = plt.subplot(426)
ax.plot(currency_list['WCN'], marker='.')
ax.set_ylabel('Payments', fontdict={'weight': 'normal', 'size': 20})
ax.set_title("WCN")

ax2 = ax.twinx()
ax2.plot(currency_list_amount['WCN'], c='r', marker='.')
xticks = list(range(0, len(payment_number_keys), 7))
xlabels = [payment_number_keys[x] for x in xticks]
xticks.append(len(payment_number_keys) - 1)
xlabels.append(payment_number_keys[-1])
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=15)
ax.set_xlabel("Date", fontdict={'weight': 'normal', 'size': 20})
ax2.set_ylabel("Payments amounts", fontdict={'weight': 'normal', 'size': 20})

ax = plt.subplot(427)
ax.plot(currency_list['ETH'], marker='.')
ax.set_ylabel('Payments', fontdict={'weight': 'normal', 'size': 20})
ax.set_title("ETH")

ax2 = ax.twinx()
ax2.plot(currency_list_amount['ETH'], c='r', marker='.')
xticks = list(range(0, len(payment_number_keys), 7))
xlabels = [payment_number_keys[x] for x in xticks]
xticks.append(len(payment_number_keys) - 1)
xlabels.append(payment_number_keys[-1])
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=15)
ax.set_xlabel("Date", fontdict={'weight': 'normal', 'size': 20})
ax2.set_ylabel("Payments amounts", fontdict={'weight': 'normal', 'size': 20})

ax = plt.subplot(428)
ax.plot(currency_list['USD'], marker='.')
ax.set_ylabel('Payments', fontdict={'weight': 'normal', 'size': 20})
ax.set_title("USD")

ax2 = ax.twinx()
ax2.plot(currency_list_amount['USD'], c='r', marker='.')
xticks = list(range(0, len(payment_number_keys), 7))
xlabels = [payment_number_keys[x] for x in xticks]
xticks.append(len(payment_number_keys) - 1)
xlabels.append(payment_number_keys[-1])
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=15)
fig.subplots_adjust(wspace=0.7, hspace=0.5)
ax.set_xlabel("Date", fontdict={'weight': 'normal', 'size': 20})
ax2.set_ylabel("Payments amounts", fontdict={'weight': 'normal', 'size': 20})
plt.savefig('figure5.pdf')
plt.show()





##################################################################
with open('offercreate_number_t_values.pickle', 'wb') as f:
    pickle.dump(offercreate_number_t_values, f)

with open('offercreate_number_values.pickle', 'wb') as f:
    pickle.dump(offercreate_number_values, f)

with open('offercreate_number_t_values.pickle', 'rb') as f:
    offercreate_number_t_values = pickle.load(f)

with open('offercreate_number_values.pickle', 'rb') as f:
    offercreate_number_values = pickle.load(f)







