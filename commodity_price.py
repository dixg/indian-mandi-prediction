import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


#paddy price
cookies = {
    'SERVERID': 'node1',
    '_ga': 'GA1.3.1698267377.1637750212',
    '_gid': 'GA1.3.843807453.1637750212',
    'ci_session': 'attmcopg1ni2hiqq5cnkm9rtsbqhj82n',
    '_gat_gtag_UA_128172535_1': '1',
}

headers = {
    'Connection': 'keep-alive',
    'sec-ch-ua': '"Google Chrome";v="95", "Chromium";v="95", ";Not A Brand";v="99"',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'X-Requested-With': 'XMLHttpRequest',
    'sec-ch-ua-mobile': '?0',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36',
    'sec-ch-ua-platform': '"macOS"',
    'Origin': 'https://enam.gov.in',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Dest': 'empty',
    'Referer': 'https://enam.gov.in/web/dashboard/trade-data',
    'Accept-Language': 'en-US,en;q=0.9',
}

data = {
    'language': 'en',
    'stateName': 'HARYANA',
    'commodityName': 'PADDY-BASMATI1121',
    'apmcName': 'BARWALA HISAR',
    'fromDate': '2018-11-24',
    'toDate': '2021-11-27'
}

response = requests.post('https://enam.gov.in/web/Ajax_ctrl/trade_data_list',
                         headers=headers, cookies=cookies, data=data)
data_apmc = response.json()
dict_list = data_apmc['data']
new_dict_list = []

for record in dict_list:
    curr_date_price = {}
    curr_date_price['date'] = datetime.strptime(
        record['created_at'], "%Y-%m-%d")
    curr_date_price['paddy_price'] = int(record['modal_price'])
    new_dict_list.append(curr_date_price)

# Plot for min_price data
filtered_modal_price_list = []
for dic_item in new_dict_list:
    if dic_item['paddy_price'] != 0:
        filtered_modal_price_list.append(dic_item)
# print("filtered_min_price_list--------> ",filtered_min_price_list)
df = pd.DataFrame(filtered_modal_price_list, columns=[
                  'date', 'paddy_price'])
print("$$$$$$$$$$$$$$$$",df)
df = df.sort_values('date', ascending=True)
ax=df.plot(x='date', y='paddy_price', c='green')
# plt.xlabel('Dates')
# plt.ylabel("Prices")
plt.autoscale(enable=True, axis='y')
# plt.show()

## For WHEAT

cookies = {
    'SERVERID': 'node1',
    '_ga': 'GA1.3.2109798933.1637126946',
    'ci_session': 'l0co83cnttd1ssgalah3fnt454su78sp',
}

headers = {
    'Connection': 'keep-alive',
    'sec-ch-ua': '"Google Chrome";v="95", "Chromium";v="95", ";Not A Brand";v="99"',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'X-Requested-With': 'XMLHttpRequest',
    'sec-ch-ua-mobile': '?0',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36',
    'sec-ch-ua-platform': '"macOS"',
    'Origin': 'https://enam.gov.in',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Dest': 'empty',
    'Referer': 'https://enam.gov.in/web/dashboard/trade-data',
    'Accept-Language': 'en-US,en;q=0.9,en-GB;q=0.8,hi;q=0.7',
}

data = {
  'language': 'en',
  'stateName': 'MADHYA PRADESH',
  'apmcName': 'INDORE',
  'commodityName': 'MOONG WHOLE (GREEN GRAM)',
  'fromDate': '2018-11-24',
  'toDate': '2021-11-27'
}

response = requests.post('https://enam.gov.in/web/Ajax_ctrl/trade_data_list', headers=headers, cookies=cookies, data=data)

data_apmc1 = response.json()
dict_list1 = data_apmc1['data']
new_dict_list1 = []

for record in dict_list1:
    curr_date_price = {}
    curr_date_price['date'] = datetime.strptime(
        record['created_at'], "%Y-%m-%d")
    curr_date_price['moong_whole_price'] = int(record['modal_price'])
    # curr_date_price['max_price'] = int(record['max_price'])
    # curr_date_price['min_price'] = int(record['min_price'])
    new_dict_list1.append(curr_date_price)

# Plot for modal_price data
filtered_modal_price_list = []
for dic_item in new_dict_list1:
    if dic_item['moong_whole_price'] != 0:
        filtered_modal_price_list.append(dic_item)
# print("filtered_min_price_list--------> ",filtered_min_price_list)
df1 = pd.DataFrame(filtered_modal_price_list, columns=[
                  'date', 'moong_whole_price'])
print("$$$$$$$$$$$$$$$$",df1)
df1 = df1.sort_values('date', ascending=True)
df1.plot.scatter(ax=ax,x='date', y='moong_whole_price', c='blue')
# plt.xlabel('Dates')
# plt.ylabel("Prices")
plt.autoscale(enable=True, axis='y')
# plt.show()

## Gram 

cookies = {
    'SERVERID': 'node1',
    '_ga': 'GA1.3.2109798933.1637126946',
    'ci_session': 'l0co83cnttd1ssgalah3fnt454su78sp',
}

headers = {
    'Connection': 'keep-alive',
    'sec-ch-ua': '"Google Chrome";v="95", "Chromium";v="95", ";Not A Brand";v="99"',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'X-Requested-With': 'XMLHttpRequest',
    'sec-ch-ua-mobile': '?0',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36',
    'sec-ch-ua-platform': '"macOS"',
    'Origin': 'https://enam.gov.in',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Dest': 'empty',
    'Referer': 'https://enam.gov.in/web/dashboard/trade-data',
    'Accept-Language': 'en-US,en;q=0.9,en-GB;q=0.8,hi;q=0.7',
}

data = {
  'language': 'en',
  'stateName': 'MADHYA PRADESH',
  'apmcName': 'INDORE',
  'commodityName': 'CHANA (BENGAL GRAM)-DESI',
  'fromDate': '2018-11-20',
  'toDate': '2021-11-27'
}

response = requests.post('https://enam.gov.in/web/Ajax_ctrl/trade_data_list', headers=headers, cookies=cookies, data=data)

data_apmc2 = response.json()

dict_list2 = data_apmc2['data']

new_dict_list2 = []

for record in dict_list2:
    curr_date_price = {}
    curr_date_price['date'] = datetime.strptime(
        record['created_at'], "%Y-%m-%d")
    curr_date_price['chana_price'] = int(record['modal_price'])
    new_dict_list2.append(curr_date_price)

# Plot for modal_price data
filtered_modal_price_list = []
for dic_item in new_dict_list2:
    if dic_item['chana_price'] != 0:
        filtered_modal_price_list.append(dic_item)
df2 = pd.DataFrame(filtered_modal_price_list, columns=[
                  'date', 'chana_price'])
# print("$$$$$$$$$$$$$$$$",df2)
df2 = df2.sort_values('date', ascending=True)
df2.plot(ax=ax, x='date', y='chana_price', c='red')
plt.autoscale(enable=True, axis='y')
plt.show()