import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# cookies = {
#     'SERVERID': 'node1',
#     '_ga': 'GA1.3.1698267377.1637750212',
#     '_gid': 'GA1.3.843807453.1637750212',
#     'ci_session': 'fh9ecn6clkus82hp2fbk6189ksre31di',
#     '_gat_gtag_UA_128172535_1': '1',
# }

# headers = {
#     'Connection': 'keep-alive',
#     'Content-Length': '0',
#     'sec-ch-ua': '"Google Chrome";v="95", "Chromium";v="95", ";Not A Brand";v="99"',
#     'Accept': 'application/json, text/javascript, */*; q=0.01',
#     'X-Requested-With': 'XMLHttpRequest',
#     'sec-ch-ua-mobile': '?0',
#     'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36',
#     'sec-ch-ua-platform': '"macOS"',
#     'Origin': 'https://enam.gov.in',
#     'Sec-Fetch-Site': 'same-origin',
#     'Sec-Fetch-Mode': 'cors',
#     'Sec-Fetch-Dest': 'empty',
#     'Referer': 'https://enam.gov.in/web/dashboard/trade-data',
#     'Accept-Language': 'en-US,en;q=0.9',
# }

# response = requests.post('https://enam.gov.in/web/ajax_ctrl/states_name', headers=headers, cookies=cookies)

# # print("State Names=",response.json())

# # API call for apmc list

# cookies = {
#     'SERVERID': 'node1',
#     '_ga': 'GA1.3.1698267377.1637750212',
#     '_gid': 'GA1.3.843807453.1637750212',
#     'ci_session': 'attmcopg1ni2hiqq5cnkm9rtsbqhj82n',
#     '_gat_gtag_UA_128172535_1': '1',
# }

# headers = {
#     'Connection': 'keep-alive',
#     'sec-ch-ua': '"Google Chrome";v="95", "Chromium";v="95", ";Not A Brand";v="99"',
#     'Accept': 'application/json, text/javascript, */*; q=0.01',
#     'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
#     'X-Requested-With': 'XMLHttpRequest',
#     'sec-ch-ua-mobile': '?0',
#     'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36',
#     'sec-ch-ua-platform': '"macOS"',
#     'Origin': 'https://enam.gov.in',
#     'Sec-Fetch-Site': 'same-origin',
#     'Sec-Fetch-Mode': 'cors',
#     'Sec-Fetch-Dest': 'empty',
#     'Referer': 'https://enam.gov.in/web/dashboard/trade-data',
#     'Accept-Language': 'en-US,en;q=0.9',
# }

# data = {
#     'state_id': '32'
# #   'apmc_id': '132'
# }

# response = requests.post('https://enam.gov.in/web/Ajax_ctrl/apmc_list', headers=headers, cookies=cookies, data=data)
# dict_data=response.json()
# print("APMC List=", dict_data["data"])



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
    'commodityName': 'MOONG WHOLE (GREEN GRAM)',
    'apmcName': 'ADAMPUR',
    'fromDate': '2018-11-24',
    'toDate': '2022-01-31'
}

response = requests.post('https://enam.gov.in/web/Ajax_ctrl/trade_data_list',
                         headers=headers, cookies=cookies, data=data)
data_apmc = response.json()
dict_list = data_apmc['data']
new_dict_list = []

for record in dict_list:
    curr_date_price = {}
    curr_date_price['date'] = datetime.strptime(record['created_at'], "%Y-%m-%d")
    curr_date_price['modal_price'] = int(record['modal_price'])
    curr_date_price['max_price'] = int(record['max_price'])
    curr_date_price['min_price'] = int(record['min_price'])
    new_dict_list.append(curr_date_price)

# Plot for min_price data
filtered_min_price_list=[]
for dic_item in new_dict_list:
    if dic_item['min_price'] != 0:
        filtered_min_price_list.append(dic_item)
# print("filtered_min_price_list--------> ",filtered_min_price_list)
df = pd.DataFrame(filtered_min_price_list, columns=['date', 'min_price','max_price','modal_price'])
print(df['min_price'])
df=df.sort_values('date', ascending=True)
# plt.plot(df['date'], df['min_price'])
df.plot(x='date',y=['min_price','max_price','modal_price'])

plt.xticks(rotation='vertical')
plt.xlabel('Dates')
plt.ylabel("Prices")
plt.autoscale(enable=True, axis='y')
plt.show()


# # Plot for max_price data
# filtered_max_price_list=[]
# for dic_item in new_dict_list:
#     if dic_item['max_price'] != 0:
#         filtered_max_price_list.append(dic_item)
# df = pd.DataFrame(filtered_max_price_list, columns=['date', 'max_price'])
# df=df[df['max_price'] != 0]
# print("-----> Filtered max_price data= ",df)
# df=df.sort_values('date', ascending=True)
# plt.plot(df['date'], df['max_price'])
# plt.xticks(rotation='vertical')
# plt.autoscale(enable=True, axis='y')
# plt.show()

#  #Plot for Modal price
# filtered_modal_price_list=[]
# for dic_item in new_dict_list:
#     if dic_item['modal_price'] != 0:
#         filtered_modal_price_list.append(dic_item)

# df = pd.DataFrame(filtered_modal_price_list, columns=['date', 'modal_price'])
# df=df.sort_values('date', ascending=True)
# plt.plot(df['date'], df['modal_price'])
# plt.xticks(rotation='vertical')
# plt.autoscale(enable=True, axis='y')
# plt.show()
