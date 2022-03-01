import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as dr
import numpy as np
from datetime import datetime
from datetime import timedelta

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
    'toDate': '2022-02-28'
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
    curr_date_price['max_price'] = int(record['max_price'])
    curr_date_price['min_price'] = int(record['min_price'])
    new_dict_list.append(curr_date_price)

# Plot for min_price data
filtered_modal_price_list = []
for dic_item in new_dict_list:
    if dic_item['paddy_price'] != 0:
        filtered_modal_price_list.append(dic_item)

df = pd.DataFrame(filtered_modal_price_list, columns=[
                  'date','paddy_price','min_price', 'max_price'])
data = df.sort_values('date')
print(data)


# Adding a new column "Prediction" corresponding to next week close price.
# Note that we are discarding the last row since we don't know tomorrows` price.
print(data["paddy_price"])
data["Prediction"] = data["paddy_price"].shift(-7)
print(data["Prediction"])
data = data[:-1]

# Let's take another look on the results
# print(data.tail())

# Removing nans and infs values
data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]

# Defining independent and dependent variables
# Note that we're ignoring the volume because it doesn't seem to 
# have a good linear relationship with the prediction price.
x = data.loc[:,~data.columns.isin(["date", "Prediction"])].values 
y = data["Prediction"].values.reshape(-1,1)



# Normalizing features
feature_scaler = MinMaxScaler(feature_range=(0,1))
x_normalized = feature_scaler.fit_transform(x)
# print(x_normalized)

# Train test split..
x_train, x_test, y_train, y_test = train_test_split(x_normalized, y, test_size = 0.3, random_state = 0)

# Creating a linear regression model and fitting on training data
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting values for the test set and evaluating the model performance
y_test_pred = regressor.predict(x_test)

# The mean absolute error
print('Mean absolute error: $ %.2f'
      % mean_absolute_error(y_test, y_test_pred) )

# The mean absolute percentage error
print('Mean absolute percentage error: %.2f'
      % ( mean_absolute_percentage_error(y_test, y_test_pred) * 100 ) + "%")

# The coefficient of determination
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_test_pred))
print(df.sort_values('date').iloc[-1])
next_week_price = regressor.predict(feature_scaler.fit_transform([[3951, 2371, 3961]]))
print(next_week_price)