# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential,load_model
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import pandas as pd
from datetime import datetime

def get_price_df_by(commodity_name, mandi_name):
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
#    date format yyyy-mm-dd
   data = {
       'language': 'en',
       'stateName': 'HARYANA',
       'commodityName': commodity_name, 
       'apmcName': mandi_name,
       'fromDate': '2022-01-01',
       'toDate': '2022-03-31'
   }
 
   response = requests.post('https://enam.gov.in/web/Ajax_ctrl/trade_data_list',
                            headers=headers, cookies=cookies, data=data)
   data_apmc = response.json()
   """
   dict_list SAMPLE:
   [
    {
        "id":"3881371",
        "state":"HARYANA",
        "apmc":"BARWALA HISAR",
        "commodity":"PADDY-BASMATI1121",
        "min_price":"1201",
        "modal_price":"2751",
        "max_price":"3150",
        "commodity_arrivals":"3281",
        "commodity_traded":"64698",
        "created_at":"2020-11-16",
        "status":"1",
        "Commodity_Uom":"Qui"
    }, ...
    ]
   """
   dict_list = data_apmc['data']
   new_dict_list = []
   
   for record in dict_list:
       curr_date_price = {}
       curr_date_price['date'] = datetime.strptime(
           record['created_at'], "%Y-%m-%d")
       curr_date_price['commodity_price'] = int(record['modal_price'])
       curr_date_price['max_price'] = int(record['max_price'])
       curr_date_price['min_price'] = int(record['min_price'])
       new_dict_list.append(curr_date_price)
 
   # Plot for min_price data
   filtered_modal_price_list = []
   for dic_item in new_dict_list:
       if dic_item['commodity_price'] != 0:
           filtered_modal_price_list.append(dic_item)
  
   df = pd.DataFrame(filtered_modal_price_list, columns=[
       'date', 'commodity_price', 'min_price', 'max_price'])
   data = df.sort_values('date')
   # set_index method will change the df index from automatically assigned id to date
   data = data.set_index('date')
   # converted string Datetime to Python Date time object
   data.index = pd.to_datetime(data.index, unit='s')
   return data

# divided data into 2 sets training set and test set with 80% and 20% data respectevly
def train_test_split(df, test_size=0.2):
   split_row = len(df) - int(test_size * len(df))
   train_data = df.iloc[:split_row]
   test_data = df.iloc[split_row:]
   return train_data, test_data

# plot func to show data on graph
def plot_line(commodity_name,line1, line2, label1=None, label2=None, title='Training-Testing', lw=2):
   print( "🦒",   line1, line2)
   fig, ax = plt.subplots(1, figsize=(13, 7))
   ax.plot(line1, label=label1, linewidth=lw)
   ax.plot(line2, label=label2, linewidth=lw)
   ax.set_ylabel(commodity_name, fontsize=14)
   ax.set_title(title, fontsize=16)
   ax.legend(loc='best', fontsize=16)
   plt.show()

#  functions to normalize the values inorder to change the values of numeric columns
#  in the dataset to a common scale, without distorting differences in the ranges of values.
 
def normalise_zero_base(df):
   return df / df.iloc[0] - 1

def normalise_min_max(df):
   return (df - df.min()) / (data.max() - df.min())

def extract_window_data(df, window_len=5, zero_base=True):
    """_summary_

    Args:
        df (_type_): _description_
        window_len (int, optional): _description_. Defaults to 5.
        zero_base (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)
"""
train_data_df:
index         value
2022-03-10    100
2022-03-11    101
2022-03-12    102
2022-03-13    103
2022-03-14    104
2022-03-15    105
2022-03-16    106
2022-03-17    107
2022-03-18    108
2022-03-19    109
2022-03-20    110
Y_train[i] = f(X_train[i])
X_train: [100,101,102,103,104,105,106,107,108,109]
Y_train: [101,102,103,104,105,106,107,108,109,110]


y=f(x)
here x is value for a date and y is predicted value for the date+(window_length)
"""
 
# fucntion to prepare data by splitting the data into set train set and test set to feed the LSTM model
def prepared_data(df, target_col, window_len=5, zero_base=True, test_size=0.2):
   train_data_df, test_data_df = train_test_split(df, test_size=test_size)
   X_train = extract_window_data(train_data_df, window_len, zero_base)
   X_test = extract_window_data(test_data_df, window_len, zero_base)
   y_train = train_data_df[target_col][window_len:].values
   y_test = test_data_df[target_col][window_len:].values
   if zero_base:
       y_train = y_train / train_data_df[target_col][:-window_len].values - 1
       y_test = y_test / test_data_df[target_col][:-window_len].values - 1
 
   return train_data_df, test_data_df, X_train, X_test, y_train, y_test
 
def build_lstm_model(input_data, output_size, neurons=100, activ_func='linear',
                    dropout=0.2, loss='mse', optimizer='adam'):
   model = Sequential()
   model.add(LSTM(neurons, input_shape=(
       input_data.shape[1], input_data.shape[2])))
   model.add(Dropout(dropout))
   model.add(Dense(units=output_size))
   model.add(Activation(activ_func))
   model.compile(loss=loss, optimizer=optimizer)
   return model
 
#  rename load_and_plot_model to prediction_func
"""_summary_
df:
index         value
2022-03-10    100
2022-03-11    101
2022-03-12    102
2022-03-13    103
2022-03-14    104
2022-03-15    105
2022-03-16    106
2022-03-17    107
2022-03-18    108
2022-03-19    109
2022-03-20    110
...
2022-04-05    160

interval = daily

y=f(x)

y = value on 2022-04-06  -> to be returned
f = saved model
x = value on date 2022-04-05
"""
def load_and_plot_model(commodity_name, mandi_name, interval):
    
    base_file_name = "models/{MANDI_NAME}_{COMMODITY_NAME}_{INTERVAL}.model"

    # Rename saved_model history
    saved_model = load_model(base_file_name.format(MANDI_NAME= mandi_name, COMMODITY_NAME=commodity_name , INTERVAL=interval))
    
    #UNDER REVIEW
    commodity_price_df = get_price_df_by(commodity_name=commodity_name, mandi_name=mandi_name)
    # train,test = train_test_split(commodity_price_df, test_size=0.2)
    target_col = 'commodity_price' 
    test_size = 0.2
    zero_base = True
    window_len = 5
 
    # training the model
    train, test, X_train, X_test, y_train, y_test = prepared_data(commodity_price_df, target_col, window_len, zero_base, test_size)
    
    ######## under edit
    
    #  # Let's check:
    # np.testing.assert_allclose(
    # model.predict(X_test),saved_model.predict(X_test)
    # )
    
    # history=saved_model.fit( X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
    # print("🍍 ", history.history)

    # plt.plot(history.history['loss'], 'r', linewidth=2, label='Train loss')
    # plt.plot(history.history['val_loss'], 'g',
    #         linewidth=2, label='Validation loss')
    # plt.title('LSTM')
    # plt.xlabel('Epochs')
    # plt.ylabel('MSE')
    # plt.show()
    print( "🐅",  test)
    targets = test[target_col][window_len:]
    preds = saved_model.predict(X_test).squeeze()
    
    # MAE and MSE is calculated just to know the accuracy. They are not actually used anywhere
    # MAE=mean_absolute_error(preds, y_test)
    # MSE = mean_squared_error(preds, y_test)
    
    # R2 = r2_score(y_test, preds)
    # preds = test[target_col].values[:-window_len] * (preds + 1)
    # preds = pd.Series(index=targets.index, data=preds)
    # plot_line(commodity_name,targets, preds, 'actual', 'prediction', lw=2)
    print(preds.to_frame())
 
def create_model_for_mandi_n_commodity(commodity_name, mandi_name, interval):
    commodity_price_df = get_price_df_by(commodity_name=commodity_name, mandi_name=mandi_name)
    
    # TODO 3: rename paddy_price to commodity_price everywhere
    target_col = 'commodity_price'
    
    # paddy_price_data.drop(["min_price", "max_price"], axis='columns', inplace=True)

    # plot_line(train[target_col], test[target_col], 'training', 'test', title='title')
    np.random.seed(42)
    window_len = 5
    test_size = 0.2
    zero_base = True
    lstm_neurons = 100
    epochs = 20
    batch_size = 32
    loss = 'mse'
    dropout = 0.2
    optimizer = 'adam'
    
    # training the model
    train, test, X_train, X_test, y_train, y_test = prepared_data(
    commodity_price_df, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)
   
    model = build_lstm_model(
    X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
    optimizer=optimizer)
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
    base_file_name = "models/{MANDI_NAME}_{COMMODITY_NAME}_{INTERVAL}.model"
    
    model.save(base_file_name.format(MANDI_NAME= mandi_name, COMMODITY_NAME=commodity_name , INTERVAL=interval))
    
   