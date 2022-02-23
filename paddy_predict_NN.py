from prediction_functions import *

def run():
    mandi_list = ['barwala' ]
    commodity_list = ['paddy']
    period = ['daily']   
    
    for mandi in mandi_list:
        for commodity in commodity_list:
            print( commodity, mandi)
            create_model_for_mandi_n_commodity(commodity_name=commodity, mandi_name=mandi, interval = "daily")

if __name__ == "__main__":
    run()