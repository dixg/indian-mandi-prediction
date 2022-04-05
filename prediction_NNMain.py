from prediction_functions import *

def run():
    # TODO 1: Add mapping for new mandi and commodity   
    locations_commodities_dict={'ADAMPUR':['AMERICAN-COTTON','GUAR SEEDS','MUSTARD'],
                            'AMBALA':['MUSTARD'],
                            'ASANDH':['PADDY-BASMATI','PADDY-BASMATI1121'],
                            'BARWALA HISAR':['PADDY-BASMATI1121','AMERICAN-COTTON','COTTON']
                            }

    for key,val in locations_commodities_dict.items():
        for commodity in val:
            print(key,commodity)
            # create_model_for_mandi_n_commodity(commodity_name=commodity, mandi_name=key, interval = "daily")
            print("_____________________________===========================))))))))))))))))))")
            load_and_plot_model(commodity_name=commodity, mandi_name=key, interval = "daily")
            return 
            
if __name__ == "__main__":
    run()