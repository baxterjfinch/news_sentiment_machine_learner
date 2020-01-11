import quandl

APIKEY="7SSnVKnZMR59ZxqGv5PZ"

def authenticate():
    return quandl.ApiConfig.api_key = APIKEY

def get_bitcoin_data(api):
    data = api.get("BTC/USD")
    print(data)

api = authenticate()
get_bitcoin_data(api)
