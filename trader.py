import psycopg2
import os
import logging
import pickle
import MetaTrader5 as mt5
from dotenv import load_dotenv
from ta import add_all_ta_features
from datetime import datetime, timedelta
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
script_start = datetime.now()

# Load the environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='trader.log',
                    encoding='utf-8', level=logging.DEBUG)
logger.info(f'trader script started at {script_start}')

# Database connection variables
database = 'fx_tradingDB'
user = os.getenv('POSTGRE_USERNAME')
password = os.getenv('POSTGRE_PASSWORD')
port = os.getenv('POSTGRE_PORT')

conn = psycopg2.connect(database=database,
                        user=user,
                        password=password,
                        port=port)
# Obtain 30 instances fin_data to calculate tech_data
fin_data_payload = '''SELECT date_time, 
                        open, 
                        high, 
                        low, 
                        close, 
                        tick_vol,
                        hist_close
                        from public.financial_data
                    ORDER BY date_time DESC
                    LIMIT 30;
                    '''
try:
    cur = conn.cursor()
    cur.execute(fin_data_payload)
    prior_fin_data = cur.fetchall()
    prior_fin_data_columns = [desc[0] for desc in cur.description]
except:
    logger.info('error')

# 30 prior data to dataframe (sorted by date-time)
df = pd.DataFrame(prior_fin_data, columns=prior_fin_data_columns)
df = df.sort_values(by='date_time').reset_index(drop=True)
df[['open',
    'high',
    'low',
    'close',
    'tick_vol',
    'hist_close']] = df[['open',
                         'high',
                         'low',
                         'close',
                         'tick_vol',
                         'hist_close']].apply(pd.to_numeric)

# Close database connection
cur.close()
conn.close()

if not mt5.initialize():
    logger.info('initialize() failed')
    mt5.shutdown()

login = os.getenv('MT5_LOGIN')
password = os.getenv('MT5_PASSWORD')
server = os.getenv('Server')

authorized = mt5.login(login, password=password, server=server)
if authorized:
    logger.info('Connected to MT5')
else:
    logger.info(f'failed to connect, error code = {mt5.last_error()}')

symbol = 'USDCAD'
rates = mt5.symbol_info_tick(symbol)

df = add_all_ta_features(df,
                         open='open',
                         high='high',
                         low='low',
                         close='hist_close',
                         volume='tick_vol'
                         )

df['hist_close'] = df['close'].shift().fillna(0)

features = df[['open', 'hist_close',
               'others_cr', 'trend_ema_fast', 'trend_ema_slow',
               'trend_ichimoku_base', 'trend_sma_fast', 'volatility_kcl']].copy()
target = df['close'].copy()

last_feature = {
    'open': rates.bid,
    'hist_close': df['hist_close'].iloc[-1],
    'others_cr': df['others_cr'].iloc[-1],
    'trend_ema_fast': df['trend_ema_fast'].iloc[-1],
    'trend_ema_slow': df['trend_ema_slow'].iloc[-1],
    'trend_ichimoku_base': df['trend_ichimoku_base'].iloc[-1],
    'trend_sma_fast': df['trend_sma_fast'].iloc[-1],
    'volatility_kcl': df['volatility_kcl'].iloc[-1]
}

last_feature = pd.DataFrame(last_feature, index=[1])


# Make predictions with saved model
filename = 'best_model.pkl'
with open(filename, 'rb') as f:
    model = pickle.load(f)

prediction = model.predict(last_feature)
logger.info(prediction[0])

lot = 0.01
symbol = "USDCAD"

# Fetch the latest market rates
rates = mt5.symbol_info_tick(symbol)

price = rates.ask  # For Buy
tp = price + 0.001
sl = price - 0.001

# Check that symbol is valid and rates were fetched
if rates is None or not rates.ask or not rates.bid:
    logger.info("Failed to fetch rates, check symbol or market status.")
else:
    # For a Buy order
    if price < prediction[0]:
        tp = price + 0.001
        sl = price - 0.001
        logger.info('that', rates.ask)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 30,
            "magic": 123456,
            "comment": f"Python buy at {datetime.now()}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

    # For a Sell order
    elif price > prediction[0]:
        tp = price - 0.001
        sl = price + 0.001
        logger.info(rates.bid)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_SELL,
            "price": rates.bid,
            "sl": sl,
            "tp": tp,
            "deviation": 30,
            "magic": 123456,
            "comment": f"Python sell at {datetime.now()}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

    else:
        logger.info("Price is equal to prediction; no action taken.")
        request = None
        logger.info('request is none')

    if request:
        # Send the trade request
        order = mt5.order_send(request)

        # Check if order is None before accessing retcode
        if order is None:
            logger.info("Order send failed, no response received.")
        else:
            # Check the result
            if order.retcode != mt5.TRADE_RETCODE_DONE:
                logger.info(f"Order failed, retcode: {order.retcode}")
                logger.info(f"Result details: {order}")
            else:
                logger.info(f"Trade successful, order ID: {order.order}")
    else:
        logger.info("Request was not created due to price condition.")


    # Check account balance
account_info = mt5.account_info()
logger.info(account_info.balance)

# Check open positions
positions = mt5.positions_get(symbol=symbol)
if positions:
    for position in positions:
        logger.info(position)


logger.info('------------------------------------------------------------------------------')
