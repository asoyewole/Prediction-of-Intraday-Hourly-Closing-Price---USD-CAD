from pandas.plotting import register_matplotlib_converters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ta import add_all_ta_features
import MetaTrader5 as mt5
import psycopg2
import pytz
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

script_start = datetime.now()

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(filename='etl_db_loader.log',
                    encoding='utf-8', level=logging.DEBUG)
logger.info(f'Etl loading started at {script_start}')

# Load the environment variables
load_dotenv()

# Mt5 connection variables
path = "C:\Program Files\MetaTrader 5\terminal64.exe"
account = os.getenv('MT5_LOGIN')
password = os.getenv('MT5_PASSWORD')
server = os.getenv('MT5_SERVER')


# Initialize mt5 connection
if not mt5.initialize():
    logger.info(f'MT5 initialization failed: {mt5.last_error()}')
    mt5.shutdown()


# Database connection variables
database = 'fx_tradingDB'
user = os.getenv('POSTGRE_USERNAME')
password = os.getenv('POSTGRE_PASSWORD')
port = os.getenv('POSTGRE_PORT')

# Connect to db
conn = psycopg2.connect(database=database,
                        user=user,
                        password=password,
                        port=port)
logger.info(f'connected to database - {database}')

# Obtain 30 instances fin_data to calculate tech_data
fin_data_payload = '''SELECT date_time, 
                        open, 
                        high, 
                        low, 
                        close, 
                        tick_vol
                        from public.financial_data
                    ORDER BY date_time DESC
                    LIMIT 30;
                    '''
try:
    cur = conn.cursor()
    cur.execute(fin_data_payload)
    prior_fin_data = cur.fetchall()
    prior_fin_data_columns = [desc[0] for desc in cur.description]
    logger.info('Payload retrieved from database')
except:
    logger.error('Error executing db query')

# 30 prior data to dataframe (sorted by date-time)
usd_df = pd.DataFrame(prior_fin_data, columns=prior_fin_data_columns)
usd_df = usd_df.sort_values(by='date_time').reset_index(drop=True)

# Retrieve data from last date time till now from mt5
symbol = 'USDCAD'
timeframe = mt5.TIMEFRAME_H1
last_date = usd_df['date_time'].iloc[-1]
logger.info(f'Last date_time on database is {last_date}')
start_date = last_date + timedelta(hours=1)

eet_timezone = pytz.timezone('Europe/Kyiv')  # EET (Eastern European Time)
end_date = datetime.now()
usdcad_rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
if usdcad_rates is not None:
    logger.info(f'Successfully retrieved USDCAD rates from {start_date} to {end_date}')
else:
    logger.info(f'No new data to retrieve at {end_date}')
# shut down connection to MetaTrader 5
mt5.shutdown()

logger.info(f'Data Transformation started at {datetime.now()}')
usdcad = pd.DataFrame(usdcad_rates)
usdcad['time'] = pd.to_datetime(usdcad['time'], unit='s')

usdcad = usdcad.sort_values(by='time').reset_index(drop=True)
usdcad = usdcad.drop(columns=['spread', 'real_volume'])

usdcad.columns = usd_df.columns
df = pd.concat([usd_df, usdcad]).reset_index(drop=True)

df[['open',
    'high',
    'low',
    'close',
    'tick_vol']] = df[['open',
                       'high',
                       'low',
                       'close',
                       'tick_vol']].apply(pd.to_numeric)

df = add_all_ta_features(df,
                         open='open',
                         high='high',
                         low='low',
                         close='close',
                         volume='tick_vol'
                         )

df_fin = df[['open', 'high', 'low', 'close', 'tick_vol', 'date_time']]
df_fin['hist_close'] = df_fin['close'].shift().fillna(0)
df_tech = df[['momentum_kama', 'others_cr', 'trend_ema_fast', 'trend_ema_slow', 'trend_ichimoku_a', 'trend_ichimoku_b', 'trend_ichimoku_base', 'trend_ichimoku_conv', 'trend_sma_fast', 'trend_sma_slow', 'trend_visual_ichimoku_a',
              'trend_visual_ichimoku_b', 'volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_dch', 'volatility_dcl', 'volatility_dcm', 'volatility_kcc', 'volatility_kch', 'volatility_kcl', 'volume_obv', 'volume_vpt', 'volume_vwap', 'date_time']]

df_date = pd.DataFrame(None)
df_date['year'] = df_tech['date_time'].dt.year
df_date['month'] = df_tech['date_time'].dt.month
df_date['day'] = df_tech['date_time'].dt.day
df_date['hour'] = df_tech['date_time'].dt.hour
df_date['day_of_week'] = df_tech['date_time'].dt.day_of_week

df_date['month_sin'] = np.sin(2 * np.pi * df_date['month'] / 12)
df_date['month_cos'] = np.sin(2 * np.pi * df_date['month'] / 12)

df_date['hour_sin'] = np.sin(2 * np.pi * df_date['hour'] / 24)
df_date['hour_cos'] = np.sin(2 * np.pi * df_date['hour'] / 24)

df_date['day_sin'] = np.sin(2 * np.pi * df_date['day'] / 31)
df_date['day_cos'] = np.sin(2 * np.pi * df_date['day'] / 31)

df_date['day_of_week_sin'] = np.sin(2 * np.pi * df_date['day_of_week'] / 7)
df_date['day_of_week_cos'] = np.sin(2 * np.pi * df_date['day_of_week'] / 7)
df_date['date_time'] = df_tech['date_time']
df_date.drop(columns=['year', 'month', 'day', 'hour',
                      'day_of_week'], inplace=True)
logger.info(f'Data transformation completed at {datetime.now()}')


new_df_fin = df_fin[df_fin['date_time'] > last_date]
new_df_tech = df_tech[df_tech['date_time'] > last_date]
new_df_date = df_date[df_date['date_time'] > last_date]

if not new_df_fin.empty:
    insert_fin_data = '''INSERT INTO PUBLIC.financial_data 
                        (open, high, low, close, tick_vol, date_time, hist_close)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)'''

    insert_tech_data = '''INSERT INTO PUBLIC.technical_data 
                        (momentum_kama,others_cr,trend_ema_fast,trend_ema_slow,
                        trend_ichimoku_a,trend_ichimoku_b,trend_ichimoku_base,
                        trend_ichimoku_conv,trend_sma_fast,trend_sma_slow,
                        trend_visual_ichimoku_a,trend_visual_ichimoku_b,volatility_bbh,
                        volatility_bbl,volatility_bbm,volatility_dch,volatility_dcl,
                        volatility_dcm,volatility_kcc,volatility_kch,volatility_kcl,
                        volume_obv,volume_vpt,volume_vwap,date_time)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'''

    insert_trans_date = '''INSERT INTO PUBLIC.transformed_date 
                        (month_sin, month_cos, hour_sin, hour_cos, day_sin, 
                        day_cos, day_of_week_sin, day_of_week_cos, date_time)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)'''
    # Convert the dataframes to lists
    fin_tuple = list(new_df_fin.itertuples(index=False, name=None))
    tech_tuple = list(new_df_tech.itertuples(index=False, name=None))
    date_tuple = list(new_df_date.itertuples(index=False, name=None))

    # Execute the insertion queries
    cur.executemany(insert_fin_data, fin_tuple)
    logger.info(f"{len(new_df_fin)} rows of financial data inserted.")

    cur.executemany(insert_tech_data, tech_tuple)
    logger.info(f"{len(new_df_tech)} rows of technical data inserted.")

    cur.executemany(insert_trans_date, date_tuple)
    logger.info(f"{len(new_df_date)} rows of transformed date inserted.")

else:
    logger.info(f'No new data to insert at {datetime.now()}')

conn.commit()

# Close database connection
cur.close()
conn.close()


script_end = datetime.now()
logger.info(f'script runtime:{script_end-script_start}')
logger.info(
    '------------------------------------------------------------------------')
