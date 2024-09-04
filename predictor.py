import psycopg2
import os
import logging
import pickle
import MetaTrader5 as mt5
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
script_start = datetime.now()

# Load the environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='predictor.log',
                    encoding='utf-8', level=logging.DEBUG)
logger.info(f'predictor script started at {script_start}')


# Database connection variables
database = 'fx_tradingDB'
user = os.getenv('POSTGRE_USERNAME')
password = os.getenv('POSTGRE_PASSWORD')
port = os.getenv('POSTGRE_PORT')

# Connect to db
try:
    conn = psycopg2.connect(database=database,
                            user=user,
                            password=password,
                            port=port)
    logger.info(f'connected to database - {database}')
    cur = conn.cursor()
    logger.info('Connected to database')
except:
    logger.info('Database connection failed!')


def db_querier(payload):
    cur.execute(payload)
    values = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    return values, columns

# retrieve the latest prediction
pred_payload = '''SELECT * FROM PUBLIC.predictions
                    ORDER BY date_time DESC
                    LIMIT 1;'''
last_pred, last_pred_cols = db_querier(pred_payload)


# Do this if prediction table is empty (start predicting from end of test data)
#Retrieve historical data from database
if not last_pred:
    hist_data_payload = '''SELECT f.date_time, f.close, f.open, f.hist_close,
                            t.others_cr, t.trend_ema_fast, t.trend_ema_slow, 
                            t.trend_ichimoku_base, t.trend_sma_fast, t.volatility_kcl
                        FROM PUBLIC.financial_data AS f
                        JOIN PUBLIC.technical_data AS t
                        ON f.date_time = t.date_time
                        WHERE f.date_time > '2024-07-27';'''
    latest_pred_time = None
    logger.info('No previous prediction')
else:
    # obtain latest prediction in datetime format and increment by 1 hr
    latest_pred_time = pd.to_datetime(last_pred[0][1])
    hist_data_payload = f'''SELECT f.date_time, f.close, f.open, f.hist_close,
                            t.others_cr, t.trend_ema_fast, t.trend_ema_slow, 
                            t.trend_ichimoku_base, t.trend_sma_fast, t.volatility_kcl
                        FROM PUBLIC.financial_data AS f
                        JOIN PUBLIC.technical_data AS t
                        ON f.date_time = t.date_time
                        WHERE f.date_time > '{latest_pred_time}';'''
    logger.info(f'Latest prediction was for timestamp: {latest_pred_time}')
    
hist_data, hist_data_cols = db_querier(hist_data_payload)
if hist_data:
    df = pd.DataFrame(hist_data, columns=hist_data_cols)
    df = df.sort_values(by='date_time').reset_index(drop=True)
    logger.info(f'Pulled {df.shape[0]} historical instances from database')

    target = df['close']
    features = df.drop(columns=['date_time', 'close'])

    # Make predictions with saved model
    filename = 'best_model.pkl'
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    pred_start_time = datetime.now()
    predictions = model.predict(features)
    pred_end_time = datetime.now()
    pred_time = pred_end_time - pred_start_time
    logger.info(f'Predictions took {pred_time} to complete')

    date_time = df['date_time']
    logger.info(f'Predicted {len(predictions)} instances')
    try:
        load_pred_payload = '''INSERT INTO PUBLIC.predictions (date_time, predictions)
                                VALUES (%s, %s);
                                '''
        for time, pred in zip(date_time, predictions):
            cur.execute(load_pred_payload, (time, pred))
        conn.commit()
    except:
        logger.error('Unable to commit predictions to DB')
else:
    logger.info('No new instance from MT5 since last prediction')
# Close database connection
cur.close()
conn.close()

script_end = datetime.now()
logger.info(f'script runtime:{script_end-script_start}')
logger.info(
    '------------------------------------------------------------------------')
