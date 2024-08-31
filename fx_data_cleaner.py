import csv
import pandas as pd

print('imported successfully')


def csv_to_columns(csv_file):
    '''Function to extract the columns in each trading data collected from MT4. The columns are seperated by "\t".
    Parameter - csv_file(file path): The relative path of the csv file.
    returns: file_dict(dict): dictionary of where value is column values and key is column name.
    '''
    date = []
    time = []
    open_ = []  # 'open' is a reserved python keyword
    high = []
    low = []
    close = []
    tick_vol = []
    vol = []
    spread = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        next(file)
        for row in reader:
            date.append(row[0])
            time.append(row[1])
            open_.append(row[2])
            high.append(row[3])
            low.append(row[4])
            close.append(row[5])
            tick_vol.append(row[6])
            vol.append(row[7])
            spread.append(row[8])

    file_dict = {'date': date,
                 'time': time,
                 'open': open_,
                 'high': high,
                 'low': low,
                 'close': close,
                 'tick_vol': tick_vol,
                 'vol': vol,
                 'spread': spread
                 }
    return file_dict


def fx_data_type(df):
    '''Function to clean up the datatypes of all columns in the forex data frame. Also convert date and time columns to a single date_time column for time series analysis.
    PARAMETER - df: The dataframe to be cleaned.
    returns: cleaned dataframe with correct datatypes.
    '''
    # Concatenate date and time, and convert to datetime
    df['date'] = df['date'].astype('str')
    df['time'] = df['time'].astype('str')
    df['date_time'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.drop(columns=['date', 'time'], inplace=True)

    # Set cut off date to july 26, 2024 - same for all datasets
    df = df[df['date_time'] < '2024-07-27 00:00:00']

    # Convert numerical columns to float and int where appropriate
    df['open'] = df['open'].astype('float')
    df['high'] = df['high'].astype('float')
    df['low'] = df['low'].astype('float')
    df['close'] = df['close'].astype('float')
    df['tick_vol'] = df['tick_vol'].astype('int')
    df['vol'] = df['vol'].astype('int64')
    df['spread'] = df['spread'].astype('int')

    df
    return (df)
