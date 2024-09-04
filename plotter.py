import psycopg2
import pickle
import os
from dotenv import load_dotenv
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from scipy.stats import gaussian_kde
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from sklearn.metrics import root_mean_squared_error
import numpy as np
import logging
from datetime import datetime


# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='plotter.log',
                    encoding='utf-8', level=logging.DEBUG)


# Load the environment variables
load_dotenv()

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
payload = '''SELECT p.date_time, 
                    p.predictions,
                    f.close
                    FROM public.predictions as p
                    JOIN public.financial_data as f
					ON p.date_time = f.date_time
                    ORDER BY p.date_time;
                    '''

kde_payload  = '''SELECT f.date_time, f.close, f.open, f.hist_close,
                            t.others_cr, t.trend_ema_fast, t.trend_ema_slow, 
                            t.trend_ichimoku_base, t.trend_sma_fast, t.volatility_kcl
                        FROM PUBLIC.financial_data AS f
                        JOIN PUBLIC.technical_data AS t
                        ON f.date_time = t.date_time
                        WHERE f.date_time > '2022-12-05'
                        '''

try:
    cur = conn.cursor()
    cur.execute(payload)
    data = cur.fetchall()
    data_columns = [desc[0] for desc in cur.description]
    logger.info('Payload retrieved from database')
except:
    logger.error('Error executing db query')


try:
    cur = conn.cursor()
    cur.execute(kde_payload)
    kde_data = cur.fetchall()
    kde_data_columns = [desc[0] for desc in cur.description]
    logger.info('kde payload retrieved from database')
except:
    logger.error('Error executing db query: kde data')

df = pd.DataFrame(data, columns=data_columns)

hist_df = pd.DataFrame(kde_data, columns=kde_data_columns)

# Close database connection
cur.close()
conn.close()

features = hist_df.drop(columns=['date_time', 'close'])
target = hist_df['close']
# Make predictions with saved model
filename = 'best_model.pkl'
with open(filename, 'rb') as f:
    model = pickle.load(f)

predictions = model.predict(features)

kde_df = pd.DataFrame({'date_time': hist_df['date_time'],
                       'predictions': predictions,
                       'close': hist_df['close']
                       })


def psi(reference, monitored, bins=None):
    """
    Calculate the Population Stability Index (PSI) between a reference dataset and a monitored dataset.
    
    Parameters:
    reference (numpy.array): The reference dataset, representing the baseline distribution.
    monitored (numpy.array): The monitored dataset, representing the distribution to compare against the reference.
    bins (int, optional): The number of bins to use for the histograms. If set to None, Doane's formula will be used to calculate the number of bins. Default is None.
    
    Returns:
    float: The calculated PSI value. A higher value indicates greater divergence between the two distributions.
    """
    # Get the full dataset
    full_dataset = np.concatenate((reference, monitored))

    # If bins is not parametrized, use Doane's formula for calculating number of bins
    if bins is None:
        _, bin_edges = np.histogram(full_dataset, bins="doane")
    else:  # If number of bins is specified
        bin_edges = np.linspace(min(min(reference), min(monitored)), max(
            max(reference), max(monitored)), bins + 1)

    # Calculate the histogram for each dataset
    reference_hist, _ = np.histogram(reference, bins=bin_edges)
    monitored_hist, _ = np.histogram(monitored, bins=bin_edges)

    # Convert histograms to proportions
    reference_proportions = reference_hist / np.sum(reference_hist)
    monitored_proportions = monitored_hist / np.sum(monitored_hist)

    # Replace zeroes to avoid division by zero or log of zero errors
    monitored_proportions = np.where(
        monitored_proportions == 0, 1e-6, monitored_proportions)
    reference_proportions = np.where(
        reference_proportions == 0, 1e-6, reference_proportions)

    # Calculate PSI
    psi_values = (monitored_proportions - reference_proportions) * \
        np.log(monitored_proportions / reference_proportions)
    psi = np.sum(psi_values)

    return psi


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([

    dbc.Row([
            dbc.Col(html.H1('USD/CAD Predictions Monitoring',
                            className='text-center'), width=12)
            ]),

    dbc.Row([
        dbc.Col(
            html.Div([
                html.H6("Slide to filter by date/time",
                        style={'margin-bottom': '10px'}),
                html.Div(
                    dcc.RangeSlider(
                        id='date-slider',
                        min=0,
                        max=len(df) - 1,
                        value=[0, len(df) - 1],
                        tooltip={"placement": "bottom",
                                 "always_visible": True},
                    ),
                    style={
                        'border': '2px solid #007bff',  # Frame color and thickness
                        'border-radius': '5px',  # rounded corners
                        'padding': '10px',  # Space between the border and the slider
                        'background-color': '#f8f9fa'  # background color inside the border
                    }
                )
            ]),
            width=12,
            style={'padding-top': '40px'}
        )
    ]),

    # time series plot of pred and actual
    dbc.Row([
        dbc.Col(dcc.Graph(id='time-series-graph'), width=12)
    ]),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='kde-plot'), width=8),
        dbc.Col(
            html.Div([
                html.H4("Population Stability Index", className='text-center'),
                html.Div(id='scorecard', style={
                         'font-size': '24px', 'padding': '10px', 'border': '2px solid #007bff', 'border-radius': '5px',
                         'background-color': '#f8f9fa',
                         'text-align': 'center'}),

                html.H4("Root Mean Square Error", className='text-center'),
                html.Div(id='scorecard2', style={
                         'font-size': '24px', 'padding': '10px', 'border': '2px solid #007bff', 'border-radius': '5px',
                         'background-color': '#f8f9fa',
                         'text-align': 'center'})
            ]),
            width=4,
            style={'padding-top': '100px'}
        )
    ])
])

# Update the graph based on the slider input

@app.callback(
    [Output('time-series-graph', 'figure'),
     Output('kde-plot', 'figure'),
     Output('scorecard', 'children'),
     Output('scorecard2', 'children')],
    [Input('date-slider', 'value')]
)
def update_graph(selected_range):
    # Extract start and end indices from the selected range
    start_index, end_index = selected_range

    # Filter the DataFrame to include only the selected range
    filtered_df = df.iloc[start_index:end_index + 1]

    # Convert columns to float
    filtered_df['close'] = filtered_df['close'].astype(float)
    filtered_df['predictions'] = filtered_df['predictions'].astype(float)

    # Create traces for the actual and predicted closing prices
    time_series_fig = go.Figure()
    time_series_fig.add_trace(go.Scatter(x=filtered_df['date_time'], y=filtered_df['close'],
                                         mode='lines', name='Actual', line=dict(color='black')))
    time_series_fig.add_trace(go.Scatter(x=filtered_df['date_time'], y=filtered_df['predictions'],
                                         mode='lines', name='Predictions', line=dict(color='red')))
    time_series_fig.update_layout(
        title='Actual vs Predictions Over Time',
        xaxis_title='Date',
        yaxis_title='Closing Price',
        hovermode='x',
        legend=dict(x=0.9, y=1),
    )

    # Convert columns to float
    kde_df['close'] = kde_df['close'].astype(float)
    kde_df['predictions'] = kde_df['predictions'].astype(float)

    # KDE Plot
    kde_fig = go.Figure()
    actual_kde = gaussian_kde(kde_df['close'])
    pred_kde = gaussian_kde(kde_df['predictions'])

    x_values = np.linspace(kde_df[['close', 'predictions']].min(
    ).min(), kde_df[['close', 'predictions']].max().max(), 1000)

    kde_fig.add_trace(go.Scatter(x=x_values, y=actual_kde(
        x_values), fill='tozeroy', fillcolor='rgba(169, 169, 169, 0.5)', name='Actual KDE'))
    kde_fig.add_trace(go.Scatter(x=x_values, y=pred_kde(
        x_values), fill='tonexty', fillcolor='rgba(255, 99, 71, 0.5)', name='Predictions KDE'))

    kde_fig.update_layout(
        title='KDE of Actual vs Predictions',
        xaxis_title='Value',
        yaxis_title='Density',
        legend=dict(x=0.1, y=0.9)
    )

    # Calculate PSI
    psi_score = psi(kde_df['close'],
                    kde_df['predictions'])
    target = 0.25
    color = 'red' if psi_score > target else 'green'

    scorecard = html.Div([
        html.H4(f"PSI Score: {psi_score:.2f}", style={'color': color}),
        html.P(f"Target: < {target}", style={'color': 'black'})
    ])

    # Calculate rmse
    rmse = root_mean_squared_error(
        kde_df['predictions'], kde_df['close'])
    
    scorecard2 = html.Div([
        html.H4(f"RMSE: {rmse:.6f}", style={'color': 'black'})
    ])

    return time_series_fig, kde_fig, scorecard, scorecard2


script_end = datetime.now()
logger.info(f'Dashboard updated at {script_end}')

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


logger.info(
    '------------------------------------------------------------------------------')







