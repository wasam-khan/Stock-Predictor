from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
from flask_bootstrap import Bootstrap
from keras.models import load_model
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from psx import stocks, tickers
import mplfinance as mpf
import os
import plotly.graph_objects as go

# tickers = tickers()




app = Flask(__name__)
Bootstrap(app)

# Dictionary that map the ticket on a the company name
ticker_to_company = {
    "BOP":"Bank of Punjab",
    "PSO":"Pakistan State Oil",
    "AKBL":"Askari bank",
    "HCAR":"Honda Car",
    "ACPL": "Attock cement",
    "LOTCHEM":"Lotte chemical",
    "PSMC":"Pak Suzuki",
    "FEROZ":"Feroz Sons",
    "PRL":"Pak Refinery",
    "HABSM":"Habib Sugar"
    }

def load_stock_data(stock_name, start_date, end_date):
    try:
        data = stocks(stock_name, start=start_date, end=end_date)
       
        data = data.reset_index()
        print(f"Downloaded data for {stock_name}:\n{data}")
        if data.empty:
            print(f"Empty data for {stock_name}. Returning None.")
            return None, None  # Return None for both data and scaler if stock data is not found

        data2 = data

        # Sort the data by date before scaling
        data = data.sort_values(by='Date')

        scaler = MinMaxScaler()
        
        data_scaled = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])

        return data_scaled, scaler , data2

    except Exception as e:
        print(f"Error loading stock data: {e}")
        return None, None


def preprocess_data(data_scaled, scaler, time_steps):
    try:
        # data_scaled = data_scaled[['Open', 'High', 'Low', 'Close', 'Volume']]

        X, y = [], []
        for i in range(len(data_scaled) - time_steps):
            X.append(data_scaled[i:i+time_steps, :5])
            y.append(data_scaled[i+time_steps, 3])  # 'Close' column is at index 3

        print(X)
        return np.array(X), np.array(y)

    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return np.array([]), np.array([])


def train_or_load_model(stock_name, X, y, time_steps):
    try:
        model_filename = f"{stock_name}.h5"
        model = load_model(model_filename)
        return model

    except Exception as e:
        print(f"Error training or loading model: {e}")
        return None


def make_predictions_future(model, latest_data, scaler, time_steps):
    try:
        

        # Use the latest available data to make predictions for the future date
        latest_data_df = pd.DataFrame(latest_data, columns=['Date','Open', 'High', 'Low', 'Close', 'Volume'])
        print(len(latest_data_df))
        latest_date = latest_data_df['Date'].max().date()
        future_date = latest_date + timedelta(days=1)
        
        input_sequence = latest_data_df[['Open', 'High', 'Low', 'Close', 'Volume']].values[-time_steps:]
        input_sequence_scaled = input_sequence.reshape(1, time_steps, 5)

        # Make predictions
        predicted_scaled = model.predict(input_sequence_scaled)

        # Inverse transform the predicted value to get the actual stock price
        predicted_price_scaled = np.array([[predicted_scaled[0, 0], 0, 0, 0, 0]])  # Add zeros for other columns
        predicted_price = scaler.inverse_transform(predicted_price_scaled)

        # Create a new DataFrame with the predicted values
        predicted_df = pd.DataFrame({'Date': [future_date], 'Close': [predicted_price[0][0]]})

        # Concatenate the new DataFrame with the existing DataFrame
        latest_data_df = pd.concat([latest_data_df, predicted_df], ignore_index=True)
    

        return latest_data_df, future_date
    
    except Exception as e:
        
        print(f"Error making future predictions: {e}")
        return pd.DataFrame()
    
def create_flag_chart(data, stock_name):
    try:
        # Extract relevant data for flag chart (adjust as needed)
        flag_data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Set 'Date' column as the index and convert it to DatetimeIndex
        flag_data['Date'] = pd.to_datetime(flag_data['Date'])
        flag_data.set_index('Date', inplace=True)

        # Create a candlestick chart using plotly.graph_objects
        fig = go.Figure(data=[go.Candlestick(x=flag_data.index,
                                             open=flag_data['Open'],
                                             high=flag_data['High'],
                                             low=flag_data['Low'],
                                             close=flag_data['Close'])])
        # Convert the plot to HTML
        chart_html = fig.to_html(full_html=False)

        return chart_html  # Return the HTML code for the chart

    except Exception as e:
        # Print the error for debugging purposes
        print(f"Error creating flag chart: {e}")
        return None


# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')



@app.route('/submit', methods=['POST'])
def submit():
    # Get user input from the form
    stock_name = request.form['stock_name']

    # Initialize future_predictions to an empty DataFrame
    future_predictions = pd.DataFrame()

    # Search for the corresponding key
    found_ticker = None
    for ticker, company in ticker_to_company.items():
        if stock_name.lower() == company.lower():
            found_ticker = ticker
            break
        
    if found_ticker is not None:


        start_date = datetime(2005,1,1).date()
        end_date = datetime.today().date()

        # Load stock data
        data, scaler,data_with_dates = load_stock_data(found_ticker, start_date, end_date)
        print("DATA : ",data_with_dates)
        if data is None:
            return render_template('index.html', error_message="Stock not found. Please enter a valid stock name.")
        
        # Preprocess data
        time_steps = 4000
        X, y = preprocess_data(data, scaler, time_steps)


        # Train or load model
        model = train_or_load_model(found_ticker, X, y, time_steps)

        if model is None:
            return render_template('index.html', error_message="Error loading model. Please try again.")
        

    # Make predictions for the future
        future_predictions,tomorrow_date = make_predictions_future(model, data_with_dates[-time_steps:], scaler, time_steps)
        print("future date : ",tomorrow_date)


        try:
            # Create a flag chart and get the generated HTML code
            flag_chart_html = create_flag_chart(data_with_dates, stock_name)
        except Exception as e:
            # Handle the exception, e.g., set flag_chart_html to None or an error message
            flag_chart_html = None
            print(f"Error creating flag chart: {e}")
        # Check if 'Close' column exists
        if 'Close' in future_predictions.columns:
            
            rounded = future_predictions.tail(1)['Close'].values[0]

            rounded = round(rounded,2)

            result = f"Predicted Close Price for {tomorrow_date}: {rounded}"
            past_15_data = data_with_dates.tail(15)

        else:
            result = "Error: 'Close' column not found in future predictions."

        return render_template('index.html', result=result, stock_name=stock_name.capitalize(),
                        flag_chart_html=flag_chart_html,past_data = past_15_data)

    else:
        error_message = "Company not found"
        return render_template('index.html', error_message=error_message)
    

if __name__ == '__main__':
    app.run(debug=True) 
    
