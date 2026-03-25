import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Download historical data
def download_data(ticker):
    end_date = datetime.today().strftime('%Y-%m-%d')  # Get current date
    try:
        # Download the data
        data = yf.download(ticker, start="2026-05-04", end=end_date)
        
        if data.empty:
            print(f"Error: No data found for {ticker}. Please check the ticker symbol or date range.")
            return None
        
        # Reset the index to make 'Date' a column
        data = data.reset_index()  # Make 'Date' a column instead of the index
        return data

    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None

# Prepare data for model
def prepare_data(data):
    if data is None or data.empty:
        print("Error: No data available to prepare.")
        return None

    data = data.copy()
    data['Date_ordinal'] = data['Date'].apply(lambda x: x.toordinal())  # Convert 'Date' to ordinal format

    return data[['Date', 'Date_ordinal', 'Close']]

# Train a linear regression model
def train_model(data):
    if data is None or data.empty:
        print("Error: No data available for training.")
        return None

    X = data[['Date_ordinal']]
    y = data['Close']

    if len(X) == 0 or len(y) == 0:
        print("Error: Not enough data points for training.")
        return None

    model = LinearRegression()
    model.fit(X, y)
    return model

# Predict future stock prices
def predict_future_prices(model, data, num_days):
    if model is None or data is None or data.empty:
        print("Error: Model or data missing for prediction.")
        return None, None

    last_date = data['Date'].iloc[-1]
    last_ordinal = data['Date_ordinal'].iloc[-1]

    future_dates_ordinal = [last_ordinal + i for i in range(1, num_days + 1)]
    future_dates = [datetime.fromordinal(date) for date in future_dates_ordinal]

    future_predictions = model.predict(np.array(future_dates_ordinal).reshape(-1, 1))

    return future_dates, future_predictions

# Visualize the predictions
def visualize_predictions(stock_data, future_dates, future_predictions):
    if stock_data is None or stock_data.empty:
        print("Error: No stock data available for visualization.")
        return

    # Reset index just in case and ensure 'Date' is available
    stock_data = stock_data.reset_index()  # This makes 'Date' a column

    if 'Date' not in stock_data.columns:
        print("Error: 'Date' column is missing even after resetting index.")
        return

    if future_dates is None or future_predictions is None or len(future_dates) == 0:
        print("Error: No future predictions available to visualize.")
        return

    plt.figure(figsize=(12, 8))  # Increase figure size for better visibility

    # Plot actual stock prices
    plt.plot(stock_data['Date'], stock_data['Close'], label='Actual Prices', color='blue')

    # Plot predicted future prices
    plt.plot(future_dates, future_predictions, label='Predicted Future Prices', color='red', linestyle='--')

    # Extend x-axis only if future dates exist
    if len(future_dates) > 0:
        plt.xlim([stock_data['Date'].iloc[0], future_dates[-1] + timedelta(days=10)])

    # Format x-axis labels
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    # Set major and minor x-axis ticks
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # Every 2 months
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())

    # Add labels, title, legend, and grid
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()

# Main function to run the program
def main():
    ticker = "TSLA"  # You can change the ticker symbol here
    data = download_data(ticker)

    if data is None:
        return

    prepared_data = prepare_data(data)

    if prepared_data is None:
        return

    model = train_model(prepared_data)

    if model is None:
        return

    future_dates, future_predictions = predict_future_prices(model, prepared_data, num_days=30)

    visualize_predictions(data, future_dates, future_predictions)

if __name__ == "__main__":
    main()
