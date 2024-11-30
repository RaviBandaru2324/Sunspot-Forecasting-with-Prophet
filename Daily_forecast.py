import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from tabulate import tabulate
import matplotlib.pyplot as plt

def determine_time_unit(data):

    if 'Day' in data.columns:
        return 'daily'
    elif 'Month' in data.columns and 'Day' not in data.columns:
        return 'monthly'
    elif 'Year' in data.columns and 'Month' not in data.columns:
        return 'yearly'
    else:
        raise ValueError("Unable to determine the time unit from the dataset structure.")

def preprocess_data(data, time_unit):

    if time_unit == 'daily':
        data['ds'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
    elif time_unit == 'monthly':
        data['ds'] = pd.to_datetime(data[['Year', 'Month']].assign(Day=1))
    elif time_unit == 'yearly':
        data['ds'] = pd.to_datetime(data['Year'], format='%Y')
    else:
        raise ValueError("Unsupported time unit detected.")
    data.rename(columns={'Sunspot_Number': 'y'}, inplace=True)
    data = data[data['y'] >= 0]  # Remove missing values
    return data[['ds', 'y']]

def get_prediction_periods(time_unit):

    if time_unit == 'daily':
        return [100, 200, 365]
    elif time_unit == 'monthly':
        return [1, 6, 9]
    elif time_unit == 'yearly':
        return [1, 10, 20]
    else:
        raise ValueError("Unsupported time unit for prediction periods.")

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return pd.DataFrame({"Metric": ["MAE", "MAPE", "R²"], "Value": [mae, mape, r2]})

# Load dataset
data = pd.read_csv("SN_d_tot_V2.0.csv", delimiter=";", header=None)
data.columns = ['Year', 'Month', 'Day', 'Date_Fraction', 'Sunspot_Number',
                'Standard_Deviation', 'Observations', 'Indicator']

# Step 1: Determine time unit
time_unit = determine_time_unit(data)
print(f"Detected time unit: {time_unit}")

# Step 2: Preprocess data
data = preprocess_data(data, time_unit)

# Step 3: Get prediction periods
prediction_periods = get_prediction_periods(time_unit)

# Train and evaluate models for each prediction period
for period in prediction_periods:
    print(f"\nPredicting for {period} {time_unit}(s) into the future...")

    # Initialize and train the model
    model = Prophet(growth='linear', changepoint_prior_scale=0.3, n_changepoints=10)
    if time_unit == 'daily':
        model.add_seasonality(name='custom_yearly', period=365, fourier_order=5)
    elif time_unit == 'monthly':
        model.add_seasonality(name='custom_monthly', period=12, fourier_order=8)
    elif time_unit == 'yearly':
        model.add_seasonality(name='custom_decade', period=11, fourier_order=3)

    model.fit(data)

    # Make future predictions
    future = model.make_future_dataframe(periods=period, freq=time_unit[0].upper())
    forecast = model.predict(future)

    # Plot historical data and forecast
    plt.figure(figsize=(12, 6))
    plt.plot(data['ds'], data['y'], label='Actual', color='blue', alpha=0.6)
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='orange', linewidth=2)
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.2, label='Confidence Interval')
    plt.title(f"Sunspot Forecast ({period} {time_unit}(s) into the future)")
    plt.xlabel("Date")
    plt.ylabel("Sunspot Count")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot forecast components
    model.plot_components(forecast)

    # Evaluate the model
    y_true = data['y']
    y_pred = forecast.loc[:len(data)-1, 'yhat']
    evaluation_metrics = evaluate_model(y_true, y_pred)

    # Display evaluation metrics
    print("\nEvaluation Metrics:")
    print(tabulate(evaluation_metrics, headers="keys", tablefmt="grid"))

# Display summary results in a table
results_df = pd.DataFrame(evaluation_metrics)
print("\nSummary of Results:")
print(tabulate(results_df, headers="keys", tablefmt="grid"))