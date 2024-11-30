import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from tabulate import tabulate
import matplotlib.pyplot as plt

# Function to preprocess the dataset
def preprocess_data(data):
    data['ds'] = pd.to_datetime(data[['Year', 'Month']].assign(Day=1))
    data.rename(columns={'Sunspot_Number': 'y'}, inplace=True)
    data = data[data['y'] >= 0]  # Remove missing values
    return data[['ds', 'y']]

# Load the monthly dataset
monthly_data = pd.read_csv("SN_m_tot_V2.0.csv", delimiter=";", header=None)
monthly_data.columns = ['Year', 'Month', 'Date_Fraction', 'Sunspot_Number',
                        'Standard_Deviation', 'Observations', 'Indicator']

# Preprocess the data
processed_data = preprocess_data(monthly_data)

# Forecasting intervals
forecast_periods = [1, 6, 9]
future_freq = 'M'

# Model configurations to test
growth_types = ["linear", "logistic", "flat"]
changepoint_prior_scales = [0.1, 0.5]
n_changepoints_list = [10, 20]
fourier_orders = [5, 10]

# Results storage
results = []

# Train and evaluate multiple configurations
for growth in growth_types:
    for cps in changepoint_prior_scales:
        for n_cp in n_changepoints_list:
            for fourier_order in fourier_orders:
                # Initialize the Prophet model
                model = Prophet(growth=growth, changepoint_prior_scale=cps, n_changepoints=n_cp)
                model.add_seasonality(name='custom_monthly', period=12, fourier_order=fourier_order)

                # Add 'cap' column if growth is logistic
                if growth == 'logistic':
                    processed_data['cap'] = processed_data['y'].max() + 10

                # Train the model
                model.fit(processed_data)

                # Make predictions
                future = model.make_future_dataframe(periods=9, freq=future_freq)
                if growth == 'logistic':
                    future['cap'] = processed_data['cap'].max()
                forecast = model.predict(future)

                # Evaluate the model
                y_true = processed_data['y']
                y_pred = forecast.loc[:len(processed_data)-1, 'yhat']
                mae = mean_absolute_error(y_true, y_pred)
                mape = mean_absolute_percentage_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)

                # Store results
                results.append({
                    "Growth": growth,
                    "Changepoint Prior Scale": cps,
                    "Number of Changepoints": n_cp,
                    "Fourier Order": fourier_order,
                    "MAE": mae,
                    "MAPE": mape,
                    "R²": r2
                })

                # Display predicted values for the last period
                predicted_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(9)
                print(f"\nPredicted Values for Growth={growth}, CPS={cps}, n_cp={n_cp}, Fourier Order={fourier_order}:")
                print(tabulate(predicted_values, headers="keys", tablefmt="grid"))

                # Plot historical and forecasted data
                plt.figure(figsize=(12, 6))
                plt.plot(processed_data['ds'], processed_data['y'], label='Historical Data', color='blue', alpha=0.6)
                plt.plot(forecast['ds'], forecast['yhat'], label='Forecasted Data', color='orange', linewidth=2)
                plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.2, label='Confidence Interval')
                plt.title(f"Monthly Sunspot Forecast (Growth={growth}, CPS={cps}, n_cp={n_cp}, Fourier Order={fourier_order})")
                plt.xlabel("Date")
                plt.ylabel("Sunspot Count")
                plt.legend()
                plt.grid()
                plt.show()

# Display summary results in a table
results_df = pd.DataFrame(results)
print("\nSummary of Results:")
print(tabulate(results_df, headers="keys", tablefmt="grid"))