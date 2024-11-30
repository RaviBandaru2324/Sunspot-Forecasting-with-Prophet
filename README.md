# Sunspot Forecasting with Prophet

**Overview**

This project aims to forecast sunspot activity using the Facebook Prophet library. The datasets provided include daily, monthly, and yearly sunspot data. The project preprocesses the data, trains machine learning models, and generates predictions for future sunspot values. The results include visualizations of historical and forecasted data, along with evaluation metrics.

**Features**

**1. Support for Multiple Time Units:**

Daily, Monthly, and Yearly datasets are supported.

Each dataset is handled dynamically with appropriate preprocessing.

**2. Preprocessing:**

Handles missing values.

Supports time-series formatting with dynamic frequency adjustments.

**3. Modeling with Prophet:**

Configurations for growth (linear, logistic, flat) and changepoints.

Seasonal adjustments using Fourier orders.

**4. Evaluation Metrics:**

Mean Absolute Error (MAE)

Mean Absolute Percentage Error (MAPE)

R² Score

**5. Visualization:**

Historical and forecasted sunspot counts.

Confidence intervals for predictions.

**6. Dynamic Configurations:**

Easily adjust forecast periods and model configurations.
