import yfinance as yf
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import ta
import joblib

# Step 1: Download Tata Steel stock data
data = yf.download('TATASTEEL.NS', start='2018-01-01', end='2023-12-31')

# Step 2: Reset index and save CSV without index
data.reset_index(inplace=True)
data.to_csv('tatasteel_stock_data.csv', index=False)

# Step 3: Load CSV and parse Date column explicitly
data = pd.read_csv('tatasteel_stock_data.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)

# Convert relevant columns to numeric values
cols_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in cols_to_convert:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna()

# Add technical indicators
data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()
macd = ta.trend.MACD(close=data['Close'])
data['MACD'] = macd.macd()
data['MACD_signal'] = macd.macd_signal()
bollinger = ta.volatility.BollingerBands(close=data['Close'], window=20)
data['Bollinger_High'] = bollinger.bollinger_hband()
data['Bollinger_Low'] = bollinger.bollinger_lband()
data['EMA_12'] = ta.trend.EMAIndicator(close=data['Close'], window=12).ema_indicator()
data['EMA_26'] = ta.trend.EMAIndicator(close=data['Close'], window=26).ema_indicator()
data['ATR'] = ta.volatility.AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=14).average_true_range()
data = data.dropna()

# Original features
data['MA7'] = data['Close'].rolling(window=7).mean()
data['Daily Return'] = data['Close'].pct_change()
data['MA30'] = data['Close'].rolling(window=30).mean()
data = data.dropna()

# Target variable for next day price
data['Target'] = data['Close'].shift(-1)
data = data.dropna()

# Define feature matrix and target vector
X = data[['MA7', 'Daily Return', 'MA30', 'RSI', 'MACD', 'MACD_signal',
          'Bollinger_High', 'Bollinger_Low', 'EMA_12', 'EMA_26', 'ATR']]
y = data['Target']

# Setup hyperparameter tuning grid
params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0]
}

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=params,
    n_iter=10,
    cv=3,
    verbose=1,
    n_jobs=-1,
    scoring='neg_mean_squared_error'
)

# Time series split for validation
tscv = TimeSeriesSplit(n_splits=5)

rmses = []
best_fold_model = None
best_fold_rmse = float('inf')

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"\nFold {fold+1}")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    print("Best parameters: ", best_params)

    model = xgb.XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Fold {fold+1} RMSE: {rmse}")
    rmses.append(rmse)

    plt.figure(figsize=(10, 5))
    plt.plot(y_test.index, y_test.values, label='Actual')
    plt.plot(y_test.index, preds, label='Predicted')
    plt.title(f'Fold {fold+1} Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Stock Close Price')
    plt.legend()
    plt.show()

    # Save the best fold model
    if rmse < best_fold_rmse:
        best_fold_rmse = rmse
        best_fold_model = model

avg_rmse = np.mean(rmses)
print(f"\nAverage RMSE across all folds: {avg_rmse}")

# Save the best model to disk for later use in the app
joblib.dump(best_fold_model, 'xgb_model.joblib')
print(f"Best model saved to 'xgb_model.joblib' with RMSE: {best_fold_rmse}")
