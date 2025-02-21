import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Define the tickers for the Magnificent Seven and the S&P 500
magnificent_seven_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA']
sp500_ticker = '^GSPC'

# Define the start date for the data
start_date = '2019-12-30'

# Download data for the Magnificent Seven and the S&P 500
magnificent_seven_data = yf.download(magnificent_seven_tickers, start=start_date)['Close']
sp500_data = yf.download(sp500_ticker, start=start_date)['Close']

# Combine the data into a single DataFrame
combined_data = magnificent_seven_data.copy()
combined_data['S&P 500'] = sp500_data

# Handle missing data (if any)
combined_data.dropna(inplace=True)

# Calculate log returns
log_returns = np.log(combined_data / combined_data.shift(1)).dropna()

# Calculate 1-year rolling betas for each stock relative to the S&P 500
rolling_window = 252  # Approximate number of trading days in a year
rolling_betas = pd.DataFrame(index=log_returns.index, columns=magnificent_seven_tickers)

for stock in magnificent_seven_tickers:
    covariance = (
        log_returns[stock].rolling(rolling_window).cov(log_returns['S&P 500'])
    )
    variance = log_returns['S&P 500'].rolling(rolling_window).var()
    rolling_betas[stock] = covariance / variance

# Plot the rolling betas
plt.figure(figsize=(14, 8))
for stock in magnificent_seven_tickers:
    plt.plot(rolling_betas.index, rolling_betas[stock], label=stock)

plt.title('1-Year Rolling Betas of Magnificent Seven Stocks to S&P 500')
plt.xlabel('Date')
plt.ylabel('Rolling Beta')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#New Data
AAPL=pd.read_csv("/Users/alexpereyma/Downloads/MANAMAA/AAPL.O.csv")
NVDA=pd.read_csv("/Users/alexpereyma/Downloads/MANAMAA/NVDA.O.csv")
META=pd.read_csv("/Users/alexpereyma/Downloads/MANAMAA/META.O.csv")
GOOG=pd.read_csv("/Users/alexpereyma/Downloads/MANAMAA/GOOG.O.csv")
AMZN=pd.read_csv("/Users/alexpereyma/Downloads/MANAMAA/AMZN.O.csv")
MSFT=pd.read_csv("/Users/alexpereyma/Downloads/MANAMAA/MSFT.O.csv")
TSLA=pd.read_csv("/Users/alexpereyma/Downloads/MANAMAA/TSLA.O.csv")

TSLA["Unnamed: 0"]=pd.to_datetime(TSLA["Unnamed: 0"])
TSLA.set_index("Unnamed: 0", inplace=True)
AAPL["Unnamed: 0"]=pd.to_datetime(AAPL["Unnamed: 0"])
AAPL.set_index("Unnamed: 0", inplace=True)
NVDA["Unnamed: 0"]=pd.to_datetime(NVDA["Unnamed: 0"])
NVDA.set_index("Unnamed: 0", inplace=True)
META["Unnamed: 0"]=pd.to_datetime(META["Unnamed: 0"])
META.set_index("Unnamed: 0", inplace=True)
GOOG["Unnamed: 0"]=pd.to_datetime(GOOG["Unnamed: 0"])
GOOG.set_index("Unnamed: 0", inplace=True)
AMZN["Unnamed: 0"]=pd.to_datetime(AMZN["Unnamed: 0"])
AMZN.set_index("Unnamed: 0", inplace=True)
MSFT["Unnamed: 0"]=pd.to_datetime(MSFT["Unnamed: 0"])
MSFT.set_index("Unnamed: 0", inplace=True)

market_weighted=TSLA["MKT_CAP_ARD"]+AAPL["MKT_CAP_ARD"]+NVDA["MKT_CAP_ARD"]+META["MKT_CAP_ARD"]+GOOG["MKT_CAP_ARD"]+AMZN["MKT_CAP_ARD"]+MSFT["MKT_CAP_ARD"]
market_weighted.dropna(inplace=True)
market_weighted.plot()
market_weighted = market_weighted.rename_axis("Date")
sp500_data_new = yf.download(sp500_ticker, start="2014-03-27")['Adj Close']
sp500_data_new.index = sp500_data_new.index.date
sp500_data_new
sp500_data_new["mag7"]=market_weighted
sp500_data_new.dropna(inplace=True)
(sp500_data_new/sp500_data_new.iloc[0]).plot()
mag7index=sp500_data_new/sp500_data_new.iloc[0]

mag7rets=np.log(mag7index).diff().dropna()
window=252

mag7rets["MarketB"]=mag7rets["mag7"].rolling(window=252).cov(mag7rets["^GSPC"])/mag7rets["^GSPC"].rolling(window=252).var()
mag7rets["MarketB"].plot()
mag7rets["MarketB"]*0.315

mag7rets

mag7rets.iloc[2398]

factorcon=0.315*(mag7index/mag7index.iloc[2399])["mag7"]/(mag7index/mag7index.iloc[2399])["^GSPC"]
factorcon.plot()

mag7rets["MarketBetacon"]=mag7rets["MarketB"]*factorcon
mag7rets["MarketBetacon"].plot()


mag7rets["mag7_est_weight"]=factorcon
mag7rets.to_csv("mag7returns_weights_beta_betacontribution.csv")

#What happens when there are significant changes

#Firstly let's see what happens in regular times
SP500=sp500_data_new.copy()

market_weighted+AAPL["CLOSE"]
AAPL["CLOSE"].index
SP500["AAPL"]=AAPL["CLOSE"]

AAPL= AAPL.rename_axis("Date")
AAPL_=AAPL
SP500
pd.concat([AAPL["CLOSE"],TSLA["CLOSE"]], axis=1)

SP500['AAPL'] = AAPL['CLOSE']
SP500['TSLA'] = TSLA['CLOSE']
SP500['GOOG'] = GOOG["CLOSE"]
SP500['AMZN'] = AMZN['CLOSE']
SP500['MSFT'] = MSFT['CLOSE']
SP500['NVDA'] = NVDA['CLOSE']
SP500['META'] = META['CLOSE']

SPrets=np.log(SP500).diff().dropna()
SPrets


# Drop rows with missing values to ensure proper regression
data = SPrets

def calculate_rolling_beta(stock_returns, benchmark_returns, window=252):
    """
    Calculate rolling beta of a stock against a benchmark.
    
    Args:
        stock_returns (pd.Series): Daily returns of the stock.
        benchmark_returns (pd.Series): Daily returns of the benchmark (e.g., ^GSPC).
        window (int): Rolling window size (default: 90 days).
    
    Returns:
        pd.Series: Rolling beta values.
    """
    # Calculate covariance and variance over the rolling window
    rolling_cov = stock_returns.rolling(window=window).cov(benchmark_returns)
    rolling_var = stock_returns.rolling(window=window).var()
    
    # Beta is covariance divided by variance
    rolling_beta = rolling_cov / rolling_var
    return rolling_beta

# Calculate daily returns for all columns
returns = SPrets

# List of stocks to calculate rolling beta for
stocks = ['AAPL', 'TSLA', 'GOOG', 'AMZN', 'MSFT', 'NVDA', "META", "mag7"]

# Create a DataFrame to store rolling betas
rolling_betas = pd.DataFrame(index=returns.index)

# Calculate rolling beta for each stock
for stock in stocks:
    rolling_betas[stock] = calculate_rolling_beta(returns[stock], returns['^GSPC'])

# Plot rolling betas for each stock
plt.figure(figsize=(12, 8))
for stock in stocks:
    plt.plot(rolling_betas[stock], label=f"{stock} Rolling Beta")

plt.title("Rolling Betas (90-day) Against S&P 500 (^GSPC)")
plt.xlabel("Date")
plt.ylabel("Beta")
plt.legend()
plt.grid()
plt.show()

#Now for the significant events
returnsshort=SPrets.iloc[-1200::]
returnsshort
returnsshort.std()

returnssig=returnsshort.loc[(abs(returnsshort)>2*returnsshort.std()).sum(axis=1)>0]

# Create a dictionary to store betas


# List of stocks
stocks = ['AAPL', 'TSLA', 'GOOG', 'AMZN', 'MSFT', 'NVDA', 'META', 'mag7']

# Create a dictionary to store betas
betas = {}

# Loop through each stock and calculate beta
for stock in stocks:
    # Independent variable (individual stock returns)
    X = returnssig[stock].values.reshape(-1, 1)  # Reshape for sklearn
    
    # Dependent variable (S&P 500 returns, ^GSPC)
    y = returnssig['^GSPC'].values
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Store the beta (slope of the regression line)
    betas[stock] = model.coef_[0]

# Convert betas to a DataFrame for better visualization
betas_df = pd.DataFrame(list(betas.items()), columns=['Stock', 'Beta'])

# Display the beta values
print(betas_df)

# Plot the betas as a bar chart
plt.figure(figsize=(10, 6))
betas_df.set_index('Stock')['Beta'].plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Beta of S&P 500 (^GSPC) Against Individual Stocks')
plt.xlabel('Stock')
plt.ylabel('Beta')
plt.grid(axis='y')
plt.show()



#Finding best ways to calculate beta
SPrets
rolling_betas
SPrets[rolling_betas.columns]
alphas=(rolling_betas*(SPrets[rolling_betas.columns].rolling(window=252).mean()))
alphas
alphas.plot()
(SPrets["^GSPC"].rolling(window=252).mean()).plot()
olsalpha=alphas.sub(SPrets["^GSPC"].rolling(window=252).mean(), axis=0)*-1
olsalpha.plot()

(olsalpha+alphas).plot()
SPrets.shift(-1)
predsp=olsalpha+rolling_betas*((SPrets[rolling_betas.columns]).shift(-1))
predsp.plot()
predsp.sub((SPrets["^GSPC"]).shift(-1))

from pykalman import KalmanFilter
def kalman_filter_beta(SPrets, target_col='^GSPC'):
    """
    Applies a Kalman filter to dynamically estimate the beta of the target column (^GSPC) 
    as the dependent variable (Y) with respect to all other columns (X).

    Args:
        SPrets (pd.DataFrame): Dataframe containing returns data.
        target_col (str): Column name of the target column (^GSPC).

    Returns:
        dict: A dictionary where keys are column names (other than target_col) and values are 
              the beta series as pandas Series.
    """
    results = {}
    target_returns = SPrets[target_col].values

    for col in SPrets.columns:
        if col == target_col:
            continue
        
        stock_returns = SPrets[col].values

        # Kalman filter parameters
        beta = np.zeros(len(SPrets))  # To store beta values
        P = 1  # Initial covariance
        Q = 0.01  # Process noise
        R = 1  # Measurement noise

        # Initial beta value set to 1
        beta_est = 0.5

        for t in range(len(SPrets)):
            # Prediction step
            beta_pred = beta_est
            P_pred = P + Q

            # Observation
            if np.isnan(target_returns[t]) or np.isnan(stock_returns[t]):
                beta[t] = np.nan
                continue

            # Kalman gain
            y = target_returns[t] - beta_pred * stock_returns[t]
            K = P_pred * stock_returns[t] / (stock_returns[t] ** 2 * P_pred + R)

            # Update step
            beta_est = beta_pred + K * y
            P = (1 - K * stock_returns[t]) * P_pred

            beta[t] = beta_est
        
        # Store beta series in results dictionary
        results[col] = pd.Series(beta, index=SPrets.index, name=f'beta_{col}')

    return results

# Example usage
# Assuming SPrets is your dataframe
beta_dict = kalman_filter_beta(SPrets)

# Combine all beta series into a single DataFrame
beta_df = pd.concat(beta_dict.values(), axis=1)
print(beta_df)
beta_df.plot()

import statsmodels as sm
from statsmodels.regression.quantile_regression import QuantReg

def rolling_quantile_beta(SPrets, target_col='^GSPC', quantile=0.5, window=252):
    """
    Calculates a rolling quantile regression beta of the target column (^GSPC)
    as the dependent variable (Y) with respect to all other columns (X).

    Args:
        SPrets (pd.DataFrame): Dataframe containing returns data.
        target_col (str): Column name of the target column (^GSPC).
        quantile (float): Quantile to estimate (e.g., 0.5 for median).
        window (int): Rolling window size.

    Returns:
        pd.DataFrame: DataFrame containing rolling beta estimates for each column.
    """
    target_returns = SPrets[target_col]
    results = {}

    for col in SPrets.columns:
        if col == target_col:
            continue

        stock_returns = SPrets[col]

        # Rolling quantile regression
        betas = []
        for start in range(len(SPrets) - window + 1):
            end = start + window
            y = target_returns.iloc[start:end]
            x = stock_returns.iloc[start:end]

            # Drop NaN values
            valid_mask = ~np.isnan(y) & ~np.isnan(x)
            y = y[valid_mask]
            x = x[valid_mask]

            if len(y) < 2:  # Skip if not enough data
                betas.append(np.nan)
                continue

            # Quantile regression
            X = sm.add_constant(x)  # Add constant for intercept
            model = QuantReg(y, X)
            result = model.fit(q=quantile)
            betas.append(result.params[1])  # Beta is the slope of the regression

        # Add NaNs to the beginning to align the results with the original index
        beta_series = pd.Series([np.nan] * (window - 1) + betas, index=SPrets.index, name=f'beta_{col}')
        results[col] = beta_series

    # Combine all beta series into a single DataFrame
    beta_df = pd.concat(results.values(), axis=1)
    return beta_df

# Example usage
# Assuming SPrets is your DataFrame
beta_df = rolling_quantile_beta(SPrets, target_col='^GSPC', quantile=0.5, window=252)

# Display the resulting beta DataFrame
print(beta_df)





