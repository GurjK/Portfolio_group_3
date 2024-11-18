import polars as pl
import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.core.interchange.dataframe_protocol import DataFrame
from sqlalchemy.dialects.mssql.information_schema import columns
from functools import partial
import statsmodels.api as sm
from functools import partial
import hvplot.polars
import datetime


# Custom Functions for Polars
def Portfolio_Statistics(excess_returns: pl.dataframe, annual_factor: int | float, quantile: int | float) -> pl.dataframe:
    "Takes in Excess Return Dataframe and Returns Portfolio Statistics-Polars where index isn't a thing"
    # excess_returns = excess_returns.lazy()
    excess_returns = excess_returns.select(pl.exclude(pl.Date))

    # This function should be used for new columns that do not rely on other columns
    summarized_stats = (
        excess_returns.select(
            mean = pl.struct(pl.all().mean() * annual_factor),
            std = pl.struct(pl.all().std() * np.sqrt(annual_factor)),
            skew = pl.struct(pl.all().skew(bias=False)),
            kurtosis = pl.struct(pl.all().kurtosis(fisher = True,bias=False)),
            VaR = pl.struct(pl.all().quantile(quantile,interpolation = "linear")),
            annualized_Var = pl.struct(pl.all().quantile(quantile,interpolation = "linear")) * np.sqrt(annual_factor),
            cVaR = pl.struct([pl.col(col).filter(pl.col(col) <= pl.col(col).quantile(quantile,interpolation = "linear")).mean()
                              for col in excess_returns.columns]),
            annualized_cVar = pl.struct([pl.col(col).filter(pl.col(col) <= pl.col(col).quantile(quantile,interpolation = "linear")).mean() * np.sqrt(annual_factor)
                              for col in excess_returns.columns]),
            min = pl.struct(pl.all().min())

        )
        .unpivot()
        .unnest("value")
        .unpivot(index = "variable", variable_name = "Index")
        # .collect()
        .pivot("variable", index = "Index")
    )

    # This part should be used for functions that rely on other columns

    summarized_stats_final = (
        summarized_stats.with_columns(
            (pl.col('mean') / pl.col('std')).alias('sharpe'),
            ()

        )

    )

    summarized_stats_final = summarized_stats_final.select(pl.exclude('literal'))

    summarized_stats_final = summarized_stats_final.rename({"Index": "Stock"})

    return summarized_stats_final


def Drawdown_Statistics(returns: pl.DataFrame) -> pl.DataFrame:
    """
    Take in a return stream of data for multiple stocks and return a dataframe
    with details about max, min drawdowns with 'Date' column.
    """

    if "date" in returns.columns:
        returns = returns.rename({"date": "Date"})

    # Unpivot to long format (if not already) to work on individual stock-level data
    returns = returns.unpivot(index = 'Date', value_name = 'Returns', variable_name = 'Stock')

    # Sort the DataFrame by Stock and Date for cumulative calculations
    returns = returns.sort(by = "Date")

    # Step 1: Calculate cumulative returns using cum_prod
    returns = returns.with_columns(
        (1 + pl.col("Returns")).cum_prod().over("Stock").alias("Cumulative_Returns")
    )

    # Step 2: Calculate the rolling maximum of cumulative returns using cum_max
    returns = returns.with_columns(
        pl.col("Cumulative_Returns").cum_max().over("Stock").alias("Rolling_Max")
    )

    # Step 3: Calculate drawdown as (Rolling_Max - Cumulative_Returns) / Rolling_Max
    returns = returns.with_columns(
        ((pl.col("Cumulative_Returns") - pl.col("Rolling_Max")) / pl.col("Rolling_Max")).alias("Drawdown")
    )

    # Step 4: Calculate max drawdown and its corresponding bottom date for each stock
    max_drawdown_df = returns.group_by("Stock").agg([
        pl.col("Drawdown").min().alias("Max_Drawdown"),
        pl.col("Date").filter(pl.col("Drawdown") == pl.col("Drawdown").min()).first().alias("Bottom")
    ])

    # Join max_drawdown_df with returns to access the "Bottom" column in the next steps
    returns_with_bottom = returns.join(max_drawdown_df, on = "Stock",how = "full", coalesce = True)

    # Step 5: Calculate peak (highest point before max drawdown)
    peak_df = returns_with_bottom.filter(
        pl.col("Date") <= pl.col("Bottom")
    ).group_by("Stock").agg([
        pl.col("Date").filter(pl.col("Cumulative_Returns") == pl.col("Rolling_Max")).last().alias("Peak")
    ])

    # Step 6: Calculate recovery date (first date when the cumulative return >= previous peak after bottom date)
    recovery_df = returns_with_bottom.filter(
        pl.col("Date") > pl.col("Bottom")
    ).filter(
        pl.col("Cumulative_Returns") >= pl.col("Rolling_Max")
    ).group_by("Stock").agg([
        pl.col("Date").first().alias("Recover")
    ])

    # Step 7: Calculate duration (recovery date - peak date)
    final_df = max_drawdown_df.join(peak_df, on = "Stock",how = "full", coalesce = True).join(recovery_df, on = "Stock",how = "full", coalesce = True).with_columns(
        (pl.col("Recover") - pl.col("Bottom")).alias("Duration (to Recover)")
    ).select([
        pl.col("Stock"),
        pl.col("Max_Drawdown"),
        pl.col("Peak"),
        pl.col("Bottom"),
        pl.col("Recover"),
        pl.col("Duration (to Recover)")
    ])

    return final_df

def filter_by_min_max(df: pl.DataFrame, col_to_filter: str, min_or_max: str, columns_to_return: list):
    """
    Filters the DataFrame based on the min or max value of a specified column and returns selected columns.

    Args:
    df (pl.DataFrame): The input Polars DataFrame.
    col_to_filter (str): The name of the column to filter by min or max value.
    min_or_max (str): 'min' to filter by minimum value, 'max' to filter by maximum value.
    columns_to_return (list): The list of column names to return.

    Returns:
    pl.DataFrame: A DataFrame filtered by min or max value of the specified column.
    """

    if min_or_max == "min":
        value = df.select(pl.col(col_to_filter).min()).to_series()[0]
    elif min_or_max == "max":
        value = df.select(pl.col(col_to_filter).max()).to_series()[0]
    else:
        raise ValueError("min_or_max must be either 'min' or 'max'")

    filtered_df = df.filter(pl.col(col_to_filter) == value).select(
        [pl.col(col) for col in columns_to_return]
    )


    return filtered_df


def plot_corr_heatmap(df: pl.DataFrame):
    """
    Converts a Polars DataFrame to a Pandas DataFrame, computes the correlation matrix,
    and plots a Seaborn heatmap of the correlation matrix.

    Args:
    df (pl.DataFrame): The input Polars DataFrame.

    Returns:
    None: Displays the heatmap using Seaborn.
    """
    df_nodate = df.select(
        pl.exclude(pl.Date))

    # Convert Polars DataFrame to Pandas DataFrame
    pandas_df = df_nodate.to_pandas()

    # Compute the correlation matrix using Pandas
    corr_matrix = pandas_df.corr()

    # Plot the heatmap using Seaborn
    plt.figure(figsize = (10, 8))
    sns.heatmap(corr_matrix, annot = True, cmap = "coolwarm", fmt = ".2f", linewidths = 0.5)

    # Display the heatmap
    plt.title("Correlation Matrix Heatmap")
    plt.show()

def min_max_correlation(df: pl.DataFrame):
    df_nodate = df.select(
        pl.exclude(pl.Date)
    )

    corr_matrix = df_nodate.corr()

    stacked_corr = corr_matrix.unpivot(variable_name = "Column_B", value_name = "Correlation")

    melted_corr = stacked_corr.with_columns(
        pl.Series("Column_A", list(corr_matrix.columns) * len(corr_matrix.columns)))
    melted_corr = melted_corr.filter(pl.col("Column_A") != pl.col("Column_B"))
    # Sort the column pairs to eliminate swapped duplicates
    melted_corr = melted_corr.with_columns([
        pl.when(pl.col("Column_A") < pl.col("Column_B"))
        .then(pl.col("Column_A"))
        .otherwise(pl.col("Column_B")).alias("Column_A"),
        pl.when(pl.col("Column_A") < pl.col("Column_B"))
        .then(pl.col("Column_B"))
        .otherwise(pl.col("Column_A")).alias("Column_B")
    ])

    # Remove duplicate pairs
    melted_corr = melted_corr.unique(subset = ["Column_A", "Column_B"])

    # Find the minimum correlation value
    min_corr_value = melted_corr["Correlation"].min()
    min_corr_rows = melted_corr.filter(pl.col("Correlation") == min_corr_value)

    # Find the maximum correlation value
    max_corr_value = melted_corr["Correlation"].max()
    max_corr_rows = melted_corr.filter(pl.col("Correlation") == max_corr_value)

    # Extract all unique column pairs and values for min and max correlations
    min_corr_pairs = list(zip(min_corr_rows.select("Column_A").to_series().to_list(),
                              min_corr_rows.select("Column_B").to_series().to_list()))

    max_corr_pairs = list(zip(max_corr_rows.select("Column_A").to_series().to_list(),
                              max_corr_rows.select("Column_B").to_series().to_list()))

    return {
        "min_corr": {"value": min_corr_value, "pairs": min_corr_pairs},
        "max_corr": {"value": max_corr_value, "pairs": max_corr_pairs}
    }


def tangency_portfolio_weights(excess_returns: pl.DataFrame):
    df = excess_returns.select(pl.exclude(pl.Date))

    # Step 1: Extract the stock names (column names) from the input dataframe
    stock_names = df.columns

    # Step 2: Convert Polars DataFrame to NumPy array for covariance calculation
    excess_returns = df.to_numpy()

    # Step 3: Compute the mean of the excess returns (vector of expected excess returns)
    expected_returns = np.mean(excess_returns, axis = 0)

    # Step 4: Compute the covariance matrix of the excess returns
    cov_matrix = np.cov(excess_returns, rowvar = False)

    # Step 5: Compute the inverse of the covariance matrix
    cov_matrix_inv = np.linalg.inv(cov_matrix)

    # Step 6: Calculate the tangency portfolio weights
    ones = np.ones(len(expected_returns))  # Vector of ones
    numerator = cov_matrix_inv @ expected_returns  # Σ⁻¹ * μ
    denominator = ones.T @ cov_matrix_inv @ expected_returns  # 1' * Σ⁻¹ * μ
    tangency_weights = numerator / denominator  # Final tangency portfolio weights

    # Step 7: Create a new Polars DataFrame with Stock names and corresponding weights
    result_df = pl.DataFrame({
        "Stock": stock_names,
        "Tangency Weights": tangency_weights
    })

    return result_df


def global_minimum_variance_portfolio_weights(excess_returns: pl.DataFrame):
    df = excess_returns.select(pl.exclude(pl.Date))

    # Step 1: Extract the stock names (column names) from the input dataframe
    stock_names = df.columns

    # Step 2: Convert Polars DataFrame to NumPy array for covariance calculation
    excess_returns = df.to_numpy()

    # Step 3: Compute the covariance matrix of the excess returns
    cov_matrix = np.cov(excess_returns, rowvar=False)

    # Step 4: Compute the inverse of the covariance matrix
    cov_matrix_inv = np.linalg.inv(cov_matrix)

    # Step 5: Replace the expected returns vector with a vector of ones
    ones = np.ones(len(stock_names))

    # Step 6: Calculate the global minimum variance portfolio weights
    numerator = cov_matrix_inv @ ones  # Σ⁻¹ * 1
    denominator = ones.T @ cov_matrix_inv @ ones  # 1' * Σ⁻¹ * 1
    global_min_variance_weights = numerator / denominator  # Final global minimum variance portfolio weights

    # Step 7: Create a new Polars DataFrame with Stock names and corresponding weights
    result_df = pl.DataFrame({
        "Stock": stock_names,
        "Global Minimum Variance Weights": global_min_variance_weights
    })

    return result_df


def calculate_tangency_portfolio_stats(excess_returns: pl.DataFrame, annual_factor: int) -> pl.DataFrame:
    """
    Calculate the tangency portfolio's annualized mean return, volatility, and Sharpe ratio,
    and return them in a Polars DataFrame.

    Parameters:
    df (pl.DataFrame): Polars DataFrame containing excess returns for multiple assets (no date column).
    annual_factor (int): The annualization factor (e.g., 252 for daily returns).

    Returns:
    pl.DataFrame: A Polars DataFrame containing the annualized mean, volatility, and Sharpe ratio.
    """

    df = excess_returns.select(pl.exclude(pl.Date))

    # Step 1: Extract the stock names (column names) from the input dataframe
    stock_names = df.columns

    # Step 2: Convert Polars DataFrame to NumPy array for covariance calculation
    excess_returns = df.to_numpy()

    # Step 3: Compute the mean of the excess returns (vector of expected excess returns)
    expected_returns = np.mean(excess_returns, axis = 0)

    # Step 4: Compute the covariance matrix of the excess returns
    cov_matrix = np.cov(excess_returns, rowvar = False)

    # Step 5: Compute the inverse of the covariance matrix
    cov_matrix_inv = np.linalg.inv(cov_matrix)

    # Step 6: Calculate the tangency portfolio weights
    ones = np.ones(len(expected_returns))  # Vector of ones
    numerator = cov_matrix_inv @ expected_returns  # Σ⁻¹ * μ
    denominator = ones.T @ cov_matrix_inv @ expected_returns  # 1' * Σ⁻¹ * μ
    tangency_weights = numerator / denominator  # Final tangency portfolio weights

    # Step 5: Calculate the portfolio's annualized mean return
    portfolio_mean = tangency_weights @ expected_returns
    annualized_mean = portfolio_mean * annual_factor

    # Step 6: Calculate the portfolio's annualized volatility
    portfolio_variance = tangency_weights.T @ cov_matrix @ tangency_weights
    annualized_volatility = np.sqrt(portfolio_variance) * np.sqrt(annual_factor)

    # Step 7: Calculate the Sharpe ratio (assuming excess returns so no risk-free rate)
    sharpe_ratio = annualized_mean / annualized_volatility

    # Step 8: Return the results as a Polars DataFrame
    result_df = pl.DataFrame({
        "Metric": ["Annualized Mean", "Annualized Volatility", "Sharpe Ratio"],
        "Value": [annualized_mean, annualized_volatility, sharpe_ratio]
    })

    return result_df


def compute_equal_weights(excess_returns : pl.DataFrame):
    """
    Compute equal weights for `n` assets.

    Parameters:
    n (int): Number of assets

    Returns:
    list: List of equal weights for each asset
    """
    n= len(excess_returns.select(pl.exclude(pl.Date)).columns)
    if n <= 0:
        raise ValueError("Number of assets must be greater than zero")
    return [1 / n] * n


def compute_risk_parity_weights(excess_returns : pl.DataFrame):
    """
    Compute risk parity weights for the assets, which are inversely proportional
    to the variance of each asset's returns.

    Parameters:
    excess_returns (pl.DataFrame): DataFrame of asset returns with the date column

    Returns:
    np.ndarray: Risk parity weights
    """
    excess_returns = excess_returns.select(pl.exclude(pl.Date))
    try:
        variance = excess_returns.var().to_numpy().flatten()
        return 1 / variance
    except ZeroDivisionError:
        raise ValueError("One or more assets have zero variance, cannot compute risk parity weights")


def compute_regularized_weights(excess_returns: pl.DataFrame, cov_matrix_coeff: float, annual_factor: float):
    """
    Compute the regularized covariance matrix and weights.

    Parameters:
    excess_returns (pl.DataFrame): DataFrame of asset returns with the date column.
    cov_matrix_coeff (float): Coefficient for the covariance matrix.
    annual_factor (float): Factor for annualizing the returns.

    Returns:
    np.ndarray: Regularized weights.
    """
    excess_returns = excess_returns.select(pl.exclude(pl.Date))

    # Calculate the mean of excess returns
    excess_returns_mean = excess_returns.mean().to_numpy().flatten()

    # Compute the covariance matrix (Σ)
    cov_matrix = np.cov(excess_returns.to_numpy(), rowvar=False)

    # Compute the regularized covariance matrix (Σ̂)
    if cov_matrix_coeff == 1:
        cov_inv = np.linalg.inv(cov_matrix * annual_factor)
    else:
        covmat_diag = np.diag(np.diag(cov_matrix))
        covmat = cov_matrix_coeff * cov_matrix + (1 - cov_matrix_coeff) * covmat_diag
        cov_inv = np.linalg.pinv(covmat * annual_factor)

    # Calculate the regularized weights: w_REG ∝ Σ̂⁻¹ * μ
    ones = np.ones(excess_returns.shape[1])
    mu = excess_returns_mean * annual_factor
    scaling = 1 / (np.transpose(ones) @ cov_inv @ mu)
    w_reg = scaling * (cov_inv @ mu)

    return w_reg


def compute_portfolio_returns(excess_returns, wts):
    """
    Compute portfolio returns based on the given weights.

    Parameters:
    excess_returns (pl.DataFrame): DataFrame of excess returns for multiple assets
    wts (pl.DataFrame): DataFrame of weights (equal, risk parity, regularized)

    Returns:
    pl.DataFrame: Portfolio returns for each weighting strategy
    """
    # Select all columns except "Date" from excess_returns
    excess_returns_wo_tips_nodate = excess_returns.select(pl.exclude("Date")).to_numpy()

    # Select all columns from wts except the "Stock" column, which contains the asset names
    if "Stock" in wts.columns:
        weight_columns = wts.select(pl.exclude("Stock"))
    else:
        weight_columns = wts

    # Convert the weight DataFrame to a NumPy array for matrix multiplication
    weights_matrix = weight_columns.to_numpy()

    # Perform matrix multiplication to calculate portfolio returns
    portfolio_returns = excess_returns_wo_tips_nodate @ weights_matrix

    # Dynamically get the names of the weight columns to use as names for the resulting portfolio DataFrame
    weight_column_names = weight_columns.columns

    # Create a Polars DataFrame by constructing a dictionary with column names and portfolio return values
    portfolio_returns_df = pl.DataFrame({
        name: portfolio_returns[:, idx] for idx, name in enumerate(weight_column_names)
    })

    return portfolio_returns_df

# Define the function to compute OOS portfolio returns
# Define the function to compute OOS portfolio returns
def compute_oos_portfolio_returns(returns, year,month,day,covmat_coef,annual_factor,mu=.01):
    """
    Fuunction expects a Date column not date
    :param returns:
    :param year:
    :param month:
    :param day:
    :param covmat_coef:
    :param annual_factor:
    :param mu:
    :return:
    """
    if "date" in returns.columns:
        returns = returns.rename({"date": "Date"})
    # Split the data into in-sample (IS) and out-of-sample (OOS) datasets

    OOS = returns.filter(pl.col('Date') > pl.datetime(year,month,day))
    OOS = OOS.filter(pl.col('Date') <= pl.datetime(year+1,month,day))
    IS = returns.filter(pl.col('Date') <= pl.datetime(year,month,day))

    # Compute the portfolio weights using the in-sample data
    tangency_weights = tangency_portfolio_weights(IS)
    tangency_weights_wo_tips = tangency_portfolio_weights(IS.select(pl.exclude("TIP")))
    tangency_weights_w_adjusted_tips = tangency_portfolio_weights(IS.with_columns(pl.col("TIP") + 0.0012))
    eq_weights = compute_equal_weights(IS)
    rp_weights = compute_risk_parity_weights(IS)
    reg_weights = compute_regularized_weights(IS, covmat_coef,annual_factor)

    # Ensure weights are Polars DataFrames and exclude 'Stock' column
    tangency_weights = tangency_weights.select(pl.exclude('Stock'))
    tangency_weights_wo_tips = tangency_weights_wo_tips.select(pl.exclude('Stock'))
    tangency_weights_w_adjusted_tips = tangency_weights_w_adjusted_tips.select(pl.exclude('Stock'))

    # Rescale the weights to set each allocation to have the same mean return
    IS_no_date = IS.select(pl.exclude('Date'))
    IS_no_date_wo_tips = IS_no_date.select(pl.exclude('TIP'))
    IS_mean = IS_no_date.mean().to_numpy().flatten()
    IS_mean_wo_tips = IS_no_date_wo_tips.mean().to_numpy().flatten()
    MU = mu  # Target Monthly Returns

    # Calculate adjustment factors
    adjustment_factors = pl.DataFrame({
        "Tangency": [MU / (IS_mean @ tangency_weights.to_numpy().flatten())],
        "Tangency_wo_TIPS": [MU / (IS_mean_wo_tips @ tangency_weights_wo_tips.to_numpy().flatten())],
        "Tangency_w_Adjusted_TIPS": [MU / (IS_mean @ tangency_weights_w_adjusted_tips.to_numpy().flatten())],
        "Equal_Weights": [MU / (IS_mean @ np.array(eq_weights))],
        "Risk_Parity": [MU / (IS_mean @ rp_weights)],
        "Regularized": [MU / (IS_mean @ reg_weights)]
    })

    # Apply the adjustments and convert to Polars DataFrames
    adjusted_weights = pl.DataFrame({
        "Stock": IS_no_date.columns,
        "Tangency": tangency_weights * adjustment_factors["Tangency"][0],
        "Tangency_w_Adjusted_TIPS": tangency_weights_w_adjusted_tips * adjustment_factors["Tangency_w_Adjusted_TIPS"][0],
        "Equal_Weights": np.array(eq_weights) * adjustment_factors["Equal_Weights"][0],
        "Risk_Parity": rp_weights * adjustment_factors["Risk_Parity"][0],
        "Regularized": reg_weights * adjustment_factors["Regularized"][0]
    })

    # Add Tangency_wo_TIPS separately to handle different length
    adjusted_weights_wo_tips = pl.DataFrame({
        "Stock": IS_no_date_wo_tips.columns,
        "Tangency_wo_TIPS": tangency_weights_wo_tips * adjustment_factors["Tangency_wo_TIPS"][0]
    })

    adjusted_weights_final = adjusted_weights.join(adjusted_weights_wo_tips, on="Stock", how="full", coalesce=True)

    adjusted_weights_final = adjusted_weights_final.with_columns(
        pl.when(pl.col("Tangency_wo_TIPS").is_null()).then(0).otherwise(pl.col("Tangency_wo_TIPS")).alias("Tangency_wo_TIPS")
    )

    adjusted_weights_final_nodate = adjusted_weights_final.select(pl.exclude(pl.Date))
    results_OOS = compute_portfolio_returns(OOS, wts=adjusted_weights_final_nodate)

    return results_OOS
"""
HW 2 Functions
"""


def Regression_Stats_Multiple(s: pl.Series, yvar: str, xvars: list[str], intercept: bool, annual_factor) -> dict:
    df = s.struct.unnest()
    yvar = df[yvar].to_numpy()
    xvars = df[xvars].to_numpy()

    if intercept == True:
        result = sm.OLS(yvar, sm.add_constant(xvars)).fit()
    else:
        result = sm.OLS(yvar, xvars).fit()
    # Extract beta (coefficient for the independent variable)
    beta = result.params[1] if intercept else result.params[0]  # Adjust index if no intercept

    alpha = result.params[0] if intercept else 0
    # Calculate Treynor ratio (12 * mean(y) / beta)
    treynor = annual_factor * yvar.mean() / beta

    if intercept:
        constant = result.params[0]
    else:
        constant = 0

    # Calculate Information Ratio (sqrt(12) * alpha / residual std)
    ir = (constant / result.resid.std()) * np.sqrt(annual_factor)
    RSquared = result.rsquared

    #Annual the alpha
    annualized_alpha = constant * annual_factor


    # Return the results as a dictionary
    return {"Beta": beta, "Alpha": alpha,"Annual Alpha":annualized_alpha,"Annual Treynor": treynor, "Annual IR": ir, "R-Squared": RSquared}

def Regression_Stats_Multiple_New(s: pl.Series, yvar: str, xvars: list[str], intercept: bool, annual_factor) -> pl.DataFrame:
    df = s.struct.unnest()
    yvar = df[yvar].to_numpy()
    xvars_df = df[xvars]
    xvars = xvars_df.to_numpy()

    if intercept:
        result = sm.OLS(yvar, sm.add_constant(xvars)).fit()
    else:
        result = sm.OLS(yvar, xvars).fit()

    # Extract beta coefficients (could be multiple, depending on the number of x variables)
    if intercept:
        betas = result.params[1:]  # Skip the intercept parameter (at index 0)
        beta_names = ["Beta_" + name for name in xvars_df.columns]  # Adjust beta names for intercept
    else:
        betas = result.params
        beta_names = ["Beta_" + name for name in xvars_df.columns]  # Adjust beta names without intercept

    alpha = result.params[0] if intercept else 0
    # Calculate Treynor ratio (12 * mean(y) / beta) for the first beta
    treynor = annual_factor * yvar.mean() / betas[0]

    # Calculate Information Ratio (sqrt(12) * alpha / residual std)
    ir = (alpha / result.resid.std()) * np.sqrt(annual_factor)
    RSquared = result.rsquared

    # Annualize the alpha
    annualized_alpha = alpha * annual_factor

    # Prepare a dictionary for the results
    results = {name: beta for name, beta in zip(beta_names, betas)}
    results.update({
        "R-Squared": RSquared,
        "Constant (Alpha)": alpha,
        "Residuals": result.resid.std(),
        "Annualized Alpha": annualized_alpha,
        "Annual IR": ir,
        "Annualized Treynor": treynor,
        "Annualized Tracking Error": result.resid.std() * np.sqrt(annual_factor),
        "Tracking Error": result.resid.std()
    })

    return results
def Regression_Stats(yvar: pl.DataFrame, xvars: pl.DataFrame, intercept: bool,
                     annual_factor: float | int) -> pl.DataFrame:
    if len(xvars) == 1:
        irmethod=1
    else:
        irmethod =2


    # Convert y and x polars DataFrames to numpy arrays
    y = yvar.to_numpy().flatten()  # Convert yvar (single column) to numpy array and flatten it
    x = xvars.to_numpy()  # Convert xvars (one or more columns) to numpy array



    # Add an intercept if needed
    if intercept:
        x = sm.add_constant(x)

    # Fit the OLS model
    result = sm.OLS(y, x).fit()

    # Extract beta coefficients (could be multiple, depending on the number of x variables)
    if intercept:
        betas = result.params[1:]  # Skip the intercept parameter (at index 0)
        beta_names = list(xvars.columns)  # Just use xvars columns, no intercept in betas list
    else:
        betas = result.params
        beta_names = list(xvars.columns)

    if intercept:
        constant = result.params[0]
    else:
        constant = 0

    residual_model = result.resid.std() * np.sqrt(annual_factor)

    # R-squared value
    RSquared = result.rsquared

    # Calculate the mean absolute error (MAE)
    MAE = np.abs(result.resid).mean()
    Annual_MAE = MAE * annual_factor

    # Calculate residuals and tracking error
    residuals = result.resid

    # Annualize alpha, IR, and Treynor Ratio
    annualized_alpha = constant * annual_factor
    annual_IR = (constant / result.resid.std()) * np.sqrt(annual_factor)
    annual_Treynor = y.mean() / betas[0] * annual_factor #Cant be cacluated for multivariate regression


    if irmethod==1 and intercept:
        if intercept:
            annual_tracking_error = constant / annual_IR
    else:
        annual_tracking_error= result.resid.std() * np.sqrt(annual_factor)

    tracking_error = annual_tracking_error / np.sqrt(annual_factor)



    # Prepare a polars DataFrame for the results
    df_result = pl.DataFrame({
        "Metric": ["Beta_" + name for name in beta_names] + ["R-Squared", "Constant (Alpha)", "Residuals",
                                                             "Annualized Alpha",
                                                             "Annual IR", "Annualized Treynor","Annualized Tracking Error","Tracking Error","CS MAE", "Annualilzed CS MAE"],
        "Value": list(betas) + [RSquared, constant, residual_model, annualized_alpha, annual_IR,
                                annual_Treynor,annual_tracking_error,tracking_error,MAE,Annual_MAE]
    }, strict = False)

    return df_result

def hedging_totals(total_amount: float, stats: pl.DataFrame) -> pl.DataFrame:
    # Extract the betas from the stats table
    betas = stats.filter(pl.col("Metric").str.starts_with("Beta_")).select("Value").to_numpy().flatten()
    beta_names = stats.filter(pl.col("Metric").str.starts_with("Beta_")).select("Metric").to_numpy().flatten()

    # Calculate the hedging amounts
    hedging_amounts = betas * total_amount * -1

    # Format the hedging amounts with a dollar sign and no scientific notation
    formatted_hedging_amounts = [f"${amount:,.2f}" for amount in hedging_amounts]

    # Create a Polars DataFrame with the results
    hedging_df = pl.DataFrame({
        "Stock": beta_names,
        "Hedging Amount": formatted_hedging_amounts
    })

    # Calculate the total hedging amount
    total_hedging_amount = sum(hedging_amounts)
    formatted_total_hedging_amount = f"${total_hedging_amount:,.2f}"

    # Add the total to the DataFrame
    total_row = pl.DataFrame({
        "Stock": ["Total"],
        "Hedging Amount": [formatted_total_hedging_amount]
    })

    hedging_df = pl.concat([hedging_df, total_row])

    return hedging_df

"""

HW 3 Functions

"""


def expanding_quantile(series, quantile):
    """
        Don't actually use this function, it's just a sub function that is used in calculate_expanding_quantile
    """
    expanding_var = series.to_list()
    return [None] + [pl.Series(expanding_var[:i]).quantile(quantile,interpolation = "linear") for i in range(1, len(expanding_var))]

def calculate_expanding_quantile(df, columns, quantile,shift_amount=1):
    """
    Calculate the expanding VaR for the specified columns and quantile. The shift amount is the amount of data to include
    up until that point. i.e if you want the VaR calculated by up until t-1 then you would set shift_amount = 1. There is no
    support for a rolling VaR calculation as of now.
    :param df:
    :param columns:
    :param quantile:
    :param shift_amount:
    :return:
    """
    for column in columns:
        df = df.with_columns(
            pl.Series(expanding_quantile(df[column], quantile)).alias(f"{column}_Expanding_VaR_{quantile}").shift(shift_amount)
        )
    return df

def calculate_rolling_quantile(df, columns, window_size, quantile):
    for column in columns:
        df = df.with_columns(
            pl.col(column).rolling_quantile(window_size=window_size, quantile=quantile, interpolation="linear").alias(f"{column}_rolling_quantile_{quantile}")
        )
    return df

def calculate_EWMA_volatility(df, columns, theta=0.94, initial_variance=0.2 / np.sqrt(252), shift_amount=1):
    initial_vol = initial_variance ** 2
    for column in columns:
        ewma_var = [initial_vol]
        for i in range(df.height):
            new_ewma_var = ewma_var[-1] * theta + (df[column][i] ** 2) * (1 - theta)
            ewma_var.append(new_ewma_var)
        ewma_var.pop(0)  # Remove initial_vol
        ewma_vol = [np.sqrt(v) for v in ewma_var]
        df = df.with_columns([
            pl.Series(ewma_vol).alias(f"{column}_ewma_volatility").shift(shift_amount)
        ])
    return df


def calculate_expanding_volatility(df, columns, shift_amount=1):
    for column in columns:
        df = df.with_columns(
            (pl.col(column).pow(2).cum_sum() / pl.arange(1, pl.len() + 1)).sqrt().alias(f"{column}_expanding_volatility").shift(shift_amount)
        )
    return df

def calculate_rolling_volatility(df, columns, window_size=252, shift_amount=1):
    for column in columns:
        df = df.with_columns(
            (pl.col(column).pow(2).rolling_mean(window_size=window_size)).sqrt().alias(f"{column}_rolling_volatility").shift(shift_amount)
        )
    return df

def expanding_cvar(series, quantile):
    expanding_cvar = series.to_list()
    return [None] + [pl.Series(expanding_cvar[:i]).filter(pl.Series(expanding_cvar[:i]) <= pl.Series(expanding_cvar[:i]).quantile(quantile, interpolation="linear")).mean() for i in range(1, len(expanding_cvar))]

def calculate_expanding_cvar(df, columns, quantile, shift_amount=1):
    for column in columns:
        df = df.with_columns(
            pl.Series(expanding_cvar(df[column], quantile)).alias(f"{column}_Expanding_CVaR_{quantile}").shift(shift_amount)
        )
    return df

def calculate_rolling_cvar(df, columns, window_size, quantile):
    for column in columns:
        df = df.with_columns(
            pl.col(column).rolling_map(
                function=lambda s: s.filter(s <= s.quantile(quantile, interpolation="linear")).mean(),
                window_size=window_size
            ).alias(f"{column}_Rolling_CVaR_{quantile}")
        )
    return df


def report_var_statistics(df, columns, var_columns, quantile_amount = 0.05):
    results = []
    for column, var_column in zip(columns, var_columns):
        # Filter out rows where var_column is None or NA
        filtered_df = df.filter(pl.col(var_column).is_not_null())

        frequency_below_var = filtered_df.filter(pl.col(column) < pl.col(var_column)).shape[0]
        total_periods = len(filtered_df)
        proportion_below_var = frequency_below_var / total_periods
        hit_ratio = proportion_below_var
        hit_error = abs(proportion_below_var / quantile_amount) - 1

        results.append({
            "column": var_column,
            "hit_ratio": hit_ratio,
            "hit_error": hit_error,
        })

    results_df = pl.DataFrame(results)
    return results_df



"""
HW 5 
"""

def calculate_mae(cross_section_results, results_multiple, portfolios, second_column_name, annual_factor=12):
    # Extract the CS MAE value from cross_section_results
    cross_section_mae = cross_section_results.filter(pl.col("Metric") == "CS MAE").select(pl.col("Value")).item()
    annualized_cross_section_mae = cross_section_mae * annual_factor

    # Extract the alphas from the time series results
    alphas = results_multiple.select("Constant (Alpha)")

    # Compute the absolute values of the alphas
    absolute_alphas = alphas.select(pl.col("Constant (Alpha)").abs())

    # Calculate the mean of the absolute values (time series MAE)
    time_series_mae = absolute_alphas.select(pl.col("Constant (Alpha)").mean()).item()
    annualized_time_series_mae = time_series_mae * annual_factor

    # Create a Polars DataFrame with the results
    mae_df = pl.DataFrame({
        "Metric": ["cross_section_mae", "annualized_cross_section_mae", "time_series_mae", "annualized_time_series_mae"],
        second_column_name: [cross_section_mae, annualized_cross_section_mae, time_series_mae, annualized_time_series_mae]
    })

    return mae_df

def Cross_Section_Stats_Clean(cross_section_results_AQR, annual_factor):
    # Rename the metrics
    renamed_df = cross_section_results_AQR.with_columns(
        pl.when(pl.col("Metric").str.contains("Beta_Beta")).then(pl.col("Metric").str.replace("Beta_Beta", "Lambda"))
        .when(pl.col("Metric") == "Constant (Alpha)").then(pl.lit("Eta"))
        .when(pl.col("Metric") == "Annualized Alpha").then(pl.lit("Annualized Eta"))
        .otherwise(pl.col("Metric")).alias("Metric")
    )

    # Keep only the specified metrics
    filtered_df = renamed_df.filter(
        pl.col("Metric").str.contains("Lambda") |
        pl.col("Metric").is_in(["Eta", "Annualized Eta", "R-Squared", "CS MAE", "Annualized CS MAE"])
    )

    # Create new rows for "Annualized" metrics
    annualized_metrics_df = filtered_df.with_columns(
        (pl.col("Value") * annual_factor).alias("Value"),
        pl.format("Annualized {}", pl.col("Metric")).alias("Metric")
    )

    # Concatenate the original filtered DataFrame with the new annualized rows
    result_df = pl.concat([filtered_df, annualized_metrics_df])

    return result_df

"""
HW 6
"""


def calculate_cs_predicted_premium(cross_section_results, results_multiple_factors, annual_factor = 12):
    # Extract CS betas where Metric contains 'Beta_Beta'
    cs_beta_rows = cross_section_results.filter(pl.col('Metric').str.contains('Beta_Beta_'))

    # Create dictionary of CS betas
    cs_betas = {
        row['Metric'].split('_')[-1]: row['Value']
        for row in cs_beta_rows.iter_rows(named = True)
    }

    # Get factor names
    factor_names = list(cs_betas.keys())

    # Get time series beta columns
    ts_beta_cols = [f'Beta_{name}' for name in factor_names]

    # Calculate predicted premium for each stock
    predicted_premiums = results_multiple_factors.with_columns([
        (pl.sum_horizontal(
            pl.col(ts_col) * cs_betas[factor_name]
            for ts_col, factor_name in zip(ts_beta_cols, factor_names)
        ) * annual_factor).alias('Predicted Premium')
    ])

    return predicted_premiums.select(['Stock', 'Predicted Premium'])


def calculate_ts_predicted_premium(results_df, factors_df, annual_factor = 12):
    # Extract the beta columns dynamically
    beta_columns = [col for col in results_df.columns if 'Beta_' in col]

    # Extract the factor names from the beta columns
    factor_names = [col.split('_')[1] for col in beta_columns]

    # Extract the betas and factors
    betas = results_df.select(['Stock'] + beta_columns)
    factors = factors_df.select(factor_names)

    # Calculate the mean of each factor
    factors_mean = factors.select([
        pl.col(col).mean().alias(col) for col in factors.columns
    ]).to_pandas().iloc[0]  # Convert to pandas series for easier multiplication

    # Calculate predicted premium for each stock
    predicted_premiums = betas.with_columns([
        (pl.sum_horizontal(
            pl.col(beta_col) * factors_mean[factor_name] * annual_factor
            for beta_col, factor_name in zip(beta_columns, factor_names)
        )).alias('Predicted Premium')
    ])

    return predicted_premiums.select(['Stock', 'Predicted Premium'])

def calculate_period_statistics(momentum, ff_factors, periods, summary_col_names, correlation_pairs, annual_factor=12):
    res = []

    # Ensure the 'Date' columns are in datetime format
    momentum = momentum.with_columns(pl.col('Date').cast(pl.Date))
    ff_factors = ff_factors.with_columns(pl.col('Date').cast(pl.Date))

    # Extract the stock column name from the momentum DataFrame
    stock_col_name = [col for col in momentum.columns if col != 'Date'][0]

    for period in periods:
        start_date = datetime.strptime(period[0], '%Y')
        end_date = datetime.strptime(period[1], '%Y')

        temp = momentum.filter((pl.col('Date').dt.year() >= start_date.year) & (pl.col('Date').dt.year() <= end_date.year))
        temp_ff = ff_factors.filter((pl.col('Date').dt.year() >= start_date.year) & (pl.col('Date').dt.year() <= end_date.year))

        # Perform a full join on the 'Date' column
        joined_data = temp.join(temp_ff, on='Date', how='outer', coalesce=True)

        # Ensure columns are of type f64 before calculating summary statistics
        temp = temp.with_columns([pl.col(col).cast(pl.Float64) for col in temp.columns if col != 'Date'])
        temp_ff = temp_ff.with_columns([pl.col(col).cast(pl.Float64) for col in temp_ff.columns if col != 'Date'])
        joined_data = joined_data.with_columns([pl.col(col).cast(pl.Float64) for col in joined_data.columns if col != 'Date'])

        # Calculate summary statistics using Portfolio_Statistics
        summary = Portfolio_Statistics(temp, annual_factor, quantile=0.05)

        # Ensure the summary contains the required columns
        if not all(col in summary.columns for col in summary_col_names):
            raise ValueError(f"One or more columns from {summary_col_names} are missing in the summary statistics")

        summary = summary.select(summary_col_names)

        # Calculate correlations
        for pair in correlation_pairs:
            corr_value = joined_data.select(pl.corr(pair[0], pair[1])).to_series()[0]
            summary = summary.with_columns(pl.lit(corr_value).alias(f'corr_{pair[0]}'))

        # Rename the summary for the period
        summary = summary.with_columns(pl.lit(f'{stock_col_name}: {period[0]} - {period[1]}').alias('Period'))

        # Append to results
        res.append(summary)

    # Concatenate all results into a final DataFrame
    summary = pl.concat(res)

    # Reorder columns to make 'Period' the first column and rename columns
    correlation_col_names = [f'corr_{pair[0]}' for pair in correlation_pairs]
    summary = summary.select(['Period'] + summary_col_names + correlation_col_names)
    summary = summary.rename({
        'mean': 'Annualized Mean',
        'std': 'Annualized Vol',
        'sharpe': 'Annualized Sharpe',
        'skew': 'Skewness'
    })

    return summary