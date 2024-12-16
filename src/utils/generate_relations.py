import os
import time
import pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import logging

parent_dir = os.path.dirname(os.path.abspath(__file__))  # Current script's directory
logs_folder = os.path.join(parent_dir, "../logs")  # Logs folder in the parent directory
os.makedirs(logs_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Configure logging
log_file_path = os.path.join(logs_folder, "generate_relations_log.log")  # Full path for the log file
logging.basicConfig(
    filename=log_file_path,
    filemode="w",  # Overwrite the log file on each run
    level=logging.DEBUG,  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)

logging.info("Script started.")

try:
    # Load pickle file
    path1 = "../../data/train_complete_9000.pkl"
    path2 = "../../data/filtered_names.pkl"
    logging.info(f"Loading pickle file from {path1}")
    with open(path1, 'rb') as f:
        df = pickle.load(f)
    with open(path2, 'rb') as f:
        filtered_names = pickle.load(f) # Dictionary with timesteps as keys and lists of tickers as values (built from train.pkl df)
    logging.info("Pickle files loaded successfully.")
except Exception as e:
    logging.error(f"Error loading pickle file: {e}")
    raise


feature_cols = ['log_rets', 'open', 'close', 'high', 'low', 'volume']


def stock_cor_matrix(ref_dict, codes, n):
    """
    Compute the stock correlation matrix efficiently by computing per-feature correlations.
    
    Args:
        ref_dict: Dictionary of {code: array} containing stock features over time.
        codes: List of stock codes.
        n: Number of observations (e.g., time steps) used in the correlation calculation.
    
    Returns:
        Pandas DataFrame containing the aggregated correlation matrix.
    """
    logging.debug(f"Computing stock correlation matrix for {len(codes)} stocks.")
    # Convert ref_dict to a single 3D array for efficient processing
    features = np.array([ref_dict[code] for code in codes])  # Shape: (num_stocks, num_features, num_timesteps)
    num_stocks, num_features, _ = features.shape

    # Initialize an array to hold per-feature correlation matrices
    feature_correlations = np.zeros((num_stocks, num_stocks, num_features))

    for f in range(num_features):
        # Extract the data for the current feature
        feature_data = features[:, f, :]  # Shape: (num_stocks, num_timesteps)
        
        # Precompute sums needed for the Pearson correlation
        sum_x = np.sum(feature_data, axis=1)  # Shape: (num_stocks,)
        sum_x2 = np.sum(feature_data**2, axis=1)  # Shape: (num_stocks,)
        sum_xy = np.dot(feature_data, feature_data.T)  # Shape: (num_stocks, num_stocks)
        
        # Compute the numerator and denominator for Pearson correlation
        numerator = n * sum_xy - np.outer(sum_x, sum_x)
        denominator = np.sqrt(
            (n * sum_x2[:, None] - sum_x[:, None]**2) *
            (n * sum_x2[None, :] - sum_x[None, :]**2)
        )
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            feature_correlation = np.where(denominator == 0, 0, numerator / denominator)
        
        # Store the correlation matrix for the current feature
        feature_correlations[:, :, f] = feature_correlation
    
    # Average the correlation matrices across features
    avg_correlation = feature_correlations.mean(axis=2)  # Shape: (num_stocks, num_stocks)
    logging.debug("Stock correlation matrix computation complete.")
    return pd.DataFrame(data=avg_correlation, index=codes, columns=codes)


def process_time_step(args):
    end_date, dt, prev_date_num, df1, filtered_names = args
    logging.info(f"Processing time step for end_date: {end_date}")
    start_date = dt[dt.index(end_date) - (prev_date_num - 1)]
    df2 = df1.loc[(df1['dt'] <= end_date) & (df1['dt'] >= start_date)]
    code = filtered_names[end_date] # Stocks to look at at the current time step based on volume filter
    df2 = df2[df2['ticker'].isin(adj_stock_set)]
    logging.info(f"Processing a total of {len(code)} names")

    ######### The following part of the function is for mean imputation of missing data for some tickers at some time steps and then calling the correlation function with complete data
    complete_times = df2['dt'].unique()
    
    # Create a DataFrame with all possible combinations of 'code' and 'dt'
    full_index = pd.MultiIndex.from_product(
        [df2['ticker'].unique(), complete_times], 
        names=['ticker', 'dt']
    )
    full_df = pd.DataFrame(index=full_index).reset_index()
    
    df_complete = full_df.merge(df2, on=['ticker', 'dt'], how='left')
    
    # Fill NaN values for each stock code with the mean across the time window for each feature
    df_complete[feature_cols] = (
        df_complete.groupby('ticker')[feature_cols]
        .transform(lambda group: group.fillna(group.mean()))
    )
    
    df_complete = df_complete.sort_values(by=['ticker', 'dt']).reset_index(drop=True)
    
    result_array = df_complete[feature_cols].values

    n_stocks = df_complete['ticker'].nunique()
    time_steps = df_complete['dt'].nunique()
    
    # Reshape the result_array into a 3D tensor
    features = result_array.reshape((n_stocks, time_steps, len(feature_cols)))

    features_transposed = features.transpose(0, 2, 1)  # Shape will be (n_stocks, n_features, time_steps)

    # Create the dictionary
    test_tmp = {stock_id: feature for stock_id, feature in zip(code, features_transposed)}

    ## Now call the correlation matrix function
    result = stock_cor_matrix(test_tmp, list(test_tmp.keys()), prev_date_num)
    result = result.fillna(0)
    np.fill_diagonal(result.values, 1)
    #result.to_csv(f"../../data/relations/{end_date}.csv")
    # Extract the upper triangular part of the matrix
    upper_triangle = result.where(np.triu(np.ones(result.shape), k=1).astype(bool))
    
    # Save to a pickle file
    upper_triangle.to_pickle(f"../../data/relations/{end_date}.pkl")
    logging.info(f"Processed and saved results for end_date: {end_date}")
    return f"Processed {end_date}"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process a subset of dates based on start and end indices.")
    parser.add_argument("start_idx", type=int, help="Start index of the date array to process.")
    parser.add_argument("processes", type=int, help="Num Processes available for parallelization.")
    args = parser.parse_args()

    try:
        df['dt'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
        prev_date_num = 50
        date_unique = df['dt'].unique()
        dt = date_unique.tolist()
        dt.sort()

        processes = args.processes
        start_idx = args.start_idx
        args_all = [(end_date, dt, prev_date_num, df, filtered_names) for end_date in dt[start_idx:]] # Exclude first 50 time steps since we don't have enough observations for rolling correlations

        logging.info("Starting parallel processing.")
        with mp.Pool(processes=processes) as pool:
            results = list(tqdm(pool.imap(process_time_step, args_all), total=len(args_all), desc="Processing Time Steps"))
        logging.info("Parallel processing completed.")
        print("\n".join(results))
    except Exception as e:
        logging.error(f"Error during script execution: {e}")
        raise
    finally:
        logging.info("Script finished.")
