import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
import networkx as nx
import pandas as pd
from torch.autograd import Variable
from datetime import datetime
import argparse
import logging

parent_dir = os.path.dirname(os.path.abspath(__file__))  # Current script's directory
logs_folder = os.path.join(parent_dir, "../logs")  # Logs folder in the parent directory
os.makedirs(logs_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Configure logging
log_file_path = os.path.join(logs_folder, "generate_training_date_log.log")  # Full path for the log file
logging.basicConfig(
    filename=log_file_path,
    filemode="w",  # Overwrite the log file on each run
    level=logging.DEBUG,  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)

logging.info("Script started.")

feature_cols = ['log_rets', 'open', 'close', 'high', 'low', 'volume', '42_vol']

path1 = "../../data/train_complete_9000.pkl"

def fun(t, df1):
    prev_date_num = 70
    ############# Load the triangular correlation matrix and recreate the full correlation matrix
    with open('../../data/relations/'+t, 'rb') as f:
        adj_all = pickle.load(f)

    t = pd.to_datetime(t[:-4])

    index = adj_all.index
    adj_all = np.array(adj_all) # Convert to array to build the full correlation matrix from upper triangular df

    # Get positive and negative relational edges among stocks
    adj_stock_set = list(index)

    np.fill_diagonal(adj_all, 1)
    
    # Copy the upper triangular part to the lower triangular part
    i, j = np.triu_indices_from(adj_all, k=1)
    adj_all[j, i] = adj_all[i, j]  # Mirror upper to lower
    
    pos_g = nx.Graph(adj_all > 0.1)
    pos_adj = nx.adjacency_matrix(pos_g).toarray()
    pos_adj = pos_adj - np.diag(np.diag(pos_adj))
    pos_adj = torch.from_numpy(pos_adj).type(torch.float32)
    neg_g = nx.Graph(adj_all < -0.1)
    neg_adj = nx.adjacency_matrix(neg_g)
    neg_adj.data = np.ones(neg_adj.data.shape)
    neg_adj = neg_adj.toarray()
    neg_adj = neg_adj - np.diag(np.diag(neg_adj))
    neg_adj = torch.from_numpy(neg_adj).type(torch.float32)
    logging.info(f"neg_adj over with shape {neg_adj.shape}.")
    
    start_data = stock_trade_data[stock_trade_data.index(t)-(prev_date_num - 1)]
    df2 = df1.loc[df1['dt'] <= t]
    df2 = df2.loc[df2['dt'] >= start_data]
    df2 = df2[df2['ticker'].isin(adj_stock_set)] # Only keep selected stock for current day in time step t
    code = adj_stock_set
    feature_all = []
    labels = []
    day_last_code = []

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
    features = torch.from_numpy(result_array.reshape((n_stocks, time_steps, len(feature_cols)))).type(torch.float32)

    logging.info(f"Features extracted with shape {features.shape}.")
    
    labels = df2.loc[df2['dt'] == t]['label'].values

    mask = [True] * len(labels)
    labels = torch.tensor(labels, dtype=torch.float32)
    result = {'pos_adj': Variable(pos_adj), 'neg_adj': Variable(neg_adj),  'features': Variable(features),
              'labels': Variable(labels), 'mask': mask}
    with open('../../data/data_train_predict/' + t.strftime('%Y-%m-%d %H:%M:%S') + '.pkl', 'wb') as f:
        pickle.dump(result, f)

    logging.info(f"File saved at {'../../data/data_train_predict/' + t.strftime('%Y-%m-%d %H:%M:%S') + '.pkl'}.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build GNN Training Set.")
    parser.add_argument("start_idx", type=int, help="Start index of the date array to process.")
    parser.add_argument("order", type=int, help="Whether to proceed forward or backward in time")
    args = parser.parse_args()

    with open(path1, 'rb') as f:
        df1 = pickle.load(f)

    logging.info("Pickle files loaded successfully.")
    
    relation = os.listdir('../../data/relations/')
    valid_relation = [x for x in relation if len(os.path.splitext(x)[0]) == 19] # Length of the datetime part is 19 characters
    
    # Sort the filenames based on the datetime extracted from each filename
    relation_sorted = sorted(valid_relation, key=lambda x: datetime.strptime(os.path.splitext(x)[0], '%Y-%m-%d %H:%M:%S'))
    
    df1['dt'] = pd.to_datetime(df1['date'].astype(str) + ' ' + df1['time'].astype(str))
    date_unique = df1['dt'].unique()
    stock_trade_data = date_unique.tolist()
    stock_trade_data.sort()

    df1['dt'] = df1['dt'].astype('datetime64[ns]')

    logging.info(f"Starting Train Set Generation from {relation_sorted[args.start_idx]}.")

    if args.order == 1:
        [fun(t,df1) for t in relation_sorted[args.start_idx:]]
    elif args.order == -1:
        [fun(t,df1) for t in relation_sorted[:args.start_idx-1:-1]]

    logging.info("Script finished.")