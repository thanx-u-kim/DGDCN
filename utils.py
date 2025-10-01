import numpy as np
import pandas as pd
import math

###########################################################################
def load_dataset(feat_path, dtype=np.float32):
    feat_dataframe = pd.read_csv(feat_path)
    feat = np.array(feat_dataframe, dtype=dtype)
    return feat
###########################################################################
def load_adjacency(adj_path):
    adj = np.load(adj_path)
    return adj ## dtype = float64
###########################################################################
def load_dijkstra(matrix_path, dtype=np.float32):
    matrix_df = pd.read_csv(matrix_path)
    matrix = np.array(matrix_df, dtype=dtype)
    return matrix
###########################################################################
def generate_dataset(data, train_len, pred_len, total_len=None, split_ratio=(0.7, 0.8), normalize=True):
    '''
    :param data : feature matrix (can be input)
    :param train_len : length of training sequence (training time step, e.g. 60min -> 6)
    :param pred_len : contents
    :param total_len : contents
    :param split_ratio : contents
    :param normalize : contents
    '''
    if total_len is None:
        total_len = data.shape[0]
    if normalize:
        max_value = np.max(data)
        min_value = np.min(data)
        data = (data - min_value)/(max_value - min_value) ## Min-Max Normalization
    
    train_size = int(total_len * split_ratio[0])
    val_size = int(total_len * split_ratio[1])
    train_data = data[:train_size]
    val_data = data[train_size:val_size]
    test_data = data[val_size:total_len]
      
    train_X, train_Y, val_X, val_Y, test_X, test_Y = list(), list(), list(), list(), list(), list()
     
    for i in range(len(train_data) - train_len - pred_len):
        train_X.append(np.array(train_data[i:i+train_len]))
        train_Y.append(np.array(train_data[i+train_len:i+train_len+pred_len]))
         
    for i in range(len(val_data) - train_len - pred_len):
        val_X.append(np.array(val_data[i:i+train_len]))
        val_Y.append(np.array(val_data[i+train_len:i+train_len+pred_len]))    
        
    for i in range(len(test_data) - train_len - pred_len):
        test_X.append(np.array(test_data[i:i+train_len]))
        test_Y.append(np.array(test_data[i+train_len:i+train_len+pred_len]))
        
    return np.array(train_X), np.array(train_Y), np.array(val_X), np.array(val_Y), np.array(test_X), np.array(test_Y)
###########################################################################
def generate_adjacency(data, train_len, pred_len, total_len=None, split_ratio=(0.7, 0.8), coeff_identity=1.0):
        
    num_node = data.shape[1]
    if total_len is None:
        total_len = data.shape[0]
        
    train_size = int(total_len * split_ratio[0])
    val_size = int(total_len * split_ratio[1])
    train_adj = data[:train_size]
    val_adj = data[train_size:val_size]
    test_adj = data[val_size:total_len]
        
    train_adj_matrix, val_adj_matrix, test_adj_matrix = list(), list(), list()
        
    for i in range(len(train_adj) - train_len - pred_len):
        sample_np = train_adj[i:i+train_len]
        sum_np = np.zeros(shape=(num_node, num_node))
        for j in range(sample_np.shape[0]):
            sum_np += sample_np[j]
        row_sum = sum_np.sum(1)
        for i in range(num_node):
            if row_sum[i] != 0:
                sum_np[i,:] = sum_np[i,:] / row_sum[i]
            else:
                pass
        result1 = sum_np + coeff_identity * np.eye(num_node)
        row_sum = np.linalg.norm(result1, ord=1, axis=1)
        result2 = (result1.T / row_sum).T
        train_adj_matrix.append(result2)
            
    for i in range(len(val_adj) - train_len - pred_len):
        sample_np = val_adj[i:i+train_len]
        sum_np = np.zeros(shape=(num_node, num_node))
        for j in range(sample_np.shape[0]):
            sum_np += sample_np[j]
        row_sum = sum_np.sum(1)
        for i in range(num_node):
            if row_sum[i] != 0:
                sum_np[i,:] = sum_np[i,:] / row_sum[i]
            else:
                pass
        result1 = sum_np + coeff_identity * np.eye(num_node)
        row_sum = np.linalg.norm(result1, ord=1, axis=1)
        result2 = (result1.T / row_sum).T
        val_adj_matrix.append(result2)
            
    for i in range(len(test_adj) - train_len - pred_len):
        sample_np = test_adj[i:i+train_len]
        sum_np = np.zeros(shape=(num_node, num_node))
        for j in range(sample_np.shape[0]):
            sum_np += sample_np[j]
        row_sum = sum_np.sum(1)
        for i in range(num_node):
            if row_sum[i] != 0:
                sum_np[i,:] = sum_np[i,:] / row_sum[i]
            else:
                pass
        result1 = sum_np + coeff_identity * np.eye(num_node)
        row_sum = np.linalg.norm(result1, ord=1, axis=1)
        result2 = (result1.T / row_sum).T
        test_adj_matrix.append(result2)
            
    return np.array(train_adj_matrix), np.array(val_adj_matrix), np.array(test_adj_matrix)
###########################################################################
def generate_dijkstra(data, coeff): ## 들어오는 data는 numpy 형태.
    num_nodes = len(data)
    arr = np.copy(data)
    for i in range(num_nodes):
        for j in range(num_nodes):
            x = arr[i][j]
            arr[i][j] = math.exp(((-1) * (x*x)) / (10**(2*coeff)))
    arr = np.where(arr == 1.0, 0.0, arr)
    output = arr + np.eye(num_nodes)
       
    return output
###########################################################################
def GDC(data, sum_len, alpha: float, eps = None, cut_range = None): ## the 'eps' value should be pre-defined
    seq_len = data.shape[0]
    num_nodes = data.shape[1]
    S = np.zeros(shape=(seq_len, num_nodes, num_nodes))
        
    # PPR-based diffusion
    for i in range(seq_len):
        sample = np.zeros(shape=(num_nodes, num_nodes))
        for k in range(sum_len+1):
            component1 = (alpha * np.eye(num_nodes))
            component2 = (math.pow(1-alpha, k) * np.linalg.matrix_power(data[i], k))
            sample += np.matmul(component1, component2)
        S[i] = sample
    
    # epsilon definition
    S_flat = S.flatten()
    S_hat = S_flat[(S_flat > 0) & (S_flat < 1)]
    S_hat = sorted(S_hat, reverse=True) ## 내림차순으로 재정렬
    if eps is None:
        if cut_range is None:
            eps = np.median(S_hat)
        else:
            cut_length = int(len(S_hat)*cut_range)
            cut_value = S_hat[max(0, cut_length - 1)]
            eps = cut_value
        
    # Sparsify using threshold epsilon
    S_tilde = np.multiply(S, (S>=eps))
    
    # Row-normalized transition matrix on graph S_tilde
    for i in range(seq_len):
        row_sum = np.linalg.norm(S_tilde[i], ord=1, axis=1)
        S_tilde[i] = (S_tilde[i].T / row_sum).T
    return S_tilde
###########################################################################

