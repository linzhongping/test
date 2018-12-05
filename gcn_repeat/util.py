from __future__ import print_function
import pickle
import numpy as np
import scipy.sparse as sp
from keras.utils import to_categorical
import scipy.io as sio
import torch
import pickle as pkl
import sys
seed = 123
np.random.seed(seed)
torch.random.manual_seed(seed)
label_set = ["Agents","AI","DB","IR","ML","HCI"]
import networkx as nx

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
def load_data_repeat(dataset_str):


    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    idx_test = [i for i in idx_test]
    idx_train = [i for i in idx_train]
    idx_val = [i for i in idx_val]


    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, labels,y_train, y_val, y_test        #, train_mask, val_mask, test_mask


def load_data(content_path,cites_path):
    label_set = ["Theory", "Neural_Networks", "Probabilistic_Methods", "Rule_Learning", "Case_Based",
                 "Genetic_Algorithms", "Reinforcement_Learning"]
    with open(content_path,'r') as f:
        lines = f.readlines()
        ids = []
        features = []
        labels = []
        for line in lines:
            id = line.split('\t')[0]
            feature = line.split('\t')[1:-1]
            label = line.split('\t')[-1].replace('\n','')
            ids.append(id)
            features.append(feature)
            labels.append(label_set.index(label))
        features = np.array(features,dtype = np.float)
        # print(features.shape)
        # features = sp.csr_matrix(features,shape=features.shape)
        # print(features)
    with open(cites_path,'r') as f:
        links = f.readlines()
        adj = np.zeros([len(ids),len(ids)],dtype = int)
        # print(ids)
        for l in links:
            u = l.split('\t')[0]
            v = l.split('\t')[1].replace('\n','')
            try:
                adj[ids.index(u)][ids.index(v)], adj[ids.index(v)][ids.index(u)]= 1 , 1
            except:
                pass

        for i in range(len(ids)):
            adj[i][i] = 1
        adj = sp.coo_matrix(adj) #有一些离群点 暂时不处理 相当于抛弃这些点
        labels = to_categorical(labels)


        # print(labels.shape)
        train_y =  range(0,140)#int(len(ids))

        test_y = range(500,1500)
        val_y = range(200,500)
        # print(train_y)
        # print(test_y)
        # print(val_y)

    return adj,features,labels,train_y,test_y,val_y

# def load_data(path):
#     dict = sio.loadmat(path)
#     adj = dict['A']
#     features = dict['X']
#     labels = dict['Y']
#     adj = normalized_adj(adj)
#     features = preprocess_features(features)
#     ids = adj.shape[0]
#     # print(ids)
#     train_y =  range(0,int(ids * 0.8))
#     test_y = range(int(ids * 0.8),int(ids * 0.9))
#     val_y = range(int(ids * 0.9),ids)
#     return adj,features,labels,train_y,test_y,val_y

# def get_indices(filename,feature_nums):
#
#     W = sio.loadmat(filename)['W']
#     select_W = W  # .detach().numpy()
#     scores = select_W.sum(axis=1).tolist()
#     import heapq
#     idx = map(scores.index, heapq.nlargest(feature_nums, scores))
#     indices = []
#     for i in idx:
#         indices.append(i)
#
#
#
#         # 挑选出的新矩阵
#     return indices


def normalized_adj(adj):
    # print(type(adj))
    # rowsum = np.array(adj.sum(1))
    # d_inv_sqrt = np.power(rowsum,-0.5).flatten()
    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def preprocess_adj(adj):
    adj_normalized = normalized_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized
    # return normalized_adj(adj + sp.eye(adj.shape[0]))

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1),dtype='float')
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # return sparse_to_tuple(features)
    return features
    # print(features)
    # rowsum = np.array(features.sum(1),dtype='float')
    #
    # r_inv = np.power(rowsum, -1).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    # r_mat_inv = np.diag(r_inv)
    # features = r_mat_inv.dot(features)
    # # return sparse_to_tuple(features)
    # return features

#
def accuracy(output,labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




