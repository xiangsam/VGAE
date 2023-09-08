import torch
from math import floor
import numpy as np

def mask_test_edges(adj):
    # remove diagonal adj
    adj = adj - torch.diagflat(torch.diag(adj))
    adj_neg = torch.ones_like(adj) - (adj !=0).int()
    adj_neg = adj_neg - torch.diagflat(torch.diag(adj))
    adj = torch.triu(adj)
    adj_neg = torch.triu(adj_neg)
    
    edges = torch.nonzero(adj)
    num_edges = edges.shape[0]
    eid = torch.randperm(num_edges)
    test_num = floor(num_edges * 0.1)
    val_num = floor(num_edges * 0.05)
    test_edges_eid = eid[:test_num]
    val_edges_eid = eid[test_num:(val_num+test_num)]
    train_eid = eid[(val_num+test_num):]
    test_edges_pos = edges[test_edges_eid, :]
    val_edges_pos = edges[val_edges_eid, :]
    train_edges = edges[train_eid, :]
    neg_edges = torch.nonzero(adj_neg)
    neg_eid = torch.randperm(neg_edges.shape[0])[:(test_num+val_num)]
    test_neg_eid = neg_eid[:test_num]
    val_neg_eid = neg_eid[test_num:]
    assert len(val_neg_eid) == val_num
    test_edges_neg = neg_edges[test_neg_eid, :]
    val_edges_neg = neg_edges[val_neg_eid, :]
    
    train_adj = torch.sparse_coo_tensor(train_edges.T, torch.ones(train_edges.shape[0]).to(adj.device), adj.shape)
    train_adj = train_adj.to_dense()
    train_adj += train_adj.clone().T

    return train_adj, train_edges, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg
