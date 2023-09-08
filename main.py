import torch
from transformers.trainer_utils import set_seed
from torch_geometric.datasets import Planetoid
from torch.optim import Adam
from model.gae import VGAE
from utils.util import mask_test_edges
from utils.lr_shced import adjust_learning_rate
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import argparse
from utils.early_stop import EarlyStopping

parser = argparse.ArgumentParser('VGAE train')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--epochs', default=2000, type=int)
parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')

set_seed(41)

def get_roc_score(edges_pos, edges_neg, adj_pred):
    adj_pred = adj_pred.detach().cpu().numpy()
    pred_pos, pred_neg = [], []
    for e in edges_pos:
        pred_pos.append(adj_pred[e[0], e[1]])
    for e in edges_neg:
        pred_neg.append(adj_pred[e[0], e[1]])
    preds = np.hstack([pred_pos, pred_neg])
    labels = np.hstack([np.ones_like(pred_pos), np.zeros_like(pred_neg)])
    roc_score = roc_auc_score(labels, preds)
    ap_score = average_precision_score(labels, preds)
    return roc_score, ap_score

def symmetrically_normalize(A):
    # A [N, N]
    I = torch.eye(A.shape[0]).to(A.device)
    A = A + I
    D = A.sum(dim=1)
    D = torch.diagflat(torch.pow(D, -0.5)).to(A.device)
    return D @ A @ D

def train(dataset_name, epoch_num=1000):
    data = Planetoid('./datasets/', name=dataset_name)[0].to('cuda:0')
    features, edge_index = data.x, data.edge_index
    adj_orig = torch.zeros((features.shape[0], features.shape[0])).to('cuda:0')
    edge_index = edge_index.T
    adj_orig[edge_index[:, 0], edge_index[:, 1]] = 1
    train_adj, train_edges, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg = mask_test_edges(adj_orig)
    norm_adj = symmetrically_normalize(train_adj)
    label_adj = train_adj + torch.eye(train_adj.shape[0], device=train_adj.device)
    model = VGAE(features.shape[1], 32, 16).to('cuda:0')
    early_stoping = EarlyStopping(patience=10, verbose=True)
    model.train()
    optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch_id in range(epoch_num):
        adjust_learning_rate(optimizer, epoch_id, args)
        optimizer.zero_grad()
        loss, gene_A = model(features, norm_adj, label_adj)
        if epoch_id % 10 == 0:
            val_roc, val_ap = get_roc_score(val_edges_pos, val_edges_neg, gene_A)
            print(f'epoch {epoch_id}/{epoch_num} | val ROC {val_roc} | val AP {val_ap} | loss {loss:.2f}')
            early_stoping(val_roc, model)
            if early_stoping.early_stop:
                print('model early stop')
                break
        loss.backward()
        optimizer.step()
        

    
    model.eval()
    _, gene_A = model(features, norm_adj, label_adj)
    test_roc, test_ap = get_roc_score(test_edges_pos, test_edges_neg, gene_A)
    print(f'ROC {test_roc} | AP {test_ap}')


if __name__ == '__main__':
    args = parser.parse_args()
    train('Cora', epoch_num=args.epochs)
