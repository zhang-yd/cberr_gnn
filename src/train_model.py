#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset_utils import DataLoader
from utils import *
from gnn_models import *
from exp_utils import RunExp


import data_utils 
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch_geometric
import copy
import numpy as np
import scipy.stats as stats
from torch_geometric.utils import to_dense_adj
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Hyperparameters for training
    parser.add_argument('--net', type=str,
                        choices=['GCN', 'GAT',  'none'],
                        default='GCN', help="Choose the GNN model to train")
    parser.add_argument('--epochs', type=int, default=10000,
                        help="Number of epochs to train (early stopping might change this")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for training")
    parser.add_argument('--weight_decay', type=float, default=0.0005, help="Weight decay for training")
    parser.add_argument('--weight_decay_type', type=str, default='gprgnn', choices=['digl', 'gprgnn'],
                        help="Type of weight decay first layer or all layers (only for GCN)")
    parser.add_argument('--early_stopping', type=int, default=200, help="Number of epochs to wait for early stopping")
    parser.add_argument('--early_stop_type', type=str, default='gprgnn', choices=['digl', 'gprgnn'],
                        help="Type of early stopping: mean (gprgnn) or just wait (digl)")
    parser.add_argument('--early_stop_loss', type=str, default='loss', choices=['acc', 'loss'],
                        help="Early stopping based on loss or accuracy")
    parser.add_argument('--hidden', type=int, default=64, help="Number of hidden units in the GNN")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate for training")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of layers in the GNN")
    parser.add_argument('--data_split', default='sparse', choices=['sparse', 'dense', 'gdl', 'sparse5', 'half'],
                        help="sparse means 0.025 for train/val, dense means 0.6/0.2")
    parser.add_argument('--set_seed', default=True, action=argparse.BooleanOptionalAction,
                        help="Set seed for reproducibility")
    parser.add_argument('--original_split', type=str,
                        choices=['Yes', 'No'],
                        default='Yes', help="Use original split from dataset")

    # GPRGNN specific hyperparameters
    parser.add_argument('--K', type=int, default=10, help="Number of hops (filter order) for GPRGNN")
    parser.add_argument('--alpha', type=float, default=0.1, help="Alpha for GPRGNN Initialization")
    parser.add_argument('--dprate', type=float, default=0.5, help="Dropout rate for GPRGNN")
    parser.add_argument('--Init', type=str,
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='PPR', help="Initialization for GPRGNN")
    parser.add_argument('--ppnp', default='GPR_prop',
                        choices=['PPNP', 'GPR_prop'], help="Choose the propagation method for GPRGNN")

    # GAT specific hyperparameters
    parser.add_argument('--heads', default=8, type=int, help="Number of attention heads for GAT")
    parser.add_argument('--output_heads', default=1, type=int, help="Number of output heads for GAT")

    # Hyperparameters of data
    parser.add_argument('--dataset', default='Cora', help="Dataset to use")
    parser.add_argument('--cuda', type=int, default=0, help="Which GPU to use")
    parser.add_argument('--RPMAX', type=int, default=100, help="Number of experiments to run (different seeds)")
    parser.add_argument('--run_num', type=int, default=0, help="Starting run number also first seed")

    parser.add_argument('--normalize_data', type=str,
                        choices=['Yes', 'No'],
                        default='No', help="Normalize the node features before training")
    parser.add_argument('--use_lcc', type=str,
                        choices=['Yes', 'No'],
                        default='No', help="Use only the largest connected component")
    parser.add_argument('--random_sort', type=str,
                        choices=['Yes', 'No'],
                        default='Yes', help="Randomly sort the nodes, old, kept for reproducibility")
    parser.add_argument('--two_class', type=str, default='None', help="Convert the dataset to two classes")

    # Parameters for joint denoising
    parser.add_argument('--denoise', type=str,
                        choices=['Yes', 'No'],
                        default='No', help="Use a denoisng method")
    parser.add_argument('--denoise_type', type=str,
                        choices=['jointly'],
                        default='jointly', help="Type of denoising, only jointly implemented")
    parser.add_argument('--denoise_A', type=str,
                        choices=['Yes', 'No'],
                        default='Yes', help="If denoise, denoise the adjacency matrix")
    parser.add_argument('--denoise_x', type=str,
                        choices=['Yes', 'No'],
                        default='Yes', help="If denoise, denoise the node features")


    # Experimental hyperparameters
    parser.add_argument('--use_edge_attr', type=str,
                        choices=['Yes', 'No'],
                        default='Yes', help="Build a weighted graph with 1d edge features (weights)")
    parser.add_argument('--denoise_A_eps', type=float,
                        default=0.01,
                        help="Threshold for sparsifying A, set A_k to zero to use this")
    parser.add_argument('--use_node_attr', type=str,
                        choices=['Yes', 'No'],
                        default='No', help="Option to sparsify the node features")
    parser.add_argument('--denoise_X_k', type=int, default=64, help="Number of largest entries in X to keep")
    parser.add_argument('--denoise_X_eps', type=float, default=0.01, help="Threshold for sparsifying X")
    parser.add_argument('--use_right_eigvec', type=str,
                        choices=['Yes', 'No'],
                        default='Yes', help="When multipling back X use the original right singluar vectors")

    # other rewire methods
    parser.add_argument('--rewire', type=str, default='none',
                        choices=['none', 'cbrr'],
                        help="Chosse a rewiring method")
    # cbrr
    parser.add_argument('--cbrr_budget_add', type=int, default=0, help='cbrr algorithm 1')
    parser.add_argument('--cbrr_budget_delete', type=int, default=0, help='myrewrie algorithm 1')
    parser.add_argument('--cbrr_edge_addition', type=int, default=0, help='myrewrie algorithm 1')
    parser.add_argument('--community_resolution', type=float, default=1, help='community_resolution')

    # WandB Hyperparameters
    parser.add_argument("--wandb_log", default=False, action=argparse.BooleanOptionalAction,
                        help="Use WandB for login; default: True")
    parser.add_argument("--wandb_log_figs", default=True, action=argparse.BooleanOptionalAction,
                        help="Use wandb logging for figures")
    parser.add_argument('--show_class_dist', default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    # Reproducibility
    if args.set_seed:
        torch.manual_seed(args.run_num)
        np.random.seed(args.run_num)

    # Data splits
    if args.data_split == "sparse":
        args.train_rate = 0.025
        args.val_rate = 0.025 
    elif args.data_split == "dense":
        args.train_rate = 0.6
        args.val_rate = 0.2 

    # Convert string to boolean (needed for wandb sweeps)
    args.denoise = True if args.denoise == "Yes" else False
    args.denoise_A = True if args.denoise_A == "Yes" else False
    args.denoise_x = True if args.denoise_x == "Yes" else False
    args.normalize_data = True if args.normalize_data == "Yes" else False
    args.random_sort = True if args.random_sort == "Yes" else False
    
    args.use_lcc = True if args.use_lcc == "Yes" else False
    args.use_edge_attr = True if args.use_edge_attr == "Yes" else False
    args.use_node_attr = True if args.use_node_attr == "Yes" else False
    args.use_right_eigvec = True if args.use_right_eigvec == "Yes" else False

    # nets
    gnn_name = args.net
    if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name == 'GAT':
        Net = GAT_Net

    dname = args.dataset
    dataset, data = DataLoader(dname, args.normalize_data)

    # Convert datasets to two classes if needed
    if not args.two_class == 'None':
        data = convert_to_two_class(data, args)
        dataset.data = data

    # Random sorting of the nodes
    if args.random_sort and (args.dataset.lower() not in ['twitch-gamers', 'penn94']):
        data = random_sort_nodes(data)
        dataset.data = data

    # Only use the largest connected component if needed
    if args.use_lcc:
        lcc = get_largest_connected_component(dataset)

        x_new = dataset.data.x[lcc]
        y_new = dataset.data.y[lcc]

        row, col = dataset.data.edge_index.numpy()
        edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
        edges = remap_edges(edges, get_node_mapper(lcc))

        data = torch_geometric.data.Data(
            x=x_new,
            edge_index=torch.LongTensor(edges),
            y=y_new,
            train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
        )
        dataset.data = data

    print("H(G): ", torch_geometric.utils.homophily(data.edge_index, data.y))
    RPMAX = args.RPMAX
    Init = args.Init

    Gamma_0 = None
    alpha = args.alpha
    train_rate = args.train_rate
    val_rate = args.val_rate
    percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
    val_lb = int(round(val_rate*len(data.y)))
    TrueLBrate = (percls_trn*dataset.num_classes+val_lb)/len(data.y)
    print(f'True Label rate in {args.dataset} based on {args.data_split} splitting: {TrueLBrate}')

    args.C = len(data.y.unique())
    args.Gamma = Gamma_0

    # plotting the class distribution
    if args.show_class_dist:
        data_plot = copy.deepcopy(data)
        data_plot.edge_attr = torch.ones(data_plot.edge_index.shape[1], 1)
        plot_class_dist(data_plot)

    # Use device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    recording_parameters = {'dataset': dname}

    # Rewiring
    if args.rewire == 'cbrr':
        import cbrr
        budget_add, budget_delete = args.cbrr_budget_add, args.cbrr_budget_delete, 
        num_additions, community_resolution = args.cbrr_edge_addition, args.community_resolution
        before_edge_cnt = dataset.data.edge_index.shape[1]

        edge_index, _ = cbrr.edge_rewire(dataset.data, budget_add, budget_delete, num_additions, community_resolution)

        dataset.data.edge_index = torch.tensor(edge_index)
        after_edge_cnt = dataset.data.edge_index.shape[1]

        recording_parameters.update({
            'budget_delete': budget_delete,
            'budget_add': budget_add,
            'num_additions': num_additions,
            'before_edge_cnt': before_edge_cnt,
            'after_edge_cnt': after_edge_cnt,
        })

    elif args.rewire == "none":
        print(" rewiring method is None ...... ")

    else:
        raise NotImplementedError(f"Rewiring method {args.rewire} not implemented")

    print(f" this rewiring method is {args.rewire}, args.net value is:   {args.net}")

    if args.show_class_dist:
        data_plot = copy.deepcopy(data)
        data_plot.edge_attr = torch.ones(data_plot.edge_index.shape[1], 1)
        plot_class_dist(data_plot)

    print("H(G) after denoise and/or rewire: ", torch_geometric.utils.homophily(data.edge_index, data.y))
    Results0 = []
    for RP in tqdm(range(RPMAX), desc='Running Experiments'):
        test_acc, best_val_acc, Gamma_0 = RunExp(RP,
            args, dataset, data, Net, percls_trn, val_lb)
        Results0.append([test_acc, best_val_acc])
        data_utils.save_result_1(str(args.rewire), recording_parameters, test_acc, best_val_acc, Gamma_0)

    test_acc_mean, val_acc_mean = np.mean(Results0, axis=0) * 100
    test_acc_std, val_acc_std = np.sqrt(np.var(Results0, axis=0)) * 100
    Results0_test = np.array(Results0)[:, 0] * 100
    Results0_val = np.array(Results0)[:, 1] * 100

    res_val = stats.bootstrap((np.array(Results0_val),), np.mean, confidence_level=0.95, n_resamples=1000)
    res_test = stats.bootstrap((np.array(Results0_test),), np.mean, confidence_level=0.95, n_resamples=1000)
    val_ci_95 = np.max(np.abs(np.array([res_val.confidence_interval.high, res_val.confidence_interval.low]) - np.mean(Results0_val)))
    test_ci_95 = np.max(np.abs(np.array([res_test.confidence_interval.high, res_test.confidence_interval.low]) - np.mean(Results0_test))) 

    print(f'{gnn_name} on dataset {args.dataset}, in {RPMAX} repeated experiment with denoising ({args.denoise}) and {args.rewire} rewiring:')
    print(f'val acc mean = {val_acc_mean:.4f} \t val acc std = {val_acc_std:.4f} \t val acc 95 conf = {val_ci_95:.4f}')
    print(f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t test acc 95 conf = {test_ci_95:.4f}')
