#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import copy

from utils import random_planetoid_splits, rand_train_test_idx



def RunExp(exp_i, args, dataset, data, Net, percls_trn, val_lb):
    """
    Args:
        exp_i: 
        args: 
        dataset:
        data:
        Net:
        percls_trn: 
        val_lb
    Return:
        test_acc, best_val_acc, Gamma_0
    """

    def train(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        if args.dataset.lower() in ['minesweeper', 'tolokers', 'questions']:
            loss = F.binary_cross_entropy_with_logits(out, data.y[data.train_mask].float())
        else:
            loss = F.nll_loss(out, data.y[data.train_mask])
        loss.backward()

        optimizer.step()
        del out

    def test(model, data):
        model.eval()
        logits, accs, losses, preds = model(data), [], [], []
        loss_name = ["train", "val", "test"]
        for index, (_, mask) in enumerate(data('train_mask', 'val_mask', 'test_mask')):
            if args.dataset.lower() in ["minesweeper", "tolokers", "questions"]:
                pred = logits[mask].squeeze()
                acc = roc_auc_score(data.y[mask].cpu(), pred.detach().cpu())
                loss = F.binary_cross_entropy_with_logits(logits[mask].squeeze(), data.y[mask].float())
            else:
                pred = logits[mask].max(1)[1]
                acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                loss = F.nll_loss(model(data)[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    appnp_net = Net(dataset, args)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.dataset.lower() in ['twitch-gamers']:
        data.train_mask, data.val_mask, data.test_mask = rand_train_test_idx(
            data, train_prop=args.train_rate, valid_prop=args.val_rate, curr_seed=exp_i + args.run_num)

    elif args.dataset.lower() in ['penn94']:
        if args.original_split:
            if exp_i == 0:
                data.train_mask_arr = copy.deepcopy(dataset.data.train_mask)
                data.val_mask_arr = copy.deepcopy(dataset.data.val_mask)
                data.test_mask_arr = copy.deepcopy(dataset.data.test_mask)
            data.train_mask = data.train_mask_arr[:, exp_i % 5]
            data.val_mask = data.val_mask_arr[:, exp_i % 5]
            data.test_mask = data.test_mask_arr[:, exp_i % 5]
        else:
            data.train_mask, data.val_mask, data.test_mask = rand_train_test_idx(
                data, train_prop=args.train_rate, valid_prop=args.val_rate, curr_seed=exp_i + args.run_num)

    elif args.dataset.lower() in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
        if args.original_split:
            if exp_i == 0:
                data.train_mask_arr = copy.deepcopy(dataset.data.train_mask)
                data.val_mask_arr = copy.deepcopy(dataset.data.val_mask)
                data.test_mask_arr = copy.deepcopy(dataset.data.test_mask)
            data.train_mask = data.train_mask_arr[:, exp_i % 10]
            data.val_mask = data.val_mask_arr[:, exp_i % 10]
            data.test_mask = data.test_mask_arr[:, exp_i % 10]
        else:
            permute_masks = random_planetoid_splits
            data = permute_masks(data, dataset.num_classes, exp_i + args.run_num, percls_trn, val_lb)
    else:
        permute_masks = random_planetoid_splits
        data = permute_masks(data, dataset.num_classes, exp_i + args.run_num, percls_trn, val_lb)

    model, data = appnp_net.to(device), data.to(device)

    if args.net in ['APPNP', 'GPRGNN']:
        optimizer = torch.optim.Adam([{
            'params': model.lin1.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.lin2.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.prop1.parameters(),
            'weight_decay': 0.0, 'lr': args.lr
        }
        ],
            lr=args.lr)
    elif args.net in ['GCN']:
        if args.weight_decay_type == "digl":
            optimizer = torch.optim.Adam([{
                'params': model.conv1.parameters(),
                'weight_decay': args.weight_decay
            },
                {
                'params': model.conv2.parameters(),
                'weight_decay': 0
            }
            ],
                lr=args.lr)
        elif args.weight_decay_type == "gprgnn":
            optimizer = torch.optim.Adam(model.parameters(),
                                         weight_decay=args.weight_decay,
                                         lr=args.lr
                                         )
    elif args.net in ['MLP']:
        if args.weight_decay_type == "digl":
            optimizer = torch.optim.Adam([{
                'params': next(model.children()).parameters(),
                'weight_decay': args.weight_decay
            },
            ],
                lr=args.lr)
        elif args.weight_decay_type == "gprgnn":
            optimizer = torch.optim.Adam(model.parameters(),
                                         weight_decay=args.weight_decay,
                                         lr=args.lr
                                         )
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    counter = 0

    for epoch in range(args.epochs):
        train(model, optimizer, data, args.dprate)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        if (args.early_stop_loss == 'acc' and (val_acc > best_val_acc)):
            best_val_acc = val_acc
            best_val_loss = val_loss 
            test_acc = tmp_test_acc

            if args.net == 'GPRGNN':
                TEST = appnp_net.prop1.temp.clone()
                Alpha = TEST.detach().cpu().numpy()
            else:
                Alpha = args.alpha
            Gamma_0 = Alpha
            counter = 0

        elif (args.early_stop_loss == 'loss' and (val_loss < best_val_loss)):
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc

            if args.net == 'GPRGNN':
                TEST = appnp_net.prop1.temp.clone()
                Alpha = TEST.detach().cpu().numpy()
            else:
                Alpha = args.alpha
            Gamma_0 = Alpha
            counter = 0

        else:
            if args.early_stop_type == 'digl':
                counter += 1
                if counter == args.early_stopping:
                    break

        if (epoch >= 0 and args.early_stop_type == 'gprgnn'):
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break

    return test_acc, best_val_acc, Gamma_0

