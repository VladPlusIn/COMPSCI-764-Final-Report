import pandas as pd
import numpy as np
import datetime
import os
import random
from sklearn.metrics import roc_auc_score
import model as Model
import data_loader as Data

import torch
import torch.nn as nn
import torch.utils.data

import logging
import sys

import config
from itertools import islice


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_model(model_name, feature_nums, field_nums, latent_dims):
    if model_name == 'LR':
        return Model.LR(feature_nums)
    elif model_name == 'FM':
        return Model.FM(feature_nums, latent_dims)
    elif model_name == 'FFM':
        return Model.FFM(feature_nums, field_nums, latent_dims)
    elif model_name == 'W&D':
        return Model.WideAndDeep(feature_nums, field_nums, latent_dims)
    elif model_name == 'DeepFM':
        return Model.DeepFM(feature_nums, field_nums, latent_dims)
    elif model_name == 'FNN':
        return Model.FNN(feature_nums, field_nums, latent_dims)
    elif model_name == 'IPNN':
        return Model.InnerPNN(feature_nums, field_nums, latent_dims)
    elif model_name == 'OPNN':
        return Model.OuterPNN(feature_nums, field_nums, latent_dims)
    elif model_name == 'DCN':
        return Model.DCN(feature_nums, field_nums, latent_dims)
    elif model_name == 'AFM':
        return Model.AFM(feature_nums, field_nums, latent_dims)


def map_fm(line):
    return line.strip().split(',')


def get_dataset(args):
    data_path = args.data_path + args.dataset_name + args.campaign_id

    # click + winning price + hour + time_fraction + timestamp
    train_data = pd.read_csv(os.path.join(data_path, 'train.bid.txt'), header=None).values
    field_nums = train_data.shape[1] - 5  # 17

    val_data = pd.read_csv(os.path.join(data_path, 'test.bid.txt'), header=None).values

    test_data = pd.read_csv(os.path.join(data_path, 'test.bid.txt'), header=None).values

    with open(os.path.join(data_path, 'feat.bid.txt')) as feat_f:
        feature_nums = int(list(islice(feat_f, 0, 1))[0].replace('\n', ''))

    return train_data, val_data, test_data, field_nums, feature_nums


def train(model, optimizer, data_loader, loss, device):
    model.train()  
    total_loss = 0  
    intervals = 0  
    for i, (features, labels) in enumerate(data_loader):
        
        features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)
        
        y = model(features)
        
        train_loss = loss(y, labels.float())

        model.zero_grad()
        train_loss.backward()
        optimizer.step()
        total_loss += train_loss.item()  

        intervals += 1

    return total_loss / intervals


def test(model, data_loader, loss, device):
    model.eval()  
    targets, predicts = list(), list()  
    intervals = 0  
    total_test_loss = 0  

    
    with torch.no_grad():
        for features, labels in data_loader:
            
            features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)
            
            y = model(features)
            
            test_loss = loss(y, labels.float())
            
            targets.extend(labels.tolist())  
            
            predicts.extend(y.tolist())
            intervals += 1
            total_test_loss += test_loss.item()

    return roc_auc_score(targets, predicts), total_test_loss / intervals  


def submission(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)
            y = model(features)

            targets.extend(labels.tolist())  。
            predicts.extend(y.tolist())

    return predicts, roc_auc_score(targets, predicts)


def main(model, model_name, train_data_loader, val_data_loader, test_data_loader, optimizer, loss, device, args):
    valid_aucs = []  
    valid_losses = []  
    early_stop_index = 0  
    is_early_stop = False  

    start_time = datetime.datetime.now()

    for epoch in range(args.epoch):
        
        torch.cuda.empty_cache()  

        train_start_time = datetime.datetime.now()
        
        train_average_loss = train(model, optimizer, train_data_loader, loss, device)
        
        torch.save(model.state_dict(), os.path.join(args.save_param_dir, args.campaign_id, model_name + str(
            np.mod(epoch, args.early_stop_iter)) + '.pth'))
        
        auc, valid_loss = test(model, val_data_loader, loss, device)
        valid_aucs.append(auc)
        valid_losses.append(valid_loss)

        train_end_time = datetime.datetime.now()
        logger.info(
            'Model {}, epoch {}, train loss {}, val auc {}, val loss {} [{}s]'.format(model_name,
                                                                                      epoch,
                                                                                      train_average_loss,
                                                                                      auc, valid_loss,
                                                                                      (train_end_time - train_start_time).seconds))

        if eva_stopping(valid_aucs, valid_losses, args.early_stop_type, args):
            early_stop_index = np.mod(epoch - args.early_stop_iter + 1, args.early_stop_iter)
            is_early_stop = True
            break

    end_time = datetime.datetime.now()

    if is_early_stop:
        
        test_model = get_model(model_name, feature_nums, field_nums, args.latent_dims).to(device)
        load_path = os.path.join(args.save_param_dir, args.campaign_id, model_name + str(
            early_stop_index) + '.pth')

        test_model.load_state_dict(torch.load(load_path, map_location=device))  
    else:
        test_model = model

    test_predicts, test_auc = submission(test_model, test_data_loader, device)
    torch.save(test_model.state_dict(),
               os.path.join(args.save_param_dir, args.campaign_id, model_name + 'best.pth'))  

    logger.info('Model {}, test auc {} [total time: {}s]'.format(model_name, test_auc,
                                                                 (end_time - start_time).seconds))

    
    #for i in range(args.early_stop_iter):
        #os.remove(os.path.join(args.save_param_dir, args.campaign_id, model_name + str(i) + '.pth'))

    #return test_predicts


def eva_stopping(valid_aucs, valid_losses, type, args):
    
    if type == 'auc':
        if len(valid_aucs) >= args.early_stop_iter:
            auc_campare_arrs = [valid_aucs[-i] < valid_aucs[-i - 1] for i in range(1, args.early_stop_iter)]
            auc_div_mean = sum([abs(valid_aucs[-i] - valid_aucs[-i - 1]) for i in
                                range(1, args.early_stop_iter)]) / args.early_stop_iter

            if (False not in auc_campare_arrs) or (auc_div_mean <= args.auc_epsilon):
                return True
    else:
        if len(valid_losses) >= args.early_stop_iter:
            loss_compare_arrs = [valid_losses[-i] > valid_losses[-i - 1] for i in range(1, args.early_stop_iter)]
            loss_div_mean = sum([abs(valid_losses[-i] - valid_losses[-i - 1]) for i in
                                 range(1, args.early_stop_iter)]) / args.early_stop_iter

            if (False not in loss_compare_arrs) or (loss_div_mean <= args.loss_epsilon):
                return True

    return False


if __name__ == '__main__':
    args = config.init_parser()
    train_data, val_data, test_data, field_nums, feature_nums = get_dataset(args)

    log_dirs = [args.save_log_dir, args.save_log_dir + args.campaign_id]
    for log_dir in log_dirs:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

    param_dirs = [args.save_param_dir, args.save_param_dir + args.campaign_id]
    for param_dir in param_dirs:
        if not os.path.exists(param_dir):
            os.mkdir(param_dir)

    setup_seed(args.seed)

    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(args.save_log_dir, args.campaign_id, 'output.log'),
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    # click + winning price + hour + time_fraction + timestamp
    test_dataset = Data.libsvm_dataset(test_data[:, 5:], test_data[:, 0])
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, num_workers=12)

    val_dataset = Data.libsvm_dataset(val_data[:, 5:], val_data[:, 0])
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, num_workers=12)

    loss = nn.BCELoss()

    device = torch.device(args.device)  

    choose_models = [args.ctr_model]
    logger.info(args.campaign_id)
    logger.info('Models ' + ','.join(choose_models) + ' have been trained')

    test_predict_arr_dicts = {}
    for model_name in choose_models:
        test_predict_arr_dicts.setdefault(model_name, [])

    # click + winning price + hour + time_fraction + timestamp
    train_dataset = Data.libsvm_dataset(train_data[:, 5:], train_data[:, 0])
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=12)

    for model_name in choose_models:
        model = get_model(model_name, feature_nums, field_nums, args.latent_dims).to(device)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate,
                                     weight_decay=args.weight_decay)

        if model_name == 'FNN':
            fm_pth_path = os.path.join(args.save_param_dir, args.campaign_id, 'FNNbest2.pth')
            if os.path.exists(fm_pth_path):
                FM_pretain_args = torch.load(fm_pth_path)
                model.load_embedding(FM_pretain_args)
            else:
                logger.warning(f"Pre-trained file {fm_pth_path} not found, skipping loading pre-trained FM model.")

        current_model_test_predicts = main(model, model_name, train_data_loader, val_data_loader, test_data_loader,
                                           optimizer, loss, device, args)

        test_predict_arr_dicts[model_name].append(current_model_test_predicts)

    # submission
    for key in test_predict_arr_dicts.keys():
        submission_path = os.path.join('results')  # ctr 

        if not os.path.exists(submission_path):
            os.mkdir(submission_path)

        final_sub = np.mean(test_predict_arr_dicts[key], axis=0)
        test_pred_df = pd.DataFrame(data=final_sub)
        test_pred_df.to_csv(os.path.join(submission_path, key + '_test_submission.csv'), header=None)

        final_auc = roc_auc_score(test_data[:, 0: 1].tolist(), final_sub.reshape(-1, 1).tolist())
        day_aucs = [[final_auc]]
        day_aucs_df = pd.DataFrame(data=day_aucs)
        day_aucs_df.to_csv(os.path.join(submission_path, key + '_auc.csv'), header=None)

        if args.dataset_name == 'ipinyou/':
            logger.info('Model {}, dataset {}, campain {}, test auc {}\n'.format(key, args.dataset_name,
                                                                                 args.campaign_id, final_auc))
        else:
            logger.info('Model {}, dataset {}, test auc {}\n'.format(key, args.dataset_name, final_auc))
