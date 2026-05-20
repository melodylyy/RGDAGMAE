import time

import torch as th
import random
import numpy as np
import torch.optim
from configs import get_args

from ggmaea import GGMAEA
from tools import make_dir, init_logger, cal_metrics
import torch.nn as nn
from loading_dataset import Data_Process,Data_divide
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# scaler = torch.amp.GradScaler()
scaler = torch.cuda.amp.GradScaler()
#mae_loss mae_loss mae_loss mae_lossmae_loss
def Prediction(model, id, m_embed, d_embed):
    pred_label, mae_loss = model(id, m_embed, d_embed)
    return pred_label

def train(args, train_id, train_label, m_embed, d_embed, model, criterion, optimizer):
    model.train()
    train_label, m_embed, d_embed = map(lambda x: x.float().to(args.device), [train_label, m_embed, d_embed])
    optimizer.zero_grad()

    with torch.amp.autocast(device_type='cuda'):
        outputs = Prediction(model, train_id, m_embed, d_embed)
        loss = criterion(outputs, train_label)


    scaler.scale(loss).backward(retain_graph=True)
    scaler.step(optimizer)
    scaler.update()

    torch.cuda.empty_cache()

    return loss.item()

def test(args, test_id, test_label, m_embed, d_embed, model):
    model.eval()
    AUC, AUPR, F1, recall, precision, ACC = [], [], [], [], [], []

    with torch.no_grad():
        outputs = Prediction(model, test_id, m_embed, d_embed)

    AUC, AUPR, F1, recall, precision, ACC = cal_metrics(test_label, outputs)

    return AUC, AUPR, F1, recall, precision, ACC, outputs



if __name__ == '__main__':

    fix_seed = 2024
    random.seed(fix_seed)
    th.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    th.backends.cudnn.deterministic = True


    args = get_args()


    cache_dir, model_dir, log_dir = make_dir(args)
    logger = init_logger(log_dir)


    final_ids, final_labels, m_embed, d_embed = Data_Process(args).to(args.device)()

    train_ids_5, test_ids_5, train_labels_5, test_labels_5 = Data_divide(args, final_ids, final_labels).to(args.device)()
    best_auc = -float('inf')
    best_params = {}




    model = GGMAEA(args).to(args.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
















































































