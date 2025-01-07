


import time
from model_DCRAKAN import *
import torch.nn as nn
from data_load import *
import numpy as np
from torch.utils.data import DataLoader
from utils.logger import init_logger
from torch.utils.tensorboard import SummaryWriter
import warnings
from tslearn.metrics import SoftDTWLossPyTorch

soft_dtw_loss = SoftDTWLossPyTorch(gamma=0.1)  # gamma是Soft-DTW的平滑参数
warnings.filterwarnings("ignore")

#TODO:  The proposed prediction advance and shape constrained loss function
class CombinedLoss(nn.Module):
    def __init__(self, qtse_weight, soft_dtw_weight, gamma, kqtse):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.soft_dtw_loss = SoftDTWLossPyTorch(gamma=gamma)
        self.qtse_weight = qtse_weight
        self.soft_dtw_weight = soft_dtw_weight
        self.kqtse = kqtse
    def forward(self, predicted, target):
        error = predicted - target
        QTSE = torch.mean((error ** 2) * torch.exp(self.kqtse * error))
        soft_dtw = self.soft_dtw_loss(predicted.unsqueeze(1).unsqueeze(1), target.unsqueeze(1).unsqueeze(1))
        soft_dtw = soft_dtw.mean()
        return self.soft_dtw_weight * soft_dtw + self.qtse_weight * QTSE

def Training(opt):
    PATH = opt.path + "-" + opt.dataset
    logger = init_logger(opt.save_path, opt, True)

    WRITER = SummaryWriter(log_dir=opt.save_path)

    ##------load parameters--------##
    dataset=opt.dataset
    num_epochs = opt.epoch  # Number of training epochs
    batch_size = opt.batch_size
    train_seq_len = opt.train_seq_len
    win = opt.win
    test_seq_len = opt.test_seq_len
    LR = opt.LR
    att = opt.att
    smooth_param = opt.smooth_param
    ##------Model to CUDA------##

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    ##------load dataset--------##
    group_train, y_test, group_test, X_test = data_processing(dataset, smooth_param)

    print("data processed")
    train_dataset = SequenceDataset(mode='train', win = win, group=group_train, sequence_train=train_seq_len,
                                    patch_size=train_seq_len)
    test_dataset = SequenceDataset(mode='test', win = win, group=group_test, y_label=y_test, sequence_train=train_seq_len,
                                   patch_size=train_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print("train loaded")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=False)
    print("test loaded")

    ##------SAVE PATH--------##
    if opt.path == '':
        PATH = "train-model-" + time.strftime("%m-%d-%H:%M:%S", time.localtime()) + ".pth"
    else:
        PATH = PATH

    logger.cprint("------Train-------")
    logger.cprint("------" + PATH + "-------")
    # result.cprint(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # result.cprint("Path Name:%s" % (PATH))

    ##------model define--------##
    train_model = DCNNKAN(in_channel=win,out_channel=64,att=att)
    print(train_model)

    # ------put model to GPU------#
    if torch.cuda.is_available():
        train_model = train_model.to(device)

    for p in train_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    # criterion = torch.nn.MSELoss(reduction="mean")
    # criterion = CustomLoss()
    optimization = torch.optim.Adam(filter(lambda p: p.requires_grad, train_model.parameters()), lr=LR,
                                    weight_decay=opt.weight_decay)
    for p in train_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    best_test_score1 = 750

    for epoch in range(num_epochs):
        train_model.train()
        train_epoch_loss = 0
        iter_num = 0
        for X, y in train_loader:
            iter_num += 1
            # print("X_input", X.shape)

            if torch.cuda.is_available():
                X = X.cuda()
                # print(X.shape)
                # X = torch.unsqueeze(X, 1)

                y = y.cuda()

            y_pred = train_model.forward(X)
            if epoch <=10:
                kqtse=0
            elif epoch >10:
                kqtse=0.05*epoch
            #criterion = torch.nn.MSELoss(reduction="mean")
            criterion = CombinedLoss(qtse_weight=0.5, soft_dtw_weight=0.5, gamma=0.1, kqtse=kqtse)
            loss = criterion(y_pred.reshape(y_pred.shape[0]), y)  # mse loss

            optimization.zero_grad()
            loss.backward()
            # lr_scheduler.step()
            optimization.step()

            train_epoch_loss = train_epoch_loss + loss.item()
        train_epoch_loss = np.sqrt(train_epoch_loss / len(train_loader))
        WRITER.add_scalar('Train RMSE', train_epoch_loss, epoch)
        #测试or验证
        train_model.eval()
        with torch.no_grad():
            test_epoch_loss = 0
            error = 0
            res = 0
            for X, y in test_loader:
                if torch.cuda.is_available():
                    X = X.cuda()
                    # print(X.shape)
                    # X = torch.unsqueeze(X, 1)
                    y = y.cuda()

                y_hat_recons = train_model.forward(X)
                y_hat_unscale = y_hat_recons * 125
                y_hat_unscale[y_hat_unscale < 0] = 0
                y_hat_unscale[y_hat_unscale > 125] = 125

                subs = y_hat_unscale.reshape(y_hat_recons.shape[0]) - y
                error = error + subs * subs
                subs = subs.cpu().detach().numpy()

                if subs[0] < 0:
                    res = res + np.exp(-subs / 13)[0] - 1
                else:
                    res = res + np.exp(subs / 10)[0] - 1

                loss = criterion(y_hat_unscale.reshape(y_hat_recons.shape[0]), y)
                test_epoch_loss = test_epoch_loss + loss
            test_rmse = torch.sqrt(error / len(test_loader))
            test_score = res
            WRITER.add_scalar('Test loss', test_rmse, epoch)
            if epoch >= 0 and test_score < best_test_score1:  # est_rmse < best_test_rmse1 or
                best_test_rmse1 = test_rmse
                best_test_score1 = res
                cur_best = train_model.state_dict()
                best_model_path = PATH + "_new_best" + ".pth"
                torch.save(cur_best, best_model_path)
                logger.cprint(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                logger.cprint(
                    "========New Best Test Loss Updata: %1.5f Best Score: %1.5f========" % (test_rmse, res))
                logger.cprint(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        logger.cprint("Epoch : %d, training loss: %1.5f, testing rmse: %1.5f, score: %1.5f" % (
        epoch+1, train_epoch_loss, test_rmse, res))
        logger.cprint("------------------------------------------------------------")
    return