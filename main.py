
import os
import argparse
from train import *
from model_DCRAKAN import *
from visualization import *
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import torchprofile
from data_load import *
import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")
torch.manual_seed(1)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FD002', help='which dataset to run, FD001-FD004')
    parser.add_argument('--modes', type=str, default='test', help='train or test')
    parser.add_argument('--path', type=str,
                        default="/home/ps/code/zhouzhihao/DCRAKAN-TEST/saved_model/model/-FD002_new_best.pth",#-FD004_DCRAKAN.pth
                        help='model save/load path')
    parser.add_argument('--save_path', type=str,
                        default='/home/ps/code/zhouzhihao/DCRAKAN-TEST/saved_model/saved_model/log/',
                        help='log save path')
    parser.add_argument('--epoch', type=int, default=30, help='epoch to train')#
    parser.add_argument('--num_features', type=int, default=14, help='number of features')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')#
    parser.add_argument('--win', type=float, default=9, help='channels for 2DCNN')
    parser.add_argument('--att', type=float, default=3, help='attition for 2DCNN')
    parser.add_argument('--LR', type=float, default=0.0008, help='learning_rate')#
    parser.add_argument('--smooth_param', type=float, default=0.15, help='none or freq')
    parser.add_argument('--train_seq_len', type=int, default=30, help='train_seq_len')
    parser.add_argument('--test_seq_len', type=int, default=30, help='test_seq_len')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='length of patch')
    parser.add_argument('--decay_step', type=float, default=100, help='length of patch')
    parser.add_argument('--decay_ratio', type=float, default=0.5, help='length of patch')
    opt = parser.parse_args()
    print(opt)
    if opt.modes == "train":
        """
        Since the proposed method has not been published in any journal, the training program is not available at this time.
        """
        for i in range(10):
            Training(opt)
    elif opt.modes == "test":
        # TODO:  Testing the proposed methodology
        PATH = opt.path
        win = opt.win
        att = opt.att
        print(PATH)
        group_train, y_test, group_test, X_test = data_processing(opt.dataset,opt.smooth_param)
        test_dataset = SequenceDataset(mode='test',win = win, group = group_test,
                                       y_label=y_test, sequence_train=opt.train_seq_len, patch_size=opt.train_seq_len)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        model = DCNNKAN(in_channel=win,out_channel=64,att=att)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        model.load_state_dict(torch.load(PATH))

        if torch.cuda.is_available():
            model = model.to(device)

        model.eval()
        result=[]
        mse_loss=0
        print(model)
        model2 = model

        with torch.no_grad():
            test_epoch_loss = 0
            for X,y in test_loader:
                if torch.cuda.is_available():
                    X=X.cuda()
                    y=y.cuda()

                y_hat_recons = model.forward(X)

                y_hat_unscale = y_hat_recons[0]*125
                result.append(y_hat_unscale.item())
        # result.clip(max=125,min=0)
        y_test.index = y_test.index
        result = y_test.join(pd.DataFrame(result))

        result.clip(upper=125, inplace=True)
        result.clip(lower=0, inplace=True)
        ## Save the last moment of prediction for each engine, if required
        # result.to_csv(opt.dataset + '_result.csv')
        error = result.iloc[:,1]-result.iloc[:,0]
        res=0
        for value in error:
            if value < 0:
                res = res + np.exp(-value / 13) - 1
            else:
                res = res + np.exp(value / 10) - 1
        rmse =  np.sqrt(np.mean(error ** 2))
        print("testing score: %1.5f" % (res))
        print("testing rmse: %1.5f" % (rmse))

        result = result.sort_values('RUL', ascending=False)
        # TODO:  visualize the testing result
        visualize(result, rmse)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = count_parameters(model)
        print("Total parameters:", total_params)
        total_flops = torchprofile.profile_macs(model, (X,))
        print("Total FLOPs:", total_flops)

        def extract_features(model, dataloader):
            features = []
            labels = []
            def hook_fn(module, input, output):
                features.append(output.view(output.size(0), -1).cpu().detach().numpy())  # Flatten the output and detach
            # Register hook on conv2 layer
            hook = model.RULPositionalEncoder.register_forward_hook(hook_fn)
            model.eval()
            with torch.no_grad():
                for X, y in dataloader:
                    if torch.cuda.is_available():
                        X = X.cuda()
                        y = y.cuda()
                    _ = model(X)  # Forward pass
                    labels.append(y)
            # Remove the hook after extraction
            hook.remove()
            features = np.concatenate(features, axis=0)
            return features, labels
        def apply_tsne(features, labels, engine_num, n_components=2):
            tsne = TSNE(perplexity=30, n_components=n_components, random_state=42)
            reduced_features = tsne.fit_transform(features)

            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['mathtext.fontset'] = 'stix'  # 设置数学公式字体为stix
            plt.figure(figsize=(8, 6), dpi=300)
            scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='jet', s=50, alpha=0.7)
            plt.colorbar(scatter)
            plt.tick_params(axis='both', which='major', labelsize=20)
            plt.title(f"{engine_num}# Test engine in FD002", fontsize=22)
            plt.axis('tight')
            plt.box(on=True)
            plt.show()

        # TODO:  Visualization of degradation characteristics
        group_train, y_test, group_test, X_test = data_processing(opt.dataset, opt.smooth_param)
        engine_num=100
        test_dataset_specific = SequenceDataset(mode='test_all_specific', win = win, group=group_test, y_label=y_test,
                                                sequence_train=opt.train_seq_len,
                                                patch_size=opt.train_seq_len, engine_num=engine_num)
        test_loader_specific = DataLoader(test_dataset_specific, batch_size=1, shuffle=False, drop_last=False)

        features, labels = extract_features(model, test_loader_specific)
        labels = [tensor.tolist() for tensor in labels]
        apply_tsne(features, labels, engine_num=engine_num)

        # TODO:  Forecasting at all moments
        group_train, y_test, group_test, X_test = data_processing(opt.dataset, opt.smooth_param)
        test_dataset_all = SequenceDataset(mode='test_all', win = win, group=group_test, y_label=y_test,
        sequence_train=opt.train_seq_len,
        patch_size=opt.train_seq_len)
        test_loader_all = DataLoader(test_dataset_all, batch_size=1, shuffle=False)
        result_all = []
        y_all = []
        aaa = test_dataset_all.X[0]
        with torch.no_grad():
            for X, y in test_loader_all:
                if torch.cuda.is_available():
                    X = X.cuda()
                    y = y.cuda()

                y_hat_recons = model.forward(X)
                y_hat_unscale = y_hat_recons[0] * 125
                result_all.append(y_hat_unscale.item())
                y_all.append(y.item())

        all = np.vstack((y_all, result_all))
        print(all.shape)
        error_all = all[0, :] - all[1, :]
        rmse_all = np.sqrt(np.mean(error_all ** 2))
        print("Error_all:", rmse_all)

        lengths = X_test.apply(lambda x: len(x)).values
        rul_len = lengths - 29-opt.win
        rul_len[rul_len<1]=1
        engine_num = [i for i, count in enumerate(rul_len, 1) for _ in range(count)]
        engine_num = np.array(engine_num)
        all = np.vstack((engine_num, all))
        df_result_all = pd.DataFrame(all.T)
        ##Save all predictions,if needed
        # df_result_all.to_csv(opt.dataset+'_result_all.csv')


