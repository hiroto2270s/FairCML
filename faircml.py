#-- coding:UTF-8 --
'''
train RMSE with ml-1m     with compositional adversary 2020/3/27
'''

import torch
import torch.nn as nn
import os
import numpy as np
import math
import sys
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.autograd import Variable
import torch.nn.functional as F
from shutil import copyfile
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from torch.nn.init import xavier_normal, xavier_uniform
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import f1_score,roc_curve,auc
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
import torch.autograd as autograd
import argparse
import pdb
import pickle
from collections import defaultdict
import time
import copy
from tqdm import tqdm
from faircml_models import fairCML, GandACMLDiscriminator, GandACMLRegression
#from utils import CMLUserTripletGenerator, CMLItemTripletGenerator, calculate_fairness_CML, freeze_model, unfreeze_model, metrics
from evaluate import hr_ndcg
import numpy as np
from torch.utils.data import DataLoader
import torch
tqdm.monitor_interval = 0
from faircml_models import *
from evaluate import *
from utils import *
import warnings
warnings.filterwarnings("ignore")
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from collections import Counter


# ランダム性を固定
SEED = 50000
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # 複数GPUを使用する場合
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cpu")
#device = torch.device("cuda")

parser = argparse.ArgumentParser(description="change gpuid")
parser.add_argument("-g", "--gpu-id", help="choose which gpu to use", type=str, default=str(3))
parser.add_argument("-t", "--times", help="choose times", type=str, default='9')
# t 5 re2 fair age
# t 6
# t 7 com
# t 8 icml age
# t 9 icml com
parser.add_argument("-d", "--D_steps", help="num of train Disc times", type=int, default=2)
parser.add_argument("--batchsize", help="batch size", type=int, default=8192)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
D_steps = args.D_steps
run_id = 't' + args.times
choice = int(args.times)
dataset_base_path = './ml-1m'
data_dir = './ml-1m'
dataset_name = 'ml-1m'
path_save_model_base = './model/' + dataset_name + '/' + run_id
if (os.path.exists(path_save_model_base)):
    print('has model save path')
else:
    os.makedirs(path_save_model_base)

path_save_log_base = './Logs/' + dataset_name + '/' + run_id

if (os.path.exists(path_save_log_base)):
    print('has log save path')
else:
    os.makedirs(path_save_log_base)

user_num = 6040
item_num = 3706
cml_factor_num = 16
factor_num = 64
batch_size = args.batchsize
lamada = 0.01

def baseline_CML(user_early_stopping=False, patience=20):
    print("baseline_user_adversary")
    with open('./preprocessed/ml-1m_gcn_genderandage.pickle', 'rb') as f:
    #with open('./preprocessed/ml-1m_gcn_rebuttal.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, val_user_list, test_user_list = dataset['train_user_list'], dataset['val_user_list'], dataset[
            'test_user_list']
        train_pair = dataset['train_pair']
        test_pair = dataset['test_pair']
        val_pair = dataset['val_pair']
        user_labels = dataset['user_labels']
        item_label_means = dataset['item_label_means']
        item_label_ratios = dataset['item_label_ratios']
        train_neg_item_lists = dataset['train_neg_item_lists']
        pos_user_lists = dataset['pos_user_lists'] 
        neg_user_lists = dataset['neg_user_lists']
        pos_item_ratio_lists = dataset['pos_item_ratio_lists']
        neg_item_ratio_lists = dataset['neg_item_ratio_lists']
    
    print('Load complete')
    pdb.set_trace()

    #トレーニング、テスト、バリデーションのペアを取得し、データセットの長さを計算
    train_set_len, test_set_len, val_set_len = len(train_pair), len(test_pair), len(val_pair)
    print(train_set_len, test_set_len, val_set_len)
    #train_pairの初期化
    #train_datasetをバッチ単位で分割
    train_loader = DataLoader(train_pair, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_pair, batch_size=test_set_len, shuffle=False)
    val_loader = DataLoader(val_pair, batch_size=val_set_len, shuffle=False)
    #pdb.set_trace()
    #ログファイルの作成
    path_save_log_base = './Logs/' + dataset_name + '/t0'
    result_file = open(path_save_log_base + '/faircf_mf_rebutaal.txt', 'a')  # ('./log/results_gcmc.txt','w+')
    result_file.write('We write this code for 2 dim pseudo item labels\n')

    
    #推薦モデルの初期化：FairCF
    model = fairCML(user_size, item_size, cml_factor_num, margin=0.5, l2_reg=0.001).to(device) 
    #モデルオプティマイザをAdamに設定し、学習率を0.0001に設定（安定性重視）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    #学習率スケジューラーの初期化（より保守的な設定）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=8, verbose=True, min_lr=1e-7, threshold=1e-4)
    
    #性別識別器の初期化(D_1)　ユーザ側の識別器
    fairD_genderandage_user = GandACMLDiscriminator(user_size, user_size, user_size, cml_factor_num, user_labels, margin=0.0, l2_reg=0.001).to(device)
    optimizer_fairD_genderandage_user = torch.optim.Adam(fairD_genderandage_user.parameters(), lr=0.0005)
    #判別器用学習率スケジューラー
    scheduler_discriminator = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_fairD_genderandage_user, mode='min', factor=0.5, patience=15, verbose=True, min_lr=1e-6)
    
    #性別回帰モデルの初期化(D_2)　アイテム側の識別器
    regression_genderandage_item = GandACMLRegression(item_size, item_size, cml_factor_num, item_label_ratios, margin=0.5, l2_reg=0.001).to(device)
    optimizer_regression_genderandage_item = torch.optim.Adam(regression_genderandage_item.parameters(), lr=0.0005)
    #回帰器用学習率スケジューラー
    scheduler_regressor = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_regression_genderandage_item, mode='min', factor=0.5, patience=15, verbose=True, min_lr=1e-6)

    ################
    #   pretrain   #
    ################
    #モデル、性別識別器、性別回帰モデルをトレーニングモードに設定
    model.train()
    fairD_genderandage_user.train()
    regression_genderandage_item.train()
    pretrain_epoch = 4
    pretrain_fairness_epoch = 4

    # --- pretrain指標記録用リスト ---
    pretrain_losses = []
    pretrain_val_rmses = []
    pretrain_test_rmses = []
    pretrain_hr_50s = []
    pretrain_ndcg_50s = []
    pretrain_ndcg_alls = []
    #推薦モデルの事前トレーニング
    for epoch in range(pretrain_epoch):
        # u と pos_i を格納するリストを初期化
        all_users = []
        all_pos_items = []
        #トレーニングデータローダーからユーザ、アイテム、評価値を取得
        for idx, (u, pos_i, r) in enumerate(train_loader):
            model.train()
            u = torch.tensor(u).to(device)
            pos_i = torch.tensor(pos_i).to(device)
            r = torch.tensor(r).to(device)

            # --- 複数ネガティブサンプリングの実装 ---
            num_neg_samples = 2  # ユーザーごとにサンプリングするネガティブアイテムの数
            # 各ユーザーに対して num_neg_samples 個のネガティブアイテムをサンプリング
            # neg_i_lists はリストのリストになります: [[n1,n2,..], [n1,n2,..], ...]
            neg_i_lists = [random.sample(train_neg_item_lists[user_id.item()], k=num_neg_samples) for user_id in u]
            # リストをフラット化し、テンソルに変換
            neg_i = torch.tensor([item for sublist in neg_i_lists for item in sublist], device=device)
            # u と pos_i をネガティブサンプルの数に合わせて拡張
            # 例: u = [u1, u2], num_neg_samples=4 の場合
            # u_expanded は [u1, u1, u1, u1, u2, u2, u2, u2] となります
            u = u.unsqueeze(1).expand(-1, num_neg_samples).reshape(-1)
            pos_i = pos_i.unsqueeze(1).expand(-1, num_neg_samples).reshape(-1)
            r = r.unsqueeze(1).expand(-1, num_neg_samples).reshape(-1)

            for user_id in u:
                J = len(train_neg_item_lists[user_id.item()])
            U = num_neg_samples

            optimizer.zero_grad()
            # 拡張されたテンソルをモデルに渡す
            all_loss, triplet_loss, dis_loss, reg_loss = model(u, pos_i, neg_i, r, None, None, None, None, None, None, J, U, cal_batch_embedding=True, cal_batch_user_embedding=True, cal_batch_item_embedding=True, return_batch_all_embedding=False, Discrminator=None, Regression =None, training_mode=True)
            #task_lossを基に勾配を計算
            all_loss.backward()
            #勾配をクリッピングして、最適
            # 化器を使用してパラメータを更新
            optimizer.step()
            # u と pos_i をリストに追加
            all_users.extend(u.to(device).tolist())
            all_pos_items.extend(pos_i.to(device).tolist())
        print(f"\nEpoch {epoch+1}/{pretrain_epoch}, Loss: {all_loss.item()}")

        #-----pretrainでの推薦精度(rmse, hitrate, ndcg)の計算--------#
        with torch.no_grad():
            model.eval()
            val_mse_list = []
            test_mse_list = []
            best_val_rmse = float('inf')
            best_test_rmse = float('inf')
            for idx, (u, pos_i, r) in enumerate(val_loader):
                u, pos_i, r = torch.tensor(u).to(device), torch.tensor(pos_i).to(device), torch.tensor(r).to(device)
                val_mse = model.predict(u, pos_i, r, return_e=False, return_user_e=False, return_item_e=False, calcurate_fairness_mode=False)
                val_mse_list.append(val_mse.item())
            
            val_rmse = torch.sqrt(torch.tensor(val_mse_list).mean()).item()

            #学習率スケジューラーを更新（検証損失に基づく）
            scheduler.step(val_rmse)

                #テストデータにおけるRMSEを計算
            for idx, (u, pos_i, r) in enumerate(test_loader):
                u, pos_i, r = torch.tensor(u).to(device), torch.tensor(pos_i).to(device), torch.tensor(r).to(device)
                test_mse  = model.predict(u, pos_i, r, return_e=False, return_user_e=False, return_item_e=False, calcurate_fairness_mode=False)
                test_mse_list.append(test_mse.item())

            test_rmse = torch.sqrt(torch.tensor(test_mse_list).mean()).item()

            # --- HR/NDCGの計算を追加 ---
            all_users = torch.tensor(range(user_num), dtype=torch.long).to(device)
            all_items = torch.tensor(range(item_num), dtype=torch.long).to(device)

            #-----------全ユーザ、全アイテムのペアを生成し、予測スコアを計算----------#
            all_predictions = model.predict(all_users, all_items, None, return_e=False, return_user_e=False, return_item_e=False, calcurate_fairness_mode=True)
            all_predictions = torch.tensor(all_predictions)
            #all_predictions = torch.cat(all_scores, dim=0)
            #fair_K_lastfm_bias__()を使用して、公平性スコア(fairness@K)を計算し取得
            #fairness_all_before, fairness_K50_before = fair_K_lastfm_bias__CML(all_predictions, userids, neg_itemids, user_labels)
            fairness_all_before, fairness_K50_before = calculate_fairness_CML(all_predictions, user_labels, train_user_list, top_k=50)
            print(fairness_all_before)
            print(fairness_K50_before)
            
            # 全ユーザー・アイテムのスコアを計算
            #all_scores = model.predict(all_users, all_items, ratings=None, calcurate_fairness_mode=True)
            all_scores_np = all_predictions.cpu().detach().numpy()

            for u in range(user_num):
                if train_user_list[u]:
                    train_items = list(train_user_list[u].keys())
                    all_scores_np[u, train_items] = -np.inf
            """
            # 検証セットのHR/NDCG
            val_ground_truth = [list(val_user_list[u].keys()) for u in range(len(val_user_list))]
            val_user_ids = [u for u, item_dict in enumerate(val_user_list) if item_dict]
            val_hr, val_ndcg = hr_ndcg(all_scores_np[val_user_ids], val_ground_truth, k=50)
            """
            # テストセットのHR/NDCG
            test_ground_truth = [list(test_user_list[u].keys()) for u in range(len(test_user_list))]
            test_user_ids = [u for u, item_dict in enumerate(test_user_list) if item_dict]
            test_hr_k50, test_ndcg_k50 = hr_ndcg(all_scores_np[test_user_ids], test_ground_truth, k=50)
            test_hr_all, test_ndcg_all = hr_ndcg(all_scores_np[test_user_ids], test_ground_truth, k=item_num)


        print(f'Epoch {epoch+1}/{pretrain_epoch}, Val RMSE: {val_rmse:.4f}, Test RMSE: {test_rmse:.4f}')
        print(f' HR@50: {test_hr_k50:.4f}, Test NDCG@50: {test_ndcg_k50:.4f}')
        print(f' HR@all: {test_hr_all:.4f}, Test NDCG@all: {test_ndcg_all:.4f}')

        # --- pretrain指標を記録 ---
        pretrain_losses.append(all_loss.item())
        pretrain_val_rmses.append(val_rmse)
        pretrain_test_rmses.append(test_rmse)
        pretrain_hr_50s.append(test_hr_k50)
        pretrain_ndcg_50s.append(test_ndcg_k50)
        pretrain_ndcg_alls.append(test_ndcg_all)

    # --- pretrain指標の可視化 ---
    plt.figure(figsize=(15, 8))
    epochs = np.arange(1, pretrain_epoch + 1)
    plt.subplot(2, 3, 1)
    plt.plot(epochs, pretrain_losses, marker='o')
    plt.title('Pretrain Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(2, 3, 2)
    plt.plot(epochs, pretrain_val_rmses, marker='o', label='Val RMSE')
    plt.plot(epochs, pretrain_test_rmses, marker='o', label='Test RMSE')
    plt.title('RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.subplot(2, 3, 3)
    plt.plot(epochs, pretrain_hr_50s, marker='o')
    plt.title('HR@50')
    plt.xlabel('Epoch')
    plt.ylabel('HR@50')
    plt.subplot(2, 3, 4)
    plt.plot(epochs, pretrain_ndcg_50s, marker='o')
    plt.title('NDCG@50')
    plt.xlabel('Epoch')
    plt.ylabel('NDCG@50')
    plt.subplot(2, 3, 5)
    plt.plot(epochs, pretrain_ndcg_alls, marker='o')
    plt.title('NDCG@all')
    plt.xlabel('Epoch')
    plt.ylabel('NDCG@all')
    plt.tight_layout()
    plt.show()
    pdb.set_trace()



    #for idx, (u, pos_i, neg_i, r) in enumerate(train_loader):
       # u = torch.tensor(u).to(device).numpy()
       #pos_i = torch.tensor(pos_i).to(device).numpy()
    
    
    # --- 改善案：トリプレット生成器をループの外で初期化 ---
    # 全ユーザーIDと全アイテムIDのリストを作成
    all_user_ids = list(range(user_size))
    all_item_ids = list(range(item_num))   
    user_triplet_generator = CMLUserTripletGenerator(all_user_ids, pos_user_lists, neg_user_lists, sample_size=1)
    item_triplet_generator = CMLItemTripletGenerator(all_item_ids, pos_item_ratio_lists, neg_item_ratio_lists, sample_size=1)
    pdb.set_trace()

    #性別識別器(D_1)の事前トレーニング
    fairD_genderandage_user.train()
    for epoch in range(pretrain_fairness_epoch):
        user_triplet_pairs = user_triplet_generator.generate_triplet_pairs(batch_size=batch_size)
        user_ids = torch.tensor([pair[0] for pair in user_triplet_pairs], dtype=torch.long).to(device)
        pos_user_ids = torch.tensor([pair[1] for pair in user_triplet_pairs], dtype=torch.long).to(device)  
        neg_user_ids = torch.tensor([pair[2] for pair in user_triplet_pairs], dtype=torch.long).to(device)
        #pdb.set_trace()
        
        user_emb, pos_user_emb, neg_user_emb = model(None, None, None, None, user_ids, pos_user_ids, neg_user_ids, None, None, None, None, None, cal_batch_embedding=False, cal_batch_user_embedding=True, cal_batch_item_embedding=False, return_batch_all_embedding=False, Discrminator=fairD_genderandage_user, Regression = regression_genderandage_item, training_mode=False)
        #pdb.set_trace()
        #性別識別器の勾配を初期化
        fairD_genderandage_user.zero_grad()
        #性別識別器の損失を計算
        l_penalty_user = fairD_genderandage_user(user_emb, pos_user_emb, neg_user_emb, user_ids, True)
        #損失を基に勾配を計算
        l_penalty_user.backward()
        #勾配をクリッピングして、最適化器を使用してパラメータを更新
        optimizer_fairD_genderandage_user.step()
        print(f"Epoch {epoch+1}/{20}, Loss: {l_penalty_user.item()}")
        print(f"Total user triplet pairs: {len(user_triplet_pairs)}")
    #トレーニング後の性別識別器のパラメータを保存
    torch.save(fairD_genderandage_user.state_dict(), './preprocessed/DISC_pretrain_genderandage.pt')
    pdb.set_trace()


    #性別回帰モデル(D_2)の事前トレーニング
    regression_genderandage_item.train()
    for epoch in range(pretrain_fairness_epoch):
        item_triplet_pairs = item_triplet_generator.generate_triplet_pairs(batch_size=batch_size)
        item_ids = torch.tensor([pair[0] for pair in item_triplet_pairs], dtype=torch.long).to(device)
        pos_item_ids = torch.tensor([pair[1] for pair in item_triplet_pairs], dtype=torch.long).to(device)
        neg_item_ids = torch.tensor([pair[2] for pair in item_triplet_pairs], dtype=torch.long).to(device)

        item_emb, pos_item_emb, neg_item_emb = model(None, None, None, None, None, None, None, item_ids, pos_item_ids, neg_item_ids, None, None, cal_batch_embedding=False, cal_batch_user_embedding=False, cal_batch_item_embedding=True, return_batch_all_embedding=False, Discrminator=fairD_genderandage_user, Regression = regression_genderandage_item, training_mode=False)
        
        #性別回帰モデルの勾配を初期化
        regression_genderandage_item.zero_grad()
        #性別回帰モデルの損失を計算
        l_penalty_item = regression_genderandage_item(item_emb, pos_item_emb, neg_item_emb, True)
        #損失を基に勾配を計算
        l_penalty_item.backward()
        #勾配をクリッピングして、最適化器を使用してパラメータを更新
        optimizer_regression_genderandage_item.step()
        print(f"Epoch {epoch+1}/{20}, Loss: {l_penalty_item.item()}")
        print(f"Total item triplet pairs: {len(item_triplet_pairs)}")
    #トレーニング後の性別回帰モデルのパラメータを保存
    torch.save(regression_genderandage_item.state_dict(), './preprocessed/REG_pretrain_genderandage.pt')


    #事前トレーニング後のモデルのパラメータをロードしてそれぞれのモデルに設定
    model.load_state_dict(torch.load("./preprocessed/recommend_pretrain_genderandage.pt"))
    fairD_genderandage_user.load_state_dict(torch.load("./preprocessed/DISC_pretrain_genderandage.pt"))
    regression_genderandage_item.load_state_dict(torch.load("./preprocessed/REG_pretrain_genderandage.pt"))
    

    pdb.set_trace()
    ################
    #  train part  #
    ################
    #トレーニング
    best_epoch = -1
    best_metrics = {}
    best_val_rmse = float('inf')
    epochs_no_improve = 0


    # --- ★修正点: トリプレット生成器をループの外で初期化 ---
    all_user_ids = list(range(user_size))
    all_item_ids = list(range(item_num))
    user_triplet_generator = CMLUserTripletGenerator(all_user_ids, pos_user_lists, neg_user_lists, sample_size=1)
    item_triplet_generator = CMLItemTripletGenerator(all_item_ids, pos_item_ratio_lists, neg_item_ratio_lists, sample_size=1)


    val_rmse_list = []
    test_rmse_list = []
    AUC_list = []
    fairness_50_list = []
    fairness_all_list = []
    epoch_metrics = []

    # --- 正解リストを事前に準備 ---
    val_ground_truth = [list(u.keys()) for u in val_user_list if u]
    val_user_ids = [u for u, item_dict in enumerate(val_user_list) if item_dict]
    test_ground_truth = [list(u.keys()) for u in test_user_list if u]
    test_user_ids = [u for u, item_dict in enumerate(test_user_list) if item_dict]
    
    for epoch in range(200):
        start_time = time.time()
        train_loss_sum = []
        train_loss_sum2 = []
        train_loss_sum3 = []
        train_loss_sum4 = []
        train_loss_sum5 = []
        train_loss_sum6 = []
        AUC_, acc_, f1_ = [], [], []
        for idx, (u, pos_i, r) in enumerate(train_loader):
            #推薦モデルをトレーニングモードに設定
            model.train()
            #性別識別器と性別回帰モデルのパラメータを凍結、更新しないようにする
            freeze_model(fairD_genderandage_user)
            freeze_model(regression_genderandage_item)
            #性別識別器と性別回帰モデルを推論モードに設定
            fairD_genderandage_user.eval()
            regression_genderandage_item.eval()
            u = torch.tensor(u).to(device)
            pos_i = torch.tensor(pos_i).to(device)
            r = torch.tensor(r).to(device)


            # --- 複数ネガティブサンプリングの実装 ---
            num_neg_samples = 2  
            neg_i_lists = [random.sample(train_neg_item_lists[user_id.item()], k=num_neg_samples) for user_id in u]
            neg_i = torch.tensor([item for sublist in neg_i_lists for item in sublist], device=device)
            u = u.unsqueeze(1).expand(-1, num_neg_samples).reshape(-1)
            pos_i = pos_i.unsqueeze(1).expand(-1, num_neg_samples).reshape(-1)
            r = r.unsqueeze(1).expand(-1, num_neg_samples).reshape(-1)

            for user_id in u:
                J = len(train_neg_item_lists[user_id.item()])
            U = num_neg_samples

            # --- ★修正点: ループ内でのジェネレータ再初期化を削除し、メソッド呼び出しのみにする ---
            user_triplet_pairs = user_triplet_generator.generate_triplet_pairs(batch_size=batch_size)
            user_ids = torch.tensor([pair[0] for pair in user_triplet_pairs], dtype=torch.long).to(device)
            pos_user_ids = torch.tensor([pair[1] for pair in user_triplet_pairs], dtype=torch.long).to(device)  
            neg_user_ids = torch.tensor([pair[2] for pair in user_triplet_pairs], dtype=torch.long).to(device)
        
            item_triplet_pairs = item_triplet_generator.generate_triplet_pairs(batch_size=batch_size)
            item_ids = torch.tensor([pair[0] for pair in item_triplet_pairs], dtype=torch.long).to(device)
            pos_item_ids = torch.tensor([pair[1] for pair in item_triplet_pairs], dtype=torch.long).to(device)
            neg_item_ids = torch.tensor([pair[2] for pair in item_triplet_pairs], dtype=torch.long).to(device)
            
            # --- ★修正点: modelの呼び出しを1回にまとめ、損失計算をループ内で行う ---
            # 1. Generatorの学習
            model.zero_grad()
            user_emb, pos_user_emb, neg_user_emb, item_emb, pos_item_emb, neg_item_emb = model(None, None, None, None, user_ids, pos_user_ids, neg_user_ids, item_ids, pos_item_ids, neg_item_ids, None, None, cal_batch_embedding=False, cal_batch_user_embedding=True, cal_batch_item_embedding=True, return_batch_all_embedding=True, Discrminator=None, Regression=None, training_mode=False)
            all_loss, triplet_loss, dis_loss, reg_loss = model(u, pos_i, neg_i, r, user_ids, pos_user_ids, neg_user_ids, item_ids, pos_item_ids, neg_item_ids, J, U, cal_batch_embedding=True, cal_batch_user_embedding=True, cal_batch_item_embedding=True, return_batch_all_embedding=False, Discrminator=fairD_genderandage_user, Regression=regression_genderandage_item, training_mode=False)
            all_loss.backward()
            optimizer.step()

            # 2. DiscriminatorとRegressorの学習
            # 埋め込みを計算グラフから切り離す
            cached_user_emb = torch.tensor(user_emb).detach().requires_grad_()
            cached_pos_user_emb = torch.tensor(pos_user_emb).detach().requires_grad_()
            cached_neg_user_emb = torch.tensor(neg_user_emb).detach().requires_grad_()
            cached_user_ids = user_ids.detach()
            cached_item_emb = torch.tensor(item_emb).detach().requires_grad_()
            cached_pos_item_emb = torch.tensor(pos_item_emb).detach().requires_grad_()
            cached_neg_item_emb = torch.tensor(neg_item_emb).detach().requires_grad_()

            #各損失をリストに記録
            train_loss_sum.append(all_loss.item())
            train_loss_sum2.append(triplet_loss.item())
            train_loss_sum3.append(dis_loss.item())
            train_loss_sum4.append(reg_loss.item())
            
            #性別識別器と性別回帰モデルのパラメータを更新可能にする
            unfreeze_model(fairD_genderandage_user)
            unfreeze_model(regression_genderandage_item)
            #pdb.set_trace()

            #性別識別器のトレーニング
            for _ in range(1):
                #性別識別器をトレーニングモードに設定
                fairD_genderandage_user.train()

                #性別識別器の勾配を初期化
                fairD_genderandage_user.zero_grad()
                #性別識別器の損失を計算
                l_penalty_user = fairD_genderandage_user(cached_user_emb, cached_pos_user_emb, cached_neg_user_emb, cached_user_ids, True)
                #損失を基に勾配を計算
                l_penalty_user.backward()

                #勾配をクリッピングして、最適化器を使用してパラメータを更新
                optimizer_fairD_genderandage_user.step()
                #最初のイテレーションのみ損失をリストに記録
                if _ == 0:
                    train_loss_sum5.append(l_penalty_user.item())
            #pdb.set_trace()
            #print("Finished training fairD_genderandage_user")
                    
            #性別回帰モデルのトレーニング
            for _ in range(1):
                #性別回帰モデルをトレーニングモードに設定
                regression_genderandage_item.train()
                #性別回帰モデルの勾配を初期化
                regression_genderandage_item.zero_grad()
                #性別回帰モデルの損失を計算
                l_penalty_item = regression_genderandage_item(cached_item_emb, cached_pos_item_emb, cached_neg_item_emb, True)
                #損失を基に勾配を計算
                l_penalty_item.backward()
                #勾配をクリッピングして、最適化器を使用してパラメータを更新
                optimizer_regression_genderandage_item.step()
                #最初のイテレーションのみ損失をリストに記録
                if _ == 0:
                    train_loss_sum6.append(l_penalty_item.item())
            #pdb.set_trace()
            #print("Finished training regression_genderandage_item")

            #勾配計算を無効化し、推論モードに切り替える
            with torch.no_grad():
                # 識別器の性能を評価するために、再度 model() を呼び出すのは非効率です。
                # 既に計算済みの埋込み（cached_user_embなど）を再利用します。
                fairD_genderandage_user.eval() # 評価モードに設定
                
                dataset_len = cached_user_emb.size(0)
                
                # y_hatには性別識別器の予測確率、yには実際のラベルを格納
                # predictメソッドには計算済みの埋め込みを渡します
                y_hat, y = fairD_genderandage_user.predict(cached_user_emb, cached_pos_user_emb, cached_neg_user_emb, cached_user_ids, return_loss=False, return_preds=False)
                
                # y_hatがLogSoftmaxの出力であると仮定
                y_hat_prob = torch.exp(y_hat)
                preds = torch.argmax(y_hat_prob, dim=1)
                
                # 正解数をカウント
                y = torch.tensor(y, device=device) # yをテンソルに変換
                correct = preds.eq(y.view_as(preds)).sum().item()
                
                # 評価指標を計算するためにリストを準備
                # バッチサイズが1なので、リストに直接テンソルを追加
                preds_list4 = [preds]
                probs_list4 = [y_hat]
                labels_list4 = [y]
                
                # 性別識別器の評価指標を計算
                AUC, acc, f1, f1_macro = metrics(preds_list4, labels_list4, probs_list4, dataset_len, correct)
                
                # 性別識別器の評価指標をリストに記録
                AUC_.append(AUC)
                acc_.append(acc)
                
                fairD_genderandage_user.train() # 訓練モードに戻す

        #pdb.set_trace()


        ###############
        #  eval part  #
        ###############
        path_save_model_b = './model/' + dataset_name + '/t0'
        PATH_model = path_save_model_b + '/2adv_' + str(epoch) + '_genderandage.pt'  # -15
        
        #モデルのパラメータを保存するためのディレクトリが存在しない場合は作成
        if not os.path.exists(path_save_model_b):
            os.makedirs(path_save_model_b)
        # Save the model state    
        torch.save(model.state_dict(), PATH_model)

        #推薦モデルを推論モードに設定
        model.eval()
        #性別識別器を推論モードに設定
        fairD_genderandage_user.eval()
        # training finish, we want to cum fairness and rmse


        #勾配計算を無効化し、推論モードに切り替える
        with torch.no_grad():
            model.eval()
            val_mse_list = []
            test_mse_list = []
            # 埋め込みを格納するリストを初期化
            all_users = torch.arange(user_size).to(device)
            all_items = torch.arange(item_size).to(device)

            #----------ユーザをバッチ単位で 生成し、予測スコアを計算----------#
            """
            eval_loader = DataLoader(all_users, batch_size=256, shuffle=False)
            all_score_list = []
            for user_batch_ids in eval_loader:
                user_batch_ids = user_batch_ids.to(device)
                scores = model.predict(user_batch_ids, all_items, None, return_e=False, return_user_e=False, return_item_e=False, calcurate_fairness_mode=True)
                all_predictions.append(scores)
            """

            #-----------全ユーザ、全アイテムのペアを生成し、予測スコアを計算----------#
            all_predictions = model.predict(all_users, all_items, None, return_e=False, return_user_e=False, return_item_e=False, calcurate_fairness_mode=True)
            all_predictions = torch.tensor(all_predictions, device=device)
            #all_predictions = torch.cat(all_predictions, dim=0)
            #fair_K_lastfm_bias__()を使用して、公平性スコア(fairness@K)を計算し取得
            #fairness_all_before, fairness_K50_before = fair_K_lastfm_bias__CML(all_predictions, userids, neg_itemids, user_labels)
            fairness_all, fairness_K50 = calculate_fairness_CML(all_predictions, user_labels, train_user_list, top_k=50)

            # --- HR/NDCG評価のための推薦リスト生成 ---
            # トレーニングデータをマスク
            for u in range(user_num):
                if train_user_list[u]:
                    train_items = list(train_user_list[u].keys())
                    all_predictions[u, train_items] = -np.inf
            
            # スコアに基づいてランキング
            _, recommend_list = torch.topk(all_predictions, item_num)
            recommend_list = recommend_list.cpu().numpy()


            # --- 検証データでのHR/NDCG計算 ---
            #val_hr_50, val_ndcg_50 = hr_ndcg(all_predictions[val_user_ids], val_ground_truth, k=50)
            #val_hr_all, val_ndcg_all = hr_ndcg(all_predictions[val_user_ids], val_ground_truth, k=item_num)
            val_hr_50, val_ndcg_50 = hr_ndcg(recommend_list, val_ground_truth, k=50)
            val_hr_all, val_ndcg_all = hr_ndcg(recommend_list, val_ground_truth, k=item_num)
  
            #if epoch == 199:
            #    visualize_embeddings_kmeans_gender(user_e, gender_data, n_clusters=2, title="User Embeddings After Training")
            #検証データにおけるRMSEを計算
            for idx, (u, pos_i, r) in enumerate(val_loader):
                u, pos_i, r = torch.tensor(u).to(device), torch.tensor(pos_i).to(device), torch.tensor(r).to(device)
                val_mse = model.predict(u, pos_i, r,  return_e=False, return_user_e=False, return_item_e=False, calcurate_fairness_mode=False)
                val_mse_list.append(val_mse.item())
                
            val_rmse = torch.sqrt(torch.tensor(val_mse_list).mean()).item()

            #学習率スケジューラーを更新（検証損失に基づく）
            scheduler.step(val_ndcg_all)
            scheduler_discriminator.step(fairness_all)
            scheduler_regressor.step(fairness_all)

            #テストデータにおけるRMSEを計算
            for idx, (u, pos_i, r) in enumerate(test_loader):
                u, pos_i, r = torch.tensor(u).to(device), torch.tensor(pos_i).to(device), torch.tensor(r).to(device)
                test_mse  = model.predict(u, pos_i, r,  return_e=False, return_user_e=False, return_item_e=False, calcurate_fairness_mode=False)
                test_mse_list.append(test_mse.item())
            
            test_rmse = torch.sqrt(torch.tensor(test_mse_list).mean()).item()
            
            # --- テストデータでのHR/NDCG計算 ---
            #test_hr_50, test_ndcg_50 = hr_ndcg(all_predictions[test_user_ids], test_ground_truth, k=50)
            #test_hr_all, test_ndcg_all = hr_ndcg(all_predictions[test_user_ids], test_ground_truth, k=item_num)
            test_hr_50, test_ndcg_50 = hr_ndcg(recommend_list, test_ground_truth, k=50)
            test_hr_all, test_ndcg_all = hr_ndcg(recommend_list, test_ground_truth, k=item_num)
 
        
        
        #各損失と評価指標の平均値を計算
        train_loss = round(np.mean(train_loss_sum[:-1]), 4)
        train_loss2 = round(np.mean(train_loss_sum2[:-1]), 4)
        train_loss3 = round(np.mean(train_loss_sum3[:-1]), 4)
        train_loss4 = round(np.mean(train_loss_sum4[:-1]), 4)
        train_loss5 = round(np.mean(train_loss_sum5[:-1]), 4)
        train_loss6 = round(np.mean(train_loss_sum6[:-1]), 4)
        AUC_mean = round(np.mean(AUC_[:-1]), 4)
        acc_mean = round(np.mean(acc_[:-1]), 4)
        elapsed_time = time.time() - start_time




        #トレーニング結果のログを出力
        str_print_train = "epoch:" + str(epoch) + ' time:' + str(round(elapsed_time, 1)) + '\t all loss:' + str(
            train_loss) + "\t triplet loss  " + str(train_loss2) + "\t user dis loss:" + str(
            train_loss3) + '\t user dis loss2:' + str(train_loss5) + "\t reg loss1:" + str(
            train_loss4) + "\t reg loss2:" + str(train_loss6) + "\t acc:" + str(acc_mean) + "\t AUC:" + str(AUC_mean)

        print('\n train_time----', elapsed_time)
        print(str_print_train)
        result_file.write(str_print_train)
        result_file.write('\n')

        #検証・テスト結果のログを出力
        str_print_eval = 'val rmse' + str(round(val_rmse, 4)) + '\t test rmse:' + str(
            round(test_rmse, 4)) + '\t fair@50:' + str(round(fairness_K50, 4)) + '\t fair@all:' + str(
            round(fairness_all, 4))
        print(str_print_eval)
        result_file.write(str_print_eval)
        result_file.write('\n')

        # --- HR/NDCGの結果を出力 ---
        str_print_hr_ndcg = (f"VAL HR@50: {val_hr_50:.4f}, NDCG@50: {val_ndcg_50:.4f} | "
                             f"TEST HR@50: {test_hr_50:.4f}, NDCG@50: {test_ndcg_50:.4f}")
        print(str_print_hr_ndcg)
        result_file.write(str_print_hr_ndcg + '\n')
        str_print_hr_ndcg_all = (f"VAL HR@all: {val_hr_all:.4f}, NDCG@all: {val_ndcg_all:.4f} | "
                                 f"TEST HR@all: {test_hr_all:.4f}, NDCG@all: {test_ndcg_all:.4f}")
        print(str_print_hr_ndcg_all)
        result_file.write(str_print_hr_ndcg_all + '\n')


        result_file.write('\n')
        result_file.flush()

        test_rmse_list.append(test_rmse)
        AUC_list.append(AUC_mean)
        fairness_50_list.append(fairness_K50)
        fairness_all_list.append(fairness_all)

        current_metrics = {
            "all_loss": train_loss,
            "triplet_loss": train_loss2,
            "user_dis_loss": train_loss3,
            "user_dis_loss2": train_loss5,
            "reg_loss1": train_loss4,
            "reg_loss2": train_loss6,
            "epoch": epoch,
            "val_rmse": val_rmse,
            "test_rmse": test_rmse,
            "AUC_mean": AUC_mean,
            "fairness_K50": fairness_K50,
            "fairness_all": fairness_all,
            "val_hr_50": val_hr_50,
            "val_ndcg_50": val_ndcg_50,
            "test_hr_50": test_hr_50,
            "test_ndcg_50": test_ndcg_50,
            "val_hr_all": val_hr_all,
            "val_ndcg_all": val_ndcg_all,
            "test_hr_all": test_hr_all,
            "test_ndcg_all": test_ndcg_all
        }
        epoch_metrics.append(current_metrics)

        # Early stopping logic
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_metrics = current_metrics
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), path_save_model_base + '/best_model.pt')
        else:
            epochs_no_improve += 1
        
        if user_early_stopping and epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break


    # ログファイルに書き込み (ループの外に移動)
    if best_metrics:
        result_file.write(f"Best epoch: {best_epoch}\n")
        result_file.write(f"Validation RMSE: {best_metrics['val_rmse']}\n")
        result_file.write(f"Test RMSE: {best_metrics['test_rmse']}\n")
        result_file.write(f"AUC: {best_metrics['AUC_mean']}\n")
        result_file.write(f"Fairness@50: {best_metrics['fairness_K50']}\n")
        result_file.write(f"Fairness@all: {best_metrics['fairness_all']}\n")
        result_file.flush()
        
    # 各評価値をリストに変換
    epochs = [data["epoch"] for data in epoch_metrics]
    all_loss = [data["all_loss"] for data in epoch_metrics]
    triplet_loss = [data["triplet_loss"] for data in epoch_metrics]
    user_dis_loss = [data["user_dis_loss"] for data in epoch_metrics]
    user_dis_loss2 = [data["user_dis_loss2"] for data in epoch_metrics]
    reg_loss1 = [data["reg_loss1"] for data in epoch_metrics]
    reg_loss2 = [data["reg_loss2"] for data in epoch_metrics]
    val_rmse = [data["val_rmse"] for data in epoch_metrics]
    test_rmse = [data["test_rmse"] for data in epoch_metrics]
    fairness_K50 = [data["fairness_K50"] for data in epoch_metrics]
    fairness_all = [data["fairness_all"] for data in epoch_metrics]
    AUC_mean = [data["AUC_mean"] for data in epoch_metrics]
    val_hr_50 = [data["val_hr_50"] for data in epoch_metrics]
    val_ndcg_50 = [data["val_ndcg_50"] for data in epoch_metrics]
    test_hr_50 = [data["test_hr_50"] for data in epoch_metrics]
    test_ndcg_50 = [data["test_ndcg_50"] for data in epoch_metrics]
    val_hr_all = [data["val_hr_all"] for data in epoch_metrics]
    val_ndcg_all = [data["val_ndcg_all"] for data in epoch_metrics]
    test_hr_all = [data["test_hr_all"] for data in epoch_metrics]
    test_ndcg_all = [data["test_ndcg_all"] for data in epoch_metrics]

    # 可視化
    plt.figure(figsize=(18, 12))

    # --- Loss Plots (Combined) ---
    plt.subplot(3, 4, 1)
    plt.plot(epochs, all_loss, label="All Loss", marker=".")
    plt.plot(epochs, triplet_loss, label="Triplet Loss", marker=".")
    plt.plot(epochs, user_dis_loss, label="User Dis Loss 1", marker=".")
    plt.plot(epochs, reg_loss1, label="Reg Loss 1", marker=".")
    plt.title("All and Component Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # --- Other Losses ---
    plt.subplot(3, 4, 2)
    plt.plot(epochs, triplet_loss, label="Triplet Loss", marker="o")
    plt.title("Triplet Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(3, 4, 3)
    plt.plot(epochs, user_dis_loss, label="User Dis Loss", marker="o")
    plt.title("User Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(3, 4, 4)
    plt.plot(epochs, reg_loss1, label="Item Reg Loss", marker="o")
    plt.title("Item Regression Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # --- RMSE Plot ---
    plt.subplot(3, 4, 3)
    plt.plot(epochs, val_rmse, label="Validation RMSE", marker="o")
    plt.plot(epochs, test_rmse, label="Test RMSE", marker="o")
    plt.title("Validation and Test RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()

    # --- Fairness Plots ---
    plt.subplot(3, 4, 4)
    plt.plot(epochs, fairness_K50, label="Fairness@50", marker="o")
    plt.plot(epochs, fairness_all, label="Fairness@all", marker="o")
    plt.title("Fairness Scores")
    plt.xlabel("Epoch")
    plt.ylabel("Fairness Score")
    plt.legend()

    # --- AUC Plot ---
    plt.subplot(3, 4, 5)
    plt.plot(epochs, AUC_mean, label="AUC Mean", marker="o")
    plt.title("Discriminator AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()

    # --- HR@50 Plot ---
    plt.subplot(3, 4, 6)
    plt.plot(epochs, val_hr_50, label="Val HR@50", marker="o")
    plt.plot(epochs, test_hr_50, label="Test HR@50", marker="o")
    plt.title("Hit Rate @50")
    plt.xlabel("Epoch")
    plt.ylabel("HR@50")
    plt.legend()

    # --- NDCG@50 Plot ---
    plt.subplot(3, 4, 7)
    plt.plot(epochs, val_ndcg_50, label="Val NDCG@50", marker="o")
    plt.plot(epochs, test_ndcg_50, label="Test NDCG@50", marker="o")
    plt.title("NDCG @50")
    plt.xlabel("Epoch")
    plt.ylabel("NDCG@50")
    plt.legend()

    # --- NDCG@all Plot ---
    plt.subplot(3, 4, 9)
    plt.plot(epochs, val_ndcg_all, label="Val NDCG@all", marker="o")
    plt.plot(epochs, test_ndcg_all, label="Test NDCG@all", marker="o")
    plt.title("NDCG @all")
    plt.xlabel("Epoch")
    plt.ylabel("NDCG@all")
    plt.legend()


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 引数パーサーの設定
    parser = argparse.ArgumentParser(description="FairCML Training")
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping")
    
    args = parser.parse_args()
    
    # baseline_CML関数を実行
    baseline_CML(user_early_stopping=args.early_stopping, patience=args.patience)
