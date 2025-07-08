import torch
import torch.nn as nn
import os
import numpy as np
import math
import sys
import torchvision.transforms as transforms
import random
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import xavier_normal, xavier_uniform, xavier_normal_
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import f1_score,roc_curve,auc
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyClassifier
import torch.autograd as autograd
import matplotlib.pyplot as plt
import pdb

lamada = 0.001
device = torch.device("cpu")


class fairCML(nn.Module):
    def __init__(self, user_size, item_size, factor_num, margin=0.5, l2_reg=0.001):
        super(fairCML, self).__init__()
        self.l2_reg = l2_reg
        self.mse = nn.MSELoss()
        self.user_embeddings = nn.Embedding(user_size, factor_num, max_norm=1.0)
        self.item_embeddings = nn.Embedding(item_size, factor_num, max_norm=1.0)
        self.user_filter = nn.Sequential(
            nn.Linear(factor_num, factor_num * 4, bias=True),
            nn.BatchNorm1d(factor_num * 4),  # Batch normalization for better training stability
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(factor_num * 4, factor_num, bias=True)
        )

            #nn.Linear(factor_num * 2, factor_num * 2, bias=True),
            #nn.BatchNorm1d(factor_num * 2),  # Batch normalization for better training stability
            #nn.LeakyReLU(0.1),
            #nn.Dropout(0.4),

            #nn.Linear(factor_num * 2, factor_num, bias=True)
        #)
        self.margin = margin
        self.init_weights()

    def init_weights(self):
        xavier_normal_(self.user_embeddings.weight.data)
        xavier_normal_(self.item_embeddings.weight.data)
        for layer in self.user_filter:
            if isinstance(layer, nn.Linear):
                xavier_normal_(layer.weight.data)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.0)

    def forward(self, user0, pos_itemi0, neg_itemi0, ratings, user_ids, pos_user_ids, neg_user_ids, item_ids, pos_item_ids, neg_item_ids, J, U, cal_batch_embedding=False, cal_batch_user_embedding=False, cal_batch_item_embedding=False, return_batch_all_embedding=False,  Discrminator=None, Regression=None, training_mode=False): 
        if cal_batch_embedding == True:
            user_e = self.user_embeddings(user0)
            item_pos_e = self.item_embeddings(pos_itemi0)
            item_neg_e = self.item_embeddings(neg_itemi0)
            #pdb.set_trace()
            user_e = self.user_filter(user_e)
            item_pos_e = self.user_filter(item_pos_e)
            item_neg_e = self.user_filter(item_neg_e)
            #pdb.set_trace()
            if cal_batch_user_embedding == False and cal_batch_item_embedding == False:
                return user_e, item_pos_e, item_neg_e
        
        if cal_batch_user_embedding == True and pos_user_ids is not None and neg_user_ids is not None:
            user_emb = self.user_embeddings(user_ids)
            pos_user_emb = self.user_embeddings(pos_user_ids)
            neg_user_emb = self.user_embeddings(neg_user_ids)
            #if training_mode == True:
                #pdb.set_trace()
            user_emb = self.user_filter(user_emb)
            pos_user_emb = self.user_filter(pos_user_emb)
            neg_user_emb = self.user_filter(neg_user_emb)
            #if training_mode == True:
                #pdb.set_trace()
            if cal_batch_item_embedding == False:
                return user_emb, pos_user_emb, neg_user_emb   
        
        if cal_batch_item_embedding == True and pos_item_ids is not None and neg_item_ids is not None:
            item_emb = self.item_embeddings(item_ids)
            pos_item_emb = self.item_embeddings(pos_item_ids)
            neg_item_emb = self.item_embeddings(neg_item_ids)
            #if training_mode == True:
                #pdb.set_trace()
            item_emb = self.user_filter(item_emb)
            pos_item_emb = self.user_filter(pos_item_emb)
            neg_item_emb = self.user_filter(neg_item_emb)
            #if training_mode == True:
                #pdb.set_trace()
            if cal_batch_user_embedding == False:
                return item_emb, pos_item_emb, neg_item_emb
        
        if return_batch_all_embedding == True:
            return user_emb, pos_user_emb, neg_user_emb, item_emb, pos_item_emb, neg_item_emb
        
        ratings = ratings.float()
        # ユークリッド距離
        dist_pos = torch.norm(user_e - item_pos_e, p=2, dim=1)
        dist_neg = torch.norm(user_e - item_neg_e, p=2, dim=1)
        #pdb.set_trace()
        # Triplet loss: minimize dist_pos, maximize dist_neg
        margin = self.margin
        #重みを正規化
        #min_rating = ratings.min()
        #max_rating = ratings.max()
        #w_ui = 1 + (ratings.float() - min_rating) / (max_rating - min_rating)
        #dist_pos = dist_pos * w_ui
        #pdb.set_trace()
        loss_triplet = torch.relu(dist_pos**2 - dist_neg**2 + margin)
        #リストの順位を考慮して重みを計算
        unique_pairs = set(zip(user0.tolist(), pos_itemi0.tolist()))
        w_list = torch.zeros_like(loss_triplet)

        for user_id, pos_item_id in unique_pairs:
            mask = (user0 == user_id) & (pos_itemi0 == pos_item_id)
            M = (loss_triplet[mask] > 0).sum().item()
            rank_d = (J * M) / U if U > 0 else 0
            w_ui = math.log(rank_d + 1)
            w_list[mask] = w_ui

        loss_triplet = loss_triplet * w_list
        loss_triplet = loss_triplet.mean()
        
        # 正則化項
        l2_reg = self.l2_reg * (user_e.norm(2).pow(2) + item_pos_e.norm(2).pow(2) + item_neg_e.norm(2).pow(2)) / user_e.shape[0]
        #loss = loss_triplet
        loss = loss_triplet + l2_reg

        l_penalty_1, l_penalty_2 = 0, 0
        # ハイパーパラメータで公平性の強度を調整
        lambda_user = 0.3 
        lambda_item = 0.2

        if Discrminator is not None:
            l_penalty_1 = Discrminator(user_emb, pos_user_emb, neg_user_emb, user0, True)
            # Discriminatorの損失を最大化するため、全体の損失からは減算する
            loss = loss - lambda_user * l_penalty_1
        if Regression is not None:
            l_penalty_2 = Regression(item_emb, pos_item_emb, neg_item_emb, True)
            # Regressionの損失を最大化するため、全体の損失からは減算する
            loss = loss - lambda_item * l_penalty_2
    
        return loss, loss_triplet, l_penalty_1, l_penalty_2

    def predict(self, userid, item_id, ratings, return_e=False, return_user_e=False, return_item_e=False, calcurate_fairness_mode=False):

        if return_e == True:
            user_e = self.user_embeddings(userid)
            item_e = self.item_embeddings(item_id) 
            user_e = self.user_filter(user_e)
            item_e = self.user_filter(item_e)
            return user_e, item_e
        if return_user_e == True:
            user_e = self.user_mebeddings(userid)
            user_e = self.user_filter(user_e)
            return user_e
        if return_item_e == True:
            item_e = self.item_embeddings(item_id) 
            item_e = self.user_filter(item_e)
            return item_e

        user_e = self.user_embeddings(userid)
        item_e = self.item_embeddings(item_id) 
        user_e = self.user_filter(user_e)
        item_e = self.user_filter(item_e)
        
        if calcurate_fairness_mode == True:
            dist = torch.cdist(user_e, item_e)
            return dist.max() - dist
        
        dist = torch.norm(user_e - item_e, p=2, dim=1)
        # 距離が小さいほど高評価とみなす
        
        score = dist
        #pdb.set_trace()

        score_min = score.min()
        score_max = score.max()
        score_scaled = 5 - 4 * (score - score_min) / (score_max - score_min + 1e-8) # ゼロ割防止のために1e-8を追加
        """
        score_sigmoid_raw = torch.sigmoid(-dist) # 距離が小さいほど高い評価（1に近い値）
        # 3. シグモイドの出力範囲 [0, 1] を目的の評価スケール [1, 5] に線形スケーリング
        # (出力 - 最小値) / (最大値 - 最小値) で [0, 1] に正規化されていると見なし、
        # 新しい範囲の最小値 + (新しい範囲の最大値 - 新しい範囲の最小値) * 正規化された値
        target_min = 1.0
        target_max = 5.0
        score_scaled = target_min + (target_max - target_min) * score_sigmoid_raw
        """
        ratings = ratings.float()
        mse = self.mse(score_scaled, ratings)
        
        return mse

class GandACMLDiscriminator(nn.Module):
    def __init__(self, user_num, pos_user_num, neg_user_num, emned_dim, inds, margin=0.0, l2_reg=0.001):
        super(GandACMLDiscriminator, self).__init__()
        self.l2_reg = l2_reg
        self.margin = margin
        self.user_sensitive = inds
        self.embed_dim = emned_dim
        self.out_dim = 6

        self.loss_fn = nn.MarginRankingLoss(margin=margin)
        #-------ser_filterによってバイアスが除去された埋め込みを受け取り、そセンシティブ属性を予測をするための変換----------#
        """
        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(int(self.embed_dim/2), int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(int(self.embed_dim / 2), int(self.embed_dim / 4), bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(int(self.embed_dim / 4), int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        )
        """
        self.net = nn.Sequential(
        nn.Linear(self.embed_dim, int(self.embed_dim /2), bias=True),
        nn.BatchNorm1d(int(self.embed_dim /2)),  # Batch normalization for better training stability
        nn.LeakyReLU(0.1),
        nn.Dropout(0.4), # 正則化のためにドロップアウトを追加
        nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        )
        self.init_weights()

    def init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                xavier_normal_(layer.weight.data)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.0)
        
    
    def forward(self, user_emb, pos_user_emb, neg_user_emb, user_ids, return_loss=False):
        anchor_e = user_emb
        pos_user_e = pos_user_emb
        neg_user_e = neg_user_emb
        #pdb.set_trace()
        anchor_e = self.net(anchor_e)
        pos_user_e = self.net(pos_user_e)
        neg_user_e = self.net(neg_user_e)
        #pdb.set_trace()
        user_ids = user_ids.long()
        sensitive_labels = [self.user_sensitive[user_id] for user_id in torch.tensor(user_ids).to(device).numpy()]

        if return_loss:
            # ユークリッド距離
            dist_pos = torch.norm(anchor_e - pos_user_e, p=2, dim=1)
            dist_neg = torch.norm(anchor_e - neg_user_e, p=2, dim=1)
            #pdb.set_trace()
            # マージンの設定
            margins = []
            for i in range(dist_pos.size(0)):
                if dist_neg[i] < dist_pos[i]:
                    margin = dist_pos[i] - dist_neg[i]
                    #print("User dist Calculated margin:", margin.item())
                else:
                    margin = 0.0
                    #print("No margin adjustment needed, using 0.0")
                margins.append(margin)
            margins = torch.tensor(margins, device=dist_pos.device)
            margin_max = torch.max(margins)
            Q1 = torch.quantile(margins, 0.25)
            margin_min = torch.min(margins) + 1e-8
            margin = Q1
            #ロスの計算
            loss_triplet = torch.relu(dist_neg**2 - dist_pos**2 + margin)
            l2_reg = self.l2_reg * (anchor_e.norm(2).pow(2) + pos_user_e.norm(2).pow(2) + neg_user_e.norm(2).pow(2)) / anchor_e.shape[0]
            loss_triplet = loss_triplet + torch.tensor(l2_reg)

            return loss_triplet.mean()
        else:
            output = nn.functional.log_softmax(anchor_e, dim=1)
            return output, sensitive_labels
    
    def predict(self, user_emb, pos_user_emb, neg_user_emb, user_ids, return_loss=False, return_preds=False):
        with torch.no_grad():

            user_ids = user_ids.long()
            sensitive_labels = [self.user_sensitive[user_id] for user_id in torch.tensor(user_ids).to(device).numpy()]
            preds = None
        if return_preds:
            user_e = self.net(user_emb)
            preds = nn.functional.log_softmax(user_e, dim=1)
            return preds, sensitive_labels
        elif return_loss:
            # ユークリッド距離
            dist_pos = torch.norm(user_emb - pos_user_emb, p=2, dim=1)
            dist_neg = torch.norm(user_emb - neg_user_emb, p=2, dim=1)
            # Triplet loss: minimize dist_pos, maximize dist_neg
            loss_triplet = torch.relu(dist_neg - self.margin - dist_pos)
            return loss_triplet.mean()
        else:
            user_e = self.net(user_emb)
            output = nn.functional.log_softmax(user_e, dim=1)
            #pdb.set_trace()
            return output, sensitive_labels


class GandACMLRegression(nn.Module):
    def __init__(self, pos_item_num, neg_item_num, emned_dim, inds, margin=0.5, l2_reg=0.001):
        super(GandACMLRegression, self).__init__()
        self.l2_reg = l2_reg
        self.margin = margin
        self.item_sensitive = inds
        self.embed_dim = emned_dim
        self.out_dim = 6
        self.loss_fn = nn.MSELoss()
        
        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim / 2), bias=True),
            nn.BatchNorm1d(int(self.embed_dim / 2)),  # Batch normalization for better training stability
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),  # 正則化のためにドロップアウトを追加
            nn.Linear(int(self.embed_dim / 2), self.embed_dim, bias=True),
        )
        self.init_weights()

    def init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                xavier_normal_(layer.weight.data)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.0)
    
    def forward(self, item_emb, pos_item_emb, neg_item_emb, return_loss=False):
        #GandA_labels = Variable(torch.LongTensor(self.item_sensitive[item_emb.cpu()]))
        anchor_e = item_emb
        pos_item_e = pos_item_emb
        neg_item_e = neg_item_emb
        #pdb.set_trace()
        anchor_e = self.net(anchor_e)
        pos_item_e = self.net(pos_item_e)
        neg_item_e = self.net(neg_item_e)
        #pdb.set_trace()
        if return_loss:
            # ユークリッド距離
            dist_pos = torch.norm(anchor_e - pos_item_e, p=2, dim=1)
            dist_neg = torch.norm(anchor_e - neg_item_e, p=2, dim=1)
            #pdb.set_trace()
            #マージンの設定
            margins = []
            for i in range(dist_pos.size(0)):
                if dist_neg[i] < dist_pos[i]:
                    margin = (dist_pos[i] - dist_neg[i])
                    #print("Item dist Calculated margin:", margin.item())
                else:
                    margin = 0.0
                    #print("No margin adjustment needed, using 0.0")
                margins.append(margin)
            margins = torch.tensor(margins, device=dist_pos.device)
            margin_max = torch.max(margins)
            Q1 = torch.quantile(margins, 0.25)
            margin_min = torch.min(margins) + 1e-8
            margin = Q1
            #ロスの計算
            loss_triplet = torch.relu(dist_neg**2 - dist_pos**2 + margin)
            l2_reg = self.l2_reg * (anchor_e.norm(2).pow(2) + pos_item_e.norm(2).pow(2) + neg_item_e.norm(2).pow(2)) / anchor_e.shape[0]
            loss_triplet = loss_triplet + torch.tensor(l2_reg)

            return loss_triplet.mean()
        else:
            output = self.net(item_emb)
            return nn.functional.log_softmax(output, dim=1), #GandA_labels
    
    def predict(self, item_emb, pos_item_emb, neg_item_emb, return_loss=True, return_preds=False):
        with torch.no_grad():
            output = self.net(item_emb)
            GandA_labels = Variable(torch.LongTensor(self.item_sensitive[torch.tensor(item_emb).to(device)])).to(device)
        
        # `forward`メソッドとの一貫性を保つために、log_softmaxを適用
        probs = nn.functional.log_softmax(output, dim=1)
        rmse = self.loss_fn(probs, GandA_labels) # 必要に応じて損失計算の入力を調整

        if return_preds:
            return rmse, probs, GandA_labels
        else:
            return rmse

class CMLUserTripletGenerator:
    """
    ユーザ ID を基にトリプレットペアを生成するクラス。
    """

    def __init__(self, user_ids, pos_user_lists, neg_user_lists, sample_size):
        """
        初期化メソッド。

        Args:
            user_ids (list): アンカーとして使用するユーザーIDのリスト。
            pos_user_lists (dict): 各ユーザに対応するポジティブユーザのリスト。
            neg_user_lists (dict): 各ユーザに対応するネガティブユーザのリスト。
            sample_size (int): 各アンカーに対してサンプリングするペアの数。
        """
        self.user_ids = user_ids
        self.pos_user_lists = pos_user_lists
        self.neg_user_lists = neg_user_lists
        self.sample_size = sample_size

    def generate_triplet_pairs(self, batch_size):
        """
        ユーザ ID を基にトリプレットペアのバッチを生成するメソッド。

        Args:
            batch_size (int): 生成するトリプレットペアの数。

        Returns:
            list: トリプレットペアのリスト [(user_id, pos_user, neg_user), ...]
        """
        triplet_pairs = []
        # アンカーユーザーをランダムに選択
        anchor_users = random.choices(self.user_ids, k=batch_size)
        #pdb.set_trace()
        
        for user_id in anchor_users:
            # Check if user_id exists as a key and the list is not empty
            #pdb.set_trace()
            # ポジティブとネガティブのユーザーをサンプリング
            pos_user = random.choice(self.pos_user_lists[user_id])
            neg_user = random.choice(self.neg_user_lists[user_id])
            #pdb.set_trace()
            
            triplet_pairs.append((user_id, pos_user, neg_user))
        
        return triplet_pairs


class CMLItemTripletGenerator:
    """
    アイテム ID を基にトリプレットペアを生成するクラス。
    """

    def __init__(self, item_ids, pos_item_ratio_lists, neg_item_ratio_lists, sample_size):
        """
        初期化メソッド。

        Args:
            item_ids (list): アンカーとして使用するアイテムIDのリスト。
            pos_item_lists (dict): 各アイテムに対応するポジティブアイテムのリスト。
            neg_item_lists (dict): 各アイテムに対応するネガティブアイテムのリスト。
            sample_size (int): 各アンカーに対してサンプリングするペアの数。
        """
        self.item_ids = item_ids
        self.pos_item_ratio_lists = pos_item_ratio_lists
        self.neg_item_ratio_lists = neg_item_ratio_lists
        self.sample_size = sample_size

    def generate_triplet_pairs(self, batch_size):
        """
        アイテム ID を基にトリプレットペアのバッチを生成するメソッド。

        Args:
            batch_size (int): 生成するトリプレットペアの数。

        Returns:
            list: トリプレットペアのリスト [(item_id, pos_item, neg_item), ...]
        """
        triplet_pairs = []
        # アンカーアイテムをランダムに選択
        anchor_items = random.choices(self.item_ids, k=batch_size)

        for item_id in anchor_items:
            # Check if item_id exists as a key and the list is not empty

            # ポジティブとネガティブのアイテムをサンプリング
            pos_item = random.choice(self.pos_item_ratio_lists[item_id])
            neg_item = random.choice(self.neg_item_ratio_lists[item_id])
            
            triplet_pairs.append((item_id, pos_item, neg_item))
        
        return triplet_pairs
