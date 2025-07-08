#-- coding:UTF-8 --
import os
import numpy as np
import math
import sys
import argparse
import pdb
import pickle
from collections import defaultdict
import time
import pandas as pd

data_dir = './ml-1m'

class MovieLens1M():
    #rating.datのパスを取得
    def __init__(self, data_dir):
        self.fpath = os.path.join(data_dir, 'ratings.dat')

    def load(self):
        # rating.datをDataaFrameとして読み込む
        df = pd.read_csv(self.fpath,
                         sep='::',
                         engine='python',
                         names=['user', 'item', 'ratings', 'time'])
        # df = df[df['ratings'] == 5]
        return df


# x:id i:range(len) id->range(len)
# for items 1193 2355 1287-> 0 1 2
def convert_unique_idx(df, column_name):
    #column_namrのユニークな値を取得し、0から始まるインデックスを割り当てた辞書を作成→{'A':0,'B':1,'C':2}
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    #列の値に対応するインデックスを割り当てる
    df[column_name] = df[column_name].apply(column_dict.get)
    #列の値を整数型に変換
    df[column_name] = df[column_name].astype('int')
    #列の最小値が0、最大値がLen(column_dict)-1であることを確認
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict

# input user0-range item0-range ratings time
def create_user_list(df, user_size):
    # user_sizeに応じた空の辞書を作成
    user_list = [dict() for u in range(user_size)]
    #ユーザIDをリストのインデックスとして使用し、アイテムIDをキー、評価値を値とする辞書を作成
    for row in df.itertuples():
        user_list[row.user][row.item] = row.ratings
    return user_list


# output:user_list:a list whose elements are all dict(save itemid for user u)
# enmurate(list)-> {itemsid1:5,   itemsid2:5......}
def split_train_test(user_list, test_size=0.2, val_size=0.1):
    #各ユーザのトレーニング、テスト、検証データを格納するためのからの辞書リストを作成
    train_user_list = [dict() for u in range(len(user_list))]
    test_user_list = [dict() for u in range(len(user_list))]
    val_user_list = [dict() for u in range(len(user_list))]
    all_user_list = [dict() for u in range(len(user_list))]
    for user, item_dict in enumerate(user_list):
        #各ユーザのアイテムをランダムに選択し、トレーニング(val_train)、テスト(test_item)、検証(val_item)データに分割
        test_item = set([])
        val_item = set([])
        test_item = set(np.random.choice(list(item_dict.keys()),
                                         size=int(len(item_dict) * test_size),
                                         replace=False))
        # pdb.set_trace()
        val_train = set(item_dict.keys()) - test_item
        val_item = set(np.random.choice(list(val_train),
                                        size=int(len(item_dict) * val_size),
                                        replace=False))
        #テスト、検証データのサイズが0より大きいことを確認
        assert len(test_item) > 0, "No test item for user %d" % user
        assert len(val_item) > 0, "No val item for user %d" % user
        #各アイテムを対応する(トレーニング、テスト、検証)データに追加
        for i in test_item:
            test_user_list[user][i] = item_dict[i]
        for i in val_item:
            val_user_list[user][i] = item_dict[i]
        for i in (set(item_dict.keys()) - test_item - val_item):
            train_user_list[user][i] = item_dict[i]
        #全てのアイテムを格納
        all_user_list[user].update(item_dict)
    return train_user_list, test_user_list, val_user_list, all_user_list

# train_user_list: list, for each user a set{} stores num of moives
# output[(user1,item1),(user_u,item_i)]
def create_pair(user_list):
    pair = []
    #user_lsitに対応したuserIDを取得し(user_id, item_id, ratings)のペアを作成
    for user, item_set in enumerate(user_list):
        pair.extend([(user, item, ratings) for item,ratings in item_set.items()])
    #pdb.set_trace()
    return pair


# --------------------------load feature data-------------------------#
#user.datから性別データを読み込み、性別を数値(0 or 1)に変換
def load_features():
    u_cols = ['user_id', 'sex', 'age', 'occupation', 'zip_code']
    users = pd.read_csv('./ml-1m/users.dat', sep='::', names=u_cols,
                        encoding='latin-1', parse_dates=True,engine='python')
    pdb.set_trace()
    users_sex = users['sex']
    users_sex_1 = [0 if i == 'M' else 1 for i in users_sex]
    users_sex_2 = [1 if i == 'M' else 0 for i in users_sex]
    gender_data = np.ascontiguousarray(users_sex_1)
    gender_data_2 = np.ascontiguousarray(users_sex_2)
    #gender_data:[0,1,0,1],男性=0,女性=1
    #gender_data_2:[1,0,1,0],男性=1,女性=0
    #pdb.set_trace()
    return gender_data,gender_data_2

def load_item_feature(gender_data,gender_data_2,train_user_list,item_size):
    train_item_list = [dict() for i in range(item_size)]
    #各アイテムの女性ユーザの評価平均を格納するリスト
    item_rating_female = [None] * item_size
    #各アイテムの男性ユーザの評価平均を格納するリスト
    item_rating_male = [None] * item_size
    #各アイテムの女性ユーザの数を格納するリスト
    item_female_inds = [None] * item_size
    #train_user_listからユーザIDとアイテムと評価値のペアを取得
    for user, item_dict in enumerate(train_user_list):
        #item_dictからアイテムIDと評価値を取得
        for item,rating in item_dict.items():
            #ex) train_item_list[101][1]=4
            train_item_list[item][user] = rating

    for item, user_dict in enumerate(train_item_list):
        tmp_list=[]
        tmp_list_2=[]
        tmp_list_3=[]
        for user, rating in user_dict.items():
            tmp_female = gender_data[user]*rating
            tmp_female_num = gender_data[user]*1
            tmp_male = gender_data_2[user]*rating
            #女性ユーザの評価値を格納
            tmp_list.append(tmp_female)
            #男性ユーザの評価値を格納
            tmp_list_2.append(tmp_male)
            #女性ユーザの数を格納
            tmp_list_3.append(tmp_female_num)

        #からのリストの場合は0を追加
        if len(tmp_list) == 0:
            tmp_list.append(0)
        if len(tmp_list_2) == 0:
            tmp_list_2.append(0)
        if len(tmp_list_3) == 0:
            tmp_list_3.append(0)
        
        #平均値を計算
        item_label_female = np.mean(tmp_list)
        item_label_male= np.mean(tmp_list_2)
        item_inds = np.mean(tmp_list_3)
        #各アイテムの女性ユーザの評価平均を格納
        item_rating_female[item] = item_label_female
        #各アイテムの男性ユーザの評価平均を格納
        item_rating_male[item] = item_label_male
        #各アイテムの女性ユーザの数を格納
        item_female_inds[item] = item_inds
    return item_rating_female,item_rating_male,item_female_inds


def data():
    #rating.datデータを読み込み、dfに変換
    #MovieLens1M()クラスをインスタンス化し、load()メソッドを呼び出す
    s = MovieLens1M(data_dir)
    pdb.set_trace()
    df = s.load()
    pdb.set_trace()
    '''
    handle data for gcn prepare
    '''
    #ユーザIDを0始まりのインデックスに変換
    df['user'] = df['user']-1
    pdb.set_trace()
    #convert_unique_idx()メソッドを呼び出す
    df, itemid = convert_unique_idx(df, 'item')
    pdb.set_trace()
    # pdb.set_trace()
    # userid:{keys:values} values are 0-n int. so we can change dict into list,and user index to represent values
    # index 0 equals to values 0, and so on
    print('Complete assigning unique index to user and item')

    #ユーザ数を所得
    user_size = len(df['user'].unique())
    #アイテム数を取得
    item_size = len(df['item'].unique())
    print(user_size)
    print(item_size)

    #create_user_list()メソッドからユーザリストを作成
    total_user_list = create_user_list(df, user_size)
    pdb.set_trace()
    #split_train_test()メソッドでユーザリストをトレーニング、テスト、検証データに分割
    train_user_list, test_user_list, val_user_list, all_user_list = split_train_test(total_user_list)
    pdb.set_trace()

    #アイテムサイズに応じた空の辞書を作成
    train_item_list = [dict() for i in range(item_size)]
    for user in range(user_size):
        for item, rating in train_user_list[user].items():
            #アイテムとユーザのペアに対応する評価値を格納する
            train_item_list[item][user] = rating

    print('Complete spliting items for training and testing')
    #デバッグ用のブレークポイント
    pdb.set_trace()

    #create_pair()メソッドでトレーニング、テスト、検証データを(user, item, rating)のペアに変換
    train_pair = create_pair(train_user_list)
    pdb.set_trace()
    test_pair = create_pair(test_user_list)
    pdb.set_trace()
    val_pair = create_pair(val_user_list)
    pdb.set_trace()

    #load_features()メソッドでユーザの性別データを数値化
    gender_data,gender_data_2 = load_features()
    pdb.set_trace()
    #load_item_feature()メソッドアイテムごとの特徴量を格納
    item_rating_female,item_rating_male,item_female_inds = load_item_feature(gender_data,gender_data_2,train_user_list,item_size)
    pdb.set_trace()
    
    #データセットを辞書形式で格納
    dataset = {'user_size': user_size, 'item_size': item_size,
               'train_user_list': train_user_list, 'test_user_list': test_user_list,
               'all_user_list': all_user_list, 'val_user_list': val_user_list,
               'train_pair': train_pair, 'test_pair': test_pair, 'val_pair': val_pair,
               'gender_data':gender_data,'item_rating_female':item_rating_female,
               'item_rating_male':item_rating_male,'item_inds':item_female_inds}
    dirname = './preprocessed/'
    filename = './preprocessed/ml-1m_gcn.pickle'
    '''
    gcn    no r
    gcn1
    gcn2    no r
    gcn3    sqrt  r
    gcn4    r/5
    gcn5  delete rating <=3
    gcn6  4 A/2  5  A 我前一次真是欧皇，这个老难分到每个item都有了（继续降低4得权重）
    7 100 50 25 10
    8 重新提高权重 50 25 10 5
    '''
    os.makedirs(dirname, exist_ok=True)
    #作成したデータセットをpickle形式で保存
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


def re_save_data():
    #user.datを読み込む
    u_cols = ['user_id', 'sex', 'age', 'occupation', 'zip_code']
    users = pd.read_csv('./ml-1m/users.dat', sep='::', names=u_cols,
                        encoding='latin-1', parse_dates=True, engine='python')
    #年齢をカテゴリに変換
    ages_dict = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
    users_age = users['age']
    users_age_ = [ages_dict[i] for i in users_age]
    #ワンホットエンコーディングを適用
    users_age_onehot = np.eye(7)[users_age_]
    pdb.set_trace()
    #保存されているデータセットを読み込む
    with open('./preprocessed/ml-1m_gcn_rebuttal.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, val_user_list, test_user_list = dataset['train_user_list'], dataset['val_user_list'], dataset[
            'test_user_list']
        all_user_list = dataset['all_user_list']
        train_pair = dataset['train_pair']
        test_pair = dataset['test_pair']
        val_pair = dataset['val_pair']
        gender_data = dataset['gender_data']
        item_inds = dataset['item_inds']
    print('Load complete')


    #アイテム総数に基づいて、空の辞書を格納
    train_item_list = [dict() for i in range(item_size)]
    #リストを作成
    item_age_inds = [None] * item_size
    #各ユーザが評価したアイテムとその評価値を取得したユーザリストから値を取得
    for user, item_dict in enumerate(train_user_list):
        for item,rating in item_dict.items():
            train_item_list[item][user] = rating
    #各アイテムに関連するユーザとその評価値を取得したアイテムリストから値を所得
    for item, user_dict in enumerate(train_item_list):
        #各年齢カテゴリに対応するユーザのデータを一時的に格納するリストを初期化
        tmp_list_0, tmp_list_1, tmp_list_2, tmp_list_3, tmp_list_4, tmp_list_5, tmp_list_6 = [], [], [], [], [], [], []
        for user, rating in user_dict.items():
            # gender_data 010 gender_data_2 101,对应项为0，算总人数已经隐藏在这里了。
            # 会 +1 / +0
            tmp_labels = users_age_onehot[user]
            tmp_list_0.append(tmp_labels[0])
            tmp_list_1.append(tmp_labels[1])
            tmp_list_2.append(tmp_labels[2])
            tmp_list_3.append(tmp_labels[3])
            tmp_list_4.append(tmp_labels[4])
            tmp_list_5.append(tmp_labels[5])
            tmp_list_6.append(tmp_labels[6])
            # pdb.set_trace()
        if len(tmp_list_0) == 0:
            tmp_list_0.append(0)
        if len(tmp_list_1) == 0:
            tmp_list_1.append(0)
        if len(tmp_list_2) == 0:
            tmp_list_2.append(0)
        if len(tmp_list_3) == 0:
            tmp_list_3.append(0)
        if len(tmp_list_4) == 0:
            tmp_list_4.append(0)
        if len(tmp_list_5) == 0:
            tmp_list_5.append(0)
        if len(tmp_list_6) == 0:
            tmp_list_6.append(0)
        item_ind_0 = np.mean(tmp_list_0)
        item_ind_1 = np.mean(tmp_list_1)
        item_ind_2 = np.mean(tmp_list_2)
        item_ind_3 = np.mean(tmp_list_3)
        item_ind_4 = np.mean(tmp_list_4)
        item_ind_5 = np.mean(tmp_list_5)
        item_ind_6 = np.mean(tmp_list_6)
        tmp_item_inds = [item_ind_0, item_ind_1, item_ind_2, item_ind_3, item_ind_4, item_ind_5, item_ind_6]
        item_age_inds[item] = tmp_item_inds
    pdb.set_trace()
    gender_data, gender_data_2 = load_features()
    item_rating_female, item_rating_male, item_female_inds = load_item_feature(gender_data, gender_data_2,
                                                                               train_user_list, item_size)

    pdb.set_trace()
    dataset = {'user_size': user_size, 'item_size': item_size,
               'train_user_list': train_user_list, 'test_user_list': test_user_list,
               'all_user_list': all_user_list, 'val_user_list': val_user_list,
               'train_pair': train_pair, 'test_pair': test_pair, 'val_pair': val_pair,
               'gender_data':gender_data,'item_inds':item_female_inds,
               'age_data':users_age_onehot,"item_age_inds":item_age_inds,
               'age_labels':users_age_,
                }
    dirname = './preprocessed/'
    filename = './preprocessed/ml-1m_gcn_re2.pickle'
    os.makedirs(dirname, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

#--------------------------FairCML用のデータセットを作成-------------------------#
def label_users_by_gender_and_age():
    """
    性別と年齢カテゴリを組み合わせて全ユーザをラベリングする関数
    :param gender_data: 性別データ (0: 男性, 1: 女性)
    :param users_age_: 年齢カテゴリデータ (0~6)
    :return: 各ユーザのラベル (0~13)
    """
    
    with open('./preprocessed/ml-1m_gcn_re2.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, val_user_list, test_user_list = dataset['train_user_list'], dataset['val_user_list'], dataset['test_user_list']
        all_user_list = dataset['all_user_list']
        train_pair = dataset['train_pair']
        test_pair = dataset['test_pair']
        val_pair = dataset['val_pair']
        gender_data = dataset['gender_data']
        item_inds = dataset['item_inds']
        age_data = dataset['age_data']
        item_age_inds = dataset['item_age_inds']
        age_labels = dataset['age_labels']
    print('Load complete')

    #CML推薦モデルのトレーニング用データの作成
    train_pos_item_lists, train_neg_item_lists = create_pos_neg_item_lists(train_user_list, item_size)
    #test_pos_item_lists, test_neg_item_lists = create_pos_neg_item_lists(test_user_list, item_size)
    #val_pos_item_lists, val_neg_item_lists = create_pos_neg_item_lists(val_user_list, item_size)
    #print(sum(len(train_pos_item_list) for train_pos_item_list in train_pos_item_lists), sum(len(test_pos_item_list) for test_pos_item_list in test_pos_item_lists), sum(len(val_pos_item_list) for val_pos_item_lists in val_pos_item_lists))
    #print(sum(len(train_neg_item_list) for train_neg_item_list in train_neg_item_lists), sum(len(test_neg_item_list) for test_neg_item_list in test_neg_item_lists), sum(len(val_neg_item_list) for val_neg_item_list in val_neg_item_lists))
    # train_pair, test_pair, val_pairを作成
    #train_pair = create_triplet_pairs(train_user_list, train_neg_item_lists, train_user_list, sample_size=7)
    #test_pair = create_triplet_pairs(test_user_list, test_neg_item_lists, test_user_list, sample_size=4)
    #val_pair = create_triplet_pairs(val_user_list, val_neg_item_lists, val_user_list, sample_size=3)
    #print(len(train_pair), len(test_pair), len(val_pair))


    #センシティブラベルの作成
    user_labels = generate_user_labels(gender_data, age_labels)
    # ポジティブユーザリストとネガティブユーザリストを生成
    pos_user_lists, neg_user_lists = generate_pos_neg_user_lists(user_labels)

        

    
    #各カテゴリのデータ数を計算
    label_counts = np.unique(user_labels, return_counts=True)
    print("Label Distribution:")
    for label, count in zip(label_counts[0], label_counts[1]):
        print(f"Label {label}: {count} users")


    # アイテムの疑似ラベルを計算
    item_label_means, item_label_ratios = calculate_item_label_metrics(item_size, train_user_list, gender_data, age_labels)


    # アイテムの公平性リストを生成
    pos_item_ratio_lists, neg_item_ratio_lists = generate_item_fairness_lists(item_size, item_label_ratios)


    print("create pos_item_ratio_list and neg_item_ratio_list complete")

    pdb.set_trace()

    dataset = {'user_size': user_size, 'item_size': item_size,
               'train_user_list': train_user_list, 'test_user_list': test_user_list,
               'all_user_list': all_user_list, 'val_user_list': val_user_list,
               'train_pair': train_pair, 'test_pair': test_pair, 'val_pair': val_pair,
               'gender_data':gender_data,'item_inds':item_inds,
               'age_data':age_data,"item_age_inds":item_age_inds,
               'age_labels':age_labels,
               'train_pos_item_lists': train_pos_item_lists,
                'train_neg_item_lists': train_neg_item_lists,
                'user_labels': user_labels,
                'pos_user_lists': pos_user_lists,
                'neg_user_lists': neg_user_lists,
                'item_label_means': item_label_means,
                'item_label_ratios': item_label_ratios,
                'pos_item_ratio_lists': pos_item_ratio_lists,
                'neg_item_ratio_lists': neg_item_ratio_lists
                }
    dirname = './preprocessed/'
    filename = './preprocessed/ml-1m_gcn_genderandage.pickle'
    os.makedirs(dirname, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)



def create_pos_neg_item_lists(train_user_list, item_size):
    """
    各ユーザごとに評価済みアイテムリスト (pos_item_lists) と未評価アイテムリスト (neg_item_lists) を作成する関数
    """
    # 全アイテムIDリストを作成
    all_item_list = list(range(item_size))

    # 各ユーザごとに評価しているアイテムのIDリストと評価していないアイテムIDのリストを作成
    pos_item_lists = []
    neg_item_lists = []

    for user, item_dict in enumerate(train_user_list):
        # 評価済みアイテムのIDリストを作成
        pos_item_list = list(item_dict.keys())
        pos_item_lists.append(pos_item_list)

        # 未評価アイテムのIDリストを作成
        neg_item_list = list(set(all_item_list) - set(pos_item_list))
        neg_item_lists.append(neg_item_list)

    print("create pos_item_lists and neg_item_lists complete")
    return pos_item_lists, neg_item_lists



def generate_user_labels(gender_data, age_labels):
    """
    性別と年齢カテゴリを組み合わせてユーザラベルを生成する関数。

    Args:
        gender_data (list or numpy.ndarray): 性別データ (0: 男性, 1: 女性)。
        age_labels (list or numpy.ndarray): 年齢カテゴリデータ (0~6)。

    Returns:
        list: 各ユーザのラベル (0~5)。
    """
    user_labels = []
    if len(gender_data) == len(age_labels):
        for gender, age in zip(gender_data, age_labels):
            # 統合クラスのラベル付け
            if gender == 0:  # 男性
                if age <= 2:  # ~24歳
                    label = 0
                elif 3 <= age <= 4:  # 25~44歳
                    label = 1
                else:  # 45歳以上
                    label = 2
            else:  # 女性
                if age <= 2:  # ~24歳
                    label = 3
                elif 3 <= age <= 4:  # 25~44歳
                    label = 4
                else:  # 45歳以上
                    label = 5
            user_labels.append(label)
    else:
        raise ValueError("Length of gender_data and age_labels do not match.")
    
    return user_labels

def generate_pos_neg_user_lists(user_labels):
    """
    ユーザラベルを基にポジティブユーザリストとネガティブユーザリストを生成する関数。

    Args:
        user_labels (list): 各ユーザのラベル。

    Returns:
        tuple: (pos_user_lists, neg_user_lists)
            - pos_user_lists: 各ユーザに対応するポジティブユーザのリスト。
            - neg_user_lists: 各ユーザに対応するネガティブユーザのリスト。
    """
    pos_user_lists = []
    neg_user_lists = []

    for user_idx, user_label in enumerate(user_labels):
        # 同じラベルを持つユーザをポジティブリストに追加
        pos_user_list = [idx for idx, label in enumerate(user_labels) if label == user_label and idx != user_idx]
        # 異なるラベルを持つユーザをネガティブリストに追加
        neg_user_list = [idx for idx, label in enumerate(user_labels) if label != user_label]

        pos_user_lists.append(pos_user_list)
        neg_user_lists.append(neg_user_list)

    return pos_user_lists, neg_user_lists


def calculate_item_label_metrics(item_size, train_user_list, gender_data, age_labels):
    """
    アイテムの疑似ラベル（カテゴリごとの評価平均と評価割合）を計算する。

    Args:
        item_size (int): アイテムの総数。
        train_user_list (list): トレーニングデータのユーザーリスト。
        gender_data (list): 性別データ。
        age_labels (list): 年齢ラベルデータ。

    Returns:
        tuple: (item_label_means, item_label_ratios)
            - item_label_means: 各アイテムのカテゴリごとの平均評価値。
            - item_label_ratios: 各アイテムのカテゴリごとの評価割合。
    """
    #各アイテムのカテゴリごとの平均値を格納リスト
    item_label_means = [None] * item_size
    #各アイテムのカテゴリごとの評価割合を格納するリスト
    item_label_ratios = [None] * item_size

    #アイテムごとの評価を集計
    train_item_list = [dict() for _ in range(item_size)]
    for user, item_dict in enumerate(train_user_list):
        for item, rating in item_dict.items():
            train_item_list[item][user] = rating
    

    #各アイテムに対してカテゴリごとの計算を実行
    for item, user_dict in enumerate(train_item_list):
        #各カテゴリの評価値と評価数を格納するリスト
        tmp_means = [0] * 6
        tmp_counts = [0] * 6

        for user, rating in user_dict.items():
            #ユーザのラベルを計算
            gender = gender_data[user]
            age = age_labels[user]
            if gender == 0:
                if age <= 2:
                    label = 0
                elif 3 <= age <= 4:
                    label = 1
                else:
                    label = 2
            else:
                if age <= 2:
                    label = 3
                elif 3 <= age <= 4:
                    label = 4
                else:
                    label = 5

            #評価値をカテゴリに追加
            tmp_means[label] += rating
            tmp_counts[label] += 1

        #各カテゴリの平均値と割合を計算
        item_label_means[item] = [np.float64(tmp_means[i] / tmp_counts[i]) if tmp_counts[i] > 0 else np.float64(0) for i in range(6)]
        total_count = sum(tmp_counts)
        item_label_ratios[item] = [np.float64(tmp_counts[i] / total_count) if total_count > 0 else np.float64(0) for i in range(6)]
    
    return item_label_means, item_label_ratios
    

def generate_item_fairness_lists(item_size, item_label_ratios, percentile_threshold=25):
    """
    アイテムの公平性に基づいてポジティブ/ネガティブリストを生成する。
    各アイテムについて、最も評価割合が高い「支配的な属性」を特定し、
    その属性における評価割合の上位・下位アイテムをそれぞれポジティブ・ネガティブリストに割り当てる。

    Args:
        item_size (int): アイテムの総数。
        item_label_ratios (list): 各アイテムのカテゴリごとの評価割合。
        percentile_threshold (int, optional): 上位・下位を判断するためのパーセンタイル。デフォルトは25。

    Returns:
        tuple: (final_pos_item_lists, final_neg_item_lists)
            - final_pos_item_lists: 各アイテムに対応する公平性上のポジティブアイテムリスト。
            - final_neg_item_lists: 各アイテムに対応する公平性上のネガティブアイテムリスト。
    """
    sensitive_classes = len(item_label_ratios[0])  # 6種類（性別×年齢層）

    # 各属性ごとに、上位・下位パーセンタイルに属するアイテムの集合を事前に計算
    pos_items_by_attr = [set() for _ in range(sensitive_classes)]
    neg_items_by_attr = [set() for _ in range(sensitive_classes)]

    for k in range(sensitive_classes):
        # 属性kの割合のみを全アイテムから抽出
        p_k = np.array([ratios[k] for ratios in item_label_ratios])

        # ゼロ除算を避けるため、p_kがすべて同じ値でないかチェック
        if np.all(p_k == p_k[0]):
            continue

        # 上位・下位の閾値を計算
        top_threshold = np.percentile(p_k, 100 - percentile_threshold)
        bottom_threshold = np.percentile(p_k, percentile_threshold)

        # 閾値に基づいてアイテムIDを集合に追加
        pos_items_by_attr[k] = {i for i, ratio in enumerate(p_k) if ratio >= top_threshold}
        neg_items_by_attr[k] = {i for i, ratio in enumerate(p_k) if ratio <= bottom_threshold}

    # 各アイテムのポジティブ・ネガティブリストを初期化
    final_pos_item_lists = [[] for _ in range(item_size)]
    final_neg_item_lists = [[] for _ in range(item_size)]

    # 各アイテムについて、支配的な属性を見つけ、リストを割り当てる
    for i in range(item_size):
        # アイテムiの支配的な属性（評価割合が最大の属性）を見つける
        dominant_attribute = np.argmax(item_label_ratios[i])

        # 支配的な属性kに対応するポジティブ・ネガティブリストを取得
        pos_set = pos_items_by_attr[dominant_attribute]
        neg_set = neg_items_by_attr[dominant_attribute]

        # 自分自身を除外してリストに変換
        final_pos_item_lists[i] = list(pos_set - {i})
        final_neg_item_lists[i] = list(neg_set - {i})
    
    pdb.set_trace()
        
    return final_pos_item_lists, final_neg_item_lists


if __name__ == '__main__':
    #元データセットを読み込み、前処理を行い、基本的なトレーニングデータセットを作成する
    #data()
    #既存の前処理済みデータセットを読み込み、追加の処理(年齢データの処理など)を行う
    #re_save_data()
    #性別と年齢カテゴリを組み合わせて全ユーザをラベリングする
    label_users_by_gender_and_age()