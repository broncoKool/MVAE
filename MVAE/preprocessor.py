"""


.------..------..------..------..------..------..------..------..------..------..------..------.
|K.--. ||O.--. ||U.--. ||Y.--. ||I.--. ||N.--. ||G.--. ||S.--. ||H.--. ||U.--. ||A.--. ||I.--. |
| :/\: || :/\: || (\/) || (\/) || (\/) || :(): || :/\: || :/\: || :/\: || (\/) || (\/) || (\/) |
| :\/: || :\/: || :\/: || :\/: || :\/: || ()() || :\/: || :\/: || (__) || :\/: || :\/: || :\/: |
| '--'K|| '--'O|| '--'U|| '--'Y|| '--'I|| '--'N|| '--'G|| '--'S|| '--'H|| '--'U|| '--'A|| '--'I|
`------'`------'`------'`------'`------'`------'`------'`------'`------'`------'`------'`------'


"""
import numpy as np
import pandas as pd


class ml1m:
    def __init__(self):
        return

    @staticmethod
    def train(n):
        train_df = pd.read_csv('./data/ml-1m/train_%d.csv' % n)
        vali_df = pd.read_csv('./data/ml-1m/vali_%d.csv' % n)
        num_users = np.max(train_df['userId'])
        num_items = np.max(train_df['movieId'])

        train_R = np.zeros((num_users, num_items))  # training rating matrix
        vali_R = np.zeros((num_users, num_items))  # validation rating matrix

        train_mat = train_df.values
        for i in range(len(train_df)):
            user_idx = int(train_mat[i, 0]) - 1
            item_idx = int(train_mat[i, 1]) - 1
            train_R[user_idx, item_idx] = 1

        vali_mat = vali_df.values
        for i in range(len(vali_df)):
            user_idx = int(vali_mat[i, 0]) - 1
            item_idx = int(vali_mat[i, 1]) - 1
            vali_R[user_idx, item_idx] = 1
        
        # matrix_completion
        # Y = np.zeros_like(train_R)
        # if not tau:
        #     tau = 5 *np.sum(train_R.shape)/2
        # if not delta:
        #     delta = 1.2* np.prod(train_R.shape) / np.sum(train_R)
        '''
        :param R: 用户-物品评分矩阵 m*n
        :param P: 用户的分解矩阵 m*k
        :param Q: 物品的分接矩阵 m*k
        :param K: 隐向量的维度
        :param aplha: 学习率
        :param beta: 正则化参数
        :param steps:
        :return:
        '''
        #R = train_R
        R=np.array(train_R)
        N=len(R)    #原矩阵R的行数
        M=len(R[0]) #原矩阵R的列数
        K=3    #K值可根据需求改变
        P=np.random.rand(N,K) #随机生成一个 N行 K列的矩阵
        Q=np.random.rand(M,K) #随机生成一个 M行 K列的矩阵
        aplha=0.0002
        beta=0.02
        steps = 500
        #开始训练  更新参数 计算损失值
        #更新参数 使用梯度下降方法
        for step in range(steps):
            ''''''
            for i in range(len(R)):
                for j in range(len(R[i])):
                    eij=R[i][j]-np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        if R[i][j]>0:
                            #更新参数
                            P[i][k]=P[i][k]+aplha*(2*eij*Q[k][j]-beta*P[i][k])
                            Q[k][j]=Q[k][j]+aplha*(2*eij*P[i][k]-beta*Q[k][j])
            eR=np.dot(P,Q)
            #计算损失值
            e=0
            for i in range(len(R)):
                for j in range(len(R[i])):
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)  # 损失函数求和
                        for k in range(K):
                            e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))  # 加入正则化后的损失函数求和
            #result.append(e)
            if e < 0.001:
                break
            train_R = P * Q
        #return P, Q.T, result


        return train_R, vali_R

       

    @staticmethod
    def test():
        test_df = pd.read_csv('./data/ml-1m/test.csv')
        num_users = np.max(test_df['userId'])
        num_items = np.max(test_df['movieId'])

        test_R = np.zeros((num_users, num_items))  # testing rating matrix

        test_mat = test_df.values
        for i in range(len(test_df)):
            user_idx = int(test_mat[i, 0]) - 1
            item_idx = int(test_mat[i, 1]) - 1
            test_R[user_idx, item_idx] = 1

        train_df = pd.read_csv('./data/ml-1m/train.csv')
        num_users = np.max(train_df['userId'])
        num_items = np.max(train_df['movieId'])

        train_R = np.zeros((num_users, num_items))  # testing rating matrix

        train_mat = train_df.values
        for i in range(len(train_df)):
            user_idx = int(train_mat[i, 0]) - 1
            item_idx = int(train_mat[i, 1]) - 1
            train_R[user_idx, item_idx] = 1
            train_R[user_idx, item_idx] = 1

        return train_R, test_R

class Pinterest:
    def __init__(self):
        return

    @staticmethod
    def train(n):
        train_df = pd.read_csv('./data/pinterest/train_%d.csv' % n)
        #vali_df = pd.read_csv('./data/p/vali_%d.csv' % n)
        num_users = np.max(train_df['userId'])
        num_items = np.max(train_df['movieId'])

        train_R = np.zeros((num_users, num_items))  # training rating matrix
        vali_R = np.zeros((num_users, num_items))  # validation rating matrix

        train_mat = train_df.values
        for i in range(len(train_df)):
            user_idx = int(train_mat[i, 0]) - 1
            item_idx = int(train_mat[i, 1]) - 1
            train_R[user_idx, item_idx] = 1

        # vali_mat = vali_df.values
        # for i in range(len(vali_df)):
        #     user_idx = int(vali_mat[i, 0]) - 1
        #     item_idx = int(vali_mat[i, 1]) - 1
        #     vali_R[user_idx, item_idx] = 1
        return train_R

    @staticmethod
    def test():
        test_df = pd.read_csv('./data/pinterest/test.csv')
        num_users = np.max(test_df['userId'])
        num_items = np.max(test_df['movieId'])

        test_R = np.zeros((num_users, num_items))  # testing rating matrix

        test_mat = test_df.values
        for i in range(len(test_df)):
            user_idx = int(test_mat[i, 0]) - 1
            item_idx = int(test_mat[i, 1]) - 1
            test_R[user_idx, item_idx] = 1

        train_df = pd.read_csv('./data/pinterest/train.csv')
        num_users = np.max(train_df['userId'])
        num_items = np.max(train_df['movieId'])

        train_R = np.zeros((num_users, num_items))  # testing rating matrix

        train_mat = train_df.values
        for i in range(len(train_df)):
            user_idx = int(train_mat[i, 0]) - 1
            item_idx = int(train_mat[i, 1]) - 1
            train_R[user_idx, item_idx] = 1
            train_R[user_idx, item_idx] = 1

        return train_R, test_R




class yelp:
    def __init__(self):
        return

    @staticmethod
    def train(n):
        train_df = pd.read_csv('./data/yelp/train_%d.csv' % n)
        vali_df = pd.read_csv('./data/yelp/vali_%d.csv' % n)
        num_users = np.max(train_df['userId'])
        num_items = np.max(train_df['itemId'])

        train_R = np.zeros((num_users, num_items))  # training rating matrix
        vali_R = np.zeros((num_users, num_items))  # validation rating matrix

        train_mat = train_df.values
        for i in range(len(train_df)):
            user_idx = int(train_mat[i, 0]) - 1
            item_idx = int(train_mat[i, 1]) - 1
            train_R[user_idx, item_idx] = 1

        vali_mat = vali_df.values
        for i in range(len(vali_df)):
            user_idx = int(vali_mat[i, 0]) - 1
            item_idx = int(vali_mat[i, 1]) - 1
            vali_R[user_idx, item_idx] = 1
        return train_R, vali_R

    @staticmethod
    def test():
        test_df = pd.read_csv('./data/yelp/test.csv')
        num_users = np.max(test_df['userId'])
        num_items = np.max(test_df['itemId'])

        test_R = np.zeros((num_users, num_items))  # testing rating matrix

        test_mat = test_df.values
        for i in range(len(test_df)):
            user_idx = int(test_mat[i, 0]) - 1
            item_idx = int(test_mat[i, 1]) - 1
            test_R[user_idx, item_idx] = 1

        train_df = pd.read_csv('./data/yelp/train.csv')
        num_users = np.max(train_df['userId'])
        num_items = np.max(train_df['itemId'])

        train_R = np.zeros((num_users, num_items))  # testing rating matrix

        train_mat = train_df.values
        for i in range(len(train_df)):
            user_idx = int(train_mat[i, 0]) - 1
            item_idx = int(train_mat[i, 1]) - 1
            train_R[user_idx, item_idx] = 1
            train_R[user_idx, item_idx] = 1

        return train_R, test_R

