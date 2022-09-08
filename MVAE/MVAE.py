"""
Yingshuai Kou

  _  __            __     ___                 _                 _ 
 | |/ /            \ \   / (_)               | |               (_)
 | ' / ___  _   _   \ \_/ / _ _ __   __ _ ___| |__  _   _  __ _ _ 
 |  < / _ \| | | |   \   / | | '_ \ / _` / __| '_ \| | | |/ _` | |
 | . \ (_) | |_| |    | |  | | | | | (_| \__ \ | | | |_| | (_| | |
 |_|\_\___/ \__,_|    |_|  |_|_| |_|\__, |___/_| |_|\__,_|\__,_|_|
                                     __/ |                        
                                    |___/                         

"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import numpy as np
import math
import evaluate
from keras.layers import Lambda, Input, Dense
from keras import backend as K
from keras.models import Model
from keras.losses import binary_crossentropy

#from sklearn.decomposition import MiniBatchNMF
import scipy
from sklearn.utils.extmath import randomized_svd


class MVAE():
    def __init__(self, sess, args,
                 num_users, num_items,
                 train_R, test_R):

        self.sess = sess
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.train_R = train_R
        self.test_R = test_R
        self.train_epoch = args.train_epoch
        self.batch_size = args.batch_size
        self.num_batch = int(math.ceil(self.num_users / float(self.batch_size)))
        self.num_batch_U = int(self.num_users / float(self.batch_size)) + 1
        self.num_batch_I = int(self.num_items / float(self.batch_size)) + 1

        self.base_lr = args.base_lr
        self.using_hinge = args.using_hinge

        self.global_step = tf.Variable(0, trainable=False)
        self.decay_epoch_step = args.decay_epoch_step
        self.decay_step = self.decay_epoch_step * self.num_batch
        self.lr = tf.train.exponential_decay(self.base_lr, self.global_step,
                                             self.decay_step, 0.96, staircase=True)
        self.beta_value = args.beta_value
        self.train_cost_list = []
        self.test_cost_list = []
        self.test_rmse_list = []
    
    # NMF 
    def NMF_prediction(self, P, Q):
        N,K = P.shape
        M,K = Q.shape

        rating_list=[]
        for u in range(N):
            u_rating = np.sum(P[u,:]*Q, axis=1)
            rating_list.append(u_rating)
        r_pred = np.array(rating_list)
        return r_pred
    
    #SVD
    def puresvd(self, R = None, # train mat
        k=150, # the number of latent factor
        ):
        P, sigma, QT = randomized_svd(R, k)
        sigma = scipy.sparse.diags(sigma, 0)
        P = P * sigma
        Q = QT.T   
        # R_= np.dot(P, QT)
        R_ = np.dot(R, np.dot(Q, QT)) #
        return R_

    def sampling(self, args):


        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def run(self):
        self.prepare_model()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch_itr in range(self.train_epoch):
            self.train_model(epoch_itr)
            self.test_model(epoch_itr)


    def prepare_model(self):

        self.input_R_U = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items], name="input_R_U")
        self.input_R_I = tf.placeholder(dtype=tf.float32, shape=[self.num_users, None], name="input_R_I")


        self.input_U_origin = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items],  name="input_U_origin")
        self.input_I_origin = tf.placeholder(dtype=tf.float32, shape=[self.num_users, None], name="input_I_origin")
        
        self.row_idx = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="row_idx") #用来传原始数据user
        self.col_idx = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="col_idx") #用来传原始数据item

        intermediate_dim = 320
        latent_dim = 80
        
        #encoder
        x1_user = Dense(intermediate_dim, activation='tanh')(Input(tensor=self.input_R_U))

        x_user = Dense(intermediate_dim, activation='sigmoid')(x1_user)
        z_mean = Dense(latent_dim, name='z_mean')(x_user)
        z_log_var = Dense(latent_dim, name='z_log_var')(x_user)

        # use reparameterization trick to push the sampling out as input

        z = Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder_user model
        encoder_user = Model(Input(tensor=self.input_R_U), [z_mean, z_log_var, z], name='encoder_user')

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x_user_decoder = Dense(intermediate_dim, activation='tanh')(latent_inputs)

        outputs_user = Dense(self.num_items, activation='sigmoid')(x_user_decoder)

        # instantiate decoder model
        decoder_user = Model(latent_inputs, outputs_user, name='decoder_user')
        # instantiate VAE model
        outputs_user = decoder_user(encoder_user(Input(tensor=self.input_R_U))[2])

        self.U_Decoder = outputs_user

        reconstruction_loss_u = binary_crossentropy(Input(tensor=self.input_U_origin),
                                                    outputs_user)

        reconstruction_loss_u *= self.num_items
        kl_loss_u = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss_u = K.sum(kl_loss_u, axis=-1)
        kl_loss_u *= -0.01
        vae_loss_u = K.mean(reconstruction_loss_u + kl_loss_u)
        #######################################################
        inp_tmp = Input(tensor=tf.transpose(self.input_R_I))
        inp_tmp_origin = tf.transpose(self.input_I_origin)

        intermediate_dim = 320
        latent_dim = 80

        x1Item = Dense(intermediate_dim, activation='tanh')(inp_tmp)

        xItem = Dense(intermediate_dim, activation='sigmoid')(x1Item)
        z_meanItem = Dense(latent_dim, name='z_mean')(xItem)
        z_log_varItem = Dense(latent_dim, name='z_log_var')(xItem)


        # use reparameterization trick to push the sampling out as input

        zItem = Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_meanItem, z_log_varItem])

        # instantiate encoder_user model
        encoderItem = Model(inp_tmp, [z_meanItem, z_log_varItem, zItem], name='encoder_user')

        # build decoder model
        latent_inputsItem = Input(shape=(latent_dim,), name='z_sampling')
        xItem = Dense(intermediate_dim, activation='tanh')(latent_inputsItem)

        outputs_item = Dense(self.num_users, activation='sigmoid')(xItem)

        # instantiate decoder model
        decoderItem = Model(latent_inputsItem, outputs_item, name='decoder')

        # instantiate VAE model
        outputs_item = decoderItem(encoderItem(inp_tmp)[2])

        self.I_Decoder = outputs_item

        self.Decoder = ((tf.transpose(tf.gather_nd(tf.transpose(self.U_Decoder), self.col_idx)))
                        + tf.gather_nd(tf.transpose(self.I_Decoder), self.row_idx)) / 2.0


        # #bpr loss
        # pos_data = tf.gather_nd(self.Decoder, self.input_P_cor)
        # neg_data = tf.gather_nd(self.Decoder, self.input_N_cor)

        # pre_cost1 = tf.maximum(neg_data - pos_data + 0.15,
        #                        tf.zeros(tf.shape(neg_data)[0]))
        # cost_hinge = tf.reduce_sum(pre_cost1)  # prediction squared error
        reconstruction_loss = binary_crossentropy(Input(tensor=inp_tmp_origin),
                                                  outputs_item)

        reconstruction_loss *= self.num_users
        kl_loss = 1 + z_log_varItem - K.square(z_meanItem) - K.exp(z_log_varItem)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.01
        vae_loss_i = K.mean(reconstruction_loss + kl_loss)

        if self.using_hinge == 0:
             self.cost = vae_loss_i + vae_loss_u 
        
        else:
             self.cost = vae_loss_i + vae_loss_u #+ self.beta_value * cost_hinge
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.optimizer = optimizer.minimize(self.cost, global_step=self.global_step)

        

    def train_model(self, itr):
        start_time = time.time()
        random_row_idx = np.random.permutation(self.num_users)  # randomly permute the rows
        random_col_idx = np.random.permutation(self.num_items)  # randomly permute the cols
        batch_cost = 0
        ts = 0

	# SVD
        train_mat = self.train_R
        completion_mat = self.puresvd(R=train_mat)
        #completion_mat = completion_mat + train_mat

	# NMF
        #train_mat = self.train_R
        #completion_mat = MiniBatchNMF(init='random',random_state=0).fit_transform(train_mat)
        
        for i in range(self.num_batch_U):  # iterate each batch
            if i == self.num_batch_U - 1:
                row_idx = random_row_idx[i * self.batch_size:]
            else:
                row_idx = random_row_idx[(i * self.batch_size):((i + 1) * self.batch_size)]
            for j in range(self.num_batch_I):
                # get the indices of the current batch
                if j == self.num_batch_I - 1:
                    col_idx = random_col_idx[j * self.batch_size:]
                else:
                    col_idx = random_col_idx[(j * self.batch_size):((j + 1) * self.batch_size)]

                    p_input, n_input = evaluate.pairwise_neg_sampling(self.train_R, row_idx, col_idx, 1)

                    input_R_U_completion = self.train_R[row_idx, :]
                    input_R_I_completion = self.train_R[:, col_idx]
                    input_R_U = completion_mat[row_idx, :]
                    input_R_I = completion_mat[: , col_idx]
                    
                    # random masked user
                    len_keep = int(self.num_items*0.95)
                    noise = np.random.uniform(0,1,(input_R_U.shape[0],input_R_U.shape[1]))                   
                    ids_shuffle = np.argsort(noise, axis=1)                    
                    ids_restore = np.argsort(ids_shuffle, axis=1)                    
                    ids_keep = ids_shuffle[:,:len_keep]                    
                    index = np.transpose(ids_keep,(1,0))                        
                    x_masked=  input_R_U[np.arange(index.shape[1],dtype=int),index].T

                    #masked choice normal
                    #mask_tokens = np.random.normal(0,0.5,(input_R_U.shape[0], self.num_items-len_keep))

                    #mask choice 1
                    #mask_tokens = np.ones((input_R_U.shape[0], self.num_items-len_keep))

                    #mask choice 0
                    mask_tokens = np.zeros((input_R_U.shape[0], self.num_items-len_keep))
                    
                    # maked choice 0-1 random uniform
                    #mask_tokens = np.random.uniform(0, 0.5,(input_R_U.shape[0], (self.num_items-len_keep)))#dtype=tf.dtypes.float64)
                    x_ = np.concatenate([x_masked,mask_tokens],1)
                    #arg1 = np.arange(0,x_.shape[0])
                    index2 = np.transpose(ids_restore,(1,0))
                    x_ = x_[np.arange(index2.shape[1],dtype=int),index2].T

                    # random masked item
                    #print("input_R_I",input_R_I.shape)
                    y = np.transpose(input_R_I,(1,0))
                    len_keep_I = int(self.num_users*0.95)
                    # print("len_keep", len_keep_I)
                    noise_I = np.random.uniform(0,1,(y.shape[0],y.shape[1])) 
                    # print("noise", noise_I.shape)                  
                    ids_shuffle_I = np.argsort(noise_I, axis=1) 
                    # print(ids_shuffle_I.shape)                   
                    ids_restore_I = np.argsort(ids_shuffle_I, axis=1)                    
                    ids_keep_I = ids_shuffle_I[:,:len_keep_I]  
                    # print("ids_keep",ids_keep_I.shape)                  
                    index_I = np.transpose(ids_keep_I,(1,0))                        
                    y_masked=  y[np.arange(index_I.shape[1],dtype=int),index_I].T

                    #masked choice normal
                    #mask_tokens_I = np.random.normal(0,0.5,(y.shape[0],self.num_users-len_keep_I))

                    #mask choice 1
                    #mask_tokens_I = np.ones((y.shape[0],self.num_users-len_keep_I))

                    #mask choice 1
                    mask_tokens_I = np.zeros((y.shape[0],self.num_users-len_keep_I))
                    
                    # maked choice 0-1 random uniform
                    #mask_tokens_I = np.random.uniform(0, 0.5,(y.shape[0],self.num_users-len_keep_I))#dtype=tf.dtypes.float64)
                    y_ = np.concatenate([y_masked,mask_tokens_I],1)
                    #arg1 = np.arange(0,x_.shape[0])
                    index2_I = np.transpose(ids_restore_I,(1,0))
                    y_ = y_[np.arange(index2_I.shape[1],dtype=int),index2_I].T
                    y_ = np.transpose(y_,(1,0))


                    _, Cost = self.sess.run(
                   [self.optimizer, self.cost],
                   feed_dict={self.input_R_U: input_R_U, self.input_R_I: y_, self.input_U_origin: input_R_U,
                           self.input_I_origin: input_R_I, self.row_idx: np.reshape(row_idx, (len(row_idx), 1)),
                           self.col_idx: np.reshape(col_idx, (len(col_idx), 1))})

     


        print ("Training //", "Epoch %d //" % (itr), " Total cost = {:.2f}".format(batch_cost),
                   "Elapsed time : %d sec" % (time.time() - start_time))

    def test_model(self, itr):
        start_time = time.time()
        Cost, Decoder = self.sess.run(
            [self.cost, self.Decoder],
            feed_dict={self.input_R_U: self.train_R, self.input_R_I: self.train_R, self.input_U_origin: self.train_R,
                       self.input_I_origin: self.train_R, self.row_idx: np.reshape(range(self.num_users), (self.num_users, 1)),
                       self.col_idx: np.reshape(range(self.num_items), (self.num_items, 1))})

        self.test_cost_list.append(Cost)

        evaluate.test_model_all(Decoder, self.test_R, self.train_R)

        print("Testing //", "Epoch %d //" % (itr), " Total cost = {:.2f}".format(Cost),
              "Elapsed time : %d sec" % (time.time() - start_time))
        print("=" * 100)
