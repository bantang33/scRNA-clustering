from tensorflow.keras.layers import GaussianNoise, Dense, Activation, BatchNormalization
from tensorflow.keras import Sequential, layers
from preprocess import *
from utils import *
from sklearn.cluster import KMeans
from loss import *
import os
import argparse
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tensorflow.python.framework import ops
from scipy.optimize import linear_sum_assignment
from tensorflow.keras import regularizers
from collections import Counter
import collections
import sys
ops.reset_default_graph()
 

tf.enable_eager_execution()
os.environ['CUDA_VISIBLE_DEVICES']="-1"


def parse_args():
    random_seed = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 10000]

    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataname", default = "Quake_10x_Trachea", type = str)
    parser.add_argument("--distribution", default = "ZINB")
    parser.add_argument("--t_alpha", default = 1.0)
    parser.add_argument("--dims", default = [500, 256, 64, 32])
    parser.add_argument("--highly_genes", default = 500)
    parser.add_argument("--learning_rate", default = 0.0001, type = float)
    parser.add_argument("--random_seed", default = random_seed)
    parser.add_argument("--batch_size", default = 256, type = int)
    parser.add_argument("--noise_sd", default = 1.5)
    return parser.parse_args()




MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)
temperature=0.07
args = parse_args()
X, Y = prepro(args.dataname)
X = np.ceil(X).astype(np.int)
count_X = X
adata = sc.AnnData(X)
adata.obs['Group'] = Y
adata = normalize(adata, copy=True, highly_genes=args.highly_genes, size_factors=True, normalize_input=True, logtrans_input=True)
X = adata.X.astype(np.float32)
Y = np.array(adata.obs["Group"])
high_variable = np.array(adata.var.highly_variable.index, dtype=np.int)
count_X = count_X[:, high_variable]
unique, counts = np.unique(count_X, return_counts=True)
#print("X.size=",X.size,"countx.size=",count_X.size,"Y.size=",Y.size)
size_factor = np.array(adata.obs.size_factors).reshape(-1, 1).astype(np.float32)  #细胞数
#print("size_factor=",size_factor)
cluster_number = int(max(Y) - min(Y) + 1)
train_batchsize=200


def _nan2zero(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x), x)

def _nan2inf(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x)+np.inf, x)

def _nelem(x):
    nelem = tf.reduce_sum(tf.cast(~tf.is_nan(x), tf.float32))
    return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)

def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return tf.divide(tf.reduce_sum(x), nelem)

def cosin(tensor1,tensor2):
    """
    tensor1:一维张量
    tensor2:二维张量
    """
    # 求模长
    tensor1_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor1)))
    tensor2_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor2),axis=1))
    # 内积
    tensor1_tensor2 = tf.reduce_sum(tf.multiply(tensor1,tensor2),axis=1)
    cosin = tensor1_tensor2/(tensor1_norm*tensor2_norm+1e-10)
    return cosin

def element(cosine,temperature):
    return tf.exp(cosine/temperature)


def NB(theta, y_true, y_pred, mask = False, debug = False, mean = False):
    eps = 1e-10
    scale_factor = 1.0
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32) * scale_factor
    if mask:
        nelem = _nelem(y_true)
        y_true = _nan2zero(y_true)
    theta = tf.minimum(theta, 1e6)
    t1 = tf.lgamma(theta + eps) + tf.lgamma(y_true + 1.0) - tf.lgamma(y_true + theta + eps)
    t2 = (theta + y_true) * tf.log(1.0 + (y_pred / (theta + eps))) + (y_true * (tf.log(theta + eps) - tf.log(y_pred + eps)))
    if debug:
        assert_ops = [tf.verify_tensor_all_finite(y_pred, 'y_pred has inf/nans'),
                      tf.verify_tensor_all_finite(t1, 't1 has inf/nans'),
                      tf.verify_tensor_all_finite(t2, 't2 has inf/nans')]
        with tf.control_dependencies(assert_ops):
            final = t1 + t2
    else:
        final = t1 + t2
    final = _nan2inf(final)
    if mean:
        if mask:
            final = tf.divide(tf.reduce_sum(final), nelem)
        else:
            final = tf.reduce_mean(final)
    return final

def ZINB(pi, theta, y_true, y_pred, ridge_lambda, mean = True, mask = False, debug = False):
    eps = 1e-10
    scale_factor = 1.0
    nb_case = NB(theta, y_true, y_pred, mean=False, debug=debug) - tf.log(1.0 - pi + eps)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32) * scale_factor
    theta = tf.minimum(theta, 1e6)

    zero_nb = tf.pow(theta / (theta + y_pred + eps), theta)
    zero_case = -tf.log(pi + ((1.0 - pi) * zero_nb) + eps)
    result = tf.where(tf.less(y_true, 1e-8), zero_case, nb_case)
    ridge = ridge_lambda * tf.square(pi)
    result += ridge
    if mean:
        if mask:
            result = _reduce_mean(result)
        else:
            result = tf.reduce_mean(result)

    result = _nan2inf(result)
    return result



#实例损失
def Instanceloss(tensor1,tensor2,temperature):
    shape = tensor1.get_shape().as_list()
    N=shape[0]
    tensor1_norm=tf.nn.l2_normalize(tensor1,axis=1)
    tensor2_norm=tf.nn.l2_normalize(tensor2,axis=1)
    tensor=tf.concat([tensor1_norm,tensor2_norm],0)
    tensor_sim=tf.matmul(tensor,tf.transpose(tensor))  #200*200
    positive_sim=tf.reduce_mean(tf.diag_part(tensor_sim[0:N,N:2*N]))  #正例对相似性，N对,
    negative_sim=tf.reduce_mean(tensor_sim[0:N,0:N])+tf.reduce_mean(tensor_sim[N:2*N,N:2*N])+tf.reduce_mean(tensor_sim[0:N,N:2*N])-tf.reduce_mean(tf.diag_part(tensor_sim))-tf.reduce_mean(tf.diag_part(tensor_sim[0:N,N:2*N]))    #a和a的负例对,b和b的负例对,a和b的负例对,减去对角,即自己和自己 
   
    tensor_exp_t=tf.exp(tensor_sim/temperature)
    zeros=tf.zeros([N,N],tf.float32)
    eye=tf.eye(N,N)
    mask=tf.concat([tf.concat([zeros,eye],0),tf.concat([eye,zeros],0)],1)
    loss=tf.reduce_sum(-tf.math.log(tf.reduce_sum(tensor_exp_t*mask,1)/tf.reduce_sum(tensor_exp_t,1)))
    return loss/(2*N),positive_sim,negative_sim,tensor_sim


def InstancelossPositivate(tensor1,tensor2,temperature):
    shape = tensor1.get_shape().as_list()
    N=shape[0]
    tensor1_norm=tf.nn.l2_normalize(tensor1,axis=1)
    tensor2_norm=tf.nn.l2_normalize(tensor2,axis=1)
    tensor=tf.concat([tensor1_norm,tensor2_norm],0)
    tensor_sim=tf.matmul(tensor,tf.transpose(tensor))  #200*200
    positive_sim=tf.reduce_mean(tf.diag_part(tensor_sim[0:N,N:2*N]))  #正例对相似性，N对,
    negative_sim=tf.reduce_mean(tensor_sim[0:N,0:N])+tf.reduce_mean(tensor_sim[N:2*N,N:2*N])+tf.reduce_mean(tensor_sim[0:N,N:2*N])-tf.reduce_mean(tf.diag_part(tensor_sim))-tf.reduce_mean(tf.diag_part(tensor_sim[0:N,N:2*N]))    #a和a的负例对,b和b的负例对,a和b的负例对,减去对角,即自己和自己

    tensor_exp_t=tf.exp(tensor_sim/temperature)
    zeros=tf.zeros([N,N],tf.float32)
    eye=tf.eye(N,N)
    mask=tf.concat([tf.concat([zeros,eye],0),tf.concat([eye,zeros],0)],1)
#    loss=tf.reduce_sum(-tf.math.log(tf.reduce_sum(tensor_exp_t*mask,1)/tf.reduce_sum(tensor_exp_t,1)))
#    loss=tf.reduce_sum(-tf.math.log(tf.reduce_sum(tensor_exp_t*mask,1)))
#    return loss/(2*N),positive_sim,negative_sim,tensor_sim
    loss=tf.maximum(0.001-tf.reduce_sum(-tf.math.log(tf.reduce_sum(tensor_sim*mask,1)))/(2*N),0.0)
    return loss,positive_sim,negative_sim,tensor_sim


#聚类损失
def Clusterloss(tensor1,tensor2,temperature):
    tensor1=tf.transpose(tensor1)
    tensor2=tf.transpose(tensor2)
    shape = tensor1.get_shape().as_list()
    M=shape[0]
    tensor1_norm=tf.nn.l2_normalize(tensor1,axis=1)
    tensor2_norm=tf.nn.l2_normalize(tensor2,axis=1)
    tensor=tf.concat([tensor1_norm,tensor2_norm],0)
    tensor_sim=tf.matmul(tensor,tf.transpose(tensor))
    positive_sim=tf.reduce_mean(tf.diag_part(tensor_sim[0:M,M:2*M]))  #正例对相似性，N对
    negative_sim=tf.reduce_mean(tensor_sim[0:M,0:M])+tf.reduce_mean(tensor_sim[M:2*M,M:2*M])+tf.reduce_mean(tensor_sim[0:M,M:2*M])-tf.reduce_mean(tf.diag_part(tensor_sim))-tf.reduce_mean(tf.diag_part(tensor_sim[0:M,M:2*M]))    #a和a的负例对,b和b的负例对,a和b的负例对,减去对角,即自己和自己 
  
    tensor_exp_t=tf.exp(tensor_sim/temperature)
    zeros=tf.zeros([M,M],tf.float32)
    eye=tf.eye(M,M)
    mask=tf.concat([tf.concat([zeros,eye],0),tf.concat([eye,zeros],0)],1)
    loss=tf.reduce_sum(-tf.math.log(tf.reduce_sum(tensor_exp_t*mask,1)/tf.reduce_sum(tensor_exp_t,1)))
    
    #p1=tf.reduce_sum(tensor1,0)/(tf.reduce_sum(tensor1)+1e-10)  
    #p2=tf.reduce_sum(tensor2,0)/(tf.reduce_sum(tensor2)+1e-10)
    #h=tf.reduce_sum(p1*tf.math.log(p1+ 1e-10))+tf.reduce_sum(p2*tf.math.log(p2+1e-10))
    
    return loss/(2*M),positive_sim,negative_sim,tensor_sim

#行的约束
def rowH(tensor1,tensor2):
    p1_max=tf.reduce_max(tensor1, 1)  #对行求最大值
    p2_max=tf.reduce_max(tensor2, 1)
    h=(1-p1_max)+(1-p2_max)
    h_m=tf.reduce_mean(h)
    return h_m,p1_max,p2_max

#求列的熵，熵最大
def colH(tensor1,tensor2):
    p1=tf.reduce_sum(tensor1,0)/(tf.reduce_sum(tensor1)+1e-10)  #对列求和
    p2=tf.reduce_sum(tensor2,0)/(tf.reduce_sum(tensor2)+1e-10)
    hh=-tf.reduce_sum(p1*tf.math.log(p1+ 1e-10))-tf.reduce_sum(p2*tf.math.log(p2+1e-10))  #熵
    return hh,p1




def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

class ParaAE(tf.keras.models.Model):
    def __init__(self,distribution, dims,learning_rate, noise_sd,t_alpha,dataname,init='glorot_uniform', act='relu'):
        super(ParaAE, self).__init__()

        self.distribution = distribution
        self.dims = dims
        self.learning_rate = learning_rate
        self.noise_sd = noise_sd
        self.t_alpha = t_alpha
        self.dataname = dataname
        self.init = init
        self.act = act
        self.n_stacks = len(self.dims) - 1
        encoder_array=dims[0]
        #self.clusters = tf.get_variable(name=self.dataname + "/clusters_rep", shape=[cluster_number, encoder_array[-1]],dtype=tf.float32, initializer=tf.glorot_uniform_initializer())


        #编码器
        center = encoder_array.pop()
        input_shape=encoder_array[0]
        self.encoder=Sequential()
        for num in encoder_array[1:]:
            self.encoder.add(layers.Dense(num, activation=tf.nn.relu,activity_regularizer=regularizers.l1(10e-5)))
            #self.encoder.add(layers.BatchNormalization())

        self.encoder.add(layers.Dense(center))

        self.decoder_list=dims[1]
        self.decoder=list()
        count=0
        for de_list in self.decoder_list:
            self.decoder.append(Sequential())
            for num in de_list:
                self.decoder[count].add(layers.Dense(num, activation=tf.nn.relu,activity_regularizer=regularizers.l1(10e-5)))
                #self.decoder[count].add(layers.BatchNormalization())
            self.decoder[count].add(layers.Dense(input_shape,activation=tf.nn.relu))
            count=count+1


         #求参数
        self.piDense = Dense(units=input_shape, activation='sigmoid', kernel_initializer=self.init, name="pi")
        self.dispDense = Dense(units=input_shape, activation=DispAct, kernel_initializer=self.init, name="dispersion")
        self.meanDense = Dense(units=input_shape, activation=MeanAct, kernel_initializer=self.init, name="mean")

       #聚类层
        self.embDense1=Dense(units=16,name="embDense1")
        self.embDense2=Dense(units=cluster_number,activation=tf.nn.softmax,name="embDense2")



    def __call__(self,x,x_count,sizefactor):
        self.x_count=x_count
        self.sf=sizefactor
        self.h = x

        #编码，求z
        self.latent = self.encoder(x)
        self.latentout=tf.nn.l2_normalize(self.latent,axis=1)
	
 
        #计算目标分布与辅助分布
        '''
        self.num, self.latent_p = cal_latent(self.latent, self.t_alpha)
        self.latent_q = target_dis(self.latent_p)
        self.latent_p = self.latent_p + tf.linalg.diag(tf.linalg.diag_part(self.num))
        self.latent_q = self.latent_q + tf.linalg.diag(tf.linalg.diag_part(self.num))
        self.latent_dist1, self.latent_dist2 = cal_dist(self.latent, self.clusters)
        '''
        #解码
        x_hat=list()
        self.likelihood_loss=tf.zeros(1)
        for i in range(len(self.decoder_list)):
            self.h=self.decoder[i](self.latent)
            self.pi = self.piDense(self.h)
            self.disp = self.dispDense(self.h)
            self.mean = self.meanDense(self.h)
            self.out = self.mean * tf.matmul(self.sf, tf.ones((1, self.mean.get_shape()[1]), dtype=tf.float32))
            self.likelihood_loss = self.likelihood_loss+ZINB(self.pi, self.disp, self.x_count, self.out, ridge_lambda=1.0)  #pi,theta,x,u
            x_hat.append(self.likelihood_loss)
    
      
        #求参数

        #kl损失
        '''
        self.cross_entropy = -tf.reduce_sum(self.latent_q * tf.log(self.latent_p))
        self.entropy = -tf.reduce_sum(self.latent_q * tf.log(self.latent_q))
        self.kl_loss = self.cross_entropy - self.entropy

        self.kmeans_loss = tf.reduce_mean(tf.reduce_sum(self.latent_dist2, axis=1))
        '''
        self.kl_loss=tf.zeros(1)
        self.kmeans_loss=tf.zeros(1)
        return self.latent,self.likelihood_loss,self.kl_loss,self.kmeans_loss


'''
#自编码器
class autoencoder(tf.keras.models.Model):
    def __init__(self,distribution, dims,learning_rate, noise_sd,t_alpha,dataname,init='glorot_uniform', act='relu'):
        super(autoencoder, self).__init__()

        self.distribution = distribution
        self.dims = dims
        self.learning_rate = learning_rate
        self.noise_sd = noise_sd
        self.t_alpha = t_alpha
        self.dataname = dataname
        self.init = init
        self.act = act
        self.n_stacks = len(self.dims) - 1
        self.clusters = tf.get_variable(name=self.dataname + "/clusters_rep", shape=[cluster_number, self.dims[-1]],dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
        
        #编码器
        self.encoderDense1 = Dense(units=self.dims[1], activation=self.act,kernel_initializer=self.init, name="encoderDense1")
        self.encoderDense2 = Dense(units=self.dims[2], activation=self.act,kernel_initializer=self.init, name="encoderDense2")
        #self.encoderDense3 = Dense(units=self.dims[3], activation=self.act,kernel_initializer=self.init, name="encoderDense3")
        self.encoderDense3 = Dense(units=self.dims[3],kernel_initializer=self.init, name="encoderDense3")  #z

        #解码
        self.decoderDense1 = Dense(units=self.dims[2], activation=self.act,kernel_initializer=self.init, name="decoderDense1")
        self.decoderDense2 = Dense(units=self.dims[1], activation=self.act,kernel_initializer=self.init, name="decoderDense2")
       # self.decoderDense3 = Dense(units=self.dims[1], activation=self.act,kernel_initializer=self.init, name="decoderDense3")
     
         #求参数
        self.piDense = Dense(units=self.dims[0], activation='sigmoid', kernel_initializer=self.init, name="pi")
        self.dispDense = Dense(units=self.dims[0], activation=DispAct, kernel_initializer=self.init, name="dispersion")
        self.meanDense = Dense(units=self.dims[0], activation=MeanAct, kernel_initializer=self.init, name="mean")

       #聚类层
        self.embDense1=Dense(units=16,name="embDense1")
        self.embDense2=Dense(units=cluster_number,activation=tf.nn.softmax,name="embDense2")



    def __call__(self,x,x_count,sizefactor):
        self.x_count=x_count
        self.sf=sizefactor
        self.h = x
        #self.h = GaussianNoise(self.noise_sd, name='input_noise')(self.h)
	
	#编码，求z
        self.h = self.encoderDense1(self.h)
        self.h = self.encoderDense2(self.h)
     
        #self.h = self.encoderDense3(self.h)
        self.latent = self.encoderDense3(self.h)
        #self.latentout = tf.norm(self.latent,1)
        self.latentout=tf.nn.l2_normalize(self.latent,axis=1)
    


        #聚类,全连接层
        #self.m=self.latent
        #self.dense1 = self.embDense1(self.m)
        #self.dense1= Activation(self.act)(self.dense1)
        #self.clusteremb= self.embDense2(self.dense1)
    
        

        #计算目标分布与辅助分布
        self.num, self.latent_p = cal_latent(self.latent, self.t_alpha)
        self.latent_q = target_dis(self.latent_p)
        self.latent_p = self.latent_p + tf.linalg.diag(tf.linalg.diag_part(self.num))
        self.latent_q = self.latent_q + tf.linalg.diag(tf.linalg.diag_part(self.num))
        self.latent_dist1, self.latent_dist2 = cal_dist(self.latent, self.clusters)

        #解码
        #self.h=self.latent
        self.h=self.latent
        self.h = self.decoderDense1(self.h)
        self.h = self.decoderDense2(self.h)
        #self.h = self.decoderDense3(self.h)
    
      
        #求参数
        self.pi = self.piDense(self.h)
        self.disp = self.dispDense(self.h)
        self.mean = self.meanDense(self.h)
        self.out = self.mean * tf.matmul(self.sf, tf.ones((1, self.mean.get_shape()[1]), dtype=tf.float32))
        self.likelihood_loss = ZINB(self.pi, self.disp, self.x_count, self.out, ridge_lambda=1.0)  #pi,theta,x,u

        #kl损失
        self.cross_entropy = -tf.reduce_sum(self.latent_q * tf.log(self.latent_p))
        self.entropy = -tf.reduce_sum(self.latent_q * tf.log(self.latent_q))
        self.kl_loss = self.cross_entropy - self.entropy

        self.kmeans_loss = tf.reduce_mean(tf.reduce_sum(self.latent_dist2, axis=1))


        return self.latent,self.likelihood_loss,self.kl_loss,self.kmeans_loss
'''

#数据增强
class augModel(tf.keras.models.Model):
    def __init__(self,rate, noise_shape=None, seed=None, **kwargs):
        super(augModel,self).__init__(**kwargs)
        self.Dropout=tf.keras.layers.Dropout(rate)

    def call(self,inputs,*args,**kwargs):
        augdata = self.Dropout(inputs)
        return augdata

class augModelNormal(tf.keras.models.Model):
    def __init__(self,scale, noise_shape=None, seed=None, **kwargs):
        super(augModelNormal,self).__init__(**kwargs)
        self.scale=scale

    def call(self,inputs,*args,**kwargs):
        augdata =  tf.add(inputs, self.scale*tf.random_normal(shape = inputs.shape))
        return augdata


#argmodel=augModel(rate=0.8)    #dropout率
X=tf.convert_to_tensor(X,dtype='float32')   #转成浮点型
sys.stdout.flush()
#count_X=tf.convert_to_tensor(count_X,dtype='float32')
#layer=tf.keras.layers.Dropout(rate=0)
#layer1=tf.keras.layers.GaussianNoise(args.noise_sd)

#argmodelDropout=augModel(rate=0.8)
#argdata1=argmodelDropout(X)   #增强1
argmodelNormal05=augModelNormal(0.05)
argdata1=argmodelNormal05(X)   #增强2
argmodelNormal=augModelNormal(0.1)
argdata2=argmodelNormal(X)   #增强2

#print("arg1=",argdata1[0:50,0:50])   #输出增强数据
#print("arg2=",argdata2)

#sys.stdout.flush()
#same?

#数据打乱
trainDataset = tf.data.Dataset.from_tensor_slices((argdata1,argdata2,count_X,size_factor)).shuffle(100).batch(train_batchsize)
#trainDataset = tf.data.Dataset.from_tensor_slices((X,count_X,size_factor)).shuffle(100).batch(train_batchsize)

#定义模型
#args.dims[3]=args.cluster_number
#model = autoencoder(args.distribution, args.dims, args.learning_rate,args.noise_sd,args.t_alpha,args.dataname)
model = ParaAE(args.distribution, [args.dims,[[64,256]]], args.learning_rate,args.noise_sd,args.t_alpha,args.dataname)

optimizer=tf.keras.optimizers.Adam(0.0001)
#print("DB:",args.dataname,cluster_number)
#exit()
#tf.config.run_functions_eagerly(True)
#kmeans = KMeans(n_clusters=cluster_number, init="k-means++")
kmeans = KMeans(n_clusters=cluster_number, init="k-means++", random_state=0)
Y_pred=model(X,count_X,size_factor)
pre_kmeans_pred = kmeans.fit_predict(Y_pred[0])

for epoch in range(300):
    #argdata1=argmodel(X)   #增强1
    #argdata2=argmodel(X)   #增强2
    #数据打乱
    #trainDataset = tf.data.Dataset.from_tensor_slices((argdata1,argdata2,count_X,size_factor)).shuffle(100).batch(train_batchsize)
    for step, (inputTensorA, inputTensorB,x_count,sizefactor) in enumerate(trainDataset):
#    for step, (inputTensorA,x_count,sizefactor) in enumerate(trainDataset):
        with tf.GradientTape() as tape:
            Y1 = model(inputTensorA,x_count,sizefactor)
            Y2 = model(inputTensorB,x_count,sizefactor)
            ##insloss = Instanceloss(Y1[0], Y2[0], temperature) 
            insloss = InstancelossPositivate(Y1[0], Y2[0], temperature)
            likeloss=(Y1[1] + Y2[1])    #似然损失
            kl_loss=Y1[2]+Y2[2]     #kl散度
            k_meansloss=Y1[3]+Y2[3]
            #likeloss=Y1[1]
            #print(insloss[1])
            #row_loss=rowH(Y1[0],Y2[0])
            #print(row_loss)
            loss = likeloss+insloss[0]*3#*0.01#+row_loss
            '''
            if likeloss < 0.0500:
                loss = likeloss+insloss[0]*0.01#+row_loss
            else:
                loss = likeloss
            '''                
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if step % 50 == 0:
            Y_pred=model(X,count_X,size_factor)
            kmeans_pred = kmeans.fit_predict(Y_pred[0])
            kaccuracy = np.around(cluster_acc(Y, kmeans_pred), 5)
            kARI = np.around(adjusted_rand_score(Y, kmeans_pred), 5)
            kNMI = np.around(normalized_mutual_info_score(Y, kmeans_pred), 5)
            pre_KNMI=np.around(normalized_mutual_info_score(pre_kmeans_pred, kmeans_pred), 5)
            pre_kmeans_pred=kmeans_pred
#            print("epoch",epoch + 1)
#            print("likeloss=", likeloss.numpy())
#            print("rowloss=",row_loss[0].numpy())
#            print("InstPosLoss=", insloss[0].numpy())
            print(args.dataname,"Epoch:",epoch+1,"likeloss=", likeloss.numpy(),"InstPosLoss=", insloss[0].numpy(),"KLloss=",kl_loss.numpy(),"kaccuracy=", kaccuracy,"kARI=",kARI, "kNMI=",kNMI,"pre_KNMI",pre_KNMI)
            sys.stdout.flush()
