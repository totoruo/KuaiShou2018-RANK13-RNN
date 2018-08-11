import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import os,time
import tensorflow as tf
from utils import *

class my_model():
    def __init__(self,num_feat,time_stage,epoch=2,batch_size=64,learning_rate=0.001,random_seed=1011,
                 hidden_size=[50,50],num_layers=2):
        self.num_feat = num_feat
        self.time_stage = time_stage
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._init_graph()
        
    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            initializer=tf.random_uniform_initializer(-0.1, 0.1)
        
        self.feat_seq = tf.placeholder(tf.float32, [None, self.time_stage, self.num_feat], name='feat_seq')
        self.label_seq = tf.placeholder(tf.int32, [None, self.time_stage], name='label_seq')
        self.register_type = tf.placeholder(tf.int32, (None,), name='register_type')
        self.device_type = tf.placeholder(tf.int32, (None,), name='device_type')
        self.seq_length = tf.placeholder(tf.int32, (None,), name='seq_length')
        self.train_phase = tf.placeholder(tf.bool, name="train_phase")
        

        cell = tf.nn.rnn_cell.LSTMCell(50,state_is_tuple=True, initializer=initializer)
        def get_lstm_cell(rnn_size):
            lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=initializer)
            return lstm_cell
        # multi_cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(self.hidden_size[i]) for i in range(self.num_layers)])
        output, state = tf.nn.dynamic_rnn(cell, self.feat_seq, dtype=tf.float32, sequence_length=self.seq_length)
        # output: -1*time_stage*rnn_size
        
        
        regType_emb = tf.gather(tf.Variable(tf.truncated_normal(shape=[12,1], mean=0.0, stddev=0.0001)), self.register_type)
        regType_emb = tf.tile(regType_emb,[1,self.time_stage])
        regType_emb = tf.expand_dims(regType_emb, -1)
        
        devType_emb = tf.gather(tf.Variable(tf.truncated_normal(shape=[2000,1], mean=0.0, stddev=0.0001)), self.device_type)
        devType_emb = tf.tile(devType_emb,[1,self.time_stage])
        devType_emb = tf.expand_dims(devType_emb, -1)
        
        output = tf.concat([output,regType_emb],axis=-1)
        output = tf.concat([output,devType_emb],axis=-1) 
        
        output = tf.reshape(output, [-1, self.hidden_size[-1]+2])

        w2 = tf.Variable(tf.random_uniform([self.hidden_size[-1]+2, 2], -0.1, 0.1))
        b2 = tf.Variable(tf.random_uniform([2], -0.1, 0.1))
        logits = tf.matmul(output, w2) + b2
        logits = tf.reshape(logits, [-1, self.time_stage, 2])
        
        # loss ignore last 7 days
        masks = tf.sequence_mask(self.seq_length-7, self.time_stage-7, dtype=tf.float32, name='masks')
        paddings = tf.constant([[0, 0,], [0, 7]])
        masks = tf.pad(masks,paddings)
        loss = tf.contrib.seq2seq.sequence_loss(logits,self.label_seq,masks)
        self.loss = tf.reduce_sum(loss)
        
        # last out
        batch_range = tf.range(tf.shape(logits)[0])
        ind = self.seq_length - 1
        indices = tf.stack([batch_range, ind], axis=1)
        logits = tf.gather_nd(logits,indices)
        self.out = tf.nn.softmax(logits)
                
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                epsilon=1e-8).minimize(self.loss)
        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess = self._init_session()
        self.sess.run(init)
        
    def _init_session(self):
        config = tf.ConfigProto()
        return tf.Session(config=config)
        
    def get_batch(self,feat_seq,label_seq,seq_length,register_type,device_type,batch_size,index):
        start = index * batch_size
        end = (index+1) * batch_size
        end = end if end < len(feat_seq) else len(feat_seq)
        return feat_seq[start:end],label_seq[start:end],seq_length[start:end],register_type[start:end],device_type[start:end]
    
    def fit_on_batch(self, feat_seq,label_seq,seq_length,register_type,device_type):
        feed_dict = {self.feat_seq: feat_seq,
                     self.label_seq: label_seq,
                     self.seq_length: seq_length,
                     self.register_type:register_type,
                     self.device_type:device_type,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss
    
    def fit(self,feat_seq,label_seq,seq_length,register_type,device_type):
        for epoch in range(self.epoch):
            total_loss = 0.0
            total_size = 0.0
            batch_begin_time = time.time()
            t1 = time.time()
            total_batch = int(len(feat_seq) / self.batch_size)
            for i in range(total_batch):
                offset = i * self.batch_size
                end = (i+1) * self.batch_size
                end = end if end < len(feat_seq) else len(feat_seq)
                _feat_seq,_label_seq,_seq_length,_register_type,_device_type\
                    = self.get_batch(feat_seq,label_seq,seq_length,register_type,device_type,self.batch_size,i)
                batch_loss = self.fit_on_batch(_feat_seq, _label_seq, _seq_length,_register_type,_device_type)
                total_loss += batch_loss * (end - offset)
                total_size += end - offset
                if i % 100 == 99:
                    print('[%d, %5d] loss: %.6f time: %.1f s' %
                          (epoch + 1, i + 1, total_loss / total_size, time.time() - batch_begin_time))
                    total_loss = 0.0
                    total_size = 0.0
                    batch_begin_time = time.time()
                    
    def predict(self,feat_seq,seq_length,register_type,device_type,y = []):
        if len(y) == 0:
            label_seq = np.zeros([feat_seq.shape[0],feat_seq.shape[1]])
        else:
            label_seq = y
        batch_index = 0
        batch_size = 4096
        _feat_seq,_label_seq,_seq_length,_register_type,_device_type\
            = self.get_batch(feat_seq,label_seq,seq_length,register_type,device_type,batch_size,batch_index)
        y_pred = None
        total_loss = 0.0
        total_size = 0.0
        while len(_seq_length) > 0:
            num_batch = len(_seq_length)
            feed_dict = {self.feat_seq: _feat_seq,
                         self.label_seq: _label_seq,
                         self.seq_length: _seq_length,
                         self.register_type:_register_type,
                         self.device_type:_device_type,
                         self.train_phase: False}
            batch_out, batch_loss = self.sess.run((self.out, self.loss), feed_dict=feed_dict)
            total_loss += batch_loss * num_batch
            total_size += num_batch
            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,2,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,2,))))
            batch_index += 1
            _feat_seq,_label_seq,_seq_length,_register_type,_device_type\
                = self.get_batch(feat_seq,label_seq,seq_length,register_type,device_type,batch_size,batch_index)
        print("valid logloss is %.6f" % (total_loss / total_size))
        print("predict end")
        return y_pred

# 线下
offline_label_seq,offline_seq_length = gen_label(1,23)
offline_lanuch_seq = gen_day_seq(1,23,'launch')
offline_video_seq = gen_day_seq(1,23,'video')
offline_reg_seq = gen_day_seq(1,23,'reg')
offline_act0_seq = gen_day_seq(1,23,'act','action_type',0)
offline_act1_seq = gen_day_seq(1,23,'act','action_type',1)
offline_act2_seq = gen_day_seq(1,23,'act','action_type',2)
offline_act3_seq = gen_day_seq(1,23,'act','action_type',3)
offline_act4_seq = gen_day_seq(1,23,'act','action_type',4)
offline_act5_seq = gen_day_seq(1,23,'act','action_type',5)
offline_page0_seq = gen_day_seq(1,23,'act','page',0)
offline_page1_seq = gen_day_seq(1,23,'act','page',1)
offline_page2_seq = gen_day_seq(1,23,'act','page',2)
offline_page3_seq = gen_day_seq(1,23,'act','page',3)
offline_page4_seq = gen_day_seq(1,23,'act','page',4)

offline_data = np.concatenate((offline_lanuch_seq.reshape(-1,1),offline_video_seq.reshape(-1,1)),axis=1)
offline_data = np.concatenate((offline_data.reshape(-1,2),offline_reg_seq.reshape(-1,1)),axis=1)
offline_data = np.concatenate((offline_data.reshape(-1,3),offline_act0_seq.reshape(-1,1)),axis=1)
offline_data = np.concatenate((offline_data.reshape(-1,4),offline_act1_seq.reshape(-1,1)),axis=1)
offline_data = np.concatenate((offline_data.reshape(-1,5),offline_act2_seq.reshape(-1,1)),axis=1)
offline_data = np.concatenate((offline_data.reshape(-1,6),offline_act3_seq.reshape(-1,1)),axis=1)
offline_data = np.concatenate((offline_data.reshape(-1,7),offline_act4_seq.reshape(-1,1)),axis=1)
offline_data = np.concatenate((offline_data.reshape(-1,8),offline_act5_seq.reshape(-1,1)),axis=1)
offline_data = np.concatenate((offline_data.reshape(-1,9),offline_page0_seq.reshape(-1,1)),axis=1)
offline_data = np.concatenate((offline_data.reshape(-1,10),offline_page1_seq.reshape(-1,1)),axis=1)
offline_data = np.concatenate((offline_data.reshape(-1,11),offline_page2_seq.reshape(-1,1)),axis=1)
offline_data = np.concatenate((offline_data.reshape(-1,12),offline_page3_seq.reshape(-1,1)),axis=1)
offline_data = np.concatenate((offline_data.reshape(-1,13),offline_page4_seq.reshape(-1,1)),axis=1)
offline_data = offline_data.reshape([-1,23,14])

sub = register[register.day<=23]
truth = gen_truth(24)
sub['device_type'] = np.where(sub['device_type']<1999,sub['device_type'],1999)
offline_register_type = sub['register_type'].values
offline_device_type = sub['device_type'].values
sub = sub[['user_id']].copy()
sub = sub.merge(truth,'left','user_id')
sub = sub.fillna(0)

tf.reset_default_graph()
model = my_model(num_feat=14,time_stage=23,epoch=35,batch_size=512,learning_rate=0.001,num_layers=2)
sub['pre'] = pre[:,1:2]
print(roc_auc_score(sub['label'],sub['pre']))

# 线上部分
train_label_seq,train_label_length = gen_label(1,30)
train_lanuch_seq = gen_day_seq(1,30,'launch')
train_video_seq = gen_day_seq(1,30,'video')
train_reg_seq = gen_day_seq(1,30,'reg')
train_act0_seq = gen_day_seq(1,30,'act','action_type',0)
train_act1_seq = gen_day_seq(1,30,'act','action_type',1)
train_act2_seq = gen_day_seq(1,30,'act','action_type',2)
train_act3_seq = gen_day_seq(1,30,'act','action_type',3)
train_act4_seq = gen_day_seq(1,30,'act','action_type',4)
train_act5_seq = gen_day_seq(1,30,'act','action_type',5)
train_page0_seq = gen_day_seq(1,30,'act','page',0)
train_page1_seq = gen_day_seq(1,30,'act','page',1)
train_page2_seq = gen_day_seq(1,30,'act','page',2)
train_page3_seq = gen_day_seq(1,30,'act','page',3)
train_page4_seq = gen_day_seq(1,30,'act','page',4)

train_data = np.concatenate((train_lanuch_seq.reshape(-1,1),train_video_seq.reshape(-1,1)),axis=1)
train_data = np.concatenate((train_data.reshape(-1,2),train_reg_seq.reshape(-1,1)),axis=1)
train_data = np.concatenate((train_data.reshape(-1,3),train_act0_seq.reshape(-1,1)),axis=1)
train_data = np.concatenate((train_data.reshape(-1,4),train_act1_seq.reshape(-1,1)),axis=1)
train_data = np.concatenate((train_data.reshape(-1,5),train_act2_seq.reshape(-1,1)),axis=1)
train_data = np.concatenate((train_data.reshape(-1,6),train_act3_seq.reshape(-1,1)),axis=1)
train_data = np.concatenate((train_data.reshape(-1,7),train_act4_seq.reshape(-1,1)),axis=1)
train_data = np.concatenate((train_data.reshape(-1,8),train_act5_seq.reshape(-1,1)),axis=1)
train_data = np.concatenate((train_data.reshape(-1,9),train_page0_seq.reshape(-1,1)),axis=1)
train_data = np.concatenate((train_data.reshape(-1,10),train_page1_seq.reshape(-1,1)),axis=1)
train_data = np.concatenate((train_data.reshape(-1,11),train_page2_seq.reshape(-1,1)),axis=1)
train_data = np.concatenate((train_data.reshape(-1,12),train_page3_seq.reshape(-1,1)),axis=1)
train_data = np.concatenate((train_data.reshape(-1,13),train_page4_seq.reshape(-1,1)),axis=1)
train_data = train_data.reshape([-1,30,14])

res = register[register.day<=30]
res['device_type'] = np.where(res['device_type']<1999,res['device_type'],1999)

register_type = res['register_type'].values
device_type = res['device_type'].values
res = res[['user_id']].copy()

tf.reset_default_graph()
model = my_model(num_feat=14,time_stage=30,epoch=30,batch_size=512,learning_rate=0.001,num_layers=2)
model.fit(train_data,train_label_seq,train_label_length,register_type,device_type)
pre = model.predict(train_data,train_label_length,register_type,device_type)
res['pre'] = pre[:,1:2].reshape([-1])
res.to_csv('submit.txt',index=False,header=None)

