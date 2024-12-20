# -*- coding: UTF-8 -*-

import math
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import time
import datetime
from sklearn.metrics import mean_squared_error
import sys
import argparse
# import files from parent directory
parent_directory = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_directory)
from models.RGBlimp_dynamics_ABNODE_p1 import *
from models.RGBlimp_dynamics import *
from utils.solvers import *
from utils.parameters import *

class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
# args_parser()
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0,
                        help="file index begin index*4+1, end index*4+4.")
    parser.add_argument('--lr_phy',type=float, default=1,
                        help="lr_phy")
    parser.add_argument('--lr_nn',type=float, default=0,
                    help="lr_nn")
    parser.add_argument('--EPOCHES',type=int, default=10,
                    help="EPOCHES")
    parser.add_argument('--decay_phy',type=float, default=0.8,
                    help="decay_phy")
    parser.add_argument('--decay_nn',type=float, default=0.9,
                    help="decay_nn")
    args = parser.parse_args()
    return args
args = args_parser()
j=args.index


log_path = './logs/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_file_name = log_path + 'abnode-phase-1-log-index-' +str(j)+'-'+ time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
sys.stdout = Logger(log_file_name)
sys.stderr = Logger(log_file_name)

print('this python file can get the prediction data and loss of the ABNODE phase1')
print('index = ',str(j))

global decay_lr_phy 
decay_lr_phy = args.decay_phy

def Lr_func_phy(epoch):
    '''
    two phases learning rate of physics parameters
    '''
    lr=1e-3*pow(decay_lr_phy,epoch)
    return lr


def Lr_func_nn(epoch):
    '''
    two phases learning rate of nn parameters
    '''
    lr = 0
    return lr

# read data
path='./data/'
file_list=os.listdir(path)
file_path=os.path.join(path,'data_'+str(j*4+1)+'.csv')
df = pd.read_csv(file_path)
data1=torch.from_numpy(df.values).detach()
file_path=os.path.join(path,'data_'+str(j*4+2)+'.csv')
df = pd.read_csv(file_path)
data2=torch.from_numpy(df.values).detach()
file_path=os.path.join(path,'data_'+str(j*4+3)+'.csv')
df = pd.read_csv(file_path)
data3=torch.from_numpy(df.values).detach()
file_path=os.path.join(path,'data_'+str(j*4+4)+'.csv')
df = pd.read_csv(file_path)
data4=torch.from_numpy(df.values).detach()

'''
data structure: 
data [0:35] 
[0:23]::::::::p[0:3], e[3:6], vb[6:9], wb[9:12], rb[12:15], rb_dot[15:18], Fl[18], Fr[19], rb_dot_dot[20:23]
[23:35]:::::::vb_dot[29:32], wb_dot[32:35],alpha,beta,vb_dot_sim,wb_dot_sim
'''

# hyper parameters
EPOCHES=args.EPOCHES
LR_phy=args.lr_phy
LR_nn=args.lr_nn
step_size=1/60

print("EPOCHES",EPOCHES)
print('learning rate phy',LR_phy)
print('learning rate nn',LR_nn)
print('decay_lr_phy',args.decay_phy)
print('decay_lr_nn',args.decay_nn)

# device
device=torch.device("cpu")
# data weight
if j >= 30:
    # straight
    df = pd.read_csv("./data/data_info/data_info_straight.csv",header=None)
    data_info_total_both_line=torch.from_numpy(df.values).detach()
    weight = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],device=device,requires_grad=False)
    weight[0,0:6]=torch.reciprocal(data_info_total_both_line[0,:6]-data_info_total_both_line[1,:6])
    weight.requires_grad=False
else:
    # spiral
    df = pd.read_csv("./data/data_info/data_info_spiral.csv",header=None)
    data_info_total=torch.from_numpy(df.values).detach()
    weight = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],device=device,requires_grad=False)
    weight[0,0:6]=torch.reciprocal(data_info_total[2,:6]-data_info_total[3,:6])
    weight.requires_grad=False


# train model
ode_train=NeuralODE(ABNODE_p1(RGBlimp_params=RGBlimp_params,device=device),RK_RGB,STEP_SIZE=step_size)
physics_param=[param for name,param in ode_train.named_parameters() if 'nn_model' not in name]
nn_model_param=[param for name,param in ode_train.named_parameters() if 'nn_model' in name]
optimizer=torch.optim.Adam([
    {'params':physics_param,'lr': LR_phy}, 
    {'params':nn_model_param,'lr': LR_nn}
])
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[Lr_func_phy, Lr_func_nn])


#### train data prepare, shape the data for training and testing
data1_t=torch.arange(0,step_size*data1.shape[0],step_size)
data1_t=(data1_t.unsqueeze(1)).unsqueeze(2)
data1_re=torch.cat([data1[:,:23]],1)
data1_re=data1_re.to(device).unsqueeze(1)

data2_t=torch.arange(0,step_size*data2.shape[0],step_size)
data2_t=(data2_t.unsqueeze(1)).unsqueeze(2)
data2_re=torch.cat([data2[:,:23]],1)
data2_re=data2_re.to(device).unsqueeze(1)

data3_t=torch.arange(0,step_size*data3.shape[0],step_size)
data3_t=(data3_t.unsqueeze(1)).unsqueeze(2)
data3_re=torch.cat([data3[:,:23]],1)
data3_re=data3_re.to(device).unsqueeze(1)

#### test data prepare
data4_t=torch.arange(0,data4.shape[0],1)*step_size
data4_t=(data4_t.unsqueeze(1)).unsqueeze(2)
data4_re=torch.cat([data4[:,:23]],1)
data4_re=data4_re.to(device).unsqueeze(1)


# train
for i in range(EPOCHES):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('=' * 8 + '%s' % nowtime + '=' * 8)
    print('Epoch{}  '.format(i+1))

    z_=ode_train(data1_re[0],data1_t,return_whole_sequence=True)
    loss=F.mse_loss(z_*weight,data1_re*weight)
    loss = loss.to(device)
    print('data1',loss.item(),end='##')
    optimizer.zero_grad()
    loss.backward(retain_graph=False)
    optimizer.step()
    
    z_=ode_train(data2_re[0],data2_t,return_whole_sequence=True)
    loss=F.mse_loss(z_*weight,data2_re*weight)
    loss = loss.to(device)
    print('data2',loss.item(),end='##')
    optimizer.zero_grad()
    loss.backward(retain_graph=False)
    optimizer.step()
    
    z_=ode_train(data3_re[0],data3_t,return_whole_sequence=True)
    loss=F.mse_loss(z_*weight,data3_re*weight)
    loss = loss.to(device)
    print('data3',loss.item())
    optimizer.zero_grad()
    loss.backward(retain_graph=False)
    optimizer.step()

    scheduler.step()



### test
ode_train.eval()
z_=ode_train(data4_re[0],data4_t,return_whole_sequence=True)
loss=F.mse_loss(z_*weight,data4_re*weight)
print('data test',loss.item(),end='##')

# over time
nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('=' * 8 + '%s' % nowtime + '=' * 8)

### save z_ and model
z_numpy=z_.data.numpy()
np.save('record/abnode/'+'ABNODE_p1'+'_index_'+str(j)+'_test_data_result_'+time.strftime("%Y%m%d", time.localtime()),z_numpy)
# store
model_name='ABNODE_p1'+'_index_'+str(j)+'_'+time.strftime("%Y%m%d", time.localtime())+'.pkl'
torch.save(ode_train, 'record/abnode/'+model_name)

