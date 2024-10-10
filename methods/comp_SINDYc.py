import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso
from pysindy.utils import enzyme
from pysindy.utils import lorenz
from pysindy.utils import lorenz_control
import pysindy as ps
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import sys
import time
import argparse
import os
from sklearn.metrics import mean_squared_error
# environment variable
PYDEVD_WARN_EVALUATION_TIMEOUT=100
# import files from parent directory
parent_directory = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_directory)
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
    parser.add_argument('--threshold', type=float, default=1,
                        help="optimizeer threshold.")
    parser.add_argument('--alpha', type=float, default=1e-5,
                        help="optimizer alpha.")
    args = parser.parse_args()
    return args
args = args_parser()
j=args.index
threshold_my=args.threshold
alpha_my=args.alpha

log_path = './logs/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_file_name = log_path + 'sindy-log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
sys.stdout = Logger(log_file_name)
sys.stderr = Logger(log_file_name)

### initial sindy
# Seed the random number generators for reproducibility
np.random.seed(100)
# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

print('this python file can get the prediction data and loss of the sindyc')
print('index = ', str(j))
print('threshold =',threshold_my)
print('alpha = ', alpha_my)


x_train=[]
x_dot_train_true=[]
u_train=[]
x_dot_train_true_error=0
x_dot_test_true_error=0

path='./data/'
file_list=os.listdir(path)
start=j*4+1
for i in range(4):
    ## train and test data
    data={}
    file_path=os.path.join(path,'data_'+str(start+i)+'.csv')
    df = pd.read_csv(file_path)

    e=df.iloc[:,3:6]
    vb=df.iloc[:,6:9]
    wb=df.iloc[:,9:12]
    rb_series=df.iloc[:,12:15]
    Fl_series=df.iloc[:,18]
    Fr_series=df.iloc[:,19]
    vb_dot=df.iloc[:,23:26]
    wb_dot=df.iloc[:,26:29]
    alpha=df.iloc[:,29]
    beta=df.iloc[:,30]
    vb_dot_sim=df.iloc[:,31:34]
    wb_dot_sim=df.iloc[:,34:37]

    # feature_names = ['vb1', 'vb2', 'vb3','wb1', 'wb2', 'wb3','Fl','Fr','alpha','beta']
    # train test 3:1

    # x | vb1 vb2 vb3 wb1 wb2 wb3
    # x | x1  x2  x3  x4  x5  x6
    # residual dynamics state variables

    # u | e1 e2 e3 rb_series1 rb_series2 rb_series3 Fl_series Fr_series alpha beta
    # u | u1 u2 u3 u4         u5         u6         u7        u8        u9    u10
    # Control variables: variables that can affect residual dynamics

    temp_x_train=np.concatenate([np.array(vb),np.array(wb)],axis=1)
    temp_x_dot_train_true=np.concatenate([np.array(vb_dot)-np.array(vb_dot_sim),np.array(wb_dot)-np.array(wb_dot_sim)],axis=1)
    temp_u_train=np.concatenate([np.array(e),np.array(rb_series),np.expand_dims(np.array(Fl_series),axis=1),np.expand_dims(np.array(Fr_series),axis=1),np.expand_dims(np.array(alpha),axis=1),np.expand_dims(np.array(beta),axis=1)],axis=1)

    if i == 3: # test data 
        x_test = temp_x_train
        x_dot_test_true = temp_x_dot_train_true
        u_test = temp_u_train
        x_dot_test_true_error = mean_squared_error(np.zeros_like(temp_x_dot_train_true),temp_x_dot_train_true)
    else: 
        x_train.append(temp_x_train)
        x_dot_train_true.append(temp_x_dot_train_true)
        u_train.append(temp_u_train)
temp = np.concatenate(x_dot_train_true,)


optimizer = ps.STLSQ(threshold=threshold_my, alpha=alpha_my, normalize_columns=True)


# New library design
# Initialize two libraries
poly_library = ps.PolynomialLibrary(include_bias=False)
fourier_library = ps.FourierLibrary()

# Initialize the default inputs, but
# don't use the v,w,rb,Fl,Fr input for generating the Fourier library
inputs_per_library = np.asarray([(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15), (6,6,6,6,6,6,6,7,8,8,8,8,8,8,14,15)])
# Tensor all the polynomial and Fourier library terms together
tensor_array = [[1, 1]]

# Initialize this generalized library, all the work hidden from the user!
generalized_library = ps.GeneralizedLibrary(
    [poly_library, fourier_library],
    tensor_array=tensor_array,
    inputs_per_library=inputs_per_library,
)

# train: fit()
step_size=1/60.0
model = ps.SINDy(optimizer=optimizer,feature_library=generalized_library)
model.fit(x_train, t=step_size, x_dot=x_dot_train_true,u=u_train,multiple_trajectories=True)
model.print()

# result
train_MSE=model.score(x_train, t=step_size, x_dot=x_dot_train_true,u=u_train,metric=mean_squared_error,multiple_trajectories=True)
test_MSE=model.score(x_test, t=step_size, x_dot=x_dot_test_true,u=u_test,metric=mean_squared_error)
intial_train_MSE=mean_squared_error(np.zeros_like(temp),temp)
intial_test_MSE=x_dot_test_true_error

print('inital x_dot_train MSE: ',intial_train_MSE)
print('inital x_dot_test MSE: ',intial_test_MSE)
print('train MSE: ', train_MSE)
print('test MSE: ', test_MSE)


device=torch.device("cpu")
data={}
file_path=os.path.join(path,'data_'+str(start+3)+'.csv')
df = pd.read_csv(file_path)
test_data=torch.from_numpy(df.values).detach()
test_data_re=torch.cat([test_data[:,:23]],1)# len*23

# generate RGBlimp dynamics data
RGBlimp_ode=RGBlimp(RGBlimp_params=RGBlimp_params,device=device)
# same as ode_train, using the first data of each train_data_set


def res_ode(temp_z):
    # combine RGBlimp dynamics and Sindyc
    eps=torch.tensor([1e-10],dtype=torch.double)
    v=temp_z[:,6:9]
    V = torch.linalg.norm(v)
    alpha=torch.atan2(v[:,2],v[:,0]+eps)
    beta=torch.asin(v[:,1]/(V+eps))
    temp_z_np=np.array(temp_z)
    alpha_np=np.array(alpha)
    beta_np=np.array(beta)
    y=RGBlimp_ode(temp_z)
    y_total=y
    u_temp=np.concatenate([temp_z_np[:,3:6].squeeze(0),temp_z_np[:,12:15].squeeze(0),temp_z_np[:,18:20].squeeze(0),alpha_np,beta_np],axis=0)
    y_total[0,6:12]=y[0,6:12]+torch.tensor(model.predict(x=np.expand_dims(temp_z_np[:,6:12],axis=0),u=np.expand_dims(u_temp,axis=0)))
    return y_total

# device
device=torch.device("cpu")
# data weight
if j >= 30:
    # straight
    df = pd.read_csv("./data/data_info/data_info_straight.csv",header=None)
    data_info_total_both_line=torch.from_numpy(df.values).detach()
    weight=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    weight[0:6]=np.reciprocal(data_info_total_both_line[0,:6]-data_info_total_both_line[1,:6])
else:
    # spiral
    df = pd.read_csv("./data/data_info/data_info_spiral.csv",header=None)
    data_info_total=torch.from_numpy(df.values).detach()
    weight=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    weight[0:6]=np.reciprocal(data_info_total[2,:6]-data_info_total[3,:6])

ode_sindy_z=[]
temp_z=test_data_re[0,:].unsqueeze(0)
ode_sindy_z.append(temp_z.tolist())
for l in range(test_data_re.shape[0]-1):
    temp_z=RK_RGB(temp_z,step_size,res_ode,step_size)
    ode_sindy_z.append(temp_z.tolist())


ode_sindy_loss=mean_squared_error(np.squeeze(np.array(ode_sindy_z)*weight,axis=1),np.array(test_data_re[:,:])*weight)
print("*"*8,'sindy_ode loss={}'.format(ode_sindy_loss),'*'*8)

# save SINDYc result
np.save('record/sindyc/'+'sindyc'+'_index_'+str(j)+'_test_data_result_'+time.strftime("%Y%m%d", time.localtime()),ode_sindy_z)
