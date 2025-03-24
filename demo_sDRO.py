from ALG.Optimizer import *
from ALG.Models import *
from ALG.Utils import *
from ALG.dataclass import *
torch.manual_seed(44)
torch.set_default_dtype(torch.float64)
import argparse

# Add argument parsing for device index
parser = argparse.ArgumentParser(description='Select GPU device.')
parser.add_argument('--gpu', type=int, default=0, help='Index of the GPU to use')
args = parser.parse_args()

try:
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    torch.cuda.get_device_name(args.gpu)
except (AssertionError, RuntimeError):
    device = 'cpu'

print(f"Using device: {device}")
# Load the data
# gisette: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2
# sido0: https://www.causality.inf.ethz.ch/data/sido0_matlab.zip, please use the *_train.mat
# a9a: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a, please save it as txt
# FMINST,MNIST,CIFAR10 will be downloaded by itself.
data_name = 'gisette' # gisette, sido0, a9a, MNIST, F-MNIST, CIFAR10,
train_set = loaddata(data_name,device)
#data normalization
if data_name == 'gisette':
    train_set.data = (train_set.data - torch.min(train_set.data,dim=1,keepdim=True)[0])/(torch.max(train_set.data,dim=1,keepdim=True)[0] - torch.min(train_set.data,dim=1,keepdim=True)[0])
    train_set.data = train_set.data[:6000]
    train_set.targets = train_set.targets[:6000]
elif data_name == 'sido0':
    train_set.data = (train_set.data - torch.min(train_set.data,dim=1,keepdim=True)[0])/(torch.max(train_set.data,dim=1,keepdim=True)[0] - torch.min(train_set.data,dim=1,keepdim=True)[0])
    #train_set.data = train_set.data/torch.norm(train_set.data)
    
print(f'data:{data_name},number_of_data:{train_set.data.shape[0]},dim_features:{train_set.data.shape[1]}')

# set the model
# Q: anydata works since it wont use it
# DRO: for gisette, sido0,a9a or any binary classification task.
# FairCNN: for FMINST,MNIST,CIFAR10 or any image m-classification task. 
for b in [200]:
    for mu_y in [0.01]:
        model_type = 'DRO' # Q: projection_y=False,DRO,FairCNN: projection_y=True
        sim_time=5
        max_iter=3000
        freq=500 # print result by freq
        my_optimizer = ALG(train_set=train_set,data_name=data_name,mu_y=mu_y,
                            sim_time=sim_time,max_iter=max_iter, b=b,
                            freq=freq,is_show_result=True,is_save_data=True,
                            projection_y=True,projection_x=False, # Q: projection_y=False; DRO,FairCNN: projection_y=True
                            maxsolver_step=min(10,1/mu_y),maxsolver_tol=1e-4,maxsolver_b=40000, # this is the setting for find y*(x)
                            isSameInitial=True, optimize_batch=False,
                            inject_noise_x=0.1,inject_noise_y=1.5, #0.1,1.5 is the noise around optimal point, 0.44,53 is the noise around initial point
                            device=device,model_type=model_type)
        my_optimizer.start_model.estimate_var(train_set.data,train_set.targets,200)
        L = my_optimizer.start_model.estimate_L(train_set.data,train_set.targets,data_name,load=True,b=b)
        if mu_y>L:
            L = mu_y
        kappa = L/mu_y
        lr_y = 1/L
        lr_x = 1/L/kappa**2
        print(f'L:{L}, mu:{mu_y}, kappa: {kappa}')
        print(f'GDA: lr_y={1/L},lr_x={1/16/(kappa+1)**2/L}')
        print(f'AGDA: lr_y={1/L},lr_x={1/3/L/(1+kappa)**2}')
        oc_tmp = 1
        sgd_b = b
        my_optimizer.max_iter = max_iter
        gamma1 = 0.9
        gamma2 = 0.95
        # result = my_optimizer.line_search(N=5,T=3)
        # result = my_optimizer.line_search(N=2,T=3)
        # result = my_optimizer.line_search(N=1,T=3)
        # my_optimizer.max_iter = max(result['total_iter'][0], my_optimizer.max_iter)
        # #
        sgd_b = b
        beta = 0.9
        lr_x_vrlm = min(kappa/40/L/(24*kappa**2+8*kappa+5), np.sqrt(beta)/48/(L+1)/(24*kappa**2+7*kappa+4))*100000
        lr_y_vrlm = np.sqrt(beta)/4/np.sqrt(2)/L
        # result = my_optimizer.optimizer(lr_x=1, lr_y=1, method='TiAda', b=sgd_b)
        result = my_optimizer.optimizer(lr_x=lr_x_vrlm,lr_y=lr_y_vrlm, b0_vrlm= 6000,beta_vrlm=beta, method='VRLM',b=sgd_b)
        # result = my_optimizer.optimizer(lr_x=1/16/(kappa+1)**2/L,lr_y=1/L,method='GDA',b=sgd_b)
        # result = my_optimizer.optimizer(lr_x=1/6/L/(1+kappa)**2,lr_y=1/L,method='AGDA',b=sgd_b)  
        # result = my_optimizer.optimizer(lr_x=1/6/L, lr_y=1/144/L,p=2*L,beta=mu_y/144/L/1600, method='Smooth-AGDA', b=sgd_b)


