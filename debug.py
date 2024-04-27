from ALG.Optimizer import *
from ALG.Models import *
from ALG.Utils import *
from ALG.dataclass import *
torch.manual_seed(123)
torch.set_default_dtype(torch.float64)

try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.get_device_name(0)
except AssertionError:
    device = 'cpu'

# Load the data
# gisette: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2
# sido0: https://www.causality.inf.ethz.ch/data/sido0_matlab.zip, please use the *_train.mat
# a9a: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a, please save it as txt
# FMINST,MNIST,CIFAR10 will be downloaded by itself.
data_name = 'sido0' # gisette, sido0, a9a, MNIST, F-MNIST, CIFAR10, 
train_set = loaddata(data_name,device)
print(f'data:{data_name},number_of_data:{train_set.data.shape[0]},dim_features:{train_set.data.shape[1]}')

# set the model
# Q: anydata works since it wont use it
# DRO: for gisette, sido0,a9a or any binary classification task.
# FairCNN: for FMINST,MNIST,CIFAR10 or any image m-classification task. 
mu_y=0.000001
for mu_y in [1,0.01]:
    model_name = 'DRO' # Q: projection_y=False,DRO,FairCNN: projection_y=True
    sim_time=10
    max_iter=1000
    freq=500 # print result by freq

    my_optimizer = ALG(train_set=train_set,data_name=data_name,mu_y=mu_y,
                        sim_time=sim_time,max_iter=max_iter,
                        freq=freq,is_show_result=True,is_save_data=True,
                        projection_y=False,projection_x=False, # Q: projection_y=False; DRO,FairCNN: projection_y=True
                        maxsolver_step=1/10/mu_y,maxsolver_tol=1e-4,maxsolver_b=40000, # this is the setting for find y*(x)
                        device=device,model_name=model_name)
    if model_name == 'Q':
        L = my_optimizer.start_model.L
        kappa = L/mu_y
        lr_y = 1/L
        lr_x = 1/L/kappa**2
    else:
        L = my_optimizer.start_model.estimate_L(train_set.data,train_set.targets,data_name,load=False)
    print(L)
    #'lr_x': 9.759657103991718e-12, 'lr_y': 3.814697265625
    # b = 500
    # iter_temp = 1
    # result = my_optimizer.line_search(gamma=0.5,N=1, min_b=b)
    # iter_temp = max(iter_temp,max(result['total_oracle_complexity'])/b)
    # result = my_optimizer.line_search(gamma=0.5,N=2, min_b=b)
    # iter_temp = max(iter_temp,max(result['total_oracle_complexity'])/b)
    # result = my_optimizer.line_search(gamma=0.5,N=5, min_b=b)
    # iter_temp = max(iter_temp,max(result['total_oracle_complexity'])/b)

    # my_optimizer.max_iter=iter_temp
    # result = my_optimizer.optimizer(lr_x=lr_x,lr_y=lr_y,method='GDA',b=b)
    # result = my_optimizer.optimizer(lr_x=lr_x,lr_y=lr_y,method='AGDA',b=b)
    # result = my_optimizer.optimizer(method='TiAda',b=b)