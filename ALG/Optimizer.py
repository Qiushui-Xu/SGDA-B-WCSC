from ctypes.wintypes import MAX_PATH
import random
from typing import final
import torch
import numpy as np
import copy
import pickle
import random
import math
from collections import OrderedDict
from importlib import reload
from ALG.Utils import projection_simplex_bisection as pj_y
#from ALG.Utils import projection_simplex_sort2 as pj_y
from ALG.Utils import projection_simplex_sort as pjy_y2
#from ALG.Utils import projection_l2 as pj_y
from ALG.Utils import projection_l2 as pj_x
from ALG.Utils import getxyFromStateModel, computeGrad
from  collections import  defaultdict

def call_model(model_type):
    if model_type == 'Q':
        from ALG.Models import ProblemQ as Model
    elif model_type == 'DRO':
        from ALG.Models import ProblemDRO as Model
    elif model_type == 'Test':
        from ALG.Models import ProblemTest as Model
    elif model_type =='FairCNN':
        from ALG.Models import FairCNN as Model
    return Model

class ALG():
    def __init__(self, train_set, data_name, mu_y, kappa = 10,
        sim_time = 3, max_epoch = 100, max_iter=100, b = 6000, 
        maxsolver_step=0.01, maxsolver_tol=1e-4, maxsolver_b = 6000,
        is_show_result = False, is_save_data = False, freq = 500,
        device = 'cuda',
        projection_y=False, projection_x=False,
        model_type = 'DRO', toymodel = False,
        inject_noise_x=0,
        inject_noise_y=0,
        isSameInitial = False, optimize_batch = False,
                 ) -> None:

        self.sim_time = sim_time    # sim_time means the times of running through the alg
        self.max_epoch = max_epoch  # an epoch means a complete pass of the data set
        self.max_iter = max_iter    # iter means the times of iteration within the alg
        self.b = b # b is the batchsize for stochastic alg
        self.is_show_result = is_show_result # whether display the training process
        self.freq = freq # the freq to show training process in terms of iter
        self.is_save_data = is_save_data # whether save data
        self.device = device
        self.maxsolver_step = maxsolver_step
        self.maxsolver_tol = maxsolver_tol
        self.maxsolver_b = maxsolver_b

        # problem parameters
        self.mu_y = mu_y
        self.kappa = kappa
        self.Dy = 1e6
        self.projection_y = projection_y # whether using projection for y
        self.projection_x = projection_x # whether using projection for x
        self.std_x = inject_noise_x
        self.std_y = inject_noise_y
        self.y_sum_eps = 0

        self.model_type = model_type
        self.data_name = data_name
        self.data = train_set.data.clone()
        self.targets = train_set.targets.clone()
        self.total_number_data = len(train_set.targets)

        self.isSameInitial = isSameInitial
        self.toymodel = toymodel
        self.optimize_batch = optimize_batch
        
        #initialize savers
        self.reset_all()

        #initilize start model
        if model_type == 'FairCNN':
            self.data_size = ((train_set.data.shape[1],train_set.data.shape[2]),len(train_set.targets))
            self.data_number_in_each_epoch = train_set.data.shape[0]
        elif model_type == 'Q':
            if toymodel:
                self.data_size = (1,1)
            else:
                self.data_size = (30,30)
            self.data_number_in_each_epoch = 1
        else:
            self.data_size = (len(train_set.data[0]),len(train_set.targets))
            self.data_number_in_each_epoch = self.total_number_data

        self.generate_initial_model()
        self.load_initial_model(0)
        
    def reset_initial_model(self):
        #initilize start model
        model_type = self.model_type
        mu_y = self.mu_y
        device = self.device
        kappa = self.kappa

        Model = call_model(model_type)
        self.start_model = Model(self.data_size, mu_y, kappa, device=device, injected_noise_x=self.std_x, injected_noise_y=self.std_y).to(device) # initial model
        self.y_opt = self.maximizer_solver(start=self.start_model,lr_y=self.maxsolver_step) # set y as y_opt for initial model
        #self.start_model.dual_y.data = self.y_opt.clone()
        self.model_copy = Model(self.data_size, mu_y, kappa, device=device, injected_noise_x=self.std_x, injected_noise_y=self.std_y).to(device) # initial model
        self.model_bk = Model(self.data_size, mu_y, device=device, injected_noise_x=self.std_x, injected_noise_y=self.std_y).to(device) # initial model

    def generate_initial_model(self):
        StartModelSet = {}
        for i in range(self.sim_time):
            #initilize start model
            model_type = self.model_type
            mu_y = self.mu_y
            device = self.device
            kappa = self.kappa

            Model = call_model(model_type)
            start_model = Model(self.data_size, mu_y, kappa, device=device,injected_noise_x=self.std_x, injected_noise_y=self.std_y).to(device) # initial model
            y_opt = self.maximizer_solver(start=start_model,lr_y=self.maxsolver_step) # set y as y_opt for initial model
            #start_model.dual_y.data = y_opt.clone()
            model_copy = Model(self.data_size, mu_y, kappa, device=device,injected_noise_x=self.std_x, injected_noise_y=self.std_y).to(device) # initial model
            model_bk = Model(self.data_size, mu_y, kappa, device=device,injected_noise_x=self.std_x, injected_noise_y=self.std_y).to(device) # initial model
            
            if self.isSameInitial:
                for i in range(self.sim_time):
                    StartModelSet[i] = [start_model, y_opt, model_copy, model_bk]
                break

            StartModelSet[i] = [start_model, y_opt, model_copy, model_bk]
            
        torch.save(StartModelSet, 'initial.pt')


    def load_initial_model(self, sim_time):
        start = torch.load('initial.pt')
        self.start_model,self.y_opt,self.model_copy,self.model_bk = start[sim_time]

    def reset_all(self,T=1):
        #the following saver will be reset for a new run with sim_times simulations
        self.sim_time = self.sim_time*T
        self.record = {}
        self.record['loss'] = [[] for _ in range(self.sim_time)]
        self.record['primalF'] =  [[] for _ in range(self.sim_time)]
        self.record['acc'] = [[] for _ in range(self.sim_time)]

        self.record['norm_square_sto_grad_x'] = [[] for _ in range(self.sim_time)]
        self.record['norm_square_sto_grad_y'] = [[] for _ in range(self.sim_time)]
        self.record['norm_square_full_grad_x'] = [[] for _ in range(self.sim_time)]
        self.record['norm_square_full_grad_y'] = [[] for _ in range(self.sim_time)]   

        self.record['total_sample_complexity'] = [0 for _ in range(self.sim_time)]
        self.record['total_oracle_complexity'] = [0 for _ in range(self.sim_time)]
        self.record['total_iter'] = [0 for _ in range(self.sim_time)]
        self.record['total_epoch'] = [0 for _ in range(self.sim_time)]

        self.record['total_sample_complexity_counter'] = [[] for _ in range(self.sim_time)]
        self.record['total_oracle_complexity_counter'] = [[] for _ in range(self.sim_time)]
        self.record['total_iter_counter'] = [[] for _ in range(self.sim_time)]
        self.record['total_epoch_counter'] = [[] for _ in range(self.sim_time)]

        self.record['sample_complexity'] = [0 for _ in range(self.sim_time)]
        self.record['oracle_complexity'] = [0 for _ in range(self.sim_time)]
        self.record['iter'] = [0 for _ in range(self.sim_time)]
        self.record['epoch'] = [0 for _ in range(self.sim_time)]

        self.record['sample_complexity_counter'] = [[] for _ in range(self.sim_time)]
        self.record['oracle_complexity_counter'] = [[] for _ in range(self.sim_time)]
        self.record['iter_counter'] = [[] for _ in range(self.sim_time)]
        self.record['epoch_counter'] = [[] for _ in range(self.sim_time)]

        self.record['contraction_times'] = [-1 for _ in range(self.sim_time)]
        self.record['config'] = [{} for _ in range(self.sim_time)]

        self.record['lr_x'] = [[] for _ in range(self.sim_time)]
        self.record['lr_y'] = [[] for _ in range(self.sim_time)]

        self.record['l(small)'] = [[] for _ in range(self.sim_time)]
        self.record['L(large)'] = [[] for _ in range(self.sim_time)]
        self.record['Deltak'] = [[] for _ in range(self.sim_time)]

        self.zt_Smooth_AGDA = {}

        self.sim_time = self.sim_time //T

    def reset_contraction(self, s):
        #the following saver will be reset for a new contraction at the s-th simulation
        self.record['sample_complexity'][s] = 0
        self.record['oracle_complexity'][s] = 0
        self.record['iter'][s] = 0
        self.record['epoch'][s] = 0

        self.record['sample_complexity_counter'][s] = []
        self.record['oracle_complexity_counter'][s] = []
        self.record['iter_counter'][s] = []
        self.record['epoch_counter'][s] = []

        self.record['total_sample_complexity_counter'][s] = []
        self.record['total_oracle_complexity_counter'][s] = []
        self.record['total_iter_counter'][s] = []
        self.record['total_epoch_counter'][s] = []

        self.record['loss'][s] = []
        self.record['primalF'][s] = []
        self.record['acc'][s] = []
        self.record['norm_square_sto_grad_x'][s] = []
        self.record['norm_square_sto_grad_y'][s] = []
        self.record['norm_square_full_grad_x'][s] = []
        self.record['norm_square_full_grad_y'][s] = [] 
        self.record['config'][s] = {}

    def pickbest(self, T):
        #the following saver will be reset for a new contraction at the s-th simulation
        idx = -1
        record_copy = {}
        for key in self.record:
            record_copy[key] = [None for _ in range(self.sim_time)]
        
        assert len(self.record['sample_complexity']) == T*self.sim_time
        for s in range(self.sim_time):
            min_idx = s*T
            min_grad_norm = np.inf
            for _ in range(T):
                idx += 1
                cur_grad_norm = self.record['norm_square_sto_grad_x'][idx][-1] + self.record['norm_square_sto_grad_y'][idx][-1]

                if cur_grad_norm<min_grad_norm:
                    min_idx = idx
                    min_grad_norm = cur_grad_norm
            
            for key in self.record:
                if key == 'config':
                    record_copy[key] = self.record[key]
                else:
                    record_copy[key][s] = copy.deepcopy(self.record[key][min_idx])
        
        return record_copy
    
    # TBD-2024-04-03-XUAN
    def pickbestParal(self, T):
        #the following saver will be reset for a new contraction at the s-th simulation
        record_copy = {}
        for key in self.record:
            record_copy[key] = [None for _ in range(self.sim_time)]
        idx = -1

        for s in range(self.sim_time):
            min_idx = s
            min_grad_norm = np.inf
            total_sample_complexity = 0
            total_oracle_complexity = 0

            for _ in range(T):
                idx += 1
                cur_grad_norm = min([self.record['norm_square_sto_grad_x'][idx][i] + self.record['norm_square_sto_grad_y'][idx][i] for i in range(len(self.record['norm_square_sto_grad_x'][idx]))])
                total_sample_complexity += self.record['total_sample_complexity'][idx]
                total_oracle_complexity += self.record['total_oracle_complexity'][idx]

                if cur_grad_norm<min_grad_norm:
                    min_idx = idx
                    min_grad_norm = cur_grad_norm
            
            for key in self.record:
                if key == 'config':
                    record_copy[key] = self.record[key]
                else:
                    record_copy[key][s] = copy.deepcopy(self.record[key][min_idx])
            record_copy['total_sample_complexity'][s] = total_sample_complexity
            record_copy['total_oracle_complexity'][s] = total_oracle_complexity
            
        return record_copy

    def line_search(self, gamma:float=0.9, N:int=1, T=3, method:str=None, min_b:int=1, force_b:int=-1, kernal='AGDA', randompick=False):
        self.reset_all(T=T)

        if not method:
            method = 'primal_line_search_N_' + str(N) + '_' + kernal
        Model = call_model(self.model_type)
        
        s = 0
        restart_simu_time = 0

        #update batchsize
        if force_b>=1:
            b = force_b
            self.b = b
        else:
            #b = 64/eps**2*(self.start_model.std_x**2+(1+6*(2-lr_y*self.mu_y)/lr_y/self.mu_y/(1-lr_y*self.mu_y)*self.start_model.std_y**2)) 
            #b = int(max(b,min_b))
            b = min(self.b,len(self.targets))
        max_iters = [random.randint(1, self.max_iter) for _ in range(T)]

        for sim in range(self.sim_time): 
            #initilize the line search parameters
            find = False # whether find finite squence until max iteration
            L = self.mu_y
            sim_find = True # whether finding lr_x,lr_y in this simluation
            self.load_initial_model(sim)
            self.start_model.dual_y.data = self.y_opt.clone()

            while not find:
                #shrink the stepsize conditions and change configs accordingly
                find = True
                L = L/gamma
                lr_y = 1/L
                rho = ((1+12/N)**(1/2)-1)/24
                lr_x = N*rho*self.mu_y**2*lr_y**3

                for t in range(T):
                    s = sim*T+t
                    max_iter= max_iters[t]

                    #load the start model
                    self.reset_contraction(s)
                    self.record['contraction_times'][s] += 1
                    self.model_curr = Model(data_size=self.data_size,mu_y=self.mu_y, kappa=self.kappa, device=self.device,injected_noise_x=self.std_x, injected_noise_y=self.std_y).to(self.device)
                    self.model_curr.load_state_dict(copy.deepcopy(self.start_model.state_dict()))

                    #initilize delta0
                    y0_opt = self.y_opt
                    delta0 = torch.norm(self.model_curr.dual_y-y0_opt)**2 # in fact, delta0 = 0, we set 1e-6 for tolerance

                    #initilize Delta0
                    full_batch = torch.arange(self.total_number_data).to(self.device)
                    self.model_curr.dual_y.data = y0_opt.clone()
                    Phi0 = self.model_curr.loss(self.data, full_batch, self.targets).item()
                    print('Phi0 =',Phi0)
                    self.model_curr.load_state_dict(copy.deepcopy(self.start_model.state_dict()))
                    #self.max_iter is the max K
                    eps = 64/self.max_iter/lr_y**3/self.mu_y*(max(0,Phi0 - self.start_model.F_lower)/rho/self.mu_y + 6*delta0) # compute the precison we can get according to iteration number
                    eps = eps**(1/2)

                    #initilize the block
                    # Generate an array of indices from 0 to N-1
                    flattened_x = torch.cat([param.flatten() for name,param in self.start_model.named_parameters() if name!='dual_y'])
                    indices = np.arange(flattened_x.shape[0])
                    # Shuffle the indices randomly
                    np.random.shuffle(indices)
                    # Split the shuffled indices into M blocks of approximately equal size
                    blocks = np.array_split(indices, N)

                    self.record['config'][s] = {'lr_x':lr_x,'lr_y':lr_y,'b':b,'eps':eps,'N':N,'K':self.max_iter,'real_K':max_iter,'L':L,'mu':self.mu_y,'std_x':self.start_model.std_x, 'std_y':self.start_model.std_y,'is_force_b':force_b>=1}
                    self.record['config'][s]['method'] = method
                    self.record['config'][s]['pjx'] = self.projection_x
                    self.record['config'][s]['pjy'] = self.projection_y

                    #initialize the data loader and full batch
                    full_batch = torch.arange(self.total_number_data).to(self.device)
                    data_loader_dumb = self.batchselect()
                    #initialize the counters for this contraction
                    batch_start = 0

                    while True:
                        #select data by batch index
                        if self.model_type != 'Q' and torch.sum(self.model_curr.dual_y.flatten()[data_loader_dumb[batch_start:batch_start+b]])<=self.y_sum_eps:
                            # skip if the batch are all invalid
                            data_loader_dumb = self.batchselect()
                            batch_start = 0
                            continue
                        elif batch_start+b <= len(data_loader_dumb):
                            batch_index = data_loader_dumb[batch_start:batch_start+b]
                            batch_start += b
                        elif b >= len(data_loader_dumb):
                            batch_index = data_loader_dumb
                            batch_start = 0
                        else:
                            #drop the incomplete data if they can not form a full batch
                            #data_loader_dumb = torch.randperm(self.total_number_data).to(self.device)
                            data_loader_dumb = self.batchselect()
                            batch_start = 0
                            continue
                        #print(torch.sum(self.model_curr.dual_y[batch_index]>0))

                        data_by_batch = torch.index_select(self.data,0,index=batch_index) #unseueeze is to make [64,28,28] to [64,1,28,28]
                        target_by_batch = torch.index_select(self.targets,0,index=batch_index)

                        #compute the gradients of current model using batches
                        self.model_curr.zero_grad()
                        computeGrad(self.model_curr,data_by_batch,target_by_batch, batch_index,b)
                        
                        #select the block
                        chosen_block = random.choice(blocks)

                        #save and show the current data before updating
                        self.save_iterates_info(s,batch_index,lr_x,lr_y,chosen_block)
                        self.record['lr_x'][s].append(lr_x)
                        self.record['lr_y'][s].append(lr_y)
                        self.record['L(large)'][s].append(L)
                        if self.is_show_result and self.record['iter'][s]%self.freq==0:
                            self.show_result(s,batch_index, sim_done=False)
                        
                        #break the if beyond max iter 
                        if self.record['iter'][s]>=max_iter:
                            break

                        #update the model by method
                        def SGDA_B(lr_x,lr_y):
                            flattened_x = torch.cat([param.data.clone().flatten() for name,param in self.model_curr.named_parameters() if name!='dual_y'])
                            flattened_grad = torch.cat([param.grad.data.clone().flatten() for name,param in self.model_curr.named_parameters() if name!='dual_y'])
                            
                            if self.projection_x:
                                projection_center =  flattened_x[chosen_block] - lr_x*flattened_grad[chosen_block]
                                flattened_x[chosen_block] = torch.tensor(pj_x(projection_center.cpu().detach().numpy()),dtype=torch.float64,device=self.device)
                            else:
                                flattened_x[chosen_block] = flattened_x[chosen_block] - lr_x*flattened_grad[chosen_block]

                            idx = 0
                            for (name,param) in self.model_curr.named_parameters():
                                if name != 'dual_y':
                                    num_params = param.numel()
                                    param.data = flattened_x[idx:idx+num_params].view(param.shape)
                                    idx += num_params
                                elif name == 'dual_y':
                                    if self.projection_y:
                                        projection_center =  param.data + lr_y*param.grad.data
                                        param.data = torch.tensor(pj_y(projection_center.cpu().detach().numpy()),dtype=torch.float64,device=self.device)
                                    else:
                                        param.data = param.data + lr_y*param.grad.data
                                            
                        #we wont use this
                        def AGDA_B(lr_x,lr_y):
                            nonlocal find
                            #update x 
                            flattened_x = torch.cat([param.data.clone().flatten() for name,param in self.model_curr.named_parameters() if name!='dual_y'])
                            flattened_grad = torch.cat([param.grad.data.clone().flatten() for name,param in self.model_curr.named_parameters() if name!='dual_y'])
                            if self.projection_x:
                                projection_center =  flattened_x[chosen_block].clone() - lr_x*flattened_grad[chosen_block].clone()
                                flattened_x[chosen_block] = torch.tensor(pj_x(projection_center.cpu().detach().numpy()),dtype=torch.float64,device=self.device)
                            else:
                                flattened_x[chosen_block] = flattened_x[chosen_block].clone() - lr_x*flattened_grad[chosen_block].clone()
                            

                            idx = 0
                            for (name,param) in self.model_curr.named_parameters():
                                if name != 'dual_y':
                                    num_params = param.numel()
                                    param.data = flattened_x[idx:idx+num_params].clone().view(param.shape)
                                    idx += num_params

                            #update y
                            computeGrad(self.model_curr,data_by_batch,target_by_batch, batch_index,b)
                            for (name,param) in self.model_curr.named_parameters():
                                if name == 'dual_y':
                                    if self.projection_y:
                                        projection_center =  param.data + lr_y*param.grad.data
                                        param.data = torch.tensor(pj_y(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device)
                                    else:
                                        param.data = param.data + lr_y*param.grad.data
                            #compute the gradients of current model using batches, not that the batch does not change here
                            self.model_curr.zero_grad()
                            
                        
                        #implement the method
                        if kernal == 'AGDA':
                            AGDA_B(lr_x,lr_y)
                        elif kernal == "GDA":
                            SGDA_B(lr_x,lr_y)

                        # update the complexity and iterations
                        self.record['total_sample_complexity'][s] += b
                        self.record['total_oracle_complexity'][s] += b/N
                        self.record['total_iter'][s] += 1
                        self.record['total_epoch'][s] += b/self.data_number_in_each_epoch

                        self.record['sample_complexity'][s] += b
                        self.record['oracle_complexity'][s] += b/N
                        self.record['iter'][s] += 1
                        self.record['epoch'][s] += b/self.data_number_in_each_epoch
                        
                        #compute the new loss for line search checking
                        f_new = self.model_curr.loss(self.data,full_batch, self.targets)
                        f_new = f_new.item()

                        check_temp = 0
                        for (name,param) in self.model_curr.named_parameters():
                                if name != 'dual_y':
                                    check_temp += torch.norm(param.data)**2

                        if not find or np.isnan(f_new)  or \
                            torch.isnan(torch.norm(self.model_curr.dual_y)) \
                            or(self.projection_y and torch.abs(torch.sum(self.model_curr.dual_y.data)-1)>1e-2) or \
                            check_temp>1e32:
                            find = False
                            break

                    # start line search after all iterations
                    # print('------------------------contraction times', self.record['contraction_times'][s], 'is finished------------------------')
                    # print('sigma =',lr_y, 'tau =',lr_x)
                
                                    #assert contraction_times<=100
                        
                if self.record['contraction_times'][s] >=100:
                    sim_find = False
                    break

                if not find:
                    print('contraction', self.record['contraction_times'][s], 'fails, the gap is nan')
                    pass
                else:
                    min_norm_square_sto_grad = min([self.record['norm_square_sto_grad_x'][idx_tmp][-1] + self.record['norm_square_sto_grad_y'][idx_tmp][-1] for idx_tmp in range(sim*T, (sim+1)*T)])
                    y_opt = self.maximizer_solver(start=self.model_curr, lr_y=self.maxsolver_step) #y_opt is the tensor
                    self.model_curr.dual_y.data = y_opt.clone()
                    Phik = self.model_curr.loss(self.data, full_batch, self.targets).item()
                    # eq (31) in the paper
                    lcondition = min_norm_square_sto_grad
                    rcondition = 4*N/lr_x*(max(0,Phi0 - Phik) + 6*rho*self.mu_y*delta0) \
                        + 4*len(self.record['norm_square_sto_grad_x'][s])/b*(self.model_curr.std**2
                                                                            +(1+6*(2-lr_y*self.mu_y)/lr_y/self.mu_y/(1-lr_y*self.mu_y)*self.model_curr.std**2)
                                                                            )
                    rcondition = rcondition/self.max_iter


                    if lcondition>rcondition:
                        find = False
                        foo = self.record['contraction_times'][s]
                        print(f'Contraction {foo} fails, gap={lcondition-rcondition}, lcondition={lcondition}, rcondition={rcondition}, Phi0={Phi0}, Phik={Phik}\n')  
                    else:
                        find = True
                        foo = self.record['contraction_times'][s]
                        print(f'Contraction {foo} successes!!! Gap={lcondition-rcondition}, lcondition={lcondition}, rcondition={rcondition}, Phi0={Phi0}, Phik={Phik}\n')       

            if not sim_find:
                print(f'{s}th sim failed ({restart_simu_time} trys), restarting...')
                continue
            
            
            #show and save this simulation result
            for idx_tmp in range(sim*T, (sim+1)*T):
                if self.record['norm_square_sto_grad_x'][idx_tmp] + self.record['norm_square_sto_grad_y'][idx_tmp] == min_norm_square_sto_grad:
                    self.show_result(idx_tmp,batch_index, sim_done=False)
                    break
                
        self.record = self.pickbestParal(T)
        if self.is_save_data:
            if self.model_type == 'Q':
                foo = self.start_model.name
                if self.toymodel:
                    foo += '_toy'
                save_kappa = self.kappa
            else:
                foo = self.data_name
                save_kappa = 1
            import os
            folder_path = './result_data/' + foo + '_muy_' + str(self.mu_y) + '_kappa_' + str(save_kappa) + '_b_' + str(self.b)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_name =  folder_path + '/' + method 
            with open(file_name , "wb") as fp:  
                pickle.dump(self.record, fp)

        return self.record

    def maximizer_solver(self,start,lr_y=None,b=None,tol=None,max_iter = 1e5):
        from torch import optim
        import time
        s = time.time()
        if not lr_y:
            lr_y=min(1,self.maxsolver_step)
        if not b:
            b = self.maxsolver_b
        if not tol:
            tol = self.maxsolver_tol

        #print(f'maximizer solver start, the stepsize is {lr_y}')
        Model = call_model(self.model_type)
        #load the start model
        model_tmp = Model(data_size=self.data_size,mu_y=self.mu_y,kappa=self.kappa, device=self.device,injected_noise_x=self.std_x, injected_noise_y=self.std_y).to(self.device)
        model_tmp.load_state_dict(copy.deepcopy(start.state_dict()))
        debug = copy.deepcopy(start.state_dict())
        optimizer = optim.Adam([model_tmp.dual_y], lr=lr_y)
        #initialize the data loader
        data_loader_dumb = self.batchselect(model=model_tmp).to(self.device)
        #initialize the batch counters
        batch_start = 0
        time_counter = 1

        if torch.norm(model_tmp.dual_y.data.clone() )**2 >1e100 or torch.isnan(torch.norm(model_tmp.dual_y.data.clone() )**2):
            print('maximizer intial point not valid!!! Please debug from maximizer_solver!!!')
        
        iter = 0
        while iter<int(max_iter):
            #select data by batch index
            if self.model_type!='Q':
                #select data by batch index
                if torch.sum(model_tmp.dual_y.flatten()[batch_start:batch_start+b])<=self.y_sum_eps:
                    # skip if the batch are all invalid
                    data_loader_dumb = self.batchselect()
                    batch_start = 0
                    continue
                elif batch_start+b <= len(data_loader_dumb):
                    batch_index = data_loader_dumb[batch_start:batch_start+b]
                    batch_start += b
                elif b >= len(data_loader_dumb):
                    batch_index = data_loader_dumb
                    batch_start = 0
                else:
                    #drop the incomplete data if they can not form a full batch
                    #data_loader_dumb = torch.randperm(self.total_number_data).to(self.device)
                    data_loader_dumb = self.batchselect(model=model_tmp).to(self.device)
                    batch_start = 0
                    continue
            else:
                batch_index = torch.tensor([],device=self.device)

            data_by_batch = torch.index_select(self.data,0,index=batch_index) #unseueeze is to make [64,28,28] to [64,1,28,28]
            target_by_batch = torch.index_select(self.targets,0,index=batch_index)

            if self.model_type == 'DRO' and 0:
                y = model_tmp.exact_y_opt(data_by_batch,batch_index, target_by_batch)
                y = torch.tensor(pj_y(y.cpu().detach().numpy()),device=self.device)
                return y
            elif self.model_type == 'Q':
                y = model_tmp.exact_y_opt(data_by_batch,batch_index, target_by_batch).clone()
                if self.projection_y:
                    y = torch.tensor(pj_y(y.cpu().detach().numpy()), device=self.device)
                return y


            #compute the gradients of current model using batches
            for (name,param) in model_tmp.named_parameters():
                if name != 'dual_y':
                    param.requires_grad_(False)
            model_tmp.zero_grad()
            loss_tmp = - model_tmp.loss(data_by_batch,batch_index, target_by_batch) #we take -loss as gradient so that it is gradient descent now.
            loss_tmp.backward()
            y_prev = model_tmp.dual_y.data.clone()

            #update the model by method
            def SGDA(lr_y,start_debug):
                for (name,param) in model_tmp.named_parameters():
                    if name == 'dual_y':
                        if self.projection_y:
                            # debug = param.data.clone()
                            # assert torch.abs(torch.sum(param.data)-1)<1e-2
                            projection_center =  param.data - lr_y*param.grad.data
                            param.data = torch.tensor(pj_y(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device)
                        else:
                            param.data = param.data - lr_y*param.grad.data 

            SGDA(lr_y,start)
            #optimizer.step()
            
            
            if abs(sum(model_tmp.dual_y.data).item()-1)>1e-4:
                model_tmp.dual_y.data =  torch.tensor(pj_y(model_tmp.dual_y.data.cpu().detach().numpy()),dtype=torch.float64).to(self.device)
            if abs(sum(model_tmp.dual_y.data).item()-1)>1e-4:
                print('projection error!!!')

            y_gap = torch.norm((model_tmp.dual_y.data.clone() - y_prev)/lr_y)**2

            if 0:
                y1 = model_tmp.exact_y_opt(data_by_batch,batch_index, target_by_batch)
                y1 = torch.tensor(pjy_y2(y1.cpu().detach().numpy()),device=self.device)
                if  torch.norm(model_tmp.dual_y.data - y1)>0.1:
                    print('error',torch.norm(model_tmp.dual_y.data - y1))
                
            #model_tmp.loss(data_by_batch,batch_index, target_by_batch)
            #model_tmp.dual_y.data=y1
            #model_tmp.loss(data_by_batch,batch_index, target_by_batch)

            if (time.time()-s)//30>=time_counter:
                time_counter += 1
                print(f"Warning!!! The maximizer solve has cost {(time.time()-s)//30*0.5} minutes!!!Consider to adjust maximizer solver!!!The gap is {y_gap}")
            if y_gap<tol:
                return model_tmp.dual_y.data.clone()
            
            iter += 1
        
        print('Fail to find the optimal y, please adjust the parameters')

    def optimizer(self, method:str, lr_x=None, lr_y=None,alpha=0.6,beta=0.4,p=0,beta_vrlm=0.99,b0_vrlm=1000, b=None, max_epoch=None,start=None,max_iter=None):
        self.reset_all()
        Model = call_model(self.model_type)
        if not b:
            b = self.b
        if not max_epoch:
            max_epoch = self.max_epoch
        if not max_iter:
            max_iter = self.max_iter

        # Generate full block
        N = 1
        flattened_x = torch.cat([param.flatten() for name,param in self.start_model.named_parameters() if name!='dual_y'])
        indices = np.arange(flattened_x.shape[0])
        full_block = copy.deepcopy(indices)

        for s in range(self.sim_time):
            self.load_initial_model(s)
            self.reset_contraction(s)
            self.record['contraction_times'][s] += 1
            self.record['config'][s] = {'b':b,'N':N,'K':self.max_iter,'std_x':self.start_model.std_x, 'std_y':self.start_model.std_y}
            self.record['config'][s]['method'] = method
            self.record['config'][s]['pjx'] = self.projection_x
            self.record['config'][s]['pjy'] = self.projection_y

            #load the start model
            start = self.start_model 
            self.model_curr = Model(data_size=self.data_size, mu_y=self.mu_y, kappa=self.kappa, device=self.device,injected_noise_x=self.std_x, injected_noise_y=self.std_y).to(self.device)
            self.model_curr.load_state_dict(copy.deepcopy(start.state_dict()))

            #initialize the data loader and full batch
            full_batch = torch.arange(self.total_number_data).to(self.device)
            data_loader_dumb = self.batchselect() # torch.randperm(self.total_number_data).to(self.device)
            batch_start = 0



            #intialize the STORM
            if method == 'VRLM':
                self.model_prev = Model(data_size=self.data_size, mu_y=self.mu_y, kappa=self.kappa, device=self.device,injected_noise_x=self.std_x, injected_noise_y=self.std_y).to(self.device)
                self.model_prev.load_state_dict(copy.deepcopy(start.state_dict()))

                b0 = b0_vrlm
                data_loader_dumb = self.batchselect()
                batch_index = data_loader_dumb[:b0]
                data_by_batch = torch.index_select(self.data,0,index=batch_index) #unseueeze is to make [64,28,28] to [64,1,28,28]
                target_by_batch = torch.index_select(self.targets,0,index=batch_index)

                self.model_curr.zero_grad()
                computeGrad(self.model_curr,data_by_batch,target_by_batch, batch_index, b0)
                computeGrad(self.model_prev,data_by_batch,target_by_batch, batch_index, b0)

                self.dx = [param.grad.data.clone() for name,param in self.model_curr.named_parameters() if name!='dual_y']
                self.dy = [param.grad.data.clone() for name,param in self.model_curr.named_parameters() if name=='dual_y']
                self.dx_prev = [param.grad.data.clone() for name,param in self.model_curr.named_parameters() if name!='dual_y']
                self.dy_prev = [param.grad.data.clone() for name,param in self.model_curr.named_parameters() if name=='dual_y']
                self.vx_vrlm = [param.grad.data.clone() for name,param in self.model_curr.named_parameters() if name!='dual_y']
                self.vy_vrlm = [param.grad.data.clone() for name,param in self.model_curr.named_parameters() if name=='dual_y']
                self.model_curr.zero_grad()
                self.model_prev.zero_grad()

            if method == 'TiAda':
                self.vx_tiada,self.vy_tiada = 1,1
                lr_x_TiAda = lr_x / math.pow(np.maximum(self.vx_tiada, self.vy_tiada), alpha)
                lr_y_TiAda = lr_y / math.pow(self.vy_tiada, beta)

            if method == 'Smooth-AGDA':
                for name, param in self.model_curr.named_parameters():
                    if name != 'dual_y':
                        self.zt_Smooth_AGDA[name] = 0# param.data.clone()
     
            while True:
                #select data by batch index
                #select data by batch index
                if self.model_type != 'Q' and torch.sum(self.model_curr.dual_y.flatten()[data_loader_dumb[batch_start:batch_start+b]])<=self.y_sum_eps:
                    # skip if the batch are all invalid
                    data_loader_dumb = self.batchselect()
                    batch_start = 0
                    continue
                elif batch_start+b <= len(data_loader_dumb):
                    batch_index = data_loader_dumb[batch_start:batch_start+b]
                    batch_start += b
                elif b >= len(data_loader_dumb):
                    batch_index = data_loader_dumb
                    batch_start = 0
                else:
                    #drop the incomplete data if they can not form a full batch
                    #data_loader_dumb = torch.randperm(self.total_number_data).to(self.device)
                    data_loader_dumb = self.batchselect()
                    batch_start = 0
                    continue
                data_by_batch = torch.index_select(self.data,0,index=batch_index) #unseueeze is to make [64,28,28] to [64,1,28,28]
                target_by_batch = torch.index_select(self.targets,0,index=batch_index)

                # compute the gradients of current model using batches
                self.model_curr.zero_grad()
                computeGrad(self.model_curr,data_by_batch,target_by_batch, batch_index,b)
                if method == 'VRLM':
                    self.model_prev.zero_grad()
                    computeGrad(self.model_prev,data_by_batch,target_by_batch, batch_index, b)
                    self.model_prev.load_state_dict(self.model_curr.state_dict()) # load state dict wont change grad information

                # save and show the current data before updating
                if method == 'TiAda':
                    self.save_iterates_info(s, batch_index, lr_x_TiAda, lr_y_TiAda, full_block)
                else:
                    self.save_iterates_info(s,batch_index,lr_x,lr_y,full_block)

                # update the model by method
                def SGDA(lr_x,lr_y):
                    for (name,param) in self.model_curr.named_parameters():
                        if name != 'dual_y':
                            if self.projection_x:
                                projection_center =  param.data - lr_x*param.grad.data
                                param.data = torch.tensor(pj_x(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device).clone()
                            else:
                                param.data = param.data - lr_x*param.grad.data
                        else:
                            if self.projection_y:
                                projection_center =  param.data + lr_y*param.grad.data
                                param.data = torch.tensor(pj_y(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device)
                            else:
                                param.data = param.data + lr_y*param.grad.data 
                    
                #update the model by method
                def ASGDA(lr_x,lr_y):
                    #update y first
                    for (name,param) in self.model_curr.named_parameters():
                        if name == 'dual_y':
                            if self.projection_y:
                                projection_center =  param.data + lr_y*param.grad.data
                                param.data = torch.tensor(pj_y(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device)
                            else:
                                param.data = param.data + lr_y*param.grad.data 

                    #compute the gradients of current model using batches, not that the batch does not change here
                    self.model_curr.zero_grad()
                    computeGrad(self.model_curr,data_by_batch,target_by_batch, batch_index,b)

                    #update x then
                    for (name,param) in self.model_curr.named_parameters():
                        if name != 'dual_y':
                            if self.projection_x:
                                projection_center =  param.data - lr_x*param.grad.data
                                param.data = torch.tensor(pj_x(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device).clone()
                            else:
                                param.data = param.data - lr_x*param.grad.data
                def ASGDA_X(lr_x,lr_y):
                    #update x then
                    for (name,param) in self.model_curr.named_parameters():
                        if name != 'dual_y':
                            if self.projection_x:
                                projection_center =  param.data - lr_x*param.grad.data
                                param.data = torch.tensor(pj_x(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device).clone()
                            else:
                                param.data = param.data - lr_x*param.grad.data

                    #compute the gradients of current model using batches, not that the batch does not change here
                    self.model_curr.zero_grad()
                    computeGrad(self.model_curr,data_by_batch,target_by_batch, batch_index,b)

                    #update y then
                    for (name,param) in self.model_curr.named_parameters():
                        if name == 'dual_y':
                            if self.projection_y:
                                projection_center =  param.data + lr_y*param.grad.data
                                param.data = torch.tensor(pj_y(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device)
                            else:
                                param.data = param.data + lr_y*param.grad.data 

                #update the model by method
                def TiAda(lr_x,lr_y,alpha,beta):
                    assert alpha>beta
                    self.vx_tiada += self.record['norm_square_sto_grad_x'][s][-1]
                    self.vy_tiada += self.record['norm_square_sto_grad_y'][s][-1]
                    lr_x_TiAda = lr_x/math.pow(np.maximum(self.vx_tiada,self.vy_tiada),alpha)
                    lr_y_TiADA = lr_y/math.pow(self.vy_tiada,beta)
                    for (name,param) in self.model_curr.named_parameters():
                        if name != 'dual_y':
                            if self.projection_x:
                                projection_center =  param.data - lr_x/math.pow(np.maximum(self.vx_tiada,self.vy_tiada),alpha)*param.grad.data 
                                param.data = torch.tensor(pj_x(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device).clone()
                            else:
                                param.data = param.data - lr_x/math.pow(np.maximum(self.vx_tiada,self.vy_tiada),alpha)*param.grad.data 
                        else:
                            if self.projection_y:
                                projection_center =  param.data + lr_y/math.pow(self.vy_tiada,beta)*param.grad.data
                                param.data = torch.tensor(pj_y(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device)
                            else:
                                param.data = param.data + lr_y/math.pow(self.vy_tiada,beta)*param.grad.data
                    return lr_x_TiAda,lr_y_TiADA

                def Smooth_AGDA(lr_x,lr_y,p,beta):
                    #update x first
                    for (name,param) in self.model_curr.named_parameters():
                        if name != 'dual_y':
                            if self.projection_x:
                                projection_center =  param.data - lr_x*param.grad.data - lr_x*p*(param.data - self.zt_Smooth_AGDA[name])
                                param.data = torch.tensor(pj_x(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device).clone()
                            else:
                                param.data = param.data - lr_x*param.grad.data - lr_x*p*(param.data - self.zt_Smooth_AGDA[name])

                            temp = self.zt_Smooth_AGDA[name] + beta*(param.data - self.zt_Smooth_AGDA[name])
                            self.zt_Smooth_AGDA[name] = temp.clone()

                    #compute the gradients of current model using batches, not that the batch does not change here
                    self.model_curr.zero_grad()
                    computeGrad(self.model_curr,data_by_batch,target_by_batch, batch_index,b)

                    #update y then
                    for (name,param) in self.model_curr.named_parameters():
                        if name == 'dual_y':
                            if self.projection_y:
                                projection_center =  param.data + lr_y*param.grad.data
                                param.data = torch.tensor(pj_y(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device)
                            else:
                                param.data = param.data + lr_y*param.grad.data
                    return lr_x,lr_y
                
                def VRLM(lr_x,lr_y,beta):
                    id_x, id_y = 0, 0
                    # dx_temp = copy.deepcopy(self.dx)
                    # dy_temp = copy.deepcopy(self.dy)
                    self.dx_prev = copy.deepcopy(self.dx)
                    self.dy_prev = copy.deepcopy(self.dy)

                    for (name,param),param_prev in zip(self.model_curr.named_parameters(),self.model_prev.parameters()):
                        if name != 'dual_y':
                            self.dx[id_x] =  param.grad.data + (1-beta)*(self.dx[id_x] - param_prev.grad.data)
                            self.vx_vrlm[id_x] = self.vx_vrlm[id_x] + self.dx[id_x] - self.dx_prev[id_x]
                            id_x += 1
                        else:
                            self.dy[id_y] =  param.grad.data + (1-beta)*(self.dy[id_y] - param_prev.grad.data)
                            self.vy_vrlm[id_y] = self.dy[id_y]
                            id_y += 1

                    id_x, id_y = 0, 0
                    for (name,param) in self.model_curr.named_parameters():
                        if name != 'dual_y':
                            if self.projection_x:
                                projection_center =  param.data - lr_x*self.vx_vrlm[id_x]
                                param.data = torch.tensor(pj_x(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device).clone()
                            else:
                                param.data = param.data - lr_x*self.vx_vrlm[id_x]
                            id_x += 1
                        else:
                            if self.projection_y:
                                projection_center =  param.data + lr_y*self.vy_vrlm[id_y]
                                param.data = torch.tensor(pj_y(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device)
                            else:
                                param.data = param.data + lr_y*self.vy_vrlm[id_y]
                            id_y += 1

                    # self.dx_prev = copy.deepcopy(dx_temp)
                    # self.dy_prev = copy.deepcopy(dy_temp)



                if method == 'GDA':
                    SGDA(lr_x,lr_y)
                elif method == 'AGDA':
                    ASGDA(lr_x,lr_y)
                elif method == 'AGDA_X':
                    ASGDA_X(lr_x,lr_y)
                elif method == 'SAPD':
                    print('method error!!!')
                elif method == 'TiAda':
                    lr_x_TiAda,lr_y_TiAda = TiAda(lr_x,lr_y,alpha,beta)
                elif method == 'Smooth-AGDA':
                    Smooth_AGDA(lr_x, lr_y, p, beta)
                elif method == 'VRLM':
                    VRLM(lr_x,lr_y,beta_vrlm)
                    self.model_prev.zero_grad()

                self.model_curr.zero_grad()
                

                if method == 'TiAda':
                    self.record['lr_x'][s].append(lr_x_TiAda)
                    self.record['lr_y'][s].append(lr_y_TiAda)
                else:
                    self.record['lr_x'][s].append(lr_x)
                    self.record['lr_y'][s].append(lr_y)

                #break the if beyond max iter
                if self.record['iter'][s]>=self.max_iter:
                    break

                if self.is_show_result and self.record['iter'][s]%self.freq==0:
                    self.show_result(s,batch_index, sim_done=False)

                # update the complexity and iterations
                if method == 'VRLM' and iter == 0:
                    self.record['total_sample_complexity'][s] += b0_vrlm
                    self.record['total_oracle_complexity'][s] += b0_vrlm/N
                    self.record['total_iter'][s] += 1
                    self.record['total_epoch'][s] += b0_vrlm/self.data_number_in_each_epoch

                    self.record['sample_complexity'][s] += b0_vrlm
                    self.record['oracle_complexity'][s] += b0_vrlm/N
                    self.record['iter'][s] += 1
                    self.record['epoch'][s] += b0_vrlm/self.data_number_in_each_epoch
                else:
                    self.record['total_sample_complexity'][s] += b
                    self.record['total_oracle_complexity'][s] += b/N
                    self.record['total_iter'][s] += 1
                    self.record['total_epoch'][s] += b/self.data_number_in_each_epoch

                    self.record['sample_complexity'][s] += b
                    self.record['oracle_complexity'][s] += b/N
                    self.record['iter'][s] += 1
                    self.record['epoch'][s] += b/self.data_number_in_each_epoch

            #show this simulation result
            self.show_result(s,batch_index,sim_done=True)
            if self.is_save_data:
                if self.model_type == 'Q':
                    foo = self.start_model.name
                    if self.toymodel:
                        foo += '_toy'
                    save_kappa = self.kappa
                else:
                    foo = self.data_name
                    save_kappa = 1
                import os
                folder_path = './result_data/' + foo + '_muy_' + str(self.mu_y) + '_kappa_' + str(save_kappa) + '_b_' + str(self.b)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                file_name =  folder_path + '/' + method 
                with open(file_name , "wb") as fp:  
                    pickle.dump(self.record, fp)
    
    def batchselect(self, model=None):
        from random import shuffle
        if not self.optimize_batch:
            return torch.randperm(self.total_number_data).to(self.device)
        if not model:
            model = self.model_curr

        x = model.dual_y.flatten()
        positive_indices = torch.nonzero(x > 0).view(-1)
        foo = torch.randperm(positive_indices.size(0)).to(self.device) #s
        positive_indices = positive_indices[foo]
        
        other_indices = torch.nonzero(x <= 0).view(-1)
        l = positive_indices.shape[0] - positive_indices.shape[0]//10
        l = positive_indices.shape[0] - positive_indices.shape[0]//30
        combined_indices = torch.cat((positive_indices[:l], other_indices,positive_indices[l:]))

        sorted_indices = torch.argsort(model.dual_y.data.flatten(),descending=True)
        
        return combined_indices
    
    def save_iterates_info(self, s, batch, lr_x, lr_y, chosen_block=None, gx=None, gy=None):
        #record the iter or complexity or epoch for plotting x-axis
        for name in ['iter','epoch','oracle_complexity','sample_complexity']:
            self.record[name+'_counter'][s].append(self.record[name][s])
            self.record['total_' + name + '_counter'][s].append(self.record['total_' + name][s])
        
        #compute and save the current norm square of sto gradients for plotting y-axis
        if gx!=None and gy!=None:
            grad_x_tmp,grad_y_tmp = gx**2,gy**2
        else:
            grad_x_tmp = 0
            grad_y_tmp = 0
            flattened_x = torch.cat([param.data.clone().flatten() for name,param in self.model_curr.named_parameters() if name!='dual_y'])
            flattened_x_grad = torch.cat([param.grad.data.clone().flatten() for name,param in self.model_curr.named_parameters() if name!='dual_y'])

            if self.projection_x:
                projection_center = flattened_x[chosen_block] - lr_x*flattened_x_grad[chosen_block]
                flattened_x_new = torch.tensor(pj_x(projection_center.cpu().detach().numpy()),dtype=torch.float64,device =self.device)
                grad_x_tmp = torch.norm(1/lr_x*(flattened_x_new - flattened_x[chosen_block])).item()**2
            else:
                grad_x_tmp = torch.norm(flattened_x_grad[chosen_block]).item()**2

            for name,param in self.model_curr.named_parameters():
                if name == 'dual_y':
                    if self.projection_y:
                        projection_center =  param.data.clone() + lr_y*param.grad.data.clone()
                        y_new = torch.tensor(pj_y(projection_center.cpu().detach().numpy()),dtype=torch.float64, device=self.device)
                        y_new = 1/lr_y*(y_new.clone() - param.data.clone())
                        grad_y_tmp += torch.norm(y_new).item()**2
                    else:
                        grad_y_tmp += torch.norm(param.grad.data).item()**2
        self.record['norm_square_sto_grad_x'][s].append(grad_x_tmp)
        self.record['norm_square_sto_grad_y'][s].append(grad_y_tmp)

        #compute the current loss and acc
        if gx!=None and gy!=None:
            grad_x_tmp, grad_y_tmp = gx**2, gy**2
        else:
            full_batch = torch.arange(self.total_number_data).to(self.device)

            self.model_copy.load_state_dict(self.model_curr.state_dict())
            self.model_copy.dual_y.data = self.maximizer_solver(start=self.model_copy,lr_y=self.maxsolver_step)
            primalF = self.model_copy.loss(self.data,full_batch,self.targets)
            self.record['primalF'][s].append(
                primalF.item())

            self.model_copy.load_state_dict(self.model_curr.state_dict())
            self.model_copy.zero_grad()
            loss = self.model_copy.loss(self.data,full_batch,self.targets)
            loss.backward()
            self.record['loss'][s].append(
                loss.item())
            self.record['acc'][s].append(
                torch.sum(self.model_copy.predict(self.data)==self.targets).item()/self.total_number_data
                )
            

            #compute and save the current norm square of full gradients for plotting y-axis
            grad_x_tmp = 0
            grad_y_tmp = 0
            flattened_x = torch.cat([param.data.clone().flatten() for name,param in self.model_copy.named_parameters() if name!='dual_y'])
            flattened_grad = torch.cat([param.grad.data.clone().flatten() for name,param in self.model_copy.named_parameters() if name!='dual_y'])

            if self.projection_x:
                projection_center =  flattened_x[chosen_block] - lr_x*flattened_grad[chosen_block]
                flattened_x_new = torch.tensor(pj_x(projection_center.cpu().detach().numpy()),dtype=torch.float64,device =self.device)
                grad_x_tmp = torch.norm(1 / lr_x * (flattened_x_new - flattened_x[chosen_block])).item() ** 2
            else:
                grad_x_tmp = torch.norm(flattened_grad[chosen_block]).item()**2

            for name,param in self.model_copy.named_parameters():
                if name == 'dual_y':
                    if self.projection_y:
                        projection_center =  param.data.clone() + lr_y*param.grad.data.clone()
                        y_new = torch.tensor(pj_y(projection_center.cpu().detach().numpy()),dtype=torch.float64, device=self.device)
                        y_new = 1/lr_y*(y_new.clone() - param.data.clone())
                        grad_y_tmp += torch.norm(y_new).item()**2
                    else:
                        grad_y_tmp += torch.norm(param.grad.data).item()**2
        self.record['norm_square_full_grad_x'][s].append(grad_x_tmp)
        self.record['norm_square_full_grad_y'][s].append(grad_y_tmp)

    def show_result(self, s, batch_index, sim_done = False):
        method = self.record['config'][s]['method']
        if sim_done:
            foo = self.record['contraction_times'][s]
            print(f'=================================================================================================')
            print(f'================={s+1}th sim is done!!! method:{method}. contraction:{foo}=================')
        else:
            print(f'-----------------------------------------------------------------------------------------------')
        print('Total Complexity:')
        print('iter:', self.record['total_iter'][s], 'epoch:', self.record['total_epoch'][s], 'OC:',  self.record['total_oracle_complexity'][s], 'SC:',  self.record['total_sample_complexity'][s])
        print('Current Complexity:')
        print('iter:', self.record['iter'][s], 'epoch:', self.record['epoch'][s], 'OC:',  self.record['oracle_complexity'][s], 'SC:',  self.record['sample_complexity'][s])
        print('Iterates Info:')
        print('acc:', self.record['acc'][s][-1], ', loss:', self.record['loss'][s][-1], ', primalF:', self.record['primalF'][s][-1])
        print('||sto grad_x||^2:', self.record['norm_square_sto_grad_x'][s][-1],'||sto grad_y||^2:', self.record['norm_square_sto_grad_y'][s][-1])
        print('||true grad_x||^2:', self.record['norm_square_full_grad_x'][s][-1], '||true grad_y||^2:',
              self.record['norm_square_full_grad_y'][s][-1])
        print('lr_x:',self.record['lr_x'][s][-1], ', lr_y:',self.record['lr_y'][s][-1])
        print('mu:', self.mu_y, ', kappa:', self.kappa)
        if 'LS' in method:
            print('l(small):', self.record['l(small)'][s][-1], \
                ', L(large):', self.record['L(large)'][s][-1],\
                  ', Deltak:', self.record['Deltak'][s][-1].item()
                  )
        print(self.record['config'][s])
        if self.model_type =='DRO':
            print('positive number of yi(batch):',torch.sum(self.model_curr.dual_y[batch_index]>0).item(), \
                    'max y_i:' , torch.max(self.model_curr.dual_y[batch_index]).item(),\
                    'min y_i:' , torch.min(self.model_curr.dual_y[batch_index]).item())
            print('positive number of yi(total):',torch.sum(self.model_curr.dual_y>0).item(), \
                    'max y_i:' , torch.max(self.model_curr.dual_y).item(),\
                    'min y_i:' , torch.min(self.model_curr.dual_y).item())
        if sim_done:
            print(f'=================================================================================================\n\n')
        else:
            print(f'-----------------------------------------------------------------------------------------------\n\n')