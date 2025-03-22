import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from ALG.Utils import *
import numpy as np
from datetime import datetime

# example of pickle
# l = [1,2,3,4]
# with open("test", "wb") as fp:   #Pickling
#     pickle.dump(l, fp)

# with open("test", "rb") as fp:   # Unpickling
#     b = pickle.load(fp)
DATA_LIMIT = 100000
PLOT_LIMIT = 100000
stdx = 1
stdy = 1
b = 10
mu_y = 1
for kappa in [5,10,50]:
    b=10
    data_name = f'Q_stdx_{stdx}_stdy_{stdy}' + '_muy_' + str(mu_y) + '_kappa_' + str(kappa) + f'_b_{b}'
    data_path = f'./result_data/{data_name}'
    newdata_path = f'./result_data/tilde_sigma_new_{data_name}'

    for plot_part in ['z']:# ['x','y','z','loss','acc','lr_x','lr_y']:
        G = {}
        G['GS-GDA-B,N=2'] = newdata_path +'/primal_line_search_N_2_AGDA'
        G['GS-GDA-B,N=5'] = newdata_path +'/primal_line_search_N_5_AGDA'
        G['GS-GDA-B,N=1'] = newdata_path +'/primal_line_search_N_1_AGDA'
        G['Smooth-AGDA'] = data_path + '/Smooth-AGDA'
        G['J-GDA'] = data_path +'/GDA'
        G['GS-GDA'] = data_path +'/AGDA'
        G['TiAda'] = data_path +'/TiAda'
        G['VRLM'] = f'./result_data/' + f'Q_stdx_{stdx}_stdy_{stdy}' + '_muy_' + str(mu_y) + '_kappa_' + str(kappa) + f'_b_{b}' +'/VRLM'


        plt.figure(dpi=150)
        fig, ax = plt.subplots()
        is_log = False
        C = 0.0  # value center for log s

        for alg_name, file_name in G.items():
            data_xLimit = DATA_LIMIT
            plot_xLimit = PLOT_LIMIT
            with open(file_name, "rb") as fp:  # Unpickling
                record = pickle.load(fp)
                # load x-axis data
                idx = most_frequent_id(record['contraction_times'])
                if 'GS-GDA-B,N=' in alg_name:
                    oracle_complexity_counter = record['oracle_complexity_counter'][idx]
                    sample_complexity_counter = record['sample_complexity_counter'][idx]
                    iter_counter = record['iter_counter'][idx]
                    epoch_counter = record['epoch_counter'][idx]
                    total_oracle_complexity_counter = record['total_oracle_complexity_counter'][idx]
                    total_sample_complexity_counter = record['total_sample_complexity_counter'][idx]
                    total_iter_counter = record['total_iter_counter'][idx]
                    total_epoch_counter = record['total_epoch_counter'][idx]
                    total_oracle_complexity_counter = [[i*record['total_oracle_complexity'][s]/(len(record['norm_square_sto_grad_x'][s])-1) for i in range(len(record['norm_square_sto_grad_x'][s]))] for s in range(len(record['total_oracle_complexity']))]
                    for s in range(len(total_oracle_complexity_counter)):
                        assert len(total_oracle_complexity_counter[s]) == len(record['norm_square_sto_grad_x'][s])
                else:
                    oracle_complexity_counter = min(record['oracle_complexity_counter'], key=len)
                    sample_complexity_counter = min(record['sample_complexity_counter'], key=len)
                    iter_counter = min(record['iter_counter'], key=len)
                    epoch_counter = min(record['epoch_counter'], key=len)
                    total_oracle_complexity_counter = min(record['total_oracle_complexity_counter'], key=len)
                    total_sample_complexity_counter = min(record['total_sample_complexity_counter'], key=len)
                    total_iter_counter = min(record['total_iter_counter'], key=len)
                    total_epoch_counter = min(record['total_epoch_counter'], key=len)
                    counter = total_oracle_complexity_counter[:data_xLimit]
                if alg_name == 'VRLM':
                    b0 = 1
                else:
                    b0= 10
                
                #data_xLimit = min(data_xLimit, len(counter))

                # load y-axis data
                if 'GS-GDA-B,N=' in alg_name:
                    valid_line_search = [i for i in range(len(record['acc'])) if record['contraction_times'][i] == record['contraction_times'][idx]]
                    valid_line_search = valid_line_search[:10]
                    valid_line_search = [i for i in range(len(record['acc']))]
                else:
                    valid_line_search = [i for i in range(len(record['acc']))]
                print(valid_line_search)

                acc = record['acc']
                acc = [acc[i][:data_xLimit] for i in valid_line_search]
                loss = [record['loss'][i][:data_xLimit] for i in valid_line_search]
                primal = [record['primalF'][i][:data_xLimit] for i in valid_line_search]
                error = [[1 - ele[i] for i in range(len(ele))] for ele in acc]
                lr_x = record['lr_x']
                lr_y = record['lr_y']
                lr_x = [lr_x[i][:data_xLimit] for i in valid_line_search]
                lr_y = [lr_y[i][:data_xLimit] for i in valid_line_search]

                norm_sqaure_sto_grad_x = [record['norm_square_sto_grad_x'][i][:data_xLimit] for i in valid_line_search]
                norm_sqaure_sto_grad_y = [record['norm_square_sto_grad_y'][i][:data_xLimit] for i in valid_line_search]                
                norm_sqaure_full_grad_x = [record['norm_square_full_grad_x'][i][:data_xLimit] for i in valid_line_search]
                norm_sqaure_full_grad_y = [record['norm_square_full_grad_x'][i][:data_xLimit] for i in valid_line_search]

                if 'GS-GDA-B,N=' in alg_name:
                    total_oracle_complexity_counter = [total_oracle_complexity_counter[i][:data_xLimit] for i in valid_line_search]
                    counter, norm_sqaure_sto_grad_x = comlementdata(total_oracle_complexity_counter, norm_sqaure_sto_grad_x)
                    counter, norm_sqaure_sto_grad_y = comlementdata(total_oracle_complexity_counter, norm_sqaure_sto_grad_y)
                    counter, norm_sqaure_full_grad_x = comlementdata(total_oracle_complexity_counter, norm_sqaure_full_grad_x)
                    counter, norm_sqaure_full_grad_y = comlementdata(total_oracle_complexity_counter, norm_sqaure_full_grad_y)
                    _, acc = comlementdata(total_oracle_complexity_counter, acc)
                    _, loss = comlementdata(total_oracle_complexity_counter, loss)
                    _, primal = comlementdata(total_oracle_complexity_counter, primal)
                    _, error = comlementdata(total_oracle_complexity_counter, error)
                    _, lr_x = comlementdata(total_oracle_complexity_counter, lr_x)
                    _, lr_y = comlementdata(total_oracle_complexity_counter, lr_y)

                norm_sqaure_sto_grad_z = [[norm_sqaure_sto_grad_x[i][j] + norm_sqaure_sto_grad_y[i][j] for j in
                                            range(len(norm_sqaure_full_grad_x[i]))] for i in
                                           range(len(norm_sqaure_full_grad_x))]
                norm_sqaure_full_grad_z = [[norm_sqaure_full_grad_x[i][j] + norm_sqaure_full_grad_y[i][j] for j in
                                            range(len(norm_sqaure_full_grad_x[i]))] for i in
                                           range(len(norm_sqaure_full_grad_x))]

                # norm_sqaure_sto_grad_x = normlize_data(norm_sqaure_sto_grad_x)
                # norm_sqaure_sto_grad_y = normlize_data(norm_sqaure_sto_grad_y)
                # norm_sqaure_sto_grad_z = normlize_data(norm_sqaure_sto_grad_z)
                # norm_sqaure_full_grad_x = normlize_data(norm_sqaure_full_grad_x)
                # norm_sqaure_full_grad_y = normlize_data(norm_sqaure_full_grad_y)
                # norm_sqaure_full_grad_z = normlize_data(norm_sqaure_full_grad_z)

                contraction_times = record['contraction_times']
                #b = record['config'][-1]['b']
                # N = record['config'][-1]['N']

                if plot_part == 'x':
                    shadowplot(counter, norm_sqaure_full_grad_x, label_input=alg_name, alpha=0.5, center=C, is_log=is_log,
                               is_var=True, alg_name=alg_name)
                elif plot_part == 'y':
                    shadowplot(counter, norm_sqaure_full_grad_y, label_input=alg_name, alpha=0.5, center=C, is_log=is_log,
                               is_var=False, alg_name=alg_name)
                elif plot_part == 'z':
                    shadowplot(counter, norm_sqaure_full_grad_z, label_input=alg_name, alpha=0.5, center=C, is_log=is_log,
                               is_var=False, alg_name=alg_name)
                elif plot_part == 'acc':
                    shadowplot(counter, error, label_input=alg_name, alpha=0.5, center=C, is_log=is_log, is_var=True,
                               alg_name=alg_name)
                elif plot_part == 'loss':
                    shadowplot(counter, primal, label_input=alg_name, alpha=0.5, center=C, is_log=is_log, is_var=False,
                               alg_name=alg_name)
                elif plot_part == 'lr_x':
                    shadowplot(counter, lr_x, label_input=alg_name, alpha=0.5, center=C, is_log=is_log, is_var=True,
                               alg_name=alg_name)
                elif plot_part == 'lr_y':
                    shadowplot(counter, lr_y, label_input=alg_name, alpha=0.5, center=C, is_log=is_log, is_var=True,
                               alg_name=alg_name)

        if plot_part == 'x':
            plt.legend(fontsize=15, loc='upper right')
        elif plot_part == 'y':
            plt.legend(fontsize=15, loc='upper right')
        elif plot_part == 'z':
            if mu_y == 0.0001:
                plt.legend(fontsize=15, loc='lower right')
            else:
                plt.legend(fontsize=15, loc='lower left')
            plt.legend(fontsize=15, loc='upper right')
        elif plot_part == 'acc':
            plt.legend(fontsize=15, loc='upper right')
        elif plot_part == 'loss':
            plt.legend(fontsize=15, loc='upper right')
        elif plot_part == 'lr_x':
            plt.legend(fontsize=15, loc='lower right')
        elif plot_part == 'lr_y':
            plt.legend(fontsize=15, loc='lower right')

        plt.xlabel("Number of gradient calls", fontsize=15)

        if plot_part == 'x':
            plt.ylabel(r"$\frac{||\nabla_x\mathcal{L}(x,y)||^2}{||\nabla_x\mathcal{L}(x_0,y_0)||^2}$", fontsize=15)
        elif plot_part == 'y':
            plt.ylabel(r"$\frac{||\nabla_y\mathcal{L}(x,y)||^2}{||\nabla_y\mathcal{L}(x_0,y_0)||^2}$", fontsize=15)
        elif plot_part == 'z':
            plt.ylabel(r"$\|\nabla\mathcal{L}(x,y)||^2$", fontsize=15)
            #plt.ylabel(r"$\frac{||\nabla\mathcal{L}(x_k,y_k)||^2}{||\nabla\mathcal{L}(x_0,y_0)||^2}$", fontsize=15)
        elif plot_part == 'acc':
            plt.ylabel(r"Train Error", fontsize=15)
        elif plot_part == 'loss':
            plt.ylabel(r"Loss", fontsize=15)
        elif plot_part == 'lr_x':
            plt.ylabel(r"Step size $\tau$", fontsize=15)
        elif plot_part == 'lr_y':
            plt.ylabel(r"Step size $\sigma$", fontsize=15)

        # set label size here
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)

        # set x,y range here
        #
        # plt.ylim(1e-2,)
        plt.xlim(0,plot_xLimit)


        # set personalized axis scale here
        if plot_part == 'x':
            plt.yscale('log')
        elif plot_part == 'y':
            plt.yscale('log')
        elif plot_part == 'z':
            plt.yscale('log')
        elif plot_part == 'acc':
            plt.ylim(0, 0.6)
        elif plot_part == 'loss':
            plt.yscale('log')
        elif plot_part == 'lr_x':
            plt.yscale('log')
        elif plot_part == 'lr_y':
            plt.yscale('log')

        # plt.xscale('log')
        if is_log:
            ax.set_yticklabels([round(np.exp(y) + C, 2) for y in ax.get_yticks()], fontsize=10)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.xaxis.offsetText.set_visible(True)
        plt.grid()
        # set title here
        # plt.title('Qudradic_Bilinear_Obj',fontsize = 15)

        data_name_tmp = list(data_name)
        for i in range(len(data_name_tmp)):
            if data_name_tmp[i] == '.':
                data_name_tmp[i] = '_'
        # # 放置图例
        # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # # 调整布局以避免裁剪图例
        # plt.tight_layout()

        current_date = datetime.now().strftime("%Y%m%d")
        # plt.savefig(f'./figure/tilde_sigma{"".join(data_name_tmp)}_{plot_part}.pdf', bbox_inches='tight', facecolor='w', dpi=150)
        plt.savefig(f'./figure/tilde_sigma_new_{"".join(data_name_tmp)}_{plot_part}_{current_date}.png', bbox_inches='tight', facecolor='w', dpi=100)
        plt.close()