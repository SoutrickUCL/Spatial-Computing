import os
import seaborn as sb
import numpy as np
import pandas as pd
from numpy import random
from matplotlib import pyplot as plt
from itertools import product 
import importlib
import spatial_logic_gate_QO_KY_automate_random_AND_NAND as slg
import csv
# Clear the terminal
os.system('cls' if os.name == 'nt' else 'clear')
###############################################################################
logic_gates=['AND', 'NAND']
###############################################################################
#       Functions for generating Traning Data for different logic gates
###############################################################################
gen_func=np.array([
       slg.generate_AND,
       slg.generate_NAND])
###################################################################################
#                Creating library for input data 
###################################################################################
ndata =5000
Xt2 = np.random.uniform(-15,-2,2*ndata)
X=10**Xt2.reshape(ndata,2)
###################################################################################
#             Optimising weights for different logic gates
###################################################################################
pref_network_topology = [None for _ in range(len(logic_gates))] 
Output_data = np.array([])
#for ii in range(4,4): # Loop over all logic gates
for ii in range(len(logic_gates)):
    runn=f"--------Running--{logic_gates[ii]}-------"
    print(runn)
    ###############################################################################
    #                    Generating the training data
    ###############################################################################
    on_cutoff=10**6.13
    off_cutoff=10**2.57
    Y = gen_func[ii](X,on_cutoff, off_cutoff) # Generates the training data
    ###############################################################################
    #Storing Optimised weights(log_w) using Genetic Algorithm for each combination
    ###############################################################################
    log_w, best_network_topology = slg.run_genetic_algorithm(X, Y, ndata)
    print(best_network_topology, 10**np.array(log_w))
    pref_network_topology[ii] = best_network_topology
    ###################################################################################
    #    Best combinations of activation function & corresponding Optimised weights
    ###################################################################################
    hidden_nodes2=best_network_topology[:-1]
    noutput2=best_network_topology[-1]
    ###################################################################################
    #         Generating predicted output from the optimised weights
    ###################################################################################
    network = slg.mlp( hidden_nodes2, noutput2 )
    #log_w= [item[0] if isinstance(item, np.ndarray) else item for item in log_w]
    log_w = np.array(log_w)
    wH = log_w[0:2*len(hidden_nodes2)].reshape(len(hidden_nodes2),2)
    wO = log_w[2*len(hidden_nodes2):]
    ntest = 50000
    Xt2 = np.random.uniform(-15,-2,2*ntest)
    Xtest=10**Xt2.reshape(ntest,2)
    #print(Xtest)
    Ytest = gen_func[ii](Xtest, on_cutoff, off_cutoff)
    YY = np.zeros([ntest])
    for i in range(ntest):
      YY[i] = network.forward(Xtest[i,:], 10**wH, 10**wO )
    Ytest=Ytest.reshape(-1,1)
    YY=YY.reshape(-1,1)
    logic_data=[]
    logic_data=np.concatenate((Xtest, Ytest, YY), axis=1)
    #np.save(f'logic_gate_output_{logic_gates[ii]}_REAL_AF.npy', logic_data)
    ###################################################################################
    #                       Ploting the predicted data 
    ###################################################################################
    x =np.log10(logic_data[:,0])
    y = np.log10(logic_data[:,1])
    z1=np.log10(logic_data[:,2])
    z2 =np.log10(logic_data[:,3])

    plt.figure(1)
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.subplot(2,3, ii+1)
    plt.scatter( x, y, c=z1, cmap='viridis', s=15, edgecolors='none')
    cbar = plt.colorbar()
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(logic_gates[ii]) 
    plt.tight_layout()


    plt.figure(2)
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.subplot(2,3, ii+1)
    plt.scatter(x, y, c=z2, cmap='viridis', s=20, edgecolors='none')
    cbar = plt.colorbar()
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(logic_gates[ii]) 
    plt.tight_layout()
    #plt.show()
    ###############################################################################
    #        Predict the loss and Fold Change with the optimised weights
    ###############################################################################
    logic_inputs=[[1e-15,1e-15],#[0,0]
               [1e-2,1e-15],    #[1,0]
               [1e-15,1e-2],    #[0,1]
               [1e-2,1e-2] ]     #[1,1]
    data_output=[]
    for jj in range(4):
      final_output = network.forward(logic_inputs[jj], 10**wH, 10**wO)
      data_output=np.append(data_output,final_output)
    categories = ['00', '10', '01', '11']
    values = np.round(data_output,2)
    if(ii==0): #OR
       off_output=data_output[0]
       on_output=np.min(data_output[1:])
    if(ii==1): #NOR
          off_output=np.max(data_output[1:])
          on_output=data_output[0]
    if(ii==2): #AND
          off_output=np.max(data_output[:2])
          on_output=data_output[3]
    if(ii==3): #NAND
          off_output=data_output[3]
          on_output=np.min(data_output[:2])
    #if(ii==4): #XOR
     #     off_output = np.max([data_output[0], data_output[3]])
     #     on_output=np.min(data_output[1:2])

    Fold_change=on_output/off_output
    Loss_logic=on_output/off_output
    Fold_change=np.round(Fold_change,2)
    Loss_logic=np.round(Loss_logic,2) 
    ###############################################################################
    #                     Ploting Output as Bar diagram 
    ###############################################################################
    normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))
    colors = plt.cm.viridis(normalized_values)
    plt.figure(3)
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.subplot(2,3, ii+1)
    bars = plt.bar(categories, values, color=colors)
    for bar, value in zip(bars, values):
     plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(value), ha='center')
    #if ii==0 and ii==2:
    #  text_box = plt.text(00, data_output[0]+400, f"Fold Change= {Fold_change}")
    #  text_box = plt.text(00, data_output[0]+125, f"Loss= {Loss_logic}")
    #else:
    #  text_box = plt.text(11, data_output[3]+400, f"Fold Change= {Fold_change}")
    #  text_box = plt.text(11, data_output[3]+125, f"Loss= {Loss_logic}")

    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.ylabel('Output', fontsize=12)
    plt.title(logic_gates[ii])
    plt.tight_layout()
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    #file_name = f'logic_data_ideal_func_{logic_gates[ii]}.csv'
    np.save(f'logic_data_output_all_logic_real_AF_{logic_gates[ii]}.npy', logic_data)
    np.save(f'logic_data_weights_all_logic_real_AF_{logic_gates[ii]}.npy', log_w)
    #with open(file_name, 'w', newline='') as f:
     # writer = csv.writer(f)
     # writer.writerow(logic_data)
###############################################################################
  
plt.figure(1)
plt.savefig("Training_automate_all_logic_real_AF_AND_NAND.png")

plt.figure(2)
plt.savefig("Predicted_automate_all_logic_real_AF_AND_NAND.png")

plt.figure(3) 
plt.savefig("Output_automate_all_logic_real_AF_AND_NAND.png")

with open('pref_network_topology_all_logic_real_AF_AND_NAND.txt', 'w') as f:
    for item in pref_network_topology:
        f.write("%s\n" % item)
