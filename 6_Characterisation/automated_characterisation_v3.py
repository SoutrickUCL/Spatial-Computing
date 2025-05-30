import os
import numpy as np
from numpy import random
from matplotlib import pyplot as plt
import importlib
import spatial_characterisation_combination_v3 as slg

# Clear the terminal
os.system('cls' if os.name == 'nt' else 'clear')
###############################################################################
analog=['circular', 'spiral','checked_box']
###############################################################################
#       Functions for generating Traning Data for different logic gates
###############################################################################
gen_func=np.array([
       slg.generate_circular,
       slg.generate_spiral,
       slg.generate_checked_box])
###################################################################################
#                Creating library for input data 
###################################################################################
#cutoff = 10**2.57
#cuton = 10**6.13
ndata =5000
Xt2 = np.random.uniform(-15,-2,2*ndata)
X=10**Xt2.reshape(ndata,2)
###################################################################################
#             Optimising weights for different logic gates
###################################################################################
pref_network_topology = [None for _ in range(len(analog))] 
Output_data = np.array([])
#for ii in range(len(analog)):
for ii in range(1,2):
    runn=f"--------Running--{analog[ii]}-------"
    print(runn)
    ###############################################################################
    #                    Generating the training data
    ###############################################################################
    Y = gen_func[ii](X)
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
    print(f"Best combination of activation function for {analog[ii]} is {hidden_nodes2} and {noutput2}")
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
    Ytest = gen_func[ii](Xtest)
    YY = np.zeros([ntest])
    for i in range(ntest):
      YY[i] = network.forward(Xtest[i,:], 10**wH, 10**wO )
    Ytest=Ytest.reshape(-1,1)
    YY=YY.reshape(-1,1)
    logic_data=[]
    logic_data=np.concatenate((Xtest, Ytest, YY), axis=1)
    #np.save(f'logic_gate_output_{analog[ii]}_REAL_AF.npy', logic_data)
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
    plt.subplot(2,2, ii+1)
    plt.scatter( x, y, c=z1, cmap='viridis', s=15, edgecolors='none')
    cbar = plt.colorbar()
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(analog[ii]) 
    plt.xticks([-2.5, -5, -7.5, -10, -12.5, -15.0])
    plt.yticks([-2.5, -5, -7.5, -10, -12.5, -15.0])
    plt.tight_layout()


    plt.figure(2)
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.subplot(2,2, ii+1)
    plt.scatter(x, y, c=z2, cmap='viridis', s=15, edgecolors='none')
    cbar = plt.colorbar()
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(analog[ii]) 
    plt.xticks([-2.5, -5, -7.5, -10, -12.5, -15.0])
    plt.yticks([-2.5, -5, -7.5, -10, -12.5, -15.0])
    plt.tight_layout()

plt.figure(1)
plt.savefig(f"Training_automate_{analog[ii]}.png")

plt.figure(2)
plt.savefig(f"Predicted_automate_{analog[ii]}.png")

# plt.figure(3) 
# plt.savefig("Output_automate_analog.png")

with open(f'pref_network_topology_{analog[ii]}.txt', 'w') as f:
    for item in pref_network_topology:
        f.write("%s\n" % item)
