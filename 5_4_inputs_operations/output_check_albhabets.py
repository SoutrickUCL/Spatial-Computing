import os
import seaborn as sb
import numpy as np
import pandas as pd
from numpy import random
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from itertools import product 
import importlib
import spatial_check_alphabets as slg
import csv
# Clear the terminal
os.system('cls' if os.name == 'nt' else 'clear')
###############################################################################
logic_gates=[ 'check_alpha']
###############################################################################
#       Functions for generating Traning Data for different logic gates
###############################################################################
gen_func=np.array([slg.generate_check_alpha])
###################################################################################
#                Creating library for input data 
###################################################################################
on_cutoff = 10**6.13
off_cutoff = 10**2.57
ndata =10
desired_word="CAFE"
X2=np.arange(10)
X3 = np.array([[int(bit) for bit in bin(x4)[2:].zfill(4)] for x4 in X2])
X = np.where(X3 == 0, off_cutoff, on_cutoff)
###################################################################################
#             Optimising weights for different logic gates
###################################################################################
pref_network_topology = [None for _ in range(len(logic_gates))] 
Output_data = np.array([])
#for ii in range(1): # Loop over all logic gates
for ii in range(len(logic_gates)):
    runn=f"--------Running--{logic_gates[ii]}-------"
    print(runn)
    ###############################################################################
    #                    Generating the training data
    ###############################################################################
    Y = gen_func[ii](desired_word,X2,on_cutoff, off_cutoff) # Generates the training output data
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
    wH = log_w[0:4*len(hidden_nodes2)].reshape(len(hidden_nodes2),4)
    wO = log_w[4*len(hidden_nodes2):]
    ntest=10
    Xtest=X2
    #print(Xtest)
    Ytest = gen_func[ii](desired_word,X2, on_cutoff, off_cutoff)
    YY = np.zeros([ntest])
    for i in range(ntest):
      YY[i] = network.forward(X[i,:], 10**wH, 10**wO )
    Xtest = Xtest.reshape(-1, 1)
    Ytest = Ytest.reshape(-1, 1)
    YY = YY.reshape(-1, 1)

    logic_data=[]
    logic_data=np.concatenate((Xtest, Ytest, YY), axis=1)
    np.save(f'inputs4_output_{logic_gates[ii]}.npy', logic_data)
    np.save(f'inputs4_weights_{logic_gates[ii]}.npy', log_w)
    ###################################################################################
    #                       Ploting the predicted data 
    ###################################################################################
    x =logic_data[:,0]
    z1=np.log10(logic_data[:,1])
    z2 =np.log10(logic_data[:,2])

    # Calculate the normalized distance to determine the color
    def get_color(y, off_cutoff, on_cutoff):
      norm = (y - off_cutoff) / (on_cutoff - off_cutoff)
      norm = np.clip(norm, 0, 1)  # Ensure values are between 0 and 1
      return plt.cm.RdYlGn(norm)  # Use a red to green colormap

    # Calculate colors based on proximity to cutoffs for z1
    colors_z1 = [get_color(y, np.log10(off_cutoff), np.log10(on_cutoff)) for y in z1]
    colors_z2 = [get_color(y, np.log10(off_cutoff), np.log10(on_cutoff)) for y in z2]
    
    plt.figure(ii)
    plt.subplot(1, 2, 1)
    bars = plt.bar(x, z1, color=colors_z1)
    plt.xlabel('Input', fontsize=14)
    plt.ylabel('Output', fontsize=14)
    plt.title('Desired', fontsize=16)
    plt.xticks(x, [chr(65 + i) for i in range(len(x))], fontsize=10)   
    plt.yticks(fontsize=12)
    
    # Add legend to the first subplot
    green_patch = mpatches.Patch(color='green', label='Yes')
    red_patch = mpatches.Patch(color='red', label='No')
    plt.legend(handles=[green_patch, red_patch], loc='upper right')

    plt.subplot(1, 2, 2)
    bars = plt.bar(x, z2, color=colors_z2)
    plt.xlabel('Input', fontsize=14)
    plt.ylabel('Output', fontsize=14)
    plt.title('Predicted', fontsize=16)
    plt.xticks(x, [chr(65 + i) for i in range(len(x))], fontsize=10) 
    plt.yticks(fontsize=12)

    cbar_ax = plt.gcf().add_axes([0.15, 0.95, 0.7, 0.02])
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='RdYlGn'), cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels([f'{np.log10(off_cutoff):.2f}', f'{(np.log10(off_cutoff) + np.log10(on_cutoff)) / 2:.2f}', f'{np.log10(on_cutoff):.2f}']) 
    plt.tight_layout(rect=[0, 0, 1, 0.93]) 

    #plt.show()
    plt.savefig(f'output_{logic_gates[ii]}.png')

# with open('pref_network_topology.txt', 'w') as f:
#    for item in pref_network_topology:
#       f.write("%s\n" % item)
