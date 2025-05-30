import os
import seaborn as sb
import numpy as np
import pandas as pd
from numpy import random
from matplotlib import pyplot as plt
import importlib
import spatial_logic_gate_fixed_positions_diff_300_real_AF2 as slg
###############################################################################
# Clear the terminal
os.system('cls' if os.name == 'nt' else 'clear')
###############################################################################
logic_gates=['OR','NOR', 'AND', 'NAND', 'XOR']
###############################################################################
#       Functions for generating Traning Data for different logic gates
###############################################################################
gen_func=np.array([
       slg.generate_OR,
       slg.generate_NOR,
       slg.generate_AND,
       slg.generate_NAND,
       slg.generate_XOR])
###############################################################################
#          Desired configuration of MLP for specfic logic gates
###############################################################################
hidden_nodes_OR=['HP', 'HP']
output_node_OR='HP'

hidden_nodes_NOR=['HP', 'HP']
output_node_NOR='LP'

hidden_nodes_AND=['LP', 'LP']
output_node_AND='LP'

hidden_nodes_NAND=['LP', 'LP']
output_node_NAND='HP'

hidden_nodes_XOR=['HP', 'LP']
output_node_XOR='LP'

hidden_layer_config=[hidden_nodes_OR, hidden_nodes_NOR, hidden_nodes_AND, hidden_nodes_NAND, hidden_nodes_XOR]
output_layer_config=[output_node_OR, output_node_NOR, output_node_AND, output_node_NAND, output_node_XOR ]
###################################################################################
#                Creating library for input data 
###################################################################################
ndata = 5000
Xt = np.random.uniform(-15,-2,2*ndata)
X=10**Xt.reshape(ndata,2)
#print (f"Input data: {X}")
###################################################################################
#             Optimising weights for different logic gates
###################################################################################
Output_data = np.array([])
for ii in range(len(logic_gates)):
#for ii in range(1):
    runn=f"---------Running {logic_gates[ii]}----------"
    print(runn)
    ###############################################################################
    #                   Imposing the architecture
    ###############################################################################
    hidden_nodes = hidden_layer_config[ii]
    noutput=output_layer_config[ii]
    network = slg.mlp( hidden_nodes, noutput )
    ###############################################################################
    #                Generating the training data
    ###############################################################################
    off_cutoff=10 ** 2.57
    on_cutoff=10 ** 6.13
    Y = gen_func[ii](X,off_cutoff, on_cutoff)
    plt.figure(1)
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.subplot(2, 3, ii+1)
    plt.scatter( np.log10(X[:,0]), np.log10(X[:,1]), c=np.log10(Y),cmap='viridis', s=15, edgecolors='none')
    cbar = plt.colorbar()
    plt.xlabel('I1')
    plt.ylabel('I2')
    plt.title(logic_gates[ii]) 
    plt.tight_layout()
    #plt.show()
    ###############################################################################
    #           Optimised weights(log_w) using Genetic Algorithm
    ###############################################################################
    best_x_indices_opt,best_y_indices_opt  = slg.run_genetic_algorithm(network, X, Y, ndata)
    np.save(f'logic_gate_x_indices_{logic_gates[ii]}_fixed_position_diff_300_REAL_AF2_test.npy', best_x_indices_opt)
    np.save(f'logic_gate_y_indices_{logic_gates[ii]}_fixed_position_diff_300_REAL_AF2_test.npy', best_y_indices_opt)
    ###############################################################################
    #         Generating predicted output from the optimised weights
    ###############################################################################
    log_w = slg.calculate_weights(best_x_indices_opt, best_y_indices_opt)
    np.save(f'logic_gate_weights_{logic_gates[ii]}_fixed_position_diff_300_REAL_AF2_test.npy', log_w)
    wH = log_w[0:2*len(hidden_nodes)].reshape(len(hidden_nodes),2)
    wO = log_w[2*len(hidden_nodes):]
    # print(f"Optimised weights for hidden layer: {wH}")
    # print(f"Optimised weights for output layer: {wO}")
    ntest = 50000
    Xt2 = np.random.uniform(-15,-2,2*ntest)
    Xtest=10**Xt2.reshape(ntest,2)
    Ytest = gen_func[ii](Xtest,off_cutoff, on_cutoff)
    YY = np.zeros([ntest])
    for i in range(ntest):
      YY[i] = network.forward(Xtest[i,:], wH, wO )
    Ytest=Ytest.reshape(-1,1)
    YY=YY.reshape(-1,1)
    logic_data=np.concatenate((Xtest, Ytest, YY), axis=1)
    np.save(f'logic_gate_output_sql_{logic_gates[ii]}_fixed_position_diff_300_REAL_AF2_test.npy', logic_data)
    ###############################################################################
    #                Ploting the predicted data 
    ###############################################################################
    x = np.log10(logic_data[:,0])
    y = np.log10(logic_data[:,1])
    z1=np.log10(logic_data[:,2])
    z2 =np.log10(logic_data[:,3])
    plt.figure(2)
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.subplot(2, 3, ii+1)
    plt.scatter(x, y, c=z2, cmap='viridis', s=20, edgecolors='none')
    cbar = plt.colorbar()
    plt.xlabel('I1')
    plt.ylabel('I2')
    plt.title(logic_gates[ii]) 
    plt.tight_layout()
    #plt.show()
    ###############################################################################
    #        Predict the loss and Fold Change with the optimised weights
    ###############################################################################
    logic_inputs=[[1e-15,1e-15],#[0,0]
               [1e-2,1e-15],    #[1,0]
               [1e-15,1e-2],    #[0,1]
               [1e-2,1e-2]      #[1,1]
               ]  # Setting OFF=0.001 & ON=1000
    data_output=[]
    for jj in range(4):
      final_output = network.forward(logic_inputs[jj], wH, wO)
      data_output=np.append(data_output,final_output)
      # print(f"Output for {logic_inputs[jj]} is {final_output}")
    categories = ['00', '10', '01', '11']
    data_output=np.log10(data_output)
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

    Fold_change=on_output/off_output
    Loss_logic=on_output/off_output
    Fold_change=np.round(Fold_change,2)
    Loss_logic=np.round(Loss_logic,2)  


    ###############################################################################
    #               Ploting Output as Bar diagram 
    ###############################################################################
    normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))
    colors = plt.cm.viridis(normalized_values)
    plt.figure(3)
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.subplot(2, 3, ii+1)
    bars = plt.bar(categories, values, color=colors)
    for bar, value in zip(bars, values):
      plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha='center', fontsize=10)
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

    #plt.show()
    np.save(f'output_values_{logic_gates[ii]}_fixed_position_diff_300_REAL_AF2_test.npy', np.array(values))
   ###############################################################################
  
plt.figure(1)
plt.savefig("Training_actual_sq_lattice_fixed_position_diff_300_REAL_AF2_test.png")

plt.figure(2)
plt.savefig("Predicted_data_actual_sq_lattice_fixed_position_diff_300_REAL_AF2_test.png")

plt.figure(3) 
plt.savefig("Output_ON_OFF_actual_sq_lattice_fixed_position_diff_300_REAL_AF2_test.png")