import numpy as np
from numpy import random
import random as py_random
from matplotlib import pyplot as plt
import math
# Create a class to define an mlp with a single hidden layer, fully connected, with two inputs and one output
# The input nodes are fixed to be HP and the output node fixed to be LP
# HP, LP define the types of nodes in the hidden layer
# The number of weights in this mlp is 3xnH where nH is the number of weights in the hidden layer

#######################################################################
#              Defining High pass function
#######################################################################
def response_hp(x, ymin, ymax, K, n):
#    return ( 10**ymin + (10**ymax-10**ymin)*( (x**n)/( K**n + x**n) ) )
    return 10**( ymin + (ymax-ymin)*( (x**n)/( K**n + x**n) ) )
#######################################################################
#              Defining Low pass function
#######################################################################
def response_lp(x, ymin, ymax, K, n):
#    return (10**ymin + (10**ymax-10**ymin)*( K**n /( (K**n + x**n) ) ))
    return 10**( ymin + (ymax-ymin)*( K**n /( K**n + x**n) ) )
#----------------------------------------------------------------------
def response_hp_s(x):
    #return response_hp(x, -14, -2, 6.98e-9, 0.866)
    return response_hp(x, 2.57, 6.13, 6.98e-9, 0.866)
    #return response_hp(x, 2.5, 47.5, 25, 6)

def response_lp_s(x):
    #return response_lp(x, -14, -2, 7.26e-8, 2.11)
    return response_lp(x, 3.05, 5.37, 7.26e-8, 2.11)
    #return response_lp(x, 2.5, 47.5, 25, 6)
#######################################################################
#def plot_activation_fns():
#    x = np.linspace(start=0.001, stop=50, num=100)
#    f = response_hp(x, 2.5, 47.5, 25, 6)
#
#    fig, ax = plt.subplots()
#    ax.plot( x , f )
#    ax.set_xlabel("log10 x (uM)")
#    ax.set_ylabel("Output")
#    
#    x =np.linspace(start=0.001, stop=50, num=100)
#    f = response_lp(x, 2.5, 47.5, 25, 6)
#
#    fig, ax = plt.subplots()
#    ax.plot( x , f )
#    ax.set_xlabel("log10 x (uM)")
#    ax.set_ylabel("Output")
#    plt.show()
#######################################################################
#                 Defining the MLP network
#######################################################################
class mlp():
    def __init__(self, hidden, noutput):
        #print("Hidden nodes:", hidden)
        #print("Output nodes:", noutput)
        self.hidden = hidden
        self.noutput = noutput
        self.nhidden = len(hidden)
        #------------------------------------------------------------
        #          Nodes will contain a list of functions
        #------------------------------------------------------------
        self.nodes = []
        for n in hidden:
            if n == 'HP':
                self.nodes.append( response_hp_s )
            else:
                self.nodes.append( response_lp_s )
        
    def forward(self, I, wH, wO):
        activationsI = np.zeros([2]) 
        activationsH = np.zeros([self.nhidden]) 
        activationsO = np.zeros([1]) 
        #------------------------------------------------------------
        #                    Input layer
        #------------------------------------------------------------
        activationsI[0] = I[0]
        activationsI[1] = I[1]
        #------------------------------------------------------------
        #                    Hidden layer
        #------------------------------------------------------------
        for ii in range(self.nhidden):
            activationsH[ii] = self.nodes[ii]( wH[ii,0]*activationsI[0] + wH[ii,1]*activationsI[1] )
            #print("Hidden node:", ii, "Activation:", wH[ii,:],activationsH[ii])
        #------------------------------------------------------------
        #                    Output layer
        #------------------------------------------------------------
        if self.noutput == 'HP':
          activationsO[0] = response_hp_s( np.dot(wO, activationsH) )
        else:
          activationsO[0] = response_lp_s( np.dot(wO, activationsH) )
        #print("Output node:", wO, activationsO[0])
        return (activationsO[0])
#######################################################################
#               Generating training data
#######################################################################
def generate_circular(X):
    log_center=-9.0, 
    radius_log=3.0
    X_log = np.log10(X)
    # Compute squared distance from center in log space
    dist_sq = (X_log[:, 0] - log_center) ** 2 + (X_log[:, 1] - log_center) ** 2
    # Label as 1 if inside the log-circle, else 0
    Y = (dist_sq <= radius_log**2).astype(int)
    return Y
#----------------------------------------------------------------------
def generate_spiral(X):
    num_spirals=3
    center_log=-9.0
    X_log = np.log10(X)
    # Compute polar coordinates in log space
    x_shifted = X_log[:, 0] - center_log
    y_shifted = X_log[:, 1] - center_log
    theta = np.arctan2(y_shifted, x_shifted)
    radius = np.sqrt(x_shifted**2 + y_shifted**2)
    # Determine if the point is in a spiral
    Y = ((radius - theta * num_spirals / (2 * np.pi)) % 2 < 1).astype(int)
    return Y
#------------------------------------------------------------------------
def generate_checked_box(X):
    grid_size = 3
    X_log = np.log10(X)
    # Normalize the data to fit into a [0, grid_size] range
    x_min, x_max = -15, -2  # Log10 range of X
    y_min, y_max = -15, -2  # Log10 range of Y
    x_normalized = ((X_log[:, 0] - x_min) / (x_max - x_min)) * grid_size
    y_normalized = ((X_log[:, 1] - y_min) / (y_max - y_min)) * grid_size
    # Convert normalized values to grid indices
    x_indices = np.floor(x_normalized).astype(int) 
    y_indices = np.floor(y_normalized).astype(int)
    # Determine if the grid cell is "black" (1) or "white" (0) based on parity
    Y = ((x_indices + y_indices) % 2).astype(int)
    return Y

#######################################################################
#                   Define Loss function (RMSLE)
#######################################################################
def loss_fn(X,Y,w,network, ndata, num_hidden_nodes, noutput):
    if noutput == 'HP':
        cutoff = 10**(2.57)
        cuton = 10**(6.13)
    else:
        cutoff = 10**(3.05)
        cuton = 10**(5.37)
    Yhat = np.zeros_like(Y, dtype=float)
    Yhat_norm = np.zeros_like(Y, dtype=float)
    Yhat_clamped = np.zeros_like(Y, dtype=float)
    w = np.array(w)
    if(len(w) != 3*num_hidden_nodes):
        print(w, num_hidden_nodes)
    wH = w[0:2*num_hidden_nodes].reshape(num_hidden_nodes,2) 
    wO = w[2*num_hidden_nodes:]
    for ii in range(ndata):
        Yhat[ii]= network.forward(X[ii,:], 10**(wH), 10**(wO) )
        #print("Yhat:", Yhat[ii], cutoff, cuton)
        #Yhat[ii] = np.clip(Yhat[ii], cutoff, cuton)
        eps = 1e-15
        Yhat_norm[ii] = (Yhat[ii] - cutoff  ) / (cuton - cutoff)
        Yhat_clamped[ii] = np.clip(Yhat_norm[ii], eps, 1 - eps)
        #print(noutput,Y[ii],Yhat[ii],Yhat_clamped[ii],-((Y[ii] * np.log(Yhat_clamped[ii] ) + (1 - Y[ii]) * np.log(1 - Yhat_clamped[ii]  ))) )
    return -((Y * np.log(Yhat_clamped ) + (1 - Y) * np.log(1 - Yhat_clamped  ))).mean()
##############################################################################################
#                                   Genetic Alogrithm
##############################################################################################
def run_genetic_algorithm(X, Y, ndata):
    ngen = 20 #Number of generations
    npop = 50000 # Number of network topologies
    pop_cut = int(npop/2) # Number of topologies to keep intact for next generation
    mutation_prob = 0.2   # Probability of mutation 20%
    nrecomb = int(npop/5) #Perform recombination on 20% of network topologies
    fitness = np.zeros([ngen,npop])
    #intact=int(npop/10) # Number of best topologies to keep intact for next generation
    #------------------------------------------------------------------------
    #Starting network topologies of 2 node in hidden layer and 1 output node
    #-----------------------------------------------------------------------
    network_topology = [[None for _ in range(npop)] for _ in range(ngen)]
    act_func = ['HP', 'LP'] # Options for activation functions
    #network_topology[0] = [[random.choice(act_func) for _ in range(3)] for _ in range(npop)]# Random combinations of 'HP' and 'LP'
    network_topology[0] = [[py_random.choice(act_func) for _ in range(3)] for _ in range(npop)]
                                      #Can change the range of random.randint to change the number of hidden nodes (3,3) to (2,4) for example
    #print(network_topology[0])
    #------------------------------------------------------------------------
    #           Initializing weights for initial topologies
    #------------------------------------------------------------------------
    weights = [[None for _ in range(npop)] for _ in range(ngen)]
    for ii in range(npop):
         num_hidden_nodes = len(network_topology[0][ii][:-1])
         weights[0][ii] = [random.uniform(-50,0) for _ in range(3*num_hidden_nodes)] # Initialize the weights with random values
    #print(weights[0])
    #------------------------------------------------------------------------
    #            Calculating fitness of initial topologies
    #------------------------------------------------------------------------
    for jj in range(npop):
        hidden_nodes = network_topology[0][jj][:-1]
        noutput = network_topology[0][jj][-1]   
        network = mlp(hidden_nodes, noutput)
        num_hidden_nodes = len(hidden_nodes)
        #print("Number of hidden nodes:", num_hidden_nodes)
        #print("Check:", noutput)
        fitness[0,jj] = loss_fn(X,Y,weights[0][jj][:],network, ndata, num_hidden_nodes, noutput)
        #print("\tFitness:",fitness[0,jj])
    #------------------------------------------------------------------------
    #              Loop over generations starts here
    #------------------------------------------------------------------------
    for ii in range(1,ngen): 
        print("Generation: ",ii)
        #--------------------------------------------------------------------
        #            Take top 50% of previous generation
        #--------------------------------------------------------------------
        srt = np.argsort( fitness[ii-1,:] )
        weights[ii][0:pop_cut] = [weights[ii-1][index] for index in srt[0:pop_cut]]
        weights[ii][pop_cut:]= [weights[ii-1][index] for index in srt[0:pop_cut]]
        network_topology[ii][0:pop_cut]= [network_topology[ii-1][index] for index in srt[0:pop_cut]]
        network_topology[ii][pop_cut:]= [network_topology[ii-1][index] for index in srt[0:pop_cut]]
        fitness[ii][0:pop_cut]= [fitness[ii-1][index] for index in srt[0:pop_cut]]

        segment_size = (npop - pop_cut) // 6
        two_thirds_size = (2 * (npop - pop_cut)) // 3
        #print("Segment size:", segment_size, "Two thirds size:", two_thirds_size)

        #-------------------------------------------------------------------------------------------------
        #                                   Mutation operator
        #   Pertubing only the first 1/6 of the population after pop_cut (Nodes)
        #-------------------------------------------------------------------------------------------------
        #for jj in range(0,intact):
        #    ipert2 = random.randint(0, len(weights[ii][jj])-1) # Selecting a random weight to perturb for intact topologies
        #    new_weight = random.normal(loc=weights[ii][jj][ipert2], scale=1e-3)
        #    weights[ii][jj][ipert2] = new_weight
        #-------------------------------------------------------------------------------------------------
        for jj in range(pop_cut,pop_cut + segment_size):
            mutation_counter = random.uniform(0, 1)
            #-------------------------------------------------------------------------------------------------
            #     Adding, deleting or modifying nodes in the network topology
            #-------------------------------------------------------------------------------------------------
            if mutation_counter < mutation_prob:
                position = random.randint(0, len(network_topology[ii][jj])-1)# Randomly choose a node in the random topology to alter
                operation = random.choice(['add', 'delete', 'modify'])# Randomly choose between addition, deletion and modification
                if operation == 'add':
                    node_to_add = random.choice(['HP', 'LP'])
                    #network_topology[ii][jj].insert(position, node_to_add) # Inserting a new node
                    network_topology[ii][jj] = np.insert(network_topology[ii][jj], position, node_to_add).tolist()
                    #print("Adding node:", node_to_add, "at position:", position)
                    #print("Network topology after addition:", network_topology[ii][jj])
                elif operation == 'delete':
                     if len(network_topology[ii][jj]) >= 3: # Ensure that after deletion there won't be fewer than 2 nodes in the network
                        network_topology[ii][jj] = np.delete(network_topology[ii][jj], position).tolist() # Deleting a node
                        #print("Network topology after deletion:", network_topology[ii][jj])
                elif operation == 'modify':
                     if network_topology[ii][jj][position] == 'LP':
                         network_topology[ii][jj][position] = 'HP'
                     elif network_topology[ii][jj][position] == 'HP':
                         network_topology[ii][jj][position] = 'LP'
        
        #-------------------------------------------------------------------------------------------------
        #               Just modifying weights in the network topology
        #       Pertubing only the next 2/3 of the population after pop_cut (Weights)
        #-------------------------------------------------------------------------------------------------
        for jj in range(pop_cut + segment_size, pop_cut + segment_size + two_thirds_size):
            ipert = random.randint(0, 3*(len(network_topology[ii][jj][:])-1)-1) # Selecting a random weight to perturb
            new_weight = random.normal(loc=weights[ii][jj][ipert], scale=1e-3)
            weights[ii][jj][ipert] = new_weight

        #-------------------------------------------------------------------------------------------------
        #      Replacing the poorest performing last 1/6 of the population with the top 1/6
        #-------------------------------------------------------------------------------------------------
        for jj in range(npop - segment_size, npop):
            index = jj - (npop - segment_size)
            network_topology[ii][jj] = network_topology[ii][index]
            weights[ii][jj] = weights[ii][index]
        #-------------------------------------------------------------------------------------------------
        #                              Recombination operator
        #                  Recombining new last 1/6 of the populations 
        #-------------------------------------------------------------------------------------------------
        for jj in range(npop - segment_size, npop):
            irecomb1 = np.random.choice(range(npop - segment_size, npop))
            irecomb2 = np.random.choice(range(npop - segment_size, npop))
            ntwrk1 = network_topology[ii][irecomb1][:]
            ntwrk2 = network_topology[ii][irecomb2][:]
            if len(ntwrk1) + len(ntwrk2) >= 3:
                new_recomb_network = np.concatenate((ntwrk1[0:2], ntwrk2[2:]))
            else:
                new_recomb_network = np.concatenate((ntwrk1[0:1], ntwrk2[1:]))
            network_topology[ii][jj] = new_recomb_network
        #-------------------------------------------------------------------------------------------------
        #                              Check the network size and weights
        #-------------------------------------------------------------------------------------------------
        for jj in range(pop_cut,npop):
            if(len(weights[ii][jj]) != (len(network_topology[ii][jj][:])-1)*3):
                    weights[ii][jj] = [random.uniform(-50,0) for _ in range((len(network_topology[ii][jj][:])-1)*3)]
        #-------------------------------------------------------------------------------------------------
        #                                   Recalculate fitness
        #-------------------------------------------------------------------------------------------------
        for jj in range(pop_cut,npop):
            hidden_nodes = network_topology[ii][jj][:-1]
            noutput = network_topology[ii][jj][-1] 
            num_hidden_nodes = len(hidden_nodes) 
            network = mlp(hidden_nodes, noutput)  
            fitness[ii,jj] = loss_fn( X,Y, weights[ii][jj], network, ndata, num_hidden_nodes, noutput)
        # print out fitness
        print("\tFitness:", np.min(fitness[ii, :]))
    #-----------------------------------------------------------------------------------------------------
    #                               Extracting best weights & least loss
    #-----------------------------------------------------------------------------------------------------
    srt = np.argsort( fitness[ngen-1,:] )
    log_w = weights[ngen-1][srt[0]][:]
    best_network_topology = network_topology[ngen-1][srt[0]][:]
    return log_w, best_network_topology
#######################################################################
