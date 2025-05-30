import numpy as np
from numpy import random
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
    return 10**( ymin + (ymax-ymin)*( (x**n)/( K**n + x**n) ) )

#######################################################################
#              Defining Low pass function
#######################################################################
def response_lp(x, ymin, ymax, K, n):
    return 10**( ymin + (ymax-ymin)*( K**n /( K**n + x**n) ) )
#----------------------------------------------------------------------
def response_hp_s(x):
    #return response_hp(x, 2.57, 6.13, 6.98e-9, 0.866)
    return response_hp(x, 2.57, 6.13, 3.53e-9, 2.532)

def response_lp_s(x):
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
        #------------------------------------------------------------
        #                    Output layer
        #------------------------------------------------------------
        if self.noutput == 'HP':
          activationsO[0] = response_hp_s( np.dot(wO, activationsH) )
        else:
          activationsO[0] = response_lp_s( np.dot(wO, activationsH) )
        return activationsO[0]
#######################################################################
#               Generating training data
#######################################################################
def generate_OR(X ,on_cutoff, off_cutoff):
    Y = np.zeros( [ X.shape[0] ] )
    for ii in range( X.shape[0] ):
        if X[ii,0] < 1e-8:
            if X[ii,1] < 1e-8:
                Y[ii] = off_cutoff
            else:
                Y[ii] =on_cutoff
        else:
            if X[ii,1] < 1e-8:
                Y[ii] = on_cutoff
            else:
                Y[ii] =on_cutoff
    return Y
#---------------------------------------------------------------------
def generate_NOR(X, on_cutoff, off_cutoff):
    Y = np.zeros( [ X.shape[0] ] )
    for ii in range( X.shape[0] ):
        if X[ii,0] < 1e-8:
            if X[ii,1] < 1e-8:
                Y[ii] = on_cutoff
            else:
                Y[ii] = off_cutoff
        else:
            if X[ii,1] < 1e-8:
                Y[ii] = off_cutoff
            else:
                Y[ii] = off_cutoff
    return Y
#---------------------------------------------------------------------
def generate_AND(X, on_cutoff, off_cutoff):
    Y = np.zeros( [ X.shape[0] ] )
    for ii in range( X.shape[0] ):
        if X[ii,0] < 1e-8:
            if X[ii,1] < 1e-8:
                Y[ii] = off_cutoff
            else:
                Y[ii] = off_cutoff
        else:
            if X[ii,1] < 1e-8:
                Y[ii] = off_cutoff
            else:
                Y[ii] = on_cutoff
    return Y
#---------------------------------------------------------------------
def generate_NAND(X, on_cutoff, off_cutoff):
    Y = np.zeros( [ X.shape[0] ] )
    for ii in range( X.shape[0] ):
        if X[ii,0] < 1e-8:
            if X[ii,1] < 1e-8:
                Y[ii] = on_cutoff
            else:
                Y[ii] = on_cutoff
        else:
            if X[ii,1] < 1e-8:
                Y[ii] = on_cutoff
            else:
                Y[ii] = off_cutoff
    return Y

def generate_XOR(X, on_cutoff, off_cutoff):
    Y = np.zeros( [ X.shape[0] ] )
    for ii in range( X.shape[0] ):
        if X[ii,0] < 1e-8:
            if X[ii,1] < 1e-8:
                Y[ii] = off_cutoff
            else:
                Y[ii] = on_cutoff
        else:
            if X[ii,1] < 1e-8:
                Y[ii] = on_cutoff
            else:
                Y[ii] = off_cutoff
    return Y
#######################################################################
def calculate_weights(x_indices, y_indices):
    Diff_co = 1e-11  # Diffusion coefficient (in m^2/s)
    time = 1e6       # Time step (in seconds)
    num_indices = len(x_indices)
    weights = np.zeros((num_indices - 3) * 2 + (num_indices - 3))# Initialize weights
    index_pairs = []# Generate index pairs
    # From the first two indices to all intermediate indices
    for i in range(2, num_indices - 1):
        index_pairs.append((i, 0))
        index_pairs.append((i, 1))
    # From the last index to all intermediate indices
    last_index = num_indices - 1
    for i in range(2, last_index):
        index_pairs.append((last_index, i))
    # Calculate delX and delY
    delX = np.abs(np.array([x_indices[i] - x_indices[j] for i, j in index_pairs])) * 1e-5
    delY = np.abs(np.array([y_indices[i] - y_indices[j] for i, j in index_pairs])) * 1e-5
    # Calculate weights
    for ii in range(len(index_pairs)):
        test1 = math.erfc(delX[ii] / (2 * (Diff_co * time) ** 0.5))
        test2 = math.erfc(delY[ii] / (2 * (Diff_co * time) ** 0.5))
        weights[ii] = test1 * test2
    
    return weights
#######################################################################
def calculate_distance(x_indices, y_indices, min_distance):
    distances = []
    # Calculate distances from indices 0 and 1 to all intermediate indices
    for i in range(2, len(x_indices) - 1):
        distances.append(((x_indices[i] - x_indices[0]) ** 2 + (y_indices[i] - y_indices[0]) ** 2) ** 0.5)
        distances.append(((x_indices[i] - x_indices[1]) ** 2 + (y_indices[i] - y_indices[1]) ** 2) ** 0.5)

    # Calculate distances from the last index to all intermediate indices
    last_index = len(x_indices) - 1
    for i in range(2, last_index):
        distances.append(((x_indices[last_index] - x_indices[i]) ** 2 + (y_indices[last_index] - y_indices[i]) ** 2) ** 0.5)

    # Calculate distances between intermediate indices
    for i in range(2, len(x_indices) - 1):
        for j in range(i + 1, len(x_indices) - 1):
            distances.append(((x_indices[i] - x_indices[j]) ** 2 + (y_indices[i] - y_indices[j]) ** 2) ** 0.5)

    return all(distance >= min_distance for distance in distances)
#######################################################################
#                   Define Loss function (RMSLE)
#######################################################################
def loss_fn(X,Y,w,network, ndata, num_hidden_nodes):
    Yhat = np.zeros_like(Y)
    w = np.array(w)
    if(len(w) != 3*num_hidden_nodes):
        print(w, num_hidden_nodes)
    wH = w[0:2*num_hidden_nodes].reshape(num_hidden_nodes,2) 
    wO = w[2*num_hidden_nodes:]
    for ii in range(ndata):
        Yhat[ii]= network.forward(X[ii,:], wH, wO)
        #print(ii, X[ii,:], Y[ii],Yhat[ii])
    return ((np.log(1+Yhat) - np.log(1+Y))**2).mean()
##############################################################################################
#                                   Genetic Alogrithm
##############################################################################################
def run_genetic_algorithm(X, Y, ndata):
    ngen = 25
    npop = 300000
    pop_cut = int(npop/2) # Number of topologies to keep intact for next generation
    mutation_topo = (npop-pop_cut)//6   
    mutation_loc = 2*(npop-pop_cut)//3 
    nrecomb = (npop-pop_cut)//6  #Perform recombination on 20% of network topologies
    fitness = np.zeros([ngen,npop])
    #intact=int(npop/10) # Number of best topologies to keep intact for next generation
    x_indices_pop = [[None for _ in range(npop)] for _ in range(ngen)]
    y_indices_pop = [[None for _ in range(npop)] for _ in range(ngen)]
    weights = [[None for _ in range(npop)] for _ in range(ngen)]

    #------------------------------------------------------------------------
    #Starting network topologies of 2 node in hidden layer and 1 output node
    #-----------------------------------------------------------------------
    network_topology = [[None for _ in range(npop)] for _ in range(ngen)]
    act_func = ['HP', 'LP'] # Options for activation functions
    network_topology[0] = [[random.choice(act_func) for _ in range(3)] for _ in range(npop)]# Random combinations of 'HP' and 'LP'
                                      #Can change the range of random.randint to change the number of hidden nodes (3,3) to (2,4) for example
    #------------------------------------------------------------------------
    #            For 1st generation, calculate weights and fitness
    #------------------------------------------------------------------------
    for jj in range(npop):
    #------------------------------------------------------------------------
    #           Initializing locations for initial topologies
    #------------------------------------------------------------------------
    #Initialing locations for two input nodes, two hidden nodes and one output node
        x_indices_pop_trail = np.array([0, 0, 0, 0, 12000])
        y_indices_pop_trail = np.array([0, 0, 0, 0, 6000])
        min_distance = 3000
        # max_iterations = 10000  # Set a maximum number of iterations to prevent infinite loop
        # iteration = 0
        while True:
            x_indices_pop_trail[0] = np.random.uniform(500, 10000)  # X index of input 1
            x_indices_pop_trail[1] = np.random.uniform(500, 10000)  # X index of input 2 
            x_indices_pop_trail[2] = np.random.uniform(500, 10000)  # X index of hidden node 1
            x_indices_pop_trail[3] = np.random.uniform(500, 10000)  # X index of hidden node 2

            y_indices_pop_trail[0] = np.random.uniform(500, 10000)  # Y index of input node 1
            y_indices_pop_trail[1] = np.random.uniform(500, 10000)  # Y index of input node 2
            y_indices_pop_trail[2] = np.random.uniform(500, 10000)  # Y index of hidden node 1
            y_indices_pop_trail[3] = np.random.uniform(500, 10000)  # Y index of hidden node 2

            if calculate_distance(x_indices_pop_trail, y_indices_pop_trail, min_distance):
            #print(j,"Condition met, breaking the loop.")
                break
            #     iteration += 1
            # if iteration == max_iterations:
            #     print("Reached maximum iterations without meeting the condition.")
        x_indices_pop[0][jj] = x_indices_pop_trail 
        y_indices_pop[0][jj] = y_indices_pop_trail 
        weights[0][jj]=calculate_weights(x_indices_pop_trail, y_indices_pop_trail)
        hidden_nodes = network_topology[0][jj][:-1]
        noutput = network_topology[0][jj][-1]   
        network = mlp(hidden_nodes, noutput)
        num_hidden_nodes = len(hidden_nodes)
        fitness[0,jj] = loss_fn(X,Y,weights[0][jj][:],network, ndata, num_hidden_nodes)
    #------------------------------------------------------------------------
    #              Loop over generations starts here
    #------------------------------------------------------------------------
    for ii in range(1,ngen): 
        print("Generation: ",ii)
        #--------------------------------------------------------------------
        #            Take top 50% of previous generation
        #--------------------------------------------------------------------
        srt = np.argsort(fitness[ii-1, :])
        x_indices_pop[ii][0:pop_cut] = [x_indices_pop[ii-1][index] for index in srt[0:pop_cut]]
        x_indices_pop[ii][pop_cut:] = [x_indices_pop[ii-1][index] for index in srt[0:pop_cut]]
        y_indices_pop[ii][0:pop_cut] = [y_indices_pop[ii-1][index] for index in srt[0:pop_cut]]
        y_indices_pop[ii][pop_cut:] = [y_indices_pop[ii-1][index] for index in srt[0:pop_cut]]
        network_topology[ii][0:pop_cut] = [network_topology[ii-1][index] for index in srt[0:pop_cut]]
        network_topology[ii][pop_cut:]= [network_topology[ii-1][index] for index in srt[0:pop_cut]]
        fitness[ii, 0:pop_cut] = [fitness[ii-1, index] for index in srt[0:pop_cut]]
        weights[ii][0:pop_cut] = [weights[ii-1][index] for index in srt[0:pop_cut]]
        # Define segment to replace 
        segment_size = (npop - pop_cut) // 6 # Number of topologies to replace (last 1/6 of the population)
        #-------------------------------------------------------------------------------------------------
        #                                  Mutation Operation
        #       Pertubing only the first 1/6 of the population after pop_cut (Nodes)
        #-------------------------------------------------------------------------------------------------
        for jj in range(pop_cut,npop):
            mutation_topo_counter = random.uniform(0, 1)
            #-------------------------------------------------------------------------------------------------
            #     Adding, deleting or modifying nodes in the network topology
            #-------------------------------------------------------------------------------------------------
            if mutation_topo_counter < mutation_topo:
                position = random.randint(0, len(network_topology[ii][jj])-1)# Randomly choose a node in the random topology to alter
                operation = random.choice(['add', 'delete', 'modify'])# Randomly choose between addition, deletion and modification
                if operation == 'add':
                    if len(network_topology[ii][jj]) < 4: # Ensure that after addition there won't be more than 4 nodes in the network
                        node_to_add = random.choice(['HP', 'LP'])
                        network_topology[ii][jj] = np.insert(network_topology[ii][jj], position, node_to_add).tolist()
                elif operation == 'delete':
                     if len(network_topology[ii][jj]) >= 3: # Ensure that after deletion there won't be fewer than 2 nodes in the network
                        network_topology[ii][jj] = np.delete(network_topology[ii][jj], position).tolist() # Deleting a node
                elif operation == 'modify':
                     if network_topology[ii][jj][position] == 'LP':
                         network_topology[ii][jj][position] = 'HP'
                     elif network_topology[ii][jj][position] == 'HP':
                         network_topology[ii][jj][position] = 'LP'
        #-------------------------------------------------------------------------------------------------
        #                               Mutation Operation
        #    Pertubing only the next 2/3 of the population after pop_cut (Location of Nodes)
        #-------------------------------------------------------------------------------------------------
        for jj in range(pop_cut,npop):
            mutation_loc_counter = random.uniform(0, 1)
            if mutation_loc_counter < mutation_loc:

                hidden_nodes = network_topology[ii][jj][:-1]
                num_hidden_nodes = len(hidden_nodes)
                # Generate new x index
                ipertx = random.randint(0, num_hidden_nodes + 2)
                new_x_indices = np.random.normal(loc=x_indices_pop[ii][jj][ipertx], scale=2, size=1).astype(int)
                x_indices_pop[ii][jj][ipertx] = new_x_indices

                # Generate new y index
                iperty = random.randint(0, num_hidden_nodes + 2)
                new_y_indices = np.random.normal(loc=y_indices_pop[ii][jj][iperty], scale=2, size=1).astype(int)
                y_indices_pop[ii][jj][iperty] = new_y_indices
        #-------------------------------------------------------------------------------------------------
        #      Replacing the poorest performing last 1/6 of the population with the top 1/6
        #-------------------------------------------------------------------------------------------------
        for jj in range(npop - segment_size, npop):
            index = jj - (npop - segment_size)
            network_topology[ii][jj] = network_topology[ii][index]
            x_indices_pop[ii][jj] = x_indices_pop[ii][index]
            y_indices_pop[ii][jj] = y_indices_pop[ii][index]
        #-------------------------------------------------------------------------------------------------
        #                                    Recombination operator
        #-------------------------------------------------------------------------------------------------
        irecomb1 = np.random.choice(range(pop_cut, npop), size=nrecomb, replace=True)
        irecomb2 = np.random.choice(range(pop_cut, npop), size=nrecomb, replace=True)
        for jj in range(nrecomb):
            ntwrk1 = network_topology[ii][irecomb1[jj]][:]
            ntwrk2 = network_topology[ii][irecomb2[jj]][:]
            if len(ntwrk1) + len(ntwrk2) >= 3:
                new_recomb_network = np.concatenate((ntwrk1[0:2], ntwrk2[2:]))
            else:
                new_recomb_network = np.concatenate((ntwrk1[0:1], ntwrk2[1:]))
            network_topology[ii][irecomb1[jj]] = new_recomb_network
        #-------------------------------------------------------------------------------------------------
        #       Check the number of location of nodes and number of nodes after mutuation
        #-------------------------------------------------------------------------------------------------
        for jj in range(pop_cut,npop):
            hidden_nodes = network_topology[ii][jj][:-1]
            num_hidden_nodes = len(hidden_nodes)
            if len(x_indices_pop[ii][jj])!= num_hidden_nodes+3:
                n_add_delete = num_hidden_nodes+3 - len(x_indices_pop[ii][jj])
                if n_add_delete > 0:
                    for _ in range(n_add_delete):
                        # Generate new indices
                        new_x_index = np.random.randint(2000, 10000)
                        new_y_index = np.random.randint(2000, 10000)
                        
                        # Insert new indices before the final location
                        x_indices_pop[ii][jj]=np.insert(x_indices_pop[ii][jj], -1, new_x_index)
                        y_indices_pop[ii][jj]=np.insert(y_indices_pop[ii][jj], -1, new_y_index)
                else:
                    # Calculate the number of indices to delete
                    n_delete = abs(n_add_delete)
                    # Ensure we delete any values apart from the first two and the last locations
                    valid_indices = list(range(2, len(x_indices_pop[ii][jj]) - 1))
                    delete_indices = np.random.choice(valid_indices, n_delete, replace=False)
                    # Delete the indices
                    x_indices_pop[ii][jj] = np.delete(x_indices_pop[ii][jj], delete_indices)
                    y_indices_pop[ii][jj] = np.delete(y_indices_pop[ii][jj], delete_indices)      
        #-------------------------------------------------------------------------------------------------
        #                                   Recalculate fitness
        #-------------------------------------------------------------------------------------------------
        for jj in range(pop_cut,npop):
            hidden_nodes = network_topology[ii][jj][:-1]
            noutput = network_topology[ii][jj][-1] 
            num_hidden_nodes = len(hidden_nodes) 
            network = mlp(hidden_nodes, noutput) 
            weights[ii][jj]=calculate_weights(x_indices_pop[ii][jj], y_indices_pop[ii][jj]) 
            fitness[ii,jj] = loss_fn( X,Y, weights[ii][jj], network, ndata, num_hidden_nodes)
        print("\tFitness:",np.min(fitness[ii,:]))
    #-----------------------------------------------------------------------------------------------------
    #                               Extracting best weights & least loss
    #-----------------------------------------------------------------------------------------------------
    srt = np.argsort( fitness[ngen-1,:] )
    log_w = weights[ngen-1][srt[0]][:]
    best_network_topology = network_topology[ngen-1][srt[0]][:]
    best_x_locations = x_indices_pop[ngen-1][srt[0]][:]
    best_y_locations = y_indices_pop[ngen-1][srt[0]][:]
    return log_w, best_network_topology,best_x_locations, best_y_locations
#######################################################################
