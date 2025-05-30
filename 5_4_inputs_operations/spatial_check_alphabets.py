import seaborn as sb
import numpy as np
from numpy import random
from matplotlib import pyplot as plt

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
    return response_hp(x, 2.57, 6.13, 6.98e-9, 0.866)

def response_lp_s(x):
    return response_lp(x, 3.05, 5.37, 7.26e-8, 2.11)
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
        activationsI = np.zeros([4]) 
        activationsH = np.zeros([self.nhidden]) 
        activationsO = np.zeros([1]) 
        #------------------------------------------------------------
        #                    Input layer
        #------------------------------------------------------------
        activationsI[0] = I[0]
        activationsI[1] = I[1]
        activationsI[2] = I[2]
        activationsI[3] = I[3]
        #------------------------------------------------------------
        #                    Hidden layer
        #------------------------------------------------------------
        for ii in range(self.nhidden):
            activationsH[ii] = self.nodes[ii](sum(wH[ii, jj] * activationsI[jj] for jj in range(len(activationsI))))
            #activationsH[ii] = self.nodes[ii]( wH[ii,0]*activationsI[0] + wH[ii,1]*activationsI[1] )
        #------------------------------------------------------------
        #                    Output layer
        #------------------------------------------------------------
        if self.noutput == 'HP':
          activationsO[0] = response_hp_s( np.dot(wO, activationsH) )
        else:
          activationsO[0] = response_lp_s( np.dot(wO, activationsH) )
        return (activationsO[0])
#######################################################################
#               Generating training data
#######################################################################
def generate_check_alpha(desired_word, X2, on_cutoff, off_cutoff):
    desired_values = [ord(char) - ord('A')  for char in desired_word]
    
    Y = np.zeros([len(X2)])
    for ii in range(len(X2)):
        if X2[ii] in desired_values:
            Y[ii] = on_cutoff
        else:
            Y[ii] = off_cutoff
    return Y
#######################################################################
#                   Define Loss function (RMSLE)
#######################################################################
def loss_fn(X,Y,w,network, ndata, num_hidden_nodes):
    Yhat = np.zeros_like(Y)
    #print(Yhat)
    w = np.array(w)
    if(len(w) != 5*num_hidden_nodes):
        print('error',w, num_hidden_nodes)
    wH = w[0:4*num_hidden_nodes].reshape(num_hidden_nodes,4) 
    wO = w[4*num_hidden_nodes:]
    for ii in range(ndata):
        Yhat[ii]= network.forward(X[ii,:], 10**(wH), 10**(wO) )
        #print(ii, X[ii,:], Y[ii],Yhat[ii])
    return ((np.log(1+Yhat) - np.log(1+Y))**2).mean()
##############################################################################################
#                                   Genetic Alogrithm
##############################################################################################
def run_genetic_algorithm(X, Y, ndata):
    ngen = 25
    npop = 10000000
    pop_cut = int(npop/2) # Number of topologies to keep intact for next generation
    mutation_prob = 0.3   # Probability of mutation 20%
    nrecomb = int(npop/5) #Perform recombination on 20% of network topologies
    fitness = np.zeros([ngen,npop])
    #intact=int(npop/10) # Number of best topologies to keep intact for next generation
    #------------------------------------------------------------------------
    #Starting network topologies of 2 node in hidden layer and 1 output node
    #-----------------------------------------------------------------------
    network_topology = [[None for _ in range(npop)] for _ in range(ngen)]
    act_func = ['HP', 'LP'] # Options for activation functions
    network_topology[0] = [[random.choice(act_func) for _ in range(3)] for _ in range(npop)]# Random combinations of 'HP' and 'LP'
                                      #Can change the range of random.randint to change the number of hidden nodes (3,3) to (2,4) for example
    #print(network_topology[0])
    #------------------------------------------------------------------------
    #           Initializing weights for initial topologies
    #------------------------------------------------------------------------
    weights = [[None for _ in range(npop)] for _ in range(ngen)]
    for ii in range(npop):
         num_hidden_nodes = len(network_topology[0][ii][:-1])
         weights[0][ii] = [random.uniform(-30, 0) for _ in range(5*num_hidden_nodes)] # Initialize the weights with random values
         #weight range for Prime and Vowel was -30,0
    #print(weights[0][1])
    #------------------------------------------------------------------------
    #            Calculating fitness of initial topologies
    #------------------------------------------------------------------------
    for jj in range(npop):
        hidden_nodes = network_topology[0][jj][:-1]
        noutput = network_topology[0][jj][-1]   
        network = mlp(hidden_nodes, noutput)
        num_hidden_nodes = len(hidden_nodes)
        fitness[0,jj] = loss_fn(X,Y,weights[0][jj][:],network, ndata, num_hidden_nodes)
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
        fitness[ii,0:pop_cut] = [fitness[ii-1][index] for index in srt[0:pop_cut]]
        #-------------------------------------------------------------------------------------------------
        #                                   Mutation operator
        #-------------------------------------------------------------------------------------------------
        #for jj in range(0,intact):
        #    ipert2 = random.randint(0, len(weights[ii][jj])-1) # Selecting a random weight to perturb for intact topologies
        #    new_weight = random.normal(loc=weights[ii][jj][ipert2], scale=1e-3)
        #    weights[ii][jj][ipert2] = new_weight
        #-------------------------------------------------------------------------------------------------
        for jj in range(pop_cut,npop):
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
                elif operation == 'delete':
                     if len(network_topology[ii][jj]) >= 3: # Ensure that after deletion there won't be fewer than 2 nodes in the network
                        network_topology[ii][jj] = np.delete(network_topology[ii][jj], position).tolist() # Deleting a node
                elif operation == 'modify':
                     if network_topology[ii][jj][position] == 'LP':
                         network_topology[ii][jj][position] = 'HP'
                     elif network_topology[ii][jj][position] == 'HP':
                         network_topology[ii][jj][position] = 'LP'
            else: 
            #-------------------------------------------------------------------------------------------------
            #     Just modifying weights in the network topology
            #-------------------------------------------------------------------------------------------------
                ipert = random.randint(0, 5*(len(network_topology[ii][jj][:])-1)-1) # Selecting a random weight to perturb
                new_weight = random.normal(loc=weights[ii][jj][ipert], scale=1e-2)
                weights[ii][jj][ipert] = new_weight
        #-------------------------------------------------------------------------------------------------
        #                              Recombination operator
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
            #new_recomb_network = np.concatenate((ntwrk1[0:int(len(ntwrk1)/2)],ntwrk2[int(len(ntwrk2)/2):]))
            #if(len(new_recomb_network) >= 3):
            network_topology[ii][irecomb1[jj]] = new_recomb_network
        #-------------------------------------------------------------------------------------------------
        #                              Check the network size and weights
        #-------------------------------------------------------------------------------------------------
        for jj in range(pop_cut,npop):
            if(len(weights[ii][jj]) != (len(network_topology[ii][jj][:])-1)*5):
                    weights[ii][jj] = [random.uniform(-30, 0) for _ in range((len(network_topology[ii][jj][:])-1)*5)]
        #-------------------------------------------------------------------------------------------------
        #                                   Recalculate fitness
        #-------------------------------------------------------------------------------------------------
        for jj in range(pop_cut,npop):
            hidden_nodes = network_topology[ii][jj][:-1]
            noutput = network_topology[ii][jj][-1] 
            num_hidden_nodes = len(hidden_nodes) 
            network = mlp(hidden_nodes, noutput)  
            fitness[ii,jj] = loss_fn( X,Y, weights[ii][jj], network, ndata, num_hidden_nodes)
            #print("All Fitness:",ii, jj, fitness[ii,jj])
        # print out fitness
        print("\tFitness:",np.min(fitness[ii,:]))
    #-----------------------------------------------------------------------------------------------------
    #                               Extracting best weights & least loss
    #-----------------------------------------------------------------------------------------------------
    srt = np.argsort( fitness[ngen-1,:] )
    log_w = weights[ngen-1][srt[0]][:]
    best_network_topology = network_topology[ngen-1][srt[0]][:]
    return log_w, best_network_topology
#######################################################################
