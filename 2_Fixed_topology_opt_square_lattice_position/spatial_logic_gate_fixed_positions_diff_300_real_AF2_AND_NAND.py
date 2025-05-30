import seaborn as sb
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
    #return response_hp(x, 1.55e3, 2.13e5, 3.86e-8, 4.35)
    return response_hp(x, 2.57, 6.13, 3.53e-9, 2.532)

def response_lp_s(x):
    return response_lp(x, 3.05, 5.38, 7.26e-8, 2.11)
#######################################################################
#def plot_activation_fns():
#    x = np.linspace(start=-9, stop=-6, num=1000)
#    f = response_hp(10**x, 1e-9, 1e-6, 1e-8, 4)

#   plt.subplot(2, 1, 1)
#    plt.plot( x , f )
#    plt.xlabel("log10 x (uM)")
#    plt.ylabel("Output")
    
#    x = np.linspace(start=-9, stop=-6, num=1000)
#    f = response_lp(10**x, 1e-9, 1e-6, 1e-8, 4 )

#   plt.subplot(2, 1, 2)
#    plt.plot( x , f )
#    plt.xlabel("log10 x (uM)")
#    plt.ylabel("Output")
#    plt.show()
#######################################################################
def calculate_weights(x_indices, y_indices):
    Diff_co=1e-11 # Diffusion coefficent (in m^2/s)
    time=1e6     # Time step (in seconds)
    weights = np.zeros(6)  # Initialize weights
    delX = np.zeros(len(x_indices))
    delY = np.zeros(len(x_indices))
    # Define the index pairs for delX and delY calculations
    index_pairs = [(2, 0), (2, 1), (3, 0), (3, 1), (4, 2), (4, 3)]
    # Calculate delX and delY
    delX = np.abs(np.array([x_indices[i] - x_indices[j] for i, j in index_pairs])) * 1e-5
    delY = np.abs(np.array([y_indices[i] - y_indices[j] for i, j in index_pairs])) * 1e-5
    #print("DelX:", x_indices, delX, len(delX))

    for ii in range(len(index_pairs)):
     #weights[ii] = (1 - D * t / (delX[ii]**2+1e-2)) * (1 - D * t / (delY[ii]**2+1e-2))
     test1=math.erfc(delX[ii]/(2*(Diff_co*time)**0.5))
     test2=math.erfc(delY[ii]/(2*(Diff_co*time)**0.5))
     weights[ii]=test1* test2
     #if weights[ii] < 0:
     #  weights[ii] = 1e-10  # or any small positive value
     
    return weights
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
        #print(wH, wO)
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
        for i in range(self.nhidden):
            
            activationsH[i] = self.nodes[i]( wH[i,0]*activationsI[0] + wH[i,1]*activationsI[1] )
            #print("1st layer:", wH,  activationsH)
            
        #print("ActivationsH:", np.log10(activationsH))
        #------------------------------------------------------------
        #                    Output layer
        #------------------------------------------------------------
        if self.noutput == 'HP':
          activationsO[0] = response_hp_s( wO[0]*activationsH[0]+ wO[1]*activationsH[1] )
          #print("Output layer:",  wO, np.log10(activationsO[0]))
        else:
          activationsO[0] = response_lp_s( np.dot(wO, activationsH) )
          #print("Output layer:",  wO, np.log10(activationsO[0]))
        return activationsO[0] 
#######################################################################
#               Generating training data
#######################################################################
def generate_OR(X, off_cutoff, on_cutoff):
    Y = np.zeros( [ X.shape[0] ] )
    for ii in range( X.shape[0] ):
        if X[ii,0] <1e-8:
            if X[ii,1] <1e-8:
                Y[ii] = off_cutoff
            else:
                Y[ii] = on_cutoff
        else:
            if X[ii,1] <1e-8:
                Y[ii] = on_cutoff
            else:
                Y[ii] = on_cutoff
    return Y
#---------------------------------------------------------------------
def generate_NOR(X, off_cutoff, on_cutoff):
    Y = np.zeros( [ X.shape[0] ] )
    for ii in range( X.shape[0] ):
        if X[ii,0] <1e-8:
            if X[ii,1] <1e-8:
                Y[ii] = on_cutoff
            else:
                Y[ii] = off_cutoff
        else:
            if X[ii,1] <1e-8:
                Y[ii] = off_cutoff
            else:
                Y[ii] = off_cutoff
    return Y
#---------------------------------------------------------------------
def generate_AND(X, off_cutoff, on_cutoff):
    Y = np.zeros( [ X.shape[0] ] )
    for ii in range( X.shape[0] ):
        if X[ii,0] <1e-8:
            if X[ii,1] <1e-8:
                Y[ii] = off_cutoff
            else:
                Y[ii] = off_cutoff
        else:
            if X[ii,1] <1e-8:
                Y[ii] = off_cutoff
            else:
                Y[ii] = on_cutoff
    return Y
#---------------------------------------------------------------------
def generate_NAND(X, off_cutoff, on_cutoff):
    Y = np.zeros( [ X.shape[0] ] )
    for ii in range( X.shape[0] ):
        if X[ii,0] <1e-8:
            if X[ii,1] <1e-8:
                Y[ii] = on_cutoff
            else:
                Y[ii] = on_cutoff
        else:
            if X[ii,1] <1e-8:
                Y[ii] = on_cutoff
            else:
                Y[ii] = off_cutoff
    return Y
#---------------------------------------------------------------------
def generate_XOR(X, off_cutoff, on_cutoff):
    Y = np.zeros( [ X.shape[0] ] )
    for ii in range( X.shape[0] ):
        if X[ii,0] <1e-8:
            if X[ii,1] <1e-8:
                Y[ii] = off_cutoff
            else:
                Y[ii] = on_cutoff
        else:
            if X[ii,1] <1e-8:
                Y[ii] = on_cutoff
            else:
                Y[ii] = off_cutoff
    return Y

#######################################################################
def calculate_distance(x_indices, y_indices, min_distance):
    # for i in range(len(x_indices)):
    #     for j in range(i + 1, len(x_indices)):
    #          distance = ((x_indices[i] - x_indices[j]) ** 2 + (y_indices[i] - y_indices[j]) ** 2) ** 0.5
    #          if distance < min_distance:
    #             return False
    # return True
    distances = [
        ((x_indices[2] - x_indices[0]) ** 2 + (y_indices[2] - y_indices[0]) ** 2) ** 0.5,
        ((x_indices[2] - x_indices[1]) ** 2 + (y_indices[2] - y_indices[1]) ** 2) ** 0.5,
        ((x_indices[2] - x_indices[4]) ** 2 + (y_indices[2] - y_indices[4]) ** 2) ** 0.5,
        ((x_indices[2] - x_indices[3]) ** 2 + (y_indices[2] - y_indices[3]) ** 2) ** 0.5,
        ((x_indices[3] - x_indices[0]) ** 2 + (y_indices[3] - y_indices[0]) ** 2) ** 0.5,
        ((x_indices[3] - x_indices[1]) ** 2 + (y_indices[3] - y_indices[1]) ** 2) ** 0.5,
        ((x_indices[3] - x_indices[4]) ** 2 + (y_indices[3] - y_indices[4]) ** 2) ** 0.5
    ]
    
    # for i, distance in enumerate(distances):
    #    print(f"Distance {i+1}: {distance}")

    return all(distance >= min_distance for distance in distances)
#---------------------------------------------------------------)
# def calculate_distance(z_indices):
#     for i in range(len(z_indices)):
#         for j in range(i + 1, len(z_indices)):
#              distance = (z_indices[i] - z_indices[j]) 
#              if distance < 2000:
#                 return False
#     return True
# def check_min_unit_distance(indices, min_distance):
#     return (abs(indices[2] - indices[0]) >= min_distance and
#             abs(indices[2] - indices[1]) >= min_distance and
#             abs(indices[2] - indices[4]) >= min_distance and
#             abs(indices[2] - indices[3]) >= min_distance and
#             abs(indices[3] - indices[0]) >= min_distance and
#             abs(indices[3] - indices[1]) >= min_distance and
#             abs(indices[3] - indices[4]) >= min_distance)
#######################################################################
#                   Define Loss function (RMSLE)
#######################################################################
def loss_fn(X,Y,w,network, ndata):
    Yhat = np.zeros_like(Y)
    wH = w[0:4].reshape(2,2) 
    wO = w[4:]
    for i in range(ndata):
        Yhat[i] = network.forward(X[i,:], wH, wO )
        #print("Yhat:", Yhat[i], Y[i])
    return ((np.log(1+Yhat) - np.log(1+Y))**2).mean()
#######################################################################
#                   Genetic Alogrithm
#######################################################################
def run_genetic_algorithm(network, X, Y, ndata):
    ngen = 25
    npop = 5000
    nrecomb = int(npop/1) # Perform recombination on 20% of population

    fitness = np.zeros([ngen,npop])
    x_indices_pop = np.zeros([ngen,npop,5])
    y_indices_pop = np.zeros([ngen,npop,5])
    pop_cut = int(npop/2)
    
    #in_weights = np.array([])
    #-----------------------------------------------------------------------
    #                        Initial generation
    #-----------------------------------------------------------------------
    for j in range(npop):
        #print("Population: ", j)
        x_indices_pop_trail = np.array([])
        y_indices_pop_trail = np.array([])

        #------------------------------------------------------------------------------------------------------
        # while len(x_indices_pop_trail) < 4:
        #      x = np.random.randint(1000,9000)
        #      if all(abs(x - x_indices_pop_trail) >= 1000) and abs(x -12000) >= 1000:
        #          x_indices_pop_trail = np.append(x_indices_pop_trail, x)
        # x_indices_pop_trail = np.append(x_indices_pop_trail, 12000) #Fixing the output node at 7800,4000 position

        # while len(y_indices_pop_trail) < 4:
        #      y = np.random.randint(1000,9000)
        #      if all(abs(y - y_indices_pop_trail) >= 1000) and abs(y -6000) >= 1000:
        #       y_indices_pop_trail = np.append(y_indices_pop_trail, y)
        # y_indices_pop_trail = np.append(y_indices_pop_trail, 6000) #Fixing the output node at 7800,4000 position
         #------------------------------------------------------------------------------------------------------
        #Initialize the arrays with fixed values
        # x_indices_pop_trail = np.array([500, 500, 0, 0, 12000])
        # y_indices_pop_trail = np.array([5500, 6500, 0, 0, 6000])
        # min_distance = 500
        # while True:
        #     x_indices_pop_trail[0] = np.random.uniform(500, 1000)  # X index of hidden node 1
        #     x_indices_pop_trail[1] = np.random.uniform(500, 1000)  # X index of hidden node 1
        #     x_indices_pop_trail[2] = np.random.uniform(1000, 10000)  # X index of hidden node 1
        #     x_indices_pop_trail[3] = np.random.uniform(1000, 10000)  # X index of hidden node 2
    
        #     if check_min_unit_distance(x_indices_pop_trail, min_distance):
        #       break

        # while True:
        #      y_indices_pop_trail[0] = np.random.uniform(100, 5000)  # Y index of hidden node 1
        #      y_indices_pop_trail[1] = np.random.uniform(100, 7000)  # Y index of hidden node 1
        #      y_indices_pop_trail[2] = np.random.uniform(2000, 10000)  # Y index of hidden node 1
        #      y_indices_pop_trail[3] = np.random.uniform(2000, 10000)  # Y index of hidden node 2
    
        #      if check_min_unit_distance(y_indices_pop_trail, min_distance):
        #         break
        #------------------------------------------------------------------------------------------------------
        #Initialize the arrays with fixed values
        x_indices_pop_trail = np.array([0, 0, 0, 0, 4000])
        y_indices_pop_trail = np.array([0, 0, 0, 0, 2000])
        min_distance = 500

        # max_iterations = 10000  # Set a maximum number of iterations to prevent infinite loop
        # iteration = 0
        while True:
            x_indices_pop_trail[0] = np.random.uniform(100, 3500)  # X index of input 1
            x_indices_pop_trail[1] = np.random.uniform(100, 3500)  # X index of input 2 
            x_indices_pop_trail[2] = np.random.uniform(500, 3500)  # X index of hidden node 1
            x_indices_pop_trail[3] = np.random.uniform(500, 3500)  # X index of hidden node 2

            y_indices_pop_trail[0] = np.random.uniform(100, 3800)  # Y index of input node 1
            y_indices_pop_trail[1] = np.random.uniform(100, 3800)  # Y index of input node 2
            y_indices_pop_trail[2] = np.random.uniform(300, 3800)  # Y index of hidden node 1
            y_indices_pop_trail[3] = np.random.uniform(300, 3800)  # Y index of hidden node 2
    
            if calculate_distance(x_indices_pop_trail, y_indices_pop_trail, min_distance):
                #print(j,"Condition met, breaking the loop.")
                break

        #     iteration += 1

        # if iteration == max_iterations:
        #     print("Reached maximum iterations without meeting the condition.")
        #------------------------------------------------------------------------------------------------------

        # Generate the middle values ensuring the distance condition
        # while True:
        #       x_indices_pop_trail[2] = np.random.uniform(3000, 9000) # X index of hidden node 1
        #       x_indices_pop_trail[3] = np.random.uniform(3000, 9000) # X index of hidden node 2
    
        #       if calculate_distance(x_indices_pop_trail):
        #           break
              
        # while True:
        #       y_indices_pop_trail[2] = np.random.uniform(2000, 9000)  # Y index of hidden node 1
        #       y_indices_pop_trail[3] = np.random.uniform(2000, 9000)  # Y index of hidden node 2
    
        #       if calculate_distance(y_indices_pop_trail):
        #         break

        weights0=calculate_weights(x_indices_pop_trail, y_indices_pop_trail)

        #print("Initial weights:", weights0)
        fitness[0,j] = loss_fn(X,Y,weights0,network, ndata)
        x_indices_pop[0,j,:] = x_indices_pop_trail 
        y_indices_pop[0,j,:] = y_indices_pop_trail 
        #in_weights.append(weights0)        
        #in_weights = np.append(in_weights, weights0)

    # np.save('initial_weights.npy', in_weights)
    #-----------------------------------------------------------------------
    #                           Generation Loop starts here
    #-----------------------------------------------------------------------
    for i in range(1,ngen):
        print("Generation: ", i)
        # Take top 50% of previous generation
        srt = np.argsort( fitness[i-1,:] )
        x_indices_pop[i,0:pop_cut,:] = x_indices_pop[i-1,srt[0:pop_cut],:]
        x_indices_pop[i,pop_cut:,:] = x_indices_pop[i-1,srt[0:pop_cut],:]
        y_indices_pop[i,0:pop_cut,:] = y_indices_pop[i-1,srt[0:pop_cut],:]
        y_indices_pop[i,pop_cut:,:] = y_indices_pop[i-1,srt[0:pop_cut],:]
        #######################################################################
        #                         Mutation operator
        #######################################################################
        for j in range(pop_cut,npop):
            #ipertx = random.randint(0, 3, size=1)
            ipertx = random.randint(0, 4, size=1)
            new_x_indices = np.random.normal(loc=x_indices_pop[i,j,ipertx], scale=5, size=1).astype(int)
            x_indices_pop[i,j,ipertx] = new_x_indices

            iperty = random.randint(0, 4, size=1)
            new_y_indices = np.random.normal(loc=x_indices_pop[i,j,ipertx], scale=5, size=1).astype(int)
            y_indices_pop[i,j,iperty] = new_y_indices
    
        #######################################################################
        #                      Recombination operator
        #######################################################################
        irecombx1 = random.choice(range(pop_cut, npop), size=nrecomb, replace=True) 
        irecombx2 = random.choice(range(pop_cut, npop), size=nrecomb, replace=True)

        irecomby1 = random.choice(range(pop_cut, npop), size=nrecomb, replace=True) 
        irecomby2 = random.choice(range(pop_cut, npop), size=nrecomb, replace=True)

        for j in range(nrecomb):
            fracx1 = x_indices_pop[i, irecombx1[j],:]
            fracx2 = x_indices_pop[i, irecombx2[j],:]
            new_indicesx = np.concatenate((fracx1[0:3],fracx2[3:]))
            #-------------------------------------------------------------------
            # x_indices_pop[i,irecombx1[j],:] = new_indicesx
            # if any(abs(new_indicesx[k] - new_indicesx[l]) < 1000 for k in range(len(new_indicesx)) for l in range(k+1, len(new_indicesx))):
            #     x_indices_pop[i,irecombx1[j],:] = fracx1
            # else:
            #     x_indices_pop[i,irecombx1[j],:] = new_indicesx

            fracy1 = y_indices_pop[i, irecomby1[j],:]
            fracy2 = y_indices_pop[i, irecomby2[j],:]
            new_indicesy = np.concatenate((fracy1[0:3],fracy2[3:]))
            # y_indices_pop[i,irecomby1[j],:] = new_indicesy
            # if any(abs(new_indicesy[k] - new_indicesy[l]) < 1000 for k in range(len(new_indicesy)) for l in range(k+1, len(new_indicesy))):
            #     y_indices_pop[i,irecomby1[j],:] = fracy1
            # else:
            #     y_indices_pop[i,irecomby1[j],:] = new_indicesy
            #-------------------------------------------------------------------

            # if calculate_distance(new_indicesx):
            #     x_indices_pop[i, irecombx1[j],:] = new_indicesx
            # else:
            #     x_indices_pop[i, irecombx1[j],:] = fracx1
            
            # if calculate_distance(new_indicesy):
            #     y_indices_pop[i, irecomby1[j],:] = new_indicesy
            # else:
            #     y_indices_pop[i, irecomby1[j],:] = fracy1
            # if check_min_unit_distance(new_indicesx, min_distance):
            #     x_indices_pop[i, irecombx1[j],:] = new_indicesx
            # else:
            #     x_indices_pop[i, irecombx1[j],:] = fracx1

            if calculate_distance(new_indicesx,new_indicesy, min_distance):
                x_indices_pop[i, irecombx1[j],:] = new_indicesx
                y_indices_pop[i, irecomby1[j],:] = new_indicesy
            else:
                x_indices_pop[i, irecombx1[j],:] = fracx1
                y_indices_pop[i, irecomby1[j],:] = fracy1

        #######################################################################
        #                   Recalculate fitness
        #######################################################################
        for j in range(npop):
            weights=calculate_weights(x_indices_pop[i,j,:], y_indices_pop[i,j,:])
            #print("Weights:", weights)
            fitness[i,j] = loss_fn( X,Y, weights, network, ndata)
        # print out fitness
        print("\tFitness:",np.min(fitness[i,:]))
    #######################################################################
    #                     Extracting best weights
    #######################################################################
    srt = np.argsort( fitness[ngen-1,:] )
    best_x_indices = x_indices_pop[ngen-1,srt[0],:]
    best_y_indices = y_indices_pop[ngen-1,srt[0],:]
    print("Best positions:", best_x_indices, best_y_indices)
    return best_x_indices, best_y_indices
######################################################################
