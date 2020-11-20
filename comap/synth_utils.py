import numpy as np
import networkx as nx

import sys

def generate_synthetic_graph(G, noise_scale=0.2, smear_func='laplace', top_k=0.5):

    """
    Create a synthetic graph modelled on a raw graph. A synthetic graph is derived through a singular-value-decomposition
    of a randomly noised adjancency matrix. Noising targets low degree nodes disproportionately, so as to reduce the reidentification risk 
    (and attribute disclosure risk?) of nodes in the aggregate map that can be traced back to only a small number of individual maps.

    Args:
        - G: A networkx graph.
        - noise_scale: controls how much noise is added to entries in the adjancency matrix. Defaults to the exponential decay parameter in a Laplace distribution.
        - smear_func: (optional) alternative noise ("smearing") function, e.g. np.random.normal( loc=0.0, scale = 0.3 ). If None, a Laplace distribution is used.
        - top_k: top k number (int) or fraction (float) of eigenvectors to use in the SVD-reconstruction of the matrix


    Yields:
        - synmap: a (laplace)-smeared, partially randomised and singular value decomposed networkX graph S, of the same type and with similar attributes as input graph G.
        - a_diff: a numpy matrix capturing the sum perturbations performed on the input adjacency matrix (NB! Should never be released with synthetic map)
        - noise: a numpy array containing random noise additions to input adjacency matrix before SVD-reconstruction (NB! should never be released with synthetic map)

    Raises:
        IOError: ...?
    """


    def _noisy_matrix(M, s, smear_func):

        """
        Function to smear entries in adjacency matrix with a Laplace-distribution (or alternatively a user provided smearing function).
        The smearing is applied to all non-zero elements. Noising zero elements causes significant distortions in the synthetic graph. 
        A zero entry corresponds to the absence of an edge in the aggregated graph and hence (unlike low non-zero values) does not 
        disclose respondents. (Todo: Risk of disclosing the absence of a property?)

        Args: 
            - M: numpy matrix representing the adjancency matrix of the graph
            - s: exponential decay parameter in laplace distribution
        
        Returns: 
            - a noised numpy matrix of A.shape, where all non-zero values are perturbed by a random number
                drawn from a laplace distribution of scale s
            - an array of noise applied to the elements of input matrix M
        """

        def _smearing(smear_func):
       
            if smear_func in ['laplace','normal']:
                if smear_func=='laplace':
                    pert = np.random.laplace( loc=0.0, scale = s)
                elif smear_func=='normal':
                    pert = np.random.normal( loc=0.0, scale = s)
            else:
            #       except ValueError:
                print("Smearing function must be either 'laplace' or 'normal'")
                sys.exit(1) # abort

            #if smear_func=='laplace':
            #    pert = np.random.laplace( loc=0.0, scale = s)
            #elif smear_func=='normal':
            #    pert = np.random.normal( loc=0.0, scale = s)

            return round(pert)

        A = M.copy()
        noise = [] # array to hold noise added to each element in matrix

        # smearing function
        #if smearing is None:   
        #    smearing = np.random.laplace( loc=0.0, scale = s )
        #    print(s, smearing, round(smearing))

        # loop over matrix elements and apply smearing to all non-zero element
        with np.nditer(A, op_flags=['readwrite']) as it:
            for x in it:
                if x!=0:
                    smear = _smearing(smear_func) #round( np.random.laplace( loc=0., scale=s) )
                    x[...] += smear
                    noise.append(smear)

#        for i in range(A.shape[0]):
#            for j in range(A.shape[1]):
#                # smear all non-zero elements
#                if A[i,j] != 0:
#                    print("Test", A[i,j], np.random.laplace( loc=A[i,j], scale=s), round(np.random.laplace( loc=A[i,j], scale=s)) )
#                    dev = round( np.random.laplace( loc=0., scale=s) ) # consider allowing user to specify arbitrary function
#                    A[i,j] += dev
#                    arr.append(dev)

        return A, noise

    def _svd_approximations(A, k):

        """
        Decompose matrix A using singular value decomposition and reconstruct an approximation A_approx using k eigenvectors

        To do: # lage en lite forkalring på hvoerdan sette k verdi (antall egenvektorer som skal brukes)
        
        Args: 
            - A: numpy matrix of shape (n,n)
            - k: number (if int) or fraction (if float) of eigenvalues to use in reconstruction. (If the full set of eigenvectors are used, matrix A 
                 is fully recovered.)

        Returns: 
            - A_approx: numpy matrix (shape (n,n)) approximation of A
        """

        if( isinstance(k,float) ):
            k = int(np.shape(A)[0] * k) # Note! Assumes n x n adjacency matrix
    
        U, sigma, V = np.linalg.svd(A) # Singular Value Decomposition
        A_approx = np.matrix(U[:, :k]) * np.diag(sigma[:k]) * np.matrix(V[:k, :]) # Reconstruct approximation using k eigenvectors
        A_approx = np.around(A_approx) # round floats to ints

        return A_approx




    # nx graph to numpy matrix for sampling
    A = nx.to_numpy_matrix(G)


    #pdf = lambda x,loc,scale : np.exp(-abs(x-loc)/scale)/(2.*scale)
    #locs = []
    #devs = []
    deg1_noise = [] # array to hold degree-1 edge perturbations
    #D = create_distance_matrix(A)

    #np.fill_diagonal(A,0)


    for i in range(A.shape[0]):
        #print(A[:,i])

        # identify nodes with in or out degree equals to 1, these corresponds to 
        # single respondents, and needs to be protected against reindetification
        if (np.sum(A[i,:]) + np.sum(A[:,i])) == 1:
            # samples an list of 4 arbitrary weights between a low and a high number
            # this is done to randomize the distribution 
            weights = np.concatenate( np.random.randint(1,10,size=(1,4)) )# [0]

            # pass the randomly sampled weight to a dirichlet distribution generator 
            s = sorted(np.random.dirichlet((weights), 1)[0],reverse=True)
        
            # use the dirichlet dustribution to sample partibations for edges in the graph
            dev = np.random.choice(np.arange(1,5,1),p=s)

            # perturb edge weights 
            edge_perturb = int( dev +(5*noise_scale) )
            A[i,:] = A[i,:] * edge_perturb #int((dev+(5*noise_scale))) #sjekk denne
            A[:,i] = A[:,i] * edge_perturb #int((dev+(5*noise_scale))) # -----||----
            deg1_noise.append( edge_perturb )
        

    # apply random noise to non-zero entries in adjacency matrix        
    A_noised, degN_noise = _noisy_matrix(A, s=noise_scale, smear_func=smear_func)

    # reduce the noised adjacency matrix by SVD and reconstruct an approximation based in k eigenvectors
    A_recon = _svd_approximations(A_noised, k=top_k)
    
    # Create a difference matrix capturing sum perturbations to original adjacency matrix
    a_diff = A_recon - A


    # create a new DiGraph from reconstructed (approximated) adjacency matrix
    synmap = nx.from_numpy_matrix(A_recon, parallel_edges=True, create_using=nx.DiGraph())
    relabelmap = {list(synmap.nodes())[i]:list(G.nodes())[i] for i in range(len(synmap.nodes()))}
    synmap = nx.relabel_nodes(synmap, relabelmap)

    return synmap, a_diff, degN_noise, deg1_noise 