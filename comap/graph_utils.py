import networkx as nx
import numpy as np




def compute_graph_deltas(G={}):
    """
    Compute differences in key statistics between a pair of graphs.

        Input: G (dict): dictionary of the form {g_A:g_a,g_B:g_b,...} of graph pairs (g_K,g_k) to be compared
               
        Output: (dict) of numpy arrays with delta statistics

    """
    

    def _convert_to_array(dictionary):
        '''Helper function to convert lists of values in a dictionary to numpy arrays'''
        return {key:np.array(val) for key, val in dictionary.items()}
    
    keylist =  [ 'delta_n_nodes'
                ,'delta_n_edges'
                ,'delta_density'
                ,'delta_radius'
                ,'delta_diameter'
                ]
    # initialise empty dict
    deltas = dict.fromkeys(keylist, []) 

    # delta: number of nodes
    deltas['delta_n_nodes'] =  [ ( nx.number_of_nodes(raw_map) - nx.number_of_nodes(recat_map) ) 
                                for raw_map,recat_map in G.items()
                                ] 
    # delta: number of edges
    deltas['delta_n_edges'] =  [ (nx.number_of_edges(raw_map) - nx.number_of_edges(recat_map)) 
                                for raw_map,recat_map in G.items()
                                ] 
    # delta: density
    deltas['delta_density'] =  [ (nx.density(raw_map) - nx.density(recat_map)) 
                                    for raw_map,recat_map in G.items() 
                                ]
    # delta: radius
    #deltas['delta_radius'] = [ (nx.radius(raw_map) - nx.radius(recat_map)) 
    #                            for raw_map,recat_map in Gc.items()
    #                            ]
     # delta: diameter
    #deltas['delta_diameter'] = [ (nx.diameter(raw_map) - nx.diameter(recat_map)) 
    #                            for raw_map,recat_map in Gc.items()
    #                            ]
        
    # convert lists to numpy arrays
    deltas = _convert_to_array(deltas)     
        
    return deltas