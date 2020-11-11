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


def plot_comparative_importance(G,  
                                var='Eigenvector_centrality',
                                col = ('orange','skyblue'),
                                compare_rank=False, 
                                excl=None
                               ):
    
    """
    Returns a lollipop plot of the difference between graph G_a and graph G_b in a user provided node property.
    
    Args:
        - G   : (tuple), of CoMap graphs to compare (e.g. (G_a, G_b))
        - var : (str), node property (default: 'Eigenvector_centrality')
        - col : (tuple), string of colors for G_a and G_b (default: ('orange','skyblue'))
        - compare_rank : (bool), optional, compare relative difference in rank of 'var' rather than difference in value (default: False)
        - excl: (list), optional, list of strings referring to nodes to exclude from plot
    
    Returns:
        - a lollipop graph
    
    """
    
    node_prpty = var
    
    # get map properties
    df_A = G[0].map_properties( sort_by=[ node_prpty ] ).drop(excl)
    df_B = G[1].map_properties( sort_by=[ node_prpty ] ).drop(excl)
    
    # compute rank differences
    if(compare_rank):
        var = var + '_rank'
        df_A[ var ] = df_A[ node_prpty ].rank()
        df_B[ var ] = df_B[ node_prpty ].rank()
        
    # compute normalised difference
    df_diff = (df_A - df_B)# / df_B
    
    # dropna to only get nodes present in both graphs
    df_diff.dropna(inplace=True)
    
    #  sort by var
    df_diff.sort_values(by=var,inplace=True)
    
    #Create a color if the group is "B"
    color_scheme = np.where( df_diff[var]>=0, col[0], col[1])
    
    # Compute range
    plt_range=range(1, len(df_diff.index)+1)

    fig,ax = plt.subplots(figsize=(1/5*len(df_diff),1/3*len(df_diff)))

    # plot
    plt.axvline(x=0,color='grey',linestyle='--') 
    plt.hlines(y=plt_range, xmin=0, xmax=df_diff[var], color=color_scheme)
    plt.scatter(df_diff[var], plt_range, color=color_scheme, alpha=1)

    # decorations
    plt.yticks(plt_range, df_diff.index)
    plt.xlabel('$\Delta$ '+var)
    plt.ylabel('Node')
    ax.yaxis.grid(True, linestyle='--')
    

    
    x = .3 * np.where( abs(df_diff[var].min()) > df_diff[var].max(), df_diff[var].max(), abs(df_diff[var].min()) ) 
    y = 1.1*len(df_diff)
    
    bbox_props_a = dict(boxstyle="rarrow,pad=0.7", fc=col[0], ec="grey", lw=1)
    ax.text(x,  y, "Viktigere i \n"+G[0].name, ha="left",  va="center", rotation=0, size=11, bbox=bbox_props_a)
    
    bbox_props_b = dict(boxstyle="larrow,pad=0.7", fc=col[1], ec="grey", lw=1)
    ax.text(-x, y, "Viktigere i \n"+G[1].name, ha="right", va="center", rotation=0, size=11, bbox=bbox_props_b)
    
    
    xlim = np.where( df_diff[var].max()>=abs(x), 1.2*df_diff[var].max(), 1.2*abs(x))
    ylim = 1.1*y
    plt.xlim(-xlim,xlim)
    plt.ylim(0,ylim)

    plt.show()
    
    return