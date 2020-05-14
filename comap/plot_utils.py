
#try:
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from matplotlib import pyplot as plt



def plot_map(G):
    """
    Draw map using networkX's draw functionality

    Input:
    - G : networkx Graph
    - node_labels : dictionary mapping node numbers to node labels
    - node_col_schema : 'defined' if nodes should be colored by a predefined color scheme; default: coloring by node degree
    - layout : networkx graph layout (options: 'circular_layout','random_layout','shell_layout','spring_layout','spectral_layout'); usage: "layout=nx.spring_layout(G)"
    - pos : optional dictionary with nodes as keys and positions as values
    """

    # set figure size
    fig = plt.figure(figsize=(26,13))
    
    # edge width by weight
    #weights = np.array( [G.get_edge_data(u,v)[0]['weight'] for u,v in G.edges()] )
    weights = list( nx.get_edge_attributes(G, name='weight').values() )
    #if not pos:
    #    pos=layout
    pos = nx.layout.shell_layout(G)

    node_colors = list( nx.get_node_attributes(G, name='node_color').values() )
    node_labels = list( nx.get_node_attributes(G, name='node_label').values() )

    # scale nodes by metric:pagerank
    #node_sizes = list( nx.pagerank(G).values() )

    # draw networkx edges, nodes and labels 
    nc = nx.draw_networkx_nodes(G 
                                , pos
                                #, nodelist=G.nodes()
                                , node_color=node_colors
                                , label=node_labels 
                                , node_size= 1000
                                , alpha=0.7
                                #, cmap=plt.cm.rainbow
                               )
    ec = nx.draw_networkx_edges(G
                                , pos
                                , alpha=0.8
                                , width=weights
                                , arrowstyle='->'
                                , arrowsize=20
                                #, arrows=True
                                #, connectionstyle='arc3,rad=0.2'
                                )
    la = nx.draw_networkx_labels(G
                                , pos
                                , labels=nx.get_node_attributes(G, name='node_label')
                                )

    plt.axis('off')

    return

def plot_graph_deltas(deltas=[]):
    """
    Helper function to plot histograms of graph deltas 

    Input: dictionary of grap deltas

    Output: plot of graph deltas
    """

    # plot with various axes scales
    plt.figure()

    # delta_n_nodes
    ax1 = plt.subplot(231)
    ax1.hist(deltas['delta_n_nodes'])
    plt.title('delta_n_nodes')
    plt.grid(True)

    # delta_n_nodes
    ax2 = plt.subplot(232)
    ax2.hist(deltas['delta_n_edges'])
    plt.title('delta_n_edges')
    plt.grid(True)

    # delta_density
    ax3 = plt.subplot(233)
    ax3.hist(deltas['delta_density'])
    plt.title('delta_density')
    plt.grid(True)

    return




def quadrant_scatter(df
                    , x='Out_degree_wg'
                    , y='In_degree_wg'
                    , z='Pagerank'
                    , skip = None
                    , prefix = None
                    , divide_canvas = True
                    , annotate = True
                    ):

    """
    Draw a scatter plot of graph nodes along node metrics.
    
    ------
    Params: 
        * df     - pd.DataFrame where index corresponds to node label and column are metrics
        * x,y,z  - columns to plot on x, y and z-axes, respectively
        * skip   - (optional) list of node labels that will not be included in scatter plot
        * prefix - (optional) prefix to axis labels. Node labels are used as axis labels. 

    """

    def _compute_scatter_size(min_new=1, max_new=50):
        return ((max_new - min_new) / ( max(Z) - min(Z) )) * ( Z - max(Z) ) + max_new

    # remove skip nodes
    if skip is not None:
        df = df.drop( skip )

    # convert prefix to 'str'
    if prefix is not None:
        prefix = str(prefix) + ' '

    X = df[x].values
    Y = df[y].values
    Z = df[z].values

    # compute ranks
    X = df[x].rank(method='first').values
    Y = df[y].rank(method='first').values
    Z = df[z].rank(method='first').values

    scatter_size = _compute_scatter_size(min_new=100, max_new=2000)

    fig,ax = plt.subplots(figsize=(20,20))
    
    # axis labels
    ax.set_xlabel(prefix + x, fontsize=25)
    ax.set_ylabel(prefix + y, fontsize=25)
    ax.tick_params(axis="both", labelsize=25)
    # axis lims
    axmax = max(max(X),max(Y))
    ax.set_xlim([0, axmax+2])
    ax.set_ylim([0, axmax+2])

    if divide_canvas:

        ax.plot( [0,1]
                ,[0,1]
                , '--'
                , color='black'
                , alpha=0.8
                , transform=ax.transAxes
                , linewidth=5
                )
        ax.fill_between( [0, axmax]
                        , 0
                        , [0, axmax]
                        , facecolor='peru'
                        , alpha=0.3
                        , transform=ax.transAxes
                        )
        ax.fill_between( [0, axmax]
                        , axmax
                        , [0, axmax]
                        , facecolor='moccasin'
                        , alpha=0.3
                        , transform=ax.transAxes
                        )
    # Draw scatter 
    sc = ax.scatter( X
                    , Y
                    , s=scatter_size
                    , c=Z
                    , edgecolors='black'
                    , linewidths=1
                    , alpha=1.
                    , cmap='coolwarm'
                    )
     # colour bar
    c_bar = fig.colorbar(sc)
    c_bar.set_label(prefix + z, rotation=270, size=25, labelpad=+30)
    c_bar.ax.tick_params(labelsize=20)


    if annotate:
        for i, txt in enumerate(df.index.values):
            ax.annotate(txt,(X[i],Y[i]), ha='right', va='baseline', size=18, name='Courier') #weight='bold'
        
    
    fig.show()
    
    return


def plot_graph_perturbations(M, noise):
	
	"""
    Helper function to draw perturbations to adjacency matrix during synthesis

    Args: 
    - M: numpy matrix of where each element represents a perturbation to the corresponding edge in the original adjacency matrix
    - noise: numpy array of noise added to adjacency matrix 

    Returns: 
    - plot of perturbations to adjacency matrix
    """
	
	# plot with various axes scales
	plt.figure(figsize=(10,10))
	
	# total perturbations to adjacency matrix
	m_pert= pd.DataFrame(M)
	ax1 = plt.subplot(221)
	ax1 = sns.heatmap(m_pert, mask = (m_pert==0.), linewidth=0.5)
	plt.title('Non-zero perturbations to graph adjacency matrix')
	
	# noise smearing
	ax2 = plt.subplot(222)
	ax2.hist(noise)
	plt.title('Noise smearing')
	
	plt.show()
	
	return



