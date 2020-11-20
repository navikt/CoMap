
#try:
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from matplotlib import pyplot as plt



def plot_map(G, layout=None):
    """
    Draw map using networkX's draw functionality

    Parameters
    ----------
    G : networkx Graph
        A networkx Graph to plot.
    node_labels : dict
        Dictionary mapping node numbers to node labels.
    layout : str 
        Networkx graph layout (options: 'circular_layout','random_layout','spring_layout','spectral_layout'). 
        If None is provided, 'shell_layout' is used. 
        usage: 'layout=nx.spring_layout(G)'

    Returns
    -------
    matplotlib figure
        A visualisation of graph G

    """

    # set figure size
    fig = plt.figure(figsize=(26,13))
    
    # edge width by weight
    #weights = np.array( [G.get_edge_data(u,v)[0]['weight'] for u,v in G.edges()] )
    weights = list( nx.get_edge_attributes(G, name='weight').values() )
    #if not pos:
    #    pos=layout
    if layout=='circular':
        pos = nx.layout.circular_layout(G)
    elif layout=='spectral':
        pos = nx.layout.spectral_layout(G)
    elif layout=='spring':
        pos = nx.layout.spring_layout(G)
    elif layout=='spiral':
        pos = nx.layout.spiral_layout(G)
    else:
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
    #plt.legend(loc='upper right')
    #plt.show()
    #colorbar()

    return fig

def plot_ego_map(G, n, direction=None, layout=None):
    """
    Helper function to draw the ego network of node n.

    Parameters
    ---------- 
    G : networkx graph
        Networkx graph containing node n
    n : int or str
        Node for which ego graph is drawn
    direction : str, optional
        Specifies direction of ego network. Options: 'incoming' or 'undirected'.
        If None, 'outgoing' ego-network is drawn
    layout : str, optional
        Specifies networkX graph visualisation layout.
        If None, 'shell_layout' is used.

    Returns
    -------
    matplotlib figure
        Plot of the ego graph of node n.
    """

    D = G.copy()
    
    if direction == 'incoming':
        ego = nx.ego_graph(D.reverse(), n)
    elif direction == 'undirected':
        ego = nx.ego_graph(D, n, undirected=True)
    else:
        ego = nx.ego_graph(D, n, undirected=False)

    # set node color of node n to lightgrey
    attr = {n : {'node_color':'lightgrey'}}

    nx.set_node_attributes(ego, attr)

    # plot ego graph
    fig = plot_map( ego, layout)

    return fig




def plot_graph_deltas(deltas=[]):
    """
    Helper function to plot histograms of graph deltas 

    Parameters
    ----------
    deltas : dict 
        Dictionary of grap deltas.

    TODO: example of dictionary

    Returns
    ------- 
    matplotlib figure
        Plot of histograms of graph deltas. 
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
                    , prefix = 'Rank of'
                    , divide_canvas = True
                    , annotate = True
                    ):

    """
    Draw a scatter plot of graph nodes along node metrics.
    
    Parameters
    ---------- 
    df : pandas DataFrame 
        Dataframe where index corresponds to node label and column are metrics.
    x : str
        Variable (column) to plot on x-axis. (Default: 'Out_degree_wg')
    y : str
        Variable (column) to plot on y-axis. (Default: 'In_degree_wg') 
    z : str 
        Variable (column) to plot on z-axes. (Default: 'Pagerank')
    skip : list, optional
        List of node labels that will not be included in scatter plot. (Default: None)
    prefix : str, optional 
        Prefix to axis labels. Node labels are used as axis labels. (Default: 'Rank of')
    divide_canvas : bool
        Split canvas in two shaded regions. (Default: True)
    annotate : bool
        Annotate scatter points with node labels. (Default: True)


    Returns
    -------
    matplotlib figure
        Scatter plot ("bubble plot") of graph nodes along desired node metrics.

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
    c_bar = fig.colorbar(sc, pad=0.15)
    c_bar.set_label(prefix + z, rotation=270, size=25, labelpad=+30)
    c_bar.ax.tick_params(labelsize=20)


    if annotate:
        for i, txt in enumerate(df.index.values):
            ax.annotate(txt,(X[i],Y[i]), ha='center', va='baseline', size=18, name='Courier') #weight='bold'
        
    
    fig.show()
    
    return fig


def plot_graph_perturbations(M, degN_noise, deg1_noise):
    """
    Helper function to draw perturbations to adjacency matrix during synthesis

    Parameters
    ----------
    M : numpy.matrix 
        Matrixx where each element represents a perturbation to the corresponding edge in the original adjacency matrix
    degN_noise : list
        List of noise added to adjancy matrix.
    deg1_noise : list
        List of noise added to degree-1 edges.

    Returns
    -------
    matplotlib figure
        Figure showing perturbations to 1) adjacency matrix, 2) degree N>1 nodes and 3) degree N==1 nodes
    """

    fig = plt.figure(figsize=(15,5))

    # Total pertubations to adjaceny matrix
    m_pert = pd.DataFrame(M)
    ax1 = plt.subplot(131)
    ax1 = sns.heatmap(m_pert, mask = (m_pert==0.), linewidth=.5)
    plt.title('Non-zero perturbations to graph adjacency matrix')
    
    # Noise smearing on degree N>1 edges
    ax2 = plt.subplot(132)
    ax2.hist(degN_noise)
    plt.title('Degree N>1 noise smearing')

    # Noise smearing on degree N=1 edges
    ax3 = plt.subplot(133)
    ax3.hist(deg1_noise)
    plt.title('Degree N=1 noise smearing')
    
    fig.show()

    return 


def plot_comparative_importance(G,  
                                var='Eigenvector_centrality',
                                col = ('orange','skyblue'),
                                compare_rank=False, 
                                excl=None
                               ):
    
    """
    Returns a lollipop plot of the difference between graph G_a and graph G_b in a user provided node property.
    
    Parameters
    ----------
    G : tuple of CoMap objects 
        Tuple of two CoMap graphs to compare (e.g. (G_a, G_b))
    var : str 
        Node property (Default: 'Eigenvector_centrality')
    col : tuple of str
        Colors for G_a and G_b (default: ('orange','skyblue'))
    compare_rank : bool, optional
        Compare relative difference in rank of 'var' rather than difference in value (default: False)
    excl : list, optional
        List of strings referring to nodes to exclude from plot.
    
    Returns
    -------
    matplotlib figure
        A lollipop graph showing the relative difference in 'var' between graphs in tuple G.

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
    plt.xlabel('$\Delta$ '+var, fontsize=18)
    plt.ylabel('Node', fontsize=18)
    ax.yaxis.grid(True, linestyle='--')
    

    
    x = .3 * np.where( abs(df_diff[var].min()) > df_diff[var].max(), df_diff[var].max(), abs(df_diff[var].min()) ) 
    y = 1.1*len(df_diff)
    
    bbox_props_a = dict(boxstyle="rarrow,pad=0.7", fc=col[0], ec="grey", lw=1)
    ax.text(x,  y, "Viktigere i \n"+G[0].name, ha="left",  va="center", bbox=bbox_props_a, rotation=0, size=11, fontsize=12 )
    
    bbox_props_b = dict(boxstyle="larrow,pad=0.7", fc=col[1], ec="grey", lw=1)
    ax.text(-x, y, "Viktigere i \n"+G[1].name, ha="right", va="center", bbox=bbox_props_b, rotation=0, size=11, fontsize=12 )
    
    
    xlim = np.where( df_diff[var].max()>=abs(x), 1.2*df_diff[var].max(), 1.2*abs(x))
    ylim = 1.1*y
    plt.xlim(-xlim,xlim)
    plt.ylim(0,ylim)

    fig.show()

    
    return fig