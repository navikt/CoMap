import networkx as nx
from comap.mapper import CoMap
from comap.graph_utils import (compute_graph_deltas)
from comap.helper_utils import (get_reduced_categories)
from comap.plot_utils import (plot_comparative_importance)


# create 800 random participant graphs
participant_maps = []
for i in range(1,50):
    n = np.random.randint(10,15)
    m = n*np.random.randint(3,5)
    m = nx.gnm_random_graph(n,m, directed=True)
    for (u,v) in m.edges():
        m.edges[u][v]['weight'] = 1
    
participant_maps.append(m)
  
# create an aggregate ensemble from maps and categories
G_agg = CoMap(name='raw_aggregate')
G_agg.aggregate_maps( participant_maps )

# plot aggregate ensemble
fig = G_agg.plot_map()
fig.show()
