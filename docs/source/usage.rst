.. _usage:

Basic usage 
===========

Import the CoMap module:

.. high-light: python
::

  from comap.mapper import CoMap

This module contains the CoMap class to represent the aggregated map object. It provides key
functionality to aggregate participant maps:


.. high-light: python
::

  #create 80 random participant graphs

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


Once the aggregated ensemble is represented as a CoMap objects, various utilities allow 
for quick inspection and study of the aggregegate graph. To get key properties or the 5 highest ranking 
nodes in each graph property:

.. high-light: python

::

  G_agg.map_properties()

  G_agg.get_n_highest_ranking_nodes(n=5)


CoMap relies on networkX for all graph related operations. The CoMap object is built on top of a networkX 
DiGraph, which can be accessed as:

::

  import networkx as nx
  
  graph = G_agg.map
  
  print(type(graph))
  print(nx.pagerank(graph))





Visualisation
^^^^^^^^^^^^^


Basic visualisation of the CoMap object:

.. high-light: python

::

  # plot a quadrant scatter of all nodes in weighted indegree, weighted outdegree and pagerank
  f = G_agg.plot_quadrant_scatter()
  f.show()


Some custom visualisation like a scatter plot of all nodes in weighted indegree, outdegree and pagerank is
also provided:

.. high-light: python

::

  # plot a quadrant scatter of all nodes in weighted indegree, weighted outdegree and pagerank
  f = G_agg.plot_quadrant_scatter()
  f.show()




.. high-light: python

::

  from comap.mapper import CoMap
  from comap.graph_utils import (compute_graph_deltas)
  from comap.helper_utils import (get_reduced_categories)
  from comap.plot_utils import (plot_comparative_importance)


  # create 80 random participant graphs
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
