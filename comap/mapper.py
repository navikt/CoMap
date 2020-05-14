#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
.. module:: mapper
   :platform: Unix, macOS
   :synopsis: Toolkit for building and examining aggregated graphs using the System Effects methodology.

.. moduleauthor:: Daniel Johannessen, 
                  Phuong Nhu Nguyen, 
                  Robindra Prabhu

.. todo::
    - plot utils
    - synt utils
"""

import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import colors

from comap.plot_utils import (plot_map,plot_graph_deltas, quadrant_scatter, plot_graph_perturbations)
from comap.synth_utils import (generate_synthetic_graph)


class CoMap():
    """
    Class to represent aggregated "System Effects" graph with associated properties.
    
    Parameters: 
    -----------
    map: a networkX DiGraph to represent an aggregated "System Effects" graph
    name: optional name to tag aggregated "System Effects" graph

    """
    
    def __init__(self, name='se_graph'):
        self.map    = nx.DiGraph()  # initialise empty DiGraph()
        self.name   = name

        # initialise dicts for saving properties
        self.graph_deltas = {}

        # node attributes
        self.node_labels = node_labels={}
        self.node_colors = node_colors={}

        # initialise empty DiGraph() for synthetic graph
        self.synmap = nx.DiGraph()  


    def set_deltas(self, deltas={}):
        """ Set graph deltas """
        self.graph_deltas = deltas

        return self


    
    def aggregate_maps(self, G=[], node_tags={}):
        """
        Takes a list of DiGraphs and aggregates common edges between nodes.
        Input: (list) of DiGraps
        Output: DiGraph
        """

        self.map = nx.DiGraph() 
        
        # loop over graphs in list of graphs
        for g in G:
            # add nodes to graph
            self.map.add_nodes_from(g.nodes())
            # loop over edges 
            for edge in g.edges(data=True):
                # if edge is already present, add weight to edge
                if self.map.has_edge(edge[0],edge[1]):
                    self.map[edge[0]][edge[1]]['weight'] += edge[2]['weight']
                # if edge is not already present, add it with weight=1
                else:
                    self.map.add_edge(edge[0],edge[1], weight=1)
        
        # relabel nodes according to mapping provided by 'node_tags'
        nx.set_node_attributes(self.map, name = 'node_label', values = node_tags) 
        nx.relabel_nodes(self.map, mapping=node_tags, copy=False)

        # assign a random color to each node
        colour_list = np.random.choice( list(colors.get_named_colors_mapping().values()), len(self.map) )
        colour_dict = dict( zip(self.map.nodes, colour_list) )
        nx.set_node_attributes(self.map, name = 'node_color', values = colour_dict)

        # save node attributes to CoMap object
        self.node_labels = nx.get_node_attributes(self.map, name = 'node_label')
        self.node_colors = nx.get_node_attributes(self.map, name = 'node_color')
    
        return
    
    def map_properties(self, sort_by=['Pagerank', 'Out_degree_wg']):
        """
        returns a pd.DataFrame with a set of graph metrics computed for each node in the graph
        This works in networkx 2.2 and potentially replaces net_stats
        todo: check backwards compatability
    
        """
    
        metrics = {
        
            # Node degree
            'Node_degree' : dict( self.map.degree )
            # Node out-degree
            ,'Out_degree' : dict( self.map.out_degree )
            # Node weighted out-degree
            ,'Out_degree_wg' : dict( self.map.out_degree(weight='weight') )
            # Node in-degree
            ,'In_degree': dict( self.map.in_degree )
            # Node weighted in-degree
            ,'In_degree_wg': dict( self.map.in_degree(weight='weight') )
            # Node pagerank
            ,'Pagerank' : dict( nx.pagerank(self.map) )
            # Node eigenvector centrality
            ,'Eigenvector_centrality' : dict( nx.eigenvector_centrality(self.map) )
            # Node degree centrality
            ,'Degree_centrality' : dict( nx.degree_centrality(self.map) )
            # Node closeness centrality
            ,'Closeness_centrality' : dict( nx.closeness_centrality(self.map) )
            # Node betweenness centrality
            ,'Betweenness_centrality' : dict( nx.betweenness_centrality( self.map.to_undirected() ) )
            # Node Katz centrality
            #,'Katz_centrality' : dict( nx.katz_centrality( self.map.to_undirected() ) )
            # Node communicability centrality
            #,'Communicability centrality' : dict( nx.communicability_centrality( self.map.to_undirected() ) )
         
        }

        df_node_properties = pd.DataFrame.from_dict(metrics)
        df_node_properties.set_index( np.array(self.map.nodes()),inplace=True )
        df_node_properties.sort_values( sort_by , ascending=False, inplace=True )
    
        return df_node_properties


    def get_n_highest_ranking_nodes(self, n ,metrics=[], color_cells=True):
        """
        Extract the n nodes with the highest rank (highest value) in a given graph metric

        Parameters:
            * n - Number of items to retrieve (int)
            * metrics - List of metrics to evaluate (list) 

        Returns: pd.DataFrame with index:rank, column:metric, cell: node
        """

        def highlight_cells(c, c_dict):
            """ 
            Get node color from supplied c_dict and return a string 
            specifying cell background color
            """
            colour= c_dict.get(c)
            return 'background-color: %s' % colour

        df_ = self.map_properties()

        if not metrics:
            metrics = list(df_.columns.values) # if a list of metrics is not specified, use all column values

        dfn = pd.DataFrame( index = range(1,n+1), columns = metrics) # create empty dataframe

        for metric in metrics:
            dfn[metric] = df_.nlargest(n, metric).index.values

        dfn.columns.name = 'Rank'

        # Colour cells by node color code
        if color_cells:
            dfn = dfn.style.applymap(lambda x: highlight_cells(x, nx.get_node_attributes(self.map, name='node_color')))


        return dfn


    def plot_map(self):

        return plot_map(self.map)

    def plot_map_deltas(self):

        return plot_graph_deltas(self.graph_deltas)

    def plot_quadrant_scatter(self, skip=None, prefix=''):

        return quadrant_scatter( self.map_properties(), skip=skip, prefix=prefix )

    def generate_synthetic_graph(self, noise_scale=0.2, smear_func='laplace' , top_k=0.5, plot=True):
        """
        Create a synthetic version of senstive CoMap-object
        """

        S = CoMap( name=self.name + '_synth' )
        # create synthetic graph object
        S.map, A_diff, noise = generate_synthetic_graph(self.map 
                                         , noise_scale = noise_scale
                                         , smear_func = smear_func
                                         , top_k = top_k
                                         )

        # copy node attributes and assign to synthetic map
        S.node_labels = self.node_labels
        S.node_colors = self.node_colors
        nx.set_node_attributes(S.map, name = 'node_label', values = S.node_labels)
        nx.relabel_nodes(S.map, mapping=S.node_labels, copy=False)
        nx.set_node_attributes(S.map, name = 'node_color', values = S.node_colors)

        # plot graph perturbations
        if plot:
            plot_graph_perturbations(A_diff, noise)
    
        return S, A_diff, noise
