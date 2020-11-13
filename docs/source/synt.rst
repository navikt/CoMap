.. _synt:

Synthetic graph generation 
==========================

System Effects relies on mapping individual experiences. Insofar as the resulting graphs contain information that can be traced back to 
an identifiable person, the individual graphs are likely to qualify as *personal data* (https://gdpr.eu/eu-gdpr-personal-data/) under 
EU's General Data Protection Regulation (GDPR) [#]_.

Even if the individual graph has been stripped of all direct identifiers (e.g. names), the remaining nodes and edges may either alone 
or in conjunction with other elements in the graph reveal the identity of a participant to an "attacker" with sufficient background 
knowledge. Once the identity of a participant is revealed ("identity disclosure"), the attacker may use the graph and its relations to 
learn something new about the participant ("attribute disclosure"). 

The System Effects methodology inherently contains two steps of data processing that (perhaps inadvertently) serve to reduce this disclosure risk:

1) Firstly, re-categorising raw nodes into standard categories ammenable to aggregation will typically distort the raw graph and result 
in a "simpler" re-categorised graph (e.g. when two "raw" nodes and their edges are merged into a common node, this may be considered a form
of microaggregation with some information loss).
2) Secondly, re-categorised individual maps are aggregated, thereby making it harder for an attacker to extract the individual graph 
pattern for an individual in the aggregate ensemble.


While aggregation may go a long way towards securing the confidentiality of participant maps, there are cases where it may fall short:

An aggregate ensemble preserves confidentiality essentially when a sufficiently large number of partipants "fall behind" each node 
and edge. Moreover, aggregates built from simple graphs, with few nodes and edges, drawn by e.g. 50 participants, are likely to preserve 
confidentialy better than more complex graphs, with many nodes and many edges, drawn by the same 50 participants. More nodes and edges 
open up for more permutations on which participants must distribute. In summary, confidentiality "increases" with the number of 
participants and "decreases" with graph complexity.

In practice, any aggregate ensemble will likely have some nodes and edges in the "fringes" which only a small number of participants have 
identified. These will typically have low weight and consequently present a higher risk of disclosure.


In order reduce the risk of the aggregate ensemble inadvertently leaking information from participant maps, CoMap implements a procedure 
to generate a "synthetic" aggregate ensemble, modelled on the original aggregate in such a way that key macrostatistical properties of 
the graph are preserved [#]_


Firstly randomised noise is selectively applied to the aggregate adjacency matrix and secondly singular value decomposition is applied 
to the noised adjacency matrix to reconstruct a low-rank approximate graph.

Method
------

Let :math:`A` represent the adjacency matrix of the original aggregate ensemble :math:`G(\mathbf{n},\mathbf{e})` where :math:`\mathbf{n}` 
is the set of nodes and :math:`\mathfrak{e}` is set of edges. 

In order to generate a synthetic approximation :math:`S`, the adjacency matrix :math:`A` undergoes a two-step procedure, whereby random 
noise is firstly selectively applied to its entries before a low-rank approximation is reconstructed through singular value decomposition 
(SVD): 

.. math::

  G(\mathbf{n},\mathbf{e})\xrightarrow[\text{random noise}]{\text{step 1}} G^{´}(\mathbf{n},\mathbf{e^{'}})\xrightarrow[\text{SVD}]{\text{step 2}} S^{´}(\mathbf{n},\mathbf{e})


Step 1: Random noise
^^^^^^^^^^^^^^^^^^^^

All non-zero elements :math:`a_{ij}` of the adjacency matrix :math:`A` are perturbed. (A zero entry in the adjacency matrix corresponds 
to the absence of an edge in the aggregate ensemble and as such, unlike low non-zero values, does not disclose information from 
participant maps. It is however possible, that the absence of a node or an edge in some cases may reveal to an attacker whether or not 
a participant has contributed to the aggregate ensemble. Perturbing zero-elements was found to significantly distort the statistical
properties of the resulting synthetic aggregate ensemble, and is therefore not handled here.)

Nodes with in- or outdegree equal to 1 correspond to single respondents and as such "high risk". These entries are therefore treated 
separately:

* First a list of 4 arbitrary weights from the set :math:`{1,2,..,10}` is drawn
* Secondly the randomly sampled weights are passed to a Dirichlet distribution generator which is used to sample
edge perturbations:

.. math::
    a_{ij}^{´} = a_{ij} * \mathcal{N_{D}}

where :math:`\mathcal{N_{D}}` is a Dirichlet random draw from the set :math:`{1,2,..,5}`

All edges related to nodes with in- or outdegree equal to 1, have fluctuated to weights between 1 and 5.


In a second step, all non-zero elements :math:`a_{ij}` of the adjacency matrix :math:`A` are randomly "smeared" by a Laplace noising 
function :math:`\mathcal{L}` centered on zero:

.. math::
    a_{ij}^{´} = a_{ij} + \mathcal{L}(a_{ij})

All non-zero entries in the adjancency matrix :math:`A` are thus subject to random perturbation. Nodes 
with in- or outdegree equal to 1, have undergone two independent steps of random perturbation, making it 
difficult for an attacker to know if correspond to an single respondent or if they are the result of 
randomised fluctuations. As such the respondent is afforded some degree of "plausible deniability".
TODO: formalise protection

Step 2: Reconstruct a low-rank approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The synthetic graph ensemble is constructed from the adjacency matrix :math:`S`, defined as an :math:`k^{th}`-rank approximation of the noised adjacency matrix :math:`A^{´}`.

.. math::
    S = U_k \Sigma_k V_k

where :math:`A^{´} = U \Sigma V`.




Usage
------

.. high-light: python


::

  from comap.mapper import CoMap

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

  # create a synthetic ensemble S_agg modelled on G_agg
  S_agg, m_diff, degN_noise, deg1_noise = G_agg.generate_synthetic_graph( noise_scale=0.9, smear_func='laplace') 


  # plot synthetic ensemble
  fig = S_agg.plot_map()


.. rubric: Footnotes

.. [#] This method is experimental and may not meet formal legal requirements of anonymisation. 
.. [#] Caveat: Only verified on a single aggregate, albeit with different parameters.