CoMap
================

**System Effects** is a research methodology to explore the 'user' or citizen 
experience of complex phenomena. Drawing on soft systems, fuzzy
cognitive mapping and graph theory, it aims to capture the varied nature 
of the individual experience to enable better intervention design.

This python package provides key utilities to support graph analyses of data 
within the framework of this methodology, such as graph relabelling, 
graph aggregation, graph analysis tools and tools to minimize the potential leakage 
of private information in aggregate graphs.

The full docs are found [here](https://navikt.github.io/comap/).

## More information

Luke K. Craven (2017) System Effects: A Hybrid Methodology for Exploring the Determinants of Food In/Security, Annals of the American Association of Geographers, 107:5, 1011-1027, DOI: 10.1080/24694452.2017.1309965

# Quick start

```python
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

  # get key properties of graph aggregate
  G_agg.get_properties()

  # plot graph ensemble
  fig = A_agg.plot_map()
  fig.show()
```

---

# Contact

* Robindra Prabhu, robindra.prabhu@nav.no

