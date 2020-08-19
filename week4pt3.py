# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 10:30:25 2020

@author: user
"""
import networkx as nx
from scipy.stats import bernoulli
#ER Graph 
def er_graph(N,p):
    G=nx.Graph()
    #add nodes
    G.add_nodes_from(range(N))
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1>node2 and bernoulli.rvs(p=p):
                 G.add_edge(node1,node2)
    return G

import numpy as np
A1=np.loadtxt('C:/Users/user/Using python for research/week4/adj_allVillageRelationships_vilno_1.csv',delimiter=',')
A2=np.loadtxt('C:/Users/user/Using python for research/week4/adj_allVillageRelationships_vilno_2.csv',delimiter=',')

G1=nx.to_networkx_graph(A1)
G2=nx.to_networkx_graph(A2)
import matplotlib.pyplot as plt
def plot_degree_distribution(G):
    degree_sequence = [d for n, d in G.degree()]
    plt.hist(degree_sequence, histtype="step")
    plt.xlabel('Degree $k$')
    plt.ylabel('$P(k)$')
    plt.title('Degree distribution')


def basic_net_stats(G):
    print('Nodes:', G.number_of_nodes())
    print('Edges:', G.number_of_edges())
    degree_sequence=[d for n, d in G.degree()]
    print("Average Degree: %.2f"%np.mean(degree_sequence))
    
    
gen=(G1.subgraph(c) for c in nx.connected_components(G1))
g=gen.__next__()
G1_LCC=max((G1.subgraph(c) for c in nx.connected_components(G1)),key=len)

plt.figure()
G1_LCC=max((G1.subgraph(c) for c in nx.connected_components(G1)),key=len)
nx.draw(G1_LCC,node_color='red',edge_color='grey',node_size=20)
plt.savefig('village1.pdf')

plt.figure()
G2_LCC=max((G2.subgraph(c) for c in nx.connected_components(G2)),key=len)
nx.draw(G2_LCC,node_color='green',edge_color='grey',node_size=20)
plt.savefig('village2.pdf')