#%%

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

#%%

## Graficas de prueba de Fath

D01 = nx.DiGraph()
D01.add_edges_from([
    ("Prawns", "Large fish", {'weight': 90.4}),
    ("Large fish", "Alligators", {'weight': 31.4})])

D02 = nx.DiGraph()
D02.add_edges_from([
    ("Prawns", "Turtles", {'weight': 74.3}),
    ("Prawns", "Snakes", {'weight': 16.1}),
    ("Turtles", "Alligators", {'weight': 7.20}),
    ("Snakes", "Alligators", {'weight': 2.06})])

D03 = nx.DiGraph()
D03.add_edges_from([
    ("Prawns", "Turtles", {'weight': 64.5}),
    ("Prawns", "Large fish", {'weight': 11.9}),
    ("Prawns", "Snakes", {'weight': 14.0}),
    ("Turtles", "Alligators", {'weight': 6.25}),
    ("Large fish", "Alligators", {'weight': 4.13}),
    ("Snakes", "Alligators", {'weight': 1.79})])

#%%

def Capacity(D, weight_col_name='weight'):
    T = nx.adjacency_matrix(D, weight = weight_col_name).todense()
    return - np.sum(T * np.log(T / np.sum(T), out=np.zeros_like(T), where=(T!=0)))

def Redundancy(D, weight_col_name='weight'):
    T = nx.adjacency_matrix(D, weight= weight_col_name).todense()
    k = np.sum(T)
    Ti_ = np.sum(T,axis=1)
    T_j = np.sum(T,axis=0)
    T2 = T*T
    Ti_T_j =  np.outer(Ti_,T_j)
    Q = np.divide(T2,Ti_T_j, out=np.zeros_like(T2), where=(T2!=0))
    return - np.sum(T * np.log(Q, out=np.zeros_like(Q), where=(Q!=0)))

def Ascendency(D, weight_col_name='weight'):
    return Capacity(D, weight_col_name) -  Redundancy(D, weight_col_name)

def TST(D, weight_col_name='weight'):
    T = nx.adjacency_matrix(D, weight=weight_col_name).todense()
    return np.sum(T)

def Robustness(D, weight_col_name='weight'):
    return - (Ascendency(D, weight_col_name)/Capacity(D, weight_col_name))*np.log(Ascendency(D, weight_col_name)/Capacity(D, weight_col_name))

print(Capacity(D01), Ascendency(D01), Redundancy(D01), TST(D01), Robustness(D01))
print(Capacity(D02), Ascendency(D02), Redundancy(D02), TST(D02), Robustness(D02))
print(Capacity(D03), Ascendency(D03), Redundancy(D03), TST(D03), Robustness(D03))

#%%

## Gráfica Con redes de prueba de Fath

plt.figure()
X = np.linspace(0.001, 1, 100)
Y = np.array([- x * np.log(x) for x in X])
plt.plot(X,Y)
redes = [D01, D02, D03]
for red in redes:
    A = Ascendency(red)
    C = Capacity(red)
    R = Robustness(red)
    plt.scatter( (A/C), R)
plt.show()

#%%
import pandas as pd

#%%

## se importan redes tróficas de web of life
base_url = "https://www.web-of-life.es/"
query = "get_networks.php?interaction_type=FoodWebs"
data = pd.read_json(base_url + query)

#%%

# se les calculan las métricas

A = []
C = []
R = []

for nombre_red in pd.unique(data.network_name):
    data_red = data[data.network_name == nombre_red ]
    red = nx.from_pandas_edgelist(data_red,
                                  source='species1',
                                  target='species2',
                                  edge_attr='connection_strength',
                                  create_using=nx.DiGraph)
    A.append(Ascendency(red, 'connection_strength'))
    C.append(Capacity(red, 'connection_strength'))
    R.append(Robustness(red, 'connection_strength'))
#%%

# se grafican para las redes de web of life
plt.figure()
X = np.linspace(0.001, 1, 100)
Y = np.array([- x * np.log(x) for x in X])
plt.plot(X,Y)
for i in range(len(A)):
    plt.scatter(A[i]/C[i], R[i])
plt.show()

#%%

# se elige una de las redes para simular extinciones quitando nodos al azar

G = nx.from_pandas_edgelist(data[data.network_name == 'FW_008' ],
                            source='species1',
                            target='species2',
                            edge_attr='connection_strength',
                            create_using=nx.DiGraph)


red_original = G.copy()
red_extinguir = red_original.copy()
num_extinciones = 200

A = []
C = []
R = []
etiquetas = [] # guarda el núero de nodos que se han removido

# weight_col_name='weight'
weight_col_name='connection_strength'
A.append(Ascendency(red_extinguir, weight_col_name))
C.append(Capacity(red_extinguir,weight_col_name))
R.append(Robustness(red_extinguir, weight_col_name))
etiquetas.append(0)

for i in range(1,num_extinciones+1):
    nodos = list(red_extinguir.nodes)
    nodo_a_remover = np.random.choice(nodos)
    red_extinguir.remove_node(nodo_a_remover)
    
    A.append(Ascendency(red_extinguir,weight_col_name))
    C.append(Capacity(red_extinguir,weight_col_name))
    R.append(Robustness(red_extinguir,weight_col_name))
    etiquetas.append(i)

# cada la etiqueta indica el número de nodos que se remueven de la red  
plt.figure()
X = np.linspace(0.001, 1, 100)
Y = np.array([- x * np.log(x) for x in X])
plt.plot(X,Y)
for i in range(len(A)):
    plt.scatter(A[i]/C[i], R[i], label = etiquetas[i])
plt.legend()
plt.show()

# redundancia vs, num de nodos removidos
plt.figure()
plt.plot(etiquetas, R)
plt.show()
