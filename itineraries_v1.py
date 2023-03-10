import queue
import zipfile
import time
import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer


time_limit = 30

# TASK 2
" We start by implementing an algorithm that computes the MST"
"based on Kruskal Method and optimized with the Union-Find by rank"

# the graph is defined by (V,E) where V is the set of vertices, and E is a list of tuples,
# where each tuple represents an edge and contains
# the vertices that the edge connects, along with the edge's weight.

def MST(V,E):
    edges = sorted(E, key=lambda x: x[2])
    # initialize of parent list (to compute the representative of the class)
    # The second coordinate is the rank of the class
    parent = {v: [v,0] for v in V}
    # initialize the empty list to store the edges of the minimum spanning tree
    mst = []
    for edge in edges :
        # find the representative of each vertices
        v1 = find(parent,edge[0])
        v2 = find(parent, edge[1])
        if v1 != v2 :
            mst.append(edge)
            union(parent,v1,v2)
    return mst

# We now define the find and union algorithm

def find(parent,v):
    if parent[v][0] == v:
        return v
    else :
        r = find(parent , parent[v][0])
        parent[v][0]=r
        return find(parent , parent[v][0])


def union(parent,x,y):
    r1 = find(parent, x)
    r2 = find(parent, y)
    # if rank[r1] > rank[r2]
    if parent[r1][1] > parent[r2][1]:
      parent[r2][0] = r1
    elif parent[r1][1] < parent[r2][1]:
      parent[r1][0] = r2
    else :
      parent[r1][0] = r2
      # increment the rank
      parent[r2][1] += 1


"Finally, we write the itineraries_v1's algorithm"

def AdjacencyList(V,E):
    Adj = {v: [] for v in V}
    # this dictionary is such that M[v] contains a list l1 of couples (x,w)
    # such that the edge (v,x) in E and w its weight
    for edge in E:
      Adj[edge[0]].append((edge[1], edge[2]))
        # edge[2] is the weight of the edge
      Adj[edge[1]].append((edge[0], edge[2]))
    return Adj

def itineraries_v1(V, Adjacency_list,start,end):
  #we suppose here that Adjacency_list = AdjacencyList(V,mst) and where mst = MST(V,E)
  Parent= {v: v for v in V}
  #the next dict will help us verify if we already visited the vertex 
  #but also to store the weight
  NoisePath ={v: -1 for v in V}
  NoisePath[start] = 0
  q= queue.Queue(maxsize=len(V))
  q.put(start)
  while q:
    vertex= q.get()
    if vertex == end :
      return computeMaxNoise(Parent,NoisePath,start,end)
    else :
      for (neighbor,w) in Adjacency_list[vertex]:
      #Verify that we didn't already visit this neighbor 
        if NoisePath[neighbor] == -1:
          Parent[neighbor] = vertex
          NoisePath[neighbor] = w
          q.put(neighbor)
    
      
      
def computeMaxNoise(Parent,NoisePath,start,end):
  current = end
    # Stocking the maximum noise
  M = 0
  while current != start:
    M = max(NoisePath[current],M)
    current = Parent[current]
  return M

