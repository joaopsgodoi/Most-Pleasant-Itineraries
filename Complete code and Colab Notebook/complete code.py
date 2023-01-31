import queue
import zipfile
import time
import signal
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


  #Given P : a list of queries. We want to test itineraries_v1


def handler(signum, frame):
    raise TimeoutError("N/A"+" itineraries_v1" + " on the test number")

def test_v1 (V, AdjList, P, k):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(time_limit)
    try :
      for (u,v,i) in P :
        itineraries_v1(V,AdjList,u,v)
    except TimeoutError as e:
      print(e,k)
    signal.alarm(0)
    
    
# we consider that the vertices are {1,....,n}
class Task3:
    def __init__(self, AdjList):
        self.n = len(AdjList)
        self.log = (self.n - 1).bit_length()
        self.parent = [[-1]*self.n for _ in range(self.log)]
        self.noise = [[0]*self.n for _ in range(self.log)]
        self.depth = [0]*self.n
        self.building(AdjList,1,-1,0,0) # we consider that 1 is the root
        self.fill_in()

    def building(self,AdjList,v,p,d,w):
        self.parent[0][v-1]= p  # v-1 because v is in {1,2,...n}
        self.noise[0][v-1]= w # the noise of (p,v)
        self.depth[v-1] = d
        for (u,weight) in AdjList[v]:
            if u != p:
                self.building(AdjList,u,v,d+1,weight)

    def fill_in(self):
        for i in range(1,self.log):
            for v in range(self.n):
                if self.parent[i-1][v-1] !=-1:
                    self.parent[i][v-1] = self.parent[i-1][self.parent[i-1][v-1]-1]
                    self.noise[i][v-1] = max(self.noise[i-1][v-1],self.noise[i-1][self.parent[i-1][v-1]-1])

    def itineraries_v2(self,u,v):
        m=0
        if self.depth[u-1] < self.depth[v-1]:
            u, v = v, u
        for i in range(self.log):
            if (self.depth[u-1] - self.depth[v-1]) >> i & 1:
                u,m = self.parent[i][u-1], max(m, self.noise[i][u-1])
        #In the end of this loup, we are sure that u and v are in the same depth
        if u == v:
            return m
        for i in range(self.log - 1, -1, -1):
            if self.parent[i][u-1] != self.parent[i][v-1]:
                u,v,m = self.parent[i][u-1],self.parent[i][v-1],max(m,self.noise[i][u-1],self.noise[i][v-1])
        # In this loop, if r= LCA(u,v) then on each step i we have depth(u-1)-depth(r-1)<=2**i
        # the Lowest common ancestor is given by parent[0][u-1]
        return max(m,self.noise[0][u-1],self.noise[0][v-1])
    
    #the function below  will test itineraries_v2 on the queries P of k th test
    def test(self, P):
      for (u,v,i) in P :
        self.itineraries_v2(u,v)

  
def handler2(signum, frame):
    raise TimeoutError("N/A"+" itineraries_v2" + " on the test number")

def test_v2 (AdjList, P, k):
    signal.signal(signal.SIGALRM, handler2)
    signal.alarm(time_limit)
    try :
      Tree = Task3(AdjList)
      Tree.test(P)
    except TimeoutError as e:
      print(e,k)
    signal.alarm(0)
    
    
    
# we still consider that the vertices are {1,....,n}



def find_v3(parent, noise, v):
    if parent[v - 1] == v:
        return v
    else:
        p = parent[v - 1]
        r = find_v3(parent, noise, p)
        # We make sure that we update the list noise so in the end noise[v-1] should contains the maximum noise on its path to the vertex find(v).
        #the 'if' bellow it's just to make sure that we really calculated the noise in the path from v to its representant r
        if p != r:
            noise[v - 1] = max(noise[p - 1], noise[v - 1])
        return r


def itineraries_v3(V, AdjList, P, n):
    parent = [-1 for v in V]
    edgeSeen = [False for e in P]
    noise = [0 for v in V]
    vertexSeen = [False for v in V]
    result = [-1] * n
    LCA = [[] for r in V]
    # LCA[r] is a list of couples (u,v) such that LCA of (u,v) is r
    TarjanLCA(V, AdjList, P, n, parent, edgeSeen, noise, vertexSeen, result, LCA, V[0], -1)
    return result
    
def TarjanLCA(V, AdjList, P, n, parent, edgeSeen, noise, vertexSeen, result, LCA, u, p):
    parent[u - 1] = u
    for (v, w) in AdjList[u]:
        if v != p:
            noise[v - 1] = w
            TarjanLCA(V, AdjList, P, n, parent, edgeSeen, noise, vertexSeen, result, LCA, v, u)
            parent[v-1] = u
    vertexSeen[u - 1] = True
    k=0
    while k < n :
        if not(edgeSeen[k]):
            (x, y, i) = P[k]
            # (x,y) is the ith query
            if x == u and vertexSeen[y - 1]:
                lca = find_v3(parent, noise, y)
                LCA[lca - 1].append((x, y, i))
                edgeSeen[k] = True
            elif y == u and vertexSeen[x - 1]:
                lca = find_v3(parent, noise, x)
                LCA[lca - 1].append((x, y, i))
                edgeSeen[k] = True
        k += 1
    for (x, y, i) in LCA[u - 1]:
        if x == u:
            find_v3(parent, noise, y)
            result[i] = noise[y - 1]
        elif y == u:
            find_v3(parent, noise, x)
            result[i] = noise[x - 1]
        else:
            find_v3(parent, noise, x)
            find_v3(parent, noise, y)
            result[i] = max(noise[x - 1], noise[y - 1])
    


def handler3(signum, frame):
    raise TimeoutError("N/A"+" itineraries_v3" + " on the test number")

def test_v3 (V, AdjList, P, l,k):
    signal.signal(signal.SIGALRM, handler3)
    signal.alarm(time_limit)
    try :
      itineraries_v3(V,AdjList,P,l)
    except TimeoutError as e:
      print(e,k)
    signal.alarm(0)
    
    
    
#Test de joao

for k in range(10):
    name_test = "C:\\Users\\jpsed\\Desktop\\tests\\tests\\itineraries." + str(k) + ".in"
    with open(name_test, "r") as test:
        n, m = map(int, test.readline().strip().split())
        V = [i for i in range(1,n+1)]
        P = []
        E = []
        for _ in range(m):
            x, y, w = map(int, test.readline().strip().split())
            E.append((x, y, w))
        
        l = int(test.readline().strip())
        for i in range(l):
            u, v = map(int,  test.readline().strip().split())
            P.append((u, v, i))
        mst = MST(V,E)
        AdjList = AdjacencyList(V,mst)
        Tree = Task3(AdjList)
        
        p=0
        print("test"+str(k))
        print(l)
        
        arquivo = open("C:\\Users\\jpsed\\Desktop\\Documentos\\" +"output"+str(k)+".txt", "a")
        for x in P:
            a = Tree.itineraries_v2(x[0],x[1])
            arquivo.write(str(a)+"\n")
    



