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