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
        parent[v-1]=r
        return r

def itineraries_v3(V, AdjList, queries, l):
    n = len(V)
    parent = [-1]*n
    noise = [0]*n
    vertexSeen = [False]*n
    result = [0] * l
    LCA = [[] for _ in range(n)]
    # LCA[r] is a list of couples (u,v) such that LCA of (u,v) is r
    TarjanLCA(V, AdjList, queries, l, parent, noise, vertexSeen, result, LCA, V[0], -1)
    return result


def TarjanLCA(V, AdjList, queries, l, parent, noise, vertexSeen, result, LCA, u, p):
    parent[u - 1] = u
    for (v, w) in AdjList[u]:
        if v != p:
            noise[v - 1] = w
            TarjanLCA(V, AdjList, queries, l, parent, noise, vertexSeen, result, LCA, v, u)
            parent[v-1] = u
    vertexSeen[u - 1] = True
    for (v, i) in queries[u-1]:
        if vertexSeen[v - 1]:
            lca = find_v3(parent, noise, v)
            LCA[lca - 1].append((u, v, i))
    for (x, y, i) in LCA[u - 1]:
        if x == y:
            result[i] = 0
        else:
            find_v3(parent, noise, x)
            if y == u:
                result[i] = noise[x - 1]
            else:
                result[i] = max(noise[x - 1], noise[y - 1])
