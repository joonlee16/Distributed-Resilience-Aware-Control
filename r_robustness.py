from gurobipy import gurobi
import numpy as np
import cvxpy as cp
from itertools import combinations, product
import networkx as nx

def add_dir_edges(edges, L):
    for (i,j) in edges:
        L[j][j]+=1
        L[j][i]=-1
    return L

def add_edges(edges, L):
    for (i,j) in edges:
        L[i][j]=-1
        L[j][i]=-1
        L[i][i]+=1
        L[j][j]+=1
    return L


def directed_milp_r_robustness(edges, n, min_r=0.0):
    L = add_edges(edges, np.zeros((n,n)))

    b = cp.Variable((2*n,1) , integer=True)#boolean = True )
    t = cp.Variable(integer=True)
    obj = cp.Minimize(t)
    L2 = np.kron( np.eye(2), L )
    eig = np.sort(np.linalg.eig(L)[0])[1]
    if min_r==0:
        min_r=np.real(eig/2)
    const = []        
    const += [t >= min_r]
    const += [ L2 @ b <= t * np.ones((2*n,1)) ]
    const += [ b >= np.zeros((2*n,1)), b <= np.ones((2*n,1)) ] # binary constraint
    const += [ np.append( np.eye(n), np.eye(n), axis=1 ) @ b <= 1  ]     
    const += [ np.append( np.ones((1,n)), np.zeros((1,n)), axis=1) @ b >= 1 ]   
    const += [ np.append( np.ones((1,n)), np.zeros((1,n)), axis=1) @ b <= n-1 ]
    const += [ np.append( np.zeros((1,n)), np.ones((1,n)), axis=1) @ b >= 1 ]
    const += [ np.append( np.zeros((1,n)), np.ones((1,n)), axis=1) @ b <= n-1 ]
    prob = cp.Problem( obj, const )
    prob.solve()

    #print(f"b: {b.value[:int(len(b.value)/2)]} and\n {b.value[int(len(b.value)/2):]}")
    return t.value


def directed_milp_rs_robustness( L, r , tell):
    n = np.shape(L)[0]
    s = cp.Variable()
    b1 = cp.Variable((n,1) , integer=True)#, boolean = True)
    b2 = cp.Variable((n,1) , integer=True)#, boolean = True)
    y1 = cp.Variable((n,1) , integer=True)#, boolean = True)
    y2 = cp.Variable((n,1) , integer=True)#, boolean = True)
    ones = np.ones((n,1))
    
    obj = cp.Minimize( s )
    
    const = []
    const += [ s >= 1, s<=(n+1) ]
    const += [ ones.T @ y1 <= ones.T @ b1 - 1 ]
    const += [ ones.T @ y2 <= ones.T @ b2 - 1 ]
    const += [ ones.T @ y1 + ones.T @ y2 <= s - 1 ]
    const += [ L @ b1 - n * y1 <= (r-1) * ones ]
    const += [ L @ b2 - n * y2 <= (r-1) * ones ]
    const += [ b1 + b2 <= ones ]
    const += [ 1 <= ones.T @ b1, ones.T @ b1 <= n-1 ]
    const += [ 1<= ones.T @ b2, ones.T @ b2 <= n-1 ]
    
    prob = cp.Problem( obj, const )
    
    prob.solve(verbose = tell)
    print(f"s status: {prob.status}, s:{s.value}")
    return s.value


def james_underbound_r(L):
    n = np.shape(L)[0]
    b = cp.Variable((n,1) , integer=True)#boolean = True )
    t = cp.Variable()
    obj = cp.Minimize(t)

    const = []        
    const += [t >= 0]
    const += [ L @ b <= t * np.ones((n,1)) ]
    const += [ b >= np.zeros((n,1)), b <= np.ones((n,1)) ] # binary constraint
    const += [ np.eye(n) @ b <= 1  ]     
    const += [ np.ones((1,n)) @ b >= 1 ]   
    const += [ np.ones((1,n)) @ b <= int(n/2) ]

    prob = cp.Problem( obj, const )
    prob.solve()

    #print(f"b: {b.value[:int(len(b.value)/2)]} and\n {b.value[int(len(b.value)/2):]}")
    return t.value

def cheegers_constant(L):
    n = np.shape(L)[0]
    b = cp.Variable((n,1) , integer=True)#boolean = True )
    t = cp.Variable()
    k = cp.Variable()
    obj = cp.Minimize(t+k)

    const = []        
    const += [t >= 1]
    const += [1 <= k, k <=  int(n/2)]
    const += [ np.ones((1,n)) @ L @ b <= t ]
    const += [ b >= np.zeros((n,1)), b <= np.ones((n,1)) ] # binary constraint
    const += [ np.eye(n) @ b <= 1  ]     
    const += [ np.ones((1,n)) @ b >= k ]   
    const += [ np.ones((1,n)) @ b <= int(n/2) ]

    prob = cp.Problem( obj, const )
    prob.solve()

    #print(f"b: {b.value[:int(len(b.value)/2)]} and\n {b.value[int(len(b.value)/2):]}")
    return t.value/k.value

def bruteforce(n,edges, r_desirable):
    r = int((n-1)/2)+1
    if r_desirable!=0:
        r = r_desirable
    
    nodes = [i for i in range(n)]
    r_min =10000000
    did=[]
    for i in range(2, n+1):
        allpossible_combo = combinations(nodes, i)
        for aa in allpossible_combo:
            for j in range(1,int(len(aa)/2)+1):
                s1_options = combinations(aa, j)
                for s1 in s1_options:
                    s2_options = [k for k in aa if k not in s1]
                    for k in range(1, len(s2_options)+1):
                        for s2 in combinations(s2_options, k):
                            if [s1,s2] in did:
                                continue
                            else:
                                did.append([s1,s2])
                                did.append([s2,s1])
                            temp=[0 for p in range(n)]
                            for (fro,two) in edges:
                                if (two in s1 and fro not in s1) or (two in s2 and fro not in s2):
                                    temp[two]+=1
                            if max(temp) < r_min:
                                r_min = max(temp)
    
    return r_min




def min_edge( n,r ):
    b = cp.Variable((2*n,1) , integer=True)#boolean = True )
    t = cp.Variable(integer=True)
    obj = cp.Minimize(t)
    L = cp.Variable((n,n), integer=True)
    L2 = np.kron( np.eye(2),L)

    
    const = [t >= 0, t<=n*(n-1)]
    edges=0
    for i in range(n):
        sum=0
        for j in range(n):
            sum+=L[j][i]
            const+=[L[j][i]<=0, L[j][i]>=-1]
            edges-=L[j][i]
        const+=[L[i][i]-sum==0]
    b1 = map(list, product([0, 1], repeat=n))
    const+=[edges <= t]
    const += [ L2 @ b >= r * np.ones((2*n,1)) ]
    const += [ b >= np.zeros((2*n,1)), b <= np.ones((2*n,1)) ] # binary constraint
    const += [ np.append( np.eye(n), np.eye(n), axis=1 ) @ b <= 1  ]     
    const += [ np.append( np.ones((1,n)), np.zeros((1,n)), axis=1) @ b >= 1 ]   
    const += [ np.append( np.ones((1,n)), np.zeros((1,n)), axis=1) @ b <= n-1 ]
    const += [ np.append( np.zeros((1,n)), np.ones((1,n)), axis=1) @ b >= 1 ]
    const += [ np.append( np.zeros((1,n)), np.ones((1,n)), axis=1) @ b <= n-1 ]

    prob = cp.Problem( obj, const )
    prob.solve()

    #print(f"b: {b.value[:int(len(b.value)/2)]} and\n {b.value[int(len(b.value)/2):]}")
    return t.value


