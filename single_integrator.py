import numpy as np
from random import randint
from scipy.integrate import solve_ivp 
import matplotlib.cm as cm


## Normal agents
class Agent:
    def __init__(self,location, color, palpha, ax, F, id):
        self.location = location.reshape(2,-1)
        self.locations = [[],[]]
        self.color = color
        self.palpha = palpha
        self.body = ax.scatter([],[],c=color,edgecolors='black',alpha=palpha,s=150)
        self.value= randint(-500,500)        
        self.original = self.value
        self.connections = []
        self.F = F
        self.values = []
        self.history = [self.value]
        self.id = id
        self.set_color()

    def f(self):
        return np.array([0,0]).reshape(-1,1)
    
    def g(self):
        return np.array([ [1, 0],[0, 1] ])
    
    # Updates the agent's state
    def step(self, U, dt):
        self.U = U
        def model(t, y):
            dydt = self.g()@ self.U
            return dydt.reshape(-1,2)[0]
        steps = solve_ivp(model, [0,dt], self.location.reshape(-1,2)[0])
        what = np.array([steps.y[0][-1], steps.y[1][-1]])
        self.location = what.reshape(2,-1)
        self.render_plot()
        return self.location

    # Updates the agent's plot 
    def render_plot(self):
        x = np.array([self.location[0][0],self.location[1][0]])
        self.locations[0] = np.append(self.locations[0], x[0])
        self.locations[1] = np.append(self.locations[1], x[1])
        self.body.set_offsets([x[0],x[1]])

    # Computes the distance function h_{ij}^{col} between the agent i and j, and also computes its derivatives w.r.t. both agents
    def agent_barrier(self,agent,d_min):
        h =  np.linalg.norm(self.location - agent.location)**2 - d_min**2 
        dh_dxi = 2*( self.location - agent.location[0:2]).T
        dh_dxj = -2*( self.location - agent.location[0:2] ).T
        return h, dh_dxi, dh_dxj

    # Forms an edge with the given agent.
    def connect(self, agent):
            self.connections.append(agent)

    # Returns its neighbor set
    def neighbors(self):
        return self.connections

    # Returns the list of neighboring robots
    def neighbors_id(self):
        return [aa.id for aa in self.connections]
    
    # Resets its neighbor set 
    def reset_neighbors(self):
        self.connections = []

    # Sets the agent's LED color based on its consensus state (It is used to color the past trajectory of the agent)
    def set_color(self):
        self.LED = cm.tab20((self.value+500)/1000)

    # Shares its consensus value with its neighbors  
    def propagate(self):
        for neigh in self.neighbors():
            neigh.receive(self.value)
        return self.value

    # Receives the consensus states from its neighbor
    def receive(self, value):
        self.values.append(value)
    
    # Performs W-MSR
    def w_msr(self):
        small_list=[];big_list=[];comb_list=[]

        for aa in self.values:
            if aa<self.value:
                small_list.append(aa)
            elif aa>self.value:
                big_list.append(aa)
            else:
                comb_list.append(aa)

        small_list = sorted(small_list)
        small_list = small_list[self.F:]

        big_list = sorted(big_list)
        big_list = big_list[:-self.F]
        comb_list = small_list+ comb_list + big_list
        total_list =len(comb_list)
        weight = 1/(total_list+1)
        weights = [weight for i in range(total_list)]
        avg = weight*self.value + sum([comb_list[i]*weights[i] for i in range(total_list)])
        self.value = avg

        self.history.append(self.value)
        self.values = []


## Malicious agents
class Malicious(Agent):
    def __init__(self, location, color, palpha, ax, F, id):
        super().__init__(location, color, palpha, ax, F, id)
        self.time = 0
        self.value = 500*np.sin(self.id+self.time/3.5)
        self.history = [self.value]

    # Sends wrong values to all of its neighbors
    def propagate(self):
        self.time+=1
        self.value = 500*np.sin(self.id+self.time/3.5)
        for neigh in self.neighbors():
            neigh.receive(self.value)
        return self.value
    
    # Does not follow W-MSR to update its consensus state
    def w_msr(self):
        self.history.append(self.value)
        self.values = []