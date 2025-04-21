import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from single_integrator import *
from helper import *

# Choose the scenario
scenario = choose()

# Sim Parameters  
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-3.25,3.25),ylim=(-1.5,1.5)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")
F = 2
R = 3
d_min = 0.5
num_constraints1  = 1
alphas = 0.1
umax = 1.5
tau = 1
T = 10
dt =0.002
max_T = T/dt
step_size = tau/dt

# Initialize the robots
robots = []
y_offset = 0.9
robots.append( Malicious(np.array([-1.1,y_offset - 0.5]),'#d62728',1.0, ax, F,0))
robots.append( Malicious(np.array([-0.8,y_offset - 1.0]),'#d62728',1.0 , ax, F,1))
robots.append( Agent(np.array([0.8,y_offset - 1.2]),'#1f77b4',1.0 , ax, F,2))
robots.append( Agent(np.array([0.1,y_offset - 2.1]),'#1f77b4',1.0 , ax, F,3))
robots.append( Agent(np.array([0.4,y_offset - 1.6]),'#1f77b4',1.0 , ax, F,4))
robots.append( Agent(np.array([-0.5,y_offset - 1.9]),'#1f77b4',1.0 , ax, F,5))
robots.append( Agent(np.array([-0.9,y_offset - 1.3]),'#1f77b4',1.0 , ax, F,6))
robots.append( Agent(np.array([1,y_offset - 0.1]),'#1f77b4',1.0 , ax, F,7))
robots.append( Agent(np.array([0.4,y_offset]),'#1f77b4',1.0 , ax, F,8))
robots.append( Agent(np.array([-0.9,y_offset - 0.4]),'#1f77b4',1.0 , ax, F,9))
robots.append( Agent(np.array([1.3,y_offset - 0.6]),'#1f77b4',1.0 , ax, F,10))

n =len(robots)
F_prime = F + n // 2

############################## Optimization problems ###################################
u1 = cp.Variable((2,1))
u1_des = cp.Parameter((2,1),value = np.zeros((2,1)) )
A1 = cp.Parameter((num_constraints1,2),value=np.zeros((num_constraints1,2)))
b1 = cp.Parameter((num_constraints1,1),value=np.zeros((num_constraints1,1)))
const1 = [A1 @ u1 >= b1]
const1+= [u1<=umax, -umax<=u1]
objective1 = cp.Minimize( cp.sum_squares( u1 - u1_des  ))
cbf_controller = cp.Problem( objective1, const1 )
########################################################################################

# Setting up the goal location
goal = np.array([[0,100],[100,0]])

# Start the simulation 
counter = 0
H =[[] for i in range(n)]
robustness_history = []
while True:   
    x = np.array([aa.location.reshape(1,-1)[0] for aa in robots])
    # Compute the actual robustness
    edges = []
    for i in range(n):
        for j in range(i+1,n):
            if np.linalg.norm(x[i]-x[j]) <=R:
                edges.append((i,j))

    # Agents form a network
    for (i,j) in edges:
        robots[i].connect(robots[j])
        robots[j].connect(robots[i])

    # Get the nominal control input
    u_des = []
    for i in range(1,n+1):
        helper = (-1)**i * np.array([100, 0])
        if i in range(6, 12):
            helper= (-1)**i * np.array([100, 0]) 
        vector = (helper - x[i-1]).reshape(-1,1)
        u_des.append(vector/np.linalg.norm(vector)) 

    # Compute the h_i and \frac{\partial h_i/} {\partial x}
    h, der_ = compute_h_and_der(n,x,edges,R)
    h = h-F_prime

    # Store the connectivity levels
    for i in range(n):
        H[i].append(h[i])

    #Set up the constraint of QP
    A1.value[:,:]=0
    b1.value[:,:]=0
    control_input = []

    # Set up the weight w and h_hat values according to the scenario
    w = [15, 25]    
    h_hat = h
    if scenario == 2:
        h_hat[0:2]+=3.5
    elif scenario ==3:
        w = [17.5, 30]    
        h_hat[0:2]-=2.5

    if np.any(h_hat<0):
        print(counter, h_hat)

    # Each robot constructs the QP and computes its contorl input u_i locally
    for i in range(n):
        u1_des.value = u_des[i]
        N_i = robots[i].neighbors_id()
        B_i = np.append(N_i,i)
        c = []; c_der_ = []
        for j in N_i:
            h_ij, dh_dxi, _ = robots[i].agent_barrier(robots[j],d_min)
            c.append(h_ij)
            c_der_.append(dh_dxi)
        c = np.array(c); c_der_ = np.array(c_der_)
        c_exp_list = np.exp(-w[1]*c).reshape((1,-1))
        c_exp_der = c_exp_list*w[1]
        exp_list = np.exp(-w[0]*(h_hat[B_i])).reshape((1,-1))
        exp_der = exp_list*w[0]
        A1.value[0,:]= exp_der @ (der_[B_i,i].reshape(-1,2)) + c_exp_der @ (c_der_.reshape(-1,2))
        b1.value[0,0]= -alphas*(1/n-sum(exp_list[0])/(F_prime+1)) + alphas*(sum(c_exp_list[0]))/2
        # b1.value[0,0]= -alphas*(1/n-exp_list1) + alphas*(sum(c_exp_list[0]))/2

        cbf_controller.solve(solver="GUROBI")
        if cbf_controller.status!='optimal':
            print("Error: should not have been infeasible here")
            print(h)
            control_input.append(np.array([[0],[0]]))
        else:
            control_input.append(u1.value)

    # Agents share their values with neighbors every tau seconds
    if counter >50 and counter % step_size ==0:
        for aa in robots:
            aa.propagate()
        # The agents perform W-MSR
        for aa in robots:
            aa.w_msr()
        # All the agents update their LED colors based on its consensus states (it is used to color the past trajectories of each agents)
        for aa in robots:
            aa.set_color()

    # Each robot implements its own control input u_i
    for i in range(n):
        robots[i].step(control_input[i], dt) 
        robots[i].reset_neighbors()
        
        '''Colors the trajectories in their current LED colors'''
        # if counter>0:
        #     plt.plot(robots[i].locations[0][counter-1:counter+1], robots[i].locations[1][counter-1:counter+1], color = robots[i].LED, zorder=0)  
    '''Plots the environment and robots: uncomment to visualize the simulation online. Will slow down the simulation significantly'''
    # fig.canvas.draw()
    # lines = []
    # for (i, j) in edges:
    #     l_color = '#555555'
    #     if i<=1 or j<=1:
    #         l_color = '#FF0000'
    #     lines.append(plt.plot(
    #         [x[i][0], x[j][0]],
    #         [x[i][1], x[j][1]],
    #         linestyle='--', color=l_color, zorder=0, linewidth=1.5
    #     ))

    # fig.canvas.flush_events()  

    # for line in lines:
    #     l = line[0]
    #     l.remove()

    # If time, terminate
    counter+=1
    if counter>=max_T:
        break

plt.ioff()
fig2 = plt.figure()

# Plot the evolutions of h_{i}'s values
for i in range(n):
    if i <=1:
        plt.plot(np.arange(counter), H[i],"r--")
    else:
        plt.plot(np.arange(counter), H[i], color="C{}".format(i % 10))
plt.plot(np.arange(counter), [0]*counter, 'k--')
plt.title("Evolution of $h_i$")
plt.show()

# Plot the evolutions of consensus states
length_of_consensus = len(robots[0].history)*step_size
for aa in robots:
    temp = np.repeat(np.array(aa.history),step_size)
    if issubclass(type(aa), Malicious):
        plt.plot(np.arange(0,length_of_consensus)*dt,temp, "r--")
    else:
        plt.plot(np.arange(0,length_of_consensus)*dt, temp)
plt.show()
