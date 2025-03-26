import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from obstacles import *
from single_integrator import *
from r_robustness import directed_milp_r_robustness

plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-3.25,3.25),ylim=(-1.5,1.5)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")

########################## Make Obatacles ###############################
obstacles = []
index = 0
x1 = -1.2#-1.0
x2 = 1.2 #1.0
radius = 0.6
y_s = 0
y_increment = 0.3
# for i in range(int( 3/y_increment )):
#     obstacles.append( circle( x1,y_s,radius,ax,0 ) ) # x,y,radius, ax, id
#     obstacles.append( circle( x2,y_s,radius,ax,1 ) )
#     y_s = y_s + y_increment

num_obstacles = len(obstacles)
########################################################################

# Sim Parameters  
F = 2
R = 3
d_min = 0.5

#Initialize the robots
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

num_robots = n =len(robots)
F_prime = F + n // 2
num_constraints1  = 1 + num_obstacles
alphas = 0.1
umax = 1.5
tau = 1
T = 10
dt =0.002
max_T = T/dt
step_size = tau/dt

############################## Optimization problems ######################################
u1 = cp.Variable((2,1))
u1_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
A1 = cp.Parameter((num_constraints1,2),value=np.zeros((num_constraints1,2)))
b1 = cp.Parameter((num_constraints1,1),value=np.zeros((num_constraints1,1)))
const1 = [A1 @ u1 >= b1]
const1+= [u1<=umax, -umax<=u1]

objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref  ))
cbf_controller = cp.Problem( objective1, const1 )
###########################################################################################


#Define the parametrized sigmoid functions
eps = 1/(n-1)- 0.001
q1 = 2+eps
q2 = 0.3
sigmoid_A1 = lambda x: q1 / (1+np.exp(-q2*x**2)) - q1/2
def compute_der(x_i, x_j):
    dist = (R**2 - np.sum((x_i-x_j)**2))**2
    exp_term = 2 * q1 * q2* np.exp(-q2 *dist)
    denominator = (1 + exp_term)**2
    coefficient = exp_term / denominator
    return - coefficient * ((x_i - x_j))*2*(R**2 - np.sum((x_i-x_j)**2))

#Setting up the goal
goal = np.array([[0,100],[100,0]])
# goal = np.array([0,10]) 


# import timeit as time
counter = 0
H =[[] for i in range(n)]
robustness_history = []
while True:   
    x = np.array([aa.location.reshape(1,-1)[0] for aa in robots])
    #Compute the actual robustness
    edges = []
    for i in range(n):
        for j in range(i+1,n):
            if np.linalg.norm(x[i]-x[j]) <=R:
                edges.append((i,j))

    #Agents form a network
    for (i,j) in edges:
        robots[i].connect(robots[j])
        robots[j].connect(robots[i])

    #Get the nominal control input
    u_des = []
    for i in range(1,n+1):
        helper = (-1)**i * np.array([100, 0])
        if i in range(6, 12):
            helper= (-1)**i * np.array([100, 0]) 
        vector = (helper - x[i-1]).reshape(-1,1)
        u_des.append(vector/np.linalg.norm(vector)) 

    #Compute the h_i and \partial h_i
    h = np.zeros(n)
    der_ = np.zeros((n,n,2))

    for (i,j) in edges:
        dist = R**2-np.sum((x[i]-x[j])**2)
        h[i]+= sigmoid_A1(dist)
        h[j]+= sigmoid_A1(dist)
        der_[i,i]+=compute_der(x[i],x[j])
        der_[i,j]= -compute_der(x[i],x[j])
        der_[j,j]+=compute_der(x[j],x[i])
        der_[j,i]= -compute_der(x[j],x[i]) 
    h = h-F_prime
    for i in range(n):
        H[i].append(h[i])
    #Set up the constraint of QP
    A1.value[:,:]=0
    b1.value[:,:]=0
    control_input = []

    w = [15, 25]
    h_hat = h
    if np.any(h_hat<0):
        print(counter, h_hat)
    for i in range(num_robots):
        u1_ref.value = u_des[i]
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
        cbf_controller.solve(solver="GUROBI")
        if cbf_controller.status!='optimal':
            print("Error: should not have been infeasible here")
            print(h)
            control_input.append(np.array([[0],[0]]))
        else:
            control_input.append(u1.value)

    if counter >50 and counter% step_size ==0:
        #Agents share their values with neighbors
        for aa in robots:
            aa.propagate()
        # The agents perform W-MSR
        for aa in robots:
            aa.w_msr()
        # All the agents update their LED colors
        for aa in robots:
            aa.set_color()

    # implement control input \mathbf u and plot the trajectory
    for i in range(n):
        robots[i].step(control_input[i], dt) 
        robots[i].reset_neighbors()
        # if counter>0:
        #     plt.plot(robots[i].locations[0][counter-1:counter+1], robots[i].locations[1][counter-1:counter+1], color = robots[i].LED, zorder=0)   
    
    #Plots the environment and robots
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

    #If time, terminate
    counter+=1

    if counter>=max_T:
        break


plt.ioff()
fig2 = plt.figure()

#Plot the evolutions of h_{i}'s values
for i in range(n):
    if i <=1:
        plt.plot(np.arange(counter), H[i],linestyle='dashdot', label="$h_{" +  str(i)+ '}$')
    else:
        plt.plot(np.arange(counter), H[i], label="$h_{" +  str(i)+ '}$')
plt.plot(np.arange(counter), [0]*counter, 'k--',label="threshold", )
plt.title("Evolution of $h_i$")
plt.yticks(np.arange(-1, 4, 1.0))
plt.show()


#Plot the evolutions of consensus values
length_of_consensus = len(robots[0].history)*step_size
for aa in robots:
    temp = np.repeat(np.array(aa.history),step_size)
    if issubclass(type(aa), Malicious):
        plt.plot(np.arange(0,length_of_consensus)*dt,temp, "r--")
    else:
        plt.plot(np.arange(0,length_of_consensus)*dt, temp)
plt.show()