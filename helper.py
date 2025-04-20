import numpy as np

# Allows to choose which scenario to simulate among three scenarios
def choose():
    scenario = input("Please press the number:\n1:nominal update\n2:overstatement \n3:understatement\n")
    while True:
        try:
            scenario = int(scenario)
            if 0<scenario<4:
                match scenario:
                    case 1:
                        print("Simulating nominal update scenario...")
                    case 2:
                        print("Simulating overstatement scenario...")
                    case 3:
                        print("Simulating understatement scenario...")       
                return scenario
            else:
                scenario = input("Wrong input, please press the number:\n1:nominal update\n2:overstatement \n3:understatement\n")
        except:
            scenario = input("Wrong input, please press the number:\n1:nominal update\n2:overstatement \n3:understatement\n")


# Compute the h_i and \frac{\partial h_i/} {\partial x}
def compute_h_and_der(n,x,edges,R):

    #Define the parametrized sigmoid function and its derivative
    eps = 1/(n-1)- 0.001
    q1 = 2+eps
    q2 = 0.9 
    sigmoid_A1 = lambda x: q1 / (1+np.exp(-q2*x)) - q1/2
    def compute_der(x_i, x_j):
        dist = (R**2 - np.sum((x_i-x_j)**2))**2
        exp_term = 2 * q1 * q2* np.exp(-q2 *dist)
        denominator = (1 + exp_term)**2
        coefficient = exp_term / denominator
        return - coefficient * ((x_i - x_j))*2*(R**2 - np.sum((x_i-x_j)**2))
    
    # Compute h and \frac{\partial h_i/} {\partial x} 
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
    return h, der_


    