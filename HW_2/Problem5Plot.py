import random
import numpy as np
import matplotlib.pyplot as plt
from Problem5 import bgd_l2, sgd_l2

if __name__ == '__main__':
    # Put the code for the plots here, you can use different functions for each
    # part

####### GRADIENT DESCENT TESTS


    w = np.random.random(2)
    data = np.load("/Users/aidanvesci/Downloads/data.npy") 
    x = np.hsplit(data, 2)
    y = x[1]
    x = x[0]

    
    new_w, history_fw = bgd_l2(x, y, w, .05, .1, .001, 50)
    #print(history_fw)
    plt.plot(history_fw)
    plt.title("History of Objective Function with Gradient Descent #1")
    plt.xlabel("# of Iterations")
    plt.ylabel("Objective Function")
    plt.show()

    
    new_w, history_fw = bgd_l2(x, y, w, .1, .01, .001, 50)
    plt.plot(history_fw)
    plt.title("History of Objective Function with Gradient Descent #2")
    plt.xlabel("# of Iterations")
    plt.ylabel("Objective Function")
    plt.show()


    new_w, history_fw = bgd_l2(x, y, w, .1, 0, .001, 100)
    plt.plot(history_fw)
    plt.title("History of Objective Function with Gradient Descent #3")
    plt.xlabel("# of Iterations")
    plt.ylabel("Objective Function")
    plt.show()

    
    new_w, history_fw = bgd_l2(x, y, w, .1, 0, 0, 100)
    plt.plot(history_fw)
    plt.title("History of Objective Function with Gradient Descent #4")
    plt.xlabel("# of Iterations")
    plt.ylabel("Objective Function")
    plt.show()

    ########### STOCHASTIC GRADIENT DESCENT TESTS

    
    new_w, history_fw = sgd_l2(x, y, w, 1, .1, .5, 800)
    plt.plot(history_fw)
    plt.title("History of Objective Function with Stochastic Gradient Descent #1")
    plt.xlabel("# of Iterations")
    plt.ylabel("Objective Function")
    plt.show()

    
    new_w, history_fw = sgd_l2(x, y, w, 1, .01, .1, 800)
    plt.plot(history_fw)
    plt.title("History of Objective Function with Stochastic Gradient Descent #2")
    plt.xlabel("# of Iterations")
    plt.ylabel("Objective Function")
    plt.show()

    
    new_w, history_fw = sgd_l2(x, y, w, 1, 0, 0, 40)
    plt.plot(history_fw)
    plt.title("History of Objective Function with Stochastic Gradient Descent #3")
    plt.xlabel("# of Iterations")
    plt.ylabel("Objective Function")
    plt.show()

    
    new_w, history_fw = sgd_l2(x, y, w, 1, 0, 0, 800)
    plt.plot(history_fw)
    plt.title("History of Objective Function with Stochastic Gradient Descent #4")
    plt.xlabel("# of Iterations")
    plt.ylabel("Objective Function")
    plt.show()