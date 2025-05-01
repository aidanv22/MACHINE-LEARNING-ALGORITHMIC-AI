import math
import random
import numpy as np


def bgd_l2(data, y, w, eta, delta, lam, num_iter):
    ones = np.full((100, 1), 1)
    x_val = np.concatenate((ones, data), axis=1)
    new_w = w
    history_fw = []

    for i in range(num_iter):
        wt = np.transpose(new_w)
        
        #calulate gradient
        gradient = 0
        for t in range(len(x_val)):
            if (y[t] >= (np.dot(wt, x_val[t]) + delta)):
                gradient += -2*(y[t] - np.dot(wt, x_val[t]) - delta) * x_val[t]
            elif (abs(y[t] - np.dot(wt, x_val[t])) < delta):
                gradient += 0
            elif (y[t] <= (np.dot(wt, x_val[t]) - delta)):
                gradient += -2*(y[t] - np.dot(wt, x_val[t]) + delta) * x_val[t]
        gradient = gradient/len(x_val)
        gradient += 2*lam*sum(wt)

        new_w = new_w - (eta * gradient)
        wt = np.transpose(new_w)

        #calulate new f(w)
        fw = 0
        for t in range(len(x_val)):
            if (y[t] >= (np.dot(wt, x_val[t]) + delta)):
                fw += ((y[t] - np.dot(wt, x_val[t]) - delta) ** 2)
            elif (abs(y[t] - np.dot(wt, x_val[t])) < delta):
                fw += 0
            elif (y[t] <= (np.dot(wt, x_val[t]) - delta)):
                fw += ((y[t] - np.dot(wt, x_val[t]) + delta) ** 2)
        fw = fw/len(x_val)
        fw += lam*sum(wt**2)
        history_fw.append(fw)
    return new_w, history_fw


def sgd_l2(data, y, w, eta, delta, lam, num_iter, i=-1):

    ones = np.full((100, 1), 1)
    x_val = np.concatenate((ones, data), axis=1)
    new_w = w
    history_fw = []

    if (i != -1):
        numer_iter = 1
    else:
        i = random.randrange(0, len(x_val))

    for j in range(1, num_iter+1):
        wt = np.transpose(new_w)
        
        #calulate gradient
        grad = 0
        for t in range(len(x_val)):
            if (y[i] >= (np.dot(wt, x_val[i]) + delta)):
                grad += -2*(y[i] - np.dot(wt, x_val[i]) - delta) * x_val[i]
            elif (abs(y[t] - np.dot(wt, x_val[i])) < delta):
                grad += 0
            elif (y[i] <= (np.dot(wt, x_val[i]) - delta)):
                grad += -2*(y[i] - np.dot(wt, x_val[i]) + delta) * x_val[i]
        grad = grad/len(x_val)
        grad += 2*lam*sum(wt)

        new_w = new_w - ((eta / math.sqrt(j))  * grad)
        wt = np.transpose(new_w)

        #calulate new f(w)
        fw = 0
        for t in range(len(x_val)):
            if (y[t] >= (np.dot(wt, x_val[t]) + delta)):
                fw += ((y[t] - np.dot(wt, x_val[t]) - delta) ** 2)
            elif (abs(y[t] - np.dot(wt, x_val[t])) < delta):
                fw += 0
            elif (y[t] <= (np.dot(wt, x_val[t]) - delta)):
                fw += ((y[t] - np.dot(wt, x_val[t]) + delta) ** 2)
        fw = fw/len(x_val)
        fw += lam*sum(wt**2)
        history_fw.append(fw)

        i = random.randrange(0, len(x_val))

    return new_w, history_fw

 
