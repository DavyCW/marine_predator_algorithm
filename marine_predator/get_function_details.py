import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import math
import pandas as pd
import time

# Define the Get_Function_Details function
def Get_Function_Details(F):
    if F == 'F1':
        lb = -100
        ub = 100
        dim = 50
        fobj = F1
    elif F == 'F2':
        lb = -10
        ub = 10
        dim = 50
        fobj = F2
    elif F == 'F3':
        lb = -100
        ub = 100
        dim = 50
        fobj = F3
    elif F == 'F4':
        lb = -100
        ub = 100
        dim = 50
        fobj = F4
    elif F == 'F5':
        lb = -30
        ub = 30
        dim = 50
        fobj = F5
    elif F == 'F6':
        lb = -100
        ub = 100
        dim = 50
        fobj = F6
    elif F == 'F7':
        lb = -1.28
        ub = 1.28
        dim = 50
        fobj = F7
    elif F == 'F8':
        lb = -500
        ub = 500
        dim = 50
        fobj = F8
    elif F == 'F9':
        lb = -5.12
        ub = 5.12
        dim = 50
        fobj = F9
    elif F == 'F10':
        lb = -32
        ub = 32
        dim = 50
        fobj = F10
    elif F == 'F11':
        lb = -600
        ub = 600
        dim = 50
        fobj = F11
    elif F == 'F12':
        lb = -50
        ub = 50
        dim = 50
        fobj = F12
    elif F == 'F13':
        lb = -50
        ub = 50
        dim = 50
        fobj = F13
    elif F == 'F14':
        lb = np.array([-65.536, -65.536])
        ub = np.array([65.536, 65.536])
        dim = 2
        fobj = F14
    elif F == 'F15':
        lb = np.array([-5, -5, -5, -5])
        ub = np.array([5, 5, 5, 5])
        dim = 4
        fobj = F15
    elif F == 'F16':
        lb = np.array([-5, -5])
        ub = np.array([5, 5])
        dim = 2
        fobj = F16
    elif F == 'F17':
        lb = np.array([-5, 0])
        ub = np.array([10, 15])
        dim = 2
        fobj = F17
    elif F == 'F18':
        lb = np.array([-2, -2])
        ub = np.array([2, 2])
        dim = 2
        fobj = F18
    elif F == 'F19':
        lb = np.array([0, 0, 0])
        ub = np.array([1, 1, 1])
        dim = 3
        fobj = F19
    elif F == 'F20':
        lb = np.array([0, 0, 0, 0, 0, 0])
        ub = np.array([1, 1, 1, 1, 1, 1])
        dim = 6
        fobj = F20
    elif F == 'F21':
        lb = np.array([0, 0, 0, 0])
        ub = np.array([10, 10, 10, 10])
        dim = 4
        fobj = F21
    elif F == 'F22':
        lb = np.array([0, 0, 0, 0])
        ub = np.array([10, 10, 10, 10])
        dim = 4
        fobj = F22
    elif F == 'F23':
        lb = np.array([0, 0, 0, 0])
        ub = np.array([10, 10, 10, 10])
        dim = 4
        fobj = F23
    elif F =='F24':
        lb = np.array([100, 1, 1])
        ub = np.array([500, 5, 188])
        dim = 3
        fobj = F24
    elif F =='F25':
        lb = np.array([100, 1, 1])
        ub = np.array([500, 5, 188])
        dim = 3
        fobj = F25
    elif F =='F26':
        lb = np.array([.000001, 2, 2, 2, .00001, 0.5001, 0.5001])
        ub = np.array([.001, 20, 20, 20, .01, .9999, .9999])
        dim = 7
        fobj = F26
    else:
        raise ValueError("Invalid function name")

    return lb, ub, dim, fobj

# You can define the benchmark functions here (similar to the previous response)

# F1
def F1(x, xtr, ytr, xt, yt):
    return x, np.sum(x**2)

# F2
def F2(x, xtr, ytr, xt, yt):
    return x, np.sum(np.abs(x)) + np.prod(np.abs(x))

# F3
def F3(x, xtr, ytr, xt, yt):
    dim = len(x)
    return x, np.sum(np.cumsum(x)**2)

# F4
def F4(x, xtr, ytr, xt, yt):
    return x, np.max(np.abs(x))

# F5
def F5(x, xtr, ytr, xt, yt):
    dim = len(x)
    return x, np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

# F6
def F6(x, xtr, ytr, xt, yt):
    return x, np.sum(np.abs(x + 0.5)**2)

# F7
def F7(x, xtr, ytr, xt, yt):
    dim = len(x)
    return x, np.sum(np.arange(1, dim + 1) * x**4) + np.random.rand()

# F8
def F8(x, xtr, ytr, xt, yt):
    return x, np.sum(-x * np.sin(np.sqrt(np.abs(x))))

# F9
def F9(x, xtr, ytr, xt, yt):
    dim = len(x)
    return x, np.sum(x**2 - 10 * np.cos(2 * np.pi * x)) + 10 * dim

# F10
def F10(x, xtr, ytr, xt, yt):
    dim = len(x)
    return x, -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / dim) + 20 + np.exp(1)

# F11
def F11(x, xtr, ytr, xt, yt):
    dim = len(x)
    return x, np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, dim + 1)))) + 1

# F12
def F12(x, xtr, ytr, xt, yt):
    dim = len(x)
    return x, (np.pi / dim) * (10 * ((np.sin(np.pi * (1 + (x[0] + 1) / 4)))**2) + np.sum((((x[:-1] + 1) / 4)**2) * (1 + 10 * ((np.sin(np.pi * (1 + (x[1:] + 1) / 4))))**2)) + ((x[-1] + 1) / 4)**2) + np.sum(Ufun(x, 10, 100, 4))

# F13
def F13(x, xtr, ytr, xt, yt):
    dim = len(x)
    return x, 0.1 * ((np.sin(3 * np.pi * x[0]))**2 + np.sum((x[:-1] - 1)**2 * (1 + (np.sin(3 * np.pi * x[1:]))**2)) + ((x[-1] - 1)**2) * (1 + (np.sin(2 * np.pi * x[-1]))**2)) + np.sum(Ufun(x, 5, 100, 4))

# F14
def F14(x, xtr, ytr, xt, yt):
    aS = np.array([[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
                    [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]])
    bS = np.zeros(25)
    for j in range(25):
        bS[j] = np.sum((x - aS[:, j])**6)
    return x, (1 / 500 + np.sum(1 / (np.arange(1, 26) + bS)))**(-1)

# F15
def F15(x, xtr, ytr, xt, yt):
    aK = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    bK = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    bK = 1 / bK
    return x, np.sum((aK - ((x[0] * (bK**2 + x[1] * bK)) / (bK**2 + x[2] * bK + x[3])))**2)

# F16
def F16(x, xtr, ytr, xt, yt):
    return x, 4 * (x[0]**2) - 2.1 * (x[0]**4) + (x[0]**6) / 3 + x[0] * x[1] - 4 * (x[1]**2) + 4 * (x[1]**4)

# F17
def F17(x, xtr, ytr, xt, yt):
    return x, (x[1] - (x[0]**2) * 5.1 / (4 * (np.pi**2)) + 5 / np.pi * x[0] - 6)**2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10

# F18
def F18(x, xtr, ytr, xt, yt):
    return x, (1 + (x[0] + x[1] + 1)**2 * (19 - 14 * x[0] + 3 * (x[0]**2) - 14 * x[1] + 6 * x[0] * x[1] + 3 * (x[1]**2))) * (30 + (2 * x[0] - 3 * x[1])**2 * (18 - 32 * x[0] + 12 * (x[0]**2) + 48 * x[1] - 36 * x[0] * x[1] + 27 * (x[1]**2)))

# F19
def F19(x, xtr, ytr, xt, yt):
    aH = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])
    o = 0
    for i in range(4):
        o = o - cH[i] * np.exp(-np.sum(aH[i, :] * ((x - pH[i, :])**2)))
    return x, o

# F20
def F20(x, xtr, ytr, xt, yt):
    aH = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886], [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991], [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650], [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    o = 0
    for i in range(4):
        o = o - cH[i] * np.exp(-np.sum(aH[i, :] * ((x - pH[i, :])**2)))
    return x, o

# F21
def F21(x, xtr, ytr, xt, yt):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    o = 0
    for i in range(5):
        o = o - (np.dot((x - aSH[i, :]), (x - aSH[i, :]).T) + cSH[i])**(-1)
    return x, o

# F22
def F22(x, xtr, ytr, xt, yt):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    o = 0
    for i in range(7):
        o = o - (np.dot((x - aSH[i, :]), (x - aSH[i, :]).T) + cSH[i])**(-1)
    return x, o

# F23
def F23(x, xtr, ytr, xt, yt):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    o = 0
    for i in range(10):
        o = o - (np.dot((x - aSH[i, :]), (x - aSH[i, :]).T) + cSH[i])**(-1)
    return x, o

# Additional helper function for F12, F13
def Ufun(x, a, k, m):
    return k * ((x - a)**m) * (x > a) + k * ((-x - a)**m) * (x < -a)

def F24(x, xtr, ytr, xt, yt):
    extra = ExtraTreesClassifier(n_estimators=int(x[0]), max_depth=None, min_samples_leaf=int(x[1]), max_features=int(x[2]), random_state=0, n_jobs=-1)
    my_time = time.time()
    print("\t\tModel Made")
    extra.fit(xtr, ytr)
    print("\t\tFitted: " + str(time.time()-my_time))
    score = extra.score(xt, yt)
    print('\t\t', x, score)
    return x, 1-score


def F25(x, xtr, ytr, xt, yt):
    extra = RandomForestClassifier(n_estimators=int(x[0]), max_depth=None, min_samples_leaf=int(x[1]), max_features=int(x[2]), random_state=0, n_jobs=-1)
    my_time = time.time()
    print("\t\tModel Made")
    extra.fit(xtr, ytr)
    print("\t\tFitted: " + str(time.time()-my_time))
    score = extra.score(xt, yt)
    print('\t\t', x, score)
    return x, 1-score

def F26(x, xtr, ytr, xt, yt):
    mlp = MLPClassifier(alpha=x[0], hidden_layer_sizes=(int(x[1])*5, int(x[2])*5, int(x[3])*5), learning_rate_init=x[4], max_iter=1500, random_state=0, beta_1=x[5], beta_2=x[6])
    my_time = time.time()
    print("\t\tModel Made")
    mlp.fit(xtr, ytr)
    print("\t\tFitted: " + str(time.time()-my_time))
    score = mlp.score(xt, yt)
    print('\t\t', x, score)
    return x, 1-score