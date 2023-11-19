import numpy as np
import matplotlib.pyplot as plt
import get_function_details


def Func_Plot(func_name, Convergence_curve, graphing, x_train, y_train, x_test, y_test):
    lb, ub, dim, fobj = get_function_details.Get_Function_Details(func_name)
    """
    if func_name == 'F1':
        x = np.arange(-100, 101, 2)
        y = x
    elif func_name == 'F2':
        x = np.arange(-10, 11, 2)
        y = x
    elif func_name == 'F3':
        x = np.arange(-100, 101, 2)
        y = x
    elif func_name == 'F4':
        x = np.arange(-100, 101, 2)
        y = x
    elif func_name == 'F5':
        x = np.arange(-200, 201, 2)
        y = x
    elif func_name == 'F6':
        x = np.arange(-100, 101, 2)
        y = x
    elif func_name == 'F7':
        x = np.arange(-1, 1.01, 0.03)
        y = x
    elif func_name == 'F8':
        x = np.arange(-500, 501, 10)
        y = x
    elif func_name == 'F9':
        x = np.arange(-5, 5.1, 0.1)
        y = x
    elif func_name == 'F10':
        x = np.arange(-20, 20.5, 0.5)
        y = x
    elif func_name == 'F11':
        x = np.arange(-500, 501, 10)
        y = x
    elif func_name == 'F12':
        x = np.arange(-10, 10.1, 0.1)
        y = x
    elif func_name == 'F13':
        x = np.arange(-5, 5.08, 0.08)
        y = x
    elif func_name == 'F14':
        x = np.arange(-100, 101, 2)
        y = x
    elif func_name == 'F15':
        x = np.arange(-5, 5.1, 0.1)
        y = x
    elif func_name == 'F16':
        x = np.arange(-1, 1.01, 0.01)
        y = x
    elif func_name == 'F17':
        x = np.arange(-5, 5.1, 0.1)
        y = x
    elif func_name == 'F18':
        x = np.arange(-5, 5.06, 0.06)
        y = x
    elif func_name == 'F19':
        x = np.arange(-5, 5.1, 0.1)
        y = x
    elif func_name == 'F20':
        x = np.arange(-5, 5.1, 0.1)
        y = x
    elif func_name == 'F21':
        x = np.arange(-5, 5.1, 0.1)
        y = x
    elif func_name == 'F22':
        x = np.arange(-5, 5.1, 0.1)
        y = x
    elif func_name == 'F23':
        x = np.arange(-5, 5.1, 0.1)
        y = x
    elif func_name == 'F24':
        x = np.arange(300, 500, 5)
        y = np.arange(1, 187, 6)
    L = len(x)
    Ly = len(y)
    f = np.zeros((L, Ly))

    for i in range(L):
        for j in range(Ly):
            if func_name not in ['F15', 'F19', 'F20', 'F21', 'F22', 'F23', 'F24']:
                f[i, j] = fobj(np.array([x[i], y[j]]), x_train, y_train, x_test, y_test)
            if func_name == 'F15':
                f[i, j] = fobj([x[i], y[j], 0, 0], x_train, y_train, x_test, y_test)
            if func_name == 'F19':
                f[i, j] = fobj([x[i], y[j], 0], x_train, y_train, x_test, y_test)
            if func_name == 'F20':
                f[i, j] = fobj([x[i], y[j], 0, 0, 0, 0], x_train, y_train, x_test, y_test)
            if func_name in ['F21', 'F22', 'F23']:
                f[i, j] = fobj([x[i], y[j], 0, 0], x_train, y_train, x_test, y_test)
            if func_name in ['F24']:
                f[i, j] = fobj([x[i], 1, y[j]], x_train, y_train, x_test, y_test)
    """

    
    fig = plt.figure(figsize=(12, 5))
    plt.plot(Convergence_curve, color='r')
    plt.title(func_name)
    plt.xlabel('Iteration')
    plt.ylabel('Best score obtained so far')
    plt.show()