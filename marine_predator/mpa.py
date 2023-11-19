import numpy as np
from initialization import initialization
from levy import levy
import pandas as pd
import time


def MPA(SearchAgents_no, Max_iter, lb, ub, dim, fobj, x_train, y_train, x_test, y_test):
    Top_predator_pos = np.zeros(dim)
    Top_predator_fit = float('inf')
    Convergence_curve = np.zeros(Max_iter)
    stepsize = np.zeros((SearchAgents_no, dim))
    fitness = np.full(SearchAgents_no, float('inf'))

    Prey = initialization(SearchAgents_no, dim, ub, lb)
    Xmin = np.tile(np.ones(dim) * lb, (SearchAgents_no, 1))
    Xmax = np.tile(np.ones(dim) * ub, (SearchAgents_no, 1))

    Iter = 0
    FADs = 0.2
    P = 0.5
    graphing = [[], []]

    while Iter < Max_iter:
        # Detecting top predator
        for i in range(Prey.shape[0]):
            Flag4ub = Prey[i, :] > ub
            Flag4lb = Prey[i, :] < lb
            Prey[i, :] = (Prey[i, :] * (~(Flag4ub + Flag4lb))) + ub * Flag4ub + lb * Flag4lb
            x_prey, fitness[i] = fobj(Prey[i, :], x_train, y_train, x_test, y_test)
            fitness[i].flatten()
            graphing.append([x_prey, fitness[i]])
            if fitness[i] < Top_predator_fit:
                Top_predator_fit = fitness[i]
                Top_predator_pos = Prey[i, :]

        # Marine Memory saving
        if Iter == 0:
            fit_old = fitness
            Prey_old = Prey

        Inx = fit_old < fitness
        Indx = np.tile(Inx, (1, dim)).reshape(Prey.shape)
        Prey = Indx * Prey_old + (~Indx) * Prey
        fitness = Inx * fit_old + (~Inx) * fitness
        fit_old = fitness
        Prey_old = Prey

        Elite = np.tile(Top_predator_pos, (SearchAgents_no, 1))
        CF = (1 - Iter / Max_iter) ** (2 * Iter / Max_iter)

        RL = 0.05 * levy(SearchAgents_no, dim, 1.5)
        RB = np.random.randn(SearchAgents_no, dim)

        for i in range(Prey.shape[0]):
            for j in range(dim):
                R = np.random.rand()

                # Phase 1
                if Iter < Max_iter / 3:
                    stepsize[i, j] = RB[i, j] * (Elite[i, j] - RB[i, j] * Prey[i, j])
                    Prey[i, j] = Prey[i, j] + P * R * stepsize[i, j]

                # Phase 2
                elif Iter > Max_iter / 3 and Iter < 2 * Max_iter / 3:
                    if i > Prey.shape[0] / 2:
                        stepsize[i, j] = RB[i, j] * (RB[i, j] * Elite[i, j] - Prey[i, j])
                        Prey[i, j] = Elite[i, j] + P * CF * stepsize[i, j]
                    else:
                        stepsize[i, j] = RL[i, j] * (Elite[i, j] - RL[i, j] * Prey[i, j])
                        Prey[i, j] = Prey[i, j] + P * R * stepsize[i, j]

                # Phase 3
                else:
                    stepsize[i, j] = RL[i, j] * (RL[i, j] * Elite[i, j] - Prey[i, j])
                    Prey[i, j] = Elite[i, j] + P * CF * stepsize[i, j]

        # Detecting top predator
        for i in range(Prey.shape[0]):
            Flag4ub = Prey[i, :] > ub
            Flag4lb = Prey[i, :] < lb
            Prey[i, :] = (Prey[i, :] * (~(Flag4ub + Flag4lb))) + ub * Flag4ub + lb * Flag4lb
            x_prey, fitness[i] = fobj(Prey[i, :], x_train, y_train, x_test, y_test)
            graphing.append([x_prey, fitness[i]])
            if fitness[i] < Top_predator_fit:
                Top_predator_fit = fitness[i]
                Top_predator_pos = Prey[i, :]

        # Marine Memory saving
        if Iter == 0:
            fit_old = fitness
            Prey_old = Prey

        Inx = fit_old < fitness
        Indx = np.tile(Inx, (1, dim)).reshape(Prey.shape)
        Prey = Indx * Prey_old + (~Indx) * Prey
        fitness = Inx * fit_old + (~Inx) * fitness
        fit_old = fitness
        Prey_old = Prey

        # Eddy formation and FADs' effect
        if np.random.rand() < FADs:
            U = np.random.rand(SearchAgents_no, dim) < FADs
            Prey = Prey + CF * ((Xmin + np.random.rand(SearchAgents_no, dim) * (Xmax - Xmin)) * U)
        else:
            r = np.random.rand()
            Rs = Prey.shape[0]
            stepsize = (FADs * (1 - r) + r) * (Prey[np.random.permutation(Rs), :] - Prey[np.random.permutation(Rs), :])

        Prey = Prey + stepsize
        Iter = Iter + 1
        Convergence_curve[Iter - 1] = Top_predator_fit

    return Top_predator_fit, Top_predator_pos, Convergence_curve, graphing
