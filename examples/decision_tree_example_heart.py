# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:28:12 2021
@author: efthi
"""
import sys
sys.path.append('../')  # Looks like mpiexec won't find discretesampling package without appending '../'
from discretesampling.domain import decision_tree as dt
from discretesampling.base.algorithms import DiscreteVariableSMC
from discretesampling.base.algorithms import DiscreteVariableMCMC
from discretesampling.base.algorithms.decision_forest import decision_forest

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
import statistics

import numpy as np
import pandas as pd
from mpi4py import MPI

df = pd.read_csv(r"datasets_smc_mcmc_CART/heart.csv")
y = df.Target
X = df.drop(['Target'], axis=1)
X = X.to_numpy()
y = y.to_numpy()

N = 1 << 10
T = 10
num_MC_runs = 10

MCMC_one_to_many = False
MCMC_many_to_one = False
SMC = True
DF = False

if MCMC_one_to_many:
    try:
        runtimes = []
        accuracies = []
        for random_state in range(num_MC_runs):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=random_state)

            a = 12
            b = 3
            target = dt.TreeTarget(a, b)
            initialProposal = dt.TreeInitialProposal(X_train, y_train)
            dtMCMC = DiscreteVariableMCMC(dt.Tree, target, initialProposal)
        
            MPI.COMM_WORLD.Barrier()
            start = MPI.Wtime()
            P = MPI.COMM_WORLD.Get_size()
            num_samples = int(N*T / P)
        
            rank = MPI.COMM_WORLD.Get_rank()
            treeSamples = dtMCMC.sample(num_samples, seed=num_samples*rank)
        
            mcmcLabels = [dt.stats(x, X_test).predict(X_test) for x in treeSamples]
            mcmcAccuracy = [dt.accuracy(y_test, x) for x in mcmcLabels]
        
            accuracy = np.zeros(1, 'd')
            MPI.COMM_WORLD.Allreduce(sendbuf=[np.sum(mcmcAccuracy), MPI.DOUBLE], recvbuf=[accuracy, MPI.DOUBLE], op=MPI.SUM)
            accuracies.append(accuracy/(N*T))
        
            MPI.COMM_WORLD.Barrier()
            end = MPI.Wtime()
            
            if MPI.COMM_WORLD.Get_rank() == 0:
                runtimes.append(end-start)
                print("MCMC mean accuracy: ", accuracies[-1])
                print("MCMC run-time: ", runtimes[-1])
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("MCMC mean of mean accuracies: ", np.mean(accuracies))
            print("MCMC median runtime: ", np.median(runtimes))
            print("MCMC standard deviation of accuracies: ", np.std(accuracies))
            print("MCMC standard deviation of runtimes: ", np.std(runtimes))
    except ZeroDivisionError:
        print("MCMC sampling failed due to division by zero")

if MCMC_many_to_one:
    try:
        runtimes = []
        accuracies = []
        for random_state in range(num_MC_runs):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=random_state)

            a = 12
            b = 5
            target = dt.TreeTarget(a, b)
            initialProposal = dt.TreeInitialProposal(X_train, y_train)
            dtMCMC = DiscreteVariableMCMC(dt.Tree, target, initialProposal)

            MPI.COMM_WORLD.Barrier()
            start = MPI.Wtime()
            P = MPI.COMM_WORLD.Get_size()
            num_samples = int(N / P)

            rank = MPI.COMM_WORLD.Get_rank()
            mcmcAccuracy = []
            for i in range(num_samples):
                treeSamples = dtMCMC.sample(T, seed=T*(num_samples * rank + i))
                mcmcLabels = [dt.stats(x, X_test).predict(X_test) for x in treeSamples]
                mcmcAccuracy += [dt.accuracy(y_test, x) for x in mcmcLabels]

            accuracy = np.zeros(1, 'd')
            MPI.COMM_WORLD.Allreduce(sendbuf=[np.sum(mcmcAccuracy), MPI.DOUBLE], recvbuf=[accuracy, MPI.DOUBLE],
                                     op=MPI.SUM)
            accuracies.append(accuracy / (N * T))

            MPI.COMM_WORLD.Barrier()
            end = MPI.Wtime()

            if MPI.COMM_WORLD.Get_rank() == 0:
                runtimes.append(end - start)
                print("MCM many to one mean accuracy: ", accuracies[-1])
                print("MCMC many to one run-time: ", runtimes[-1])
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("MCMC many to one mean of mean accuracies: ", np.mean(accuracies))
            print("MCMC many to one median runtime: ", np.median(runtimes))
            print("MCMC many to one standard deviation of accuracies: ", np.std(accuracies))
            print("MCMC many to one standard deviation of runtimes: ", np.std(runtimes))
    except ZeroDivisionError:
        print("MCMC sampling failed due to division by zero")

if SMC:
    try:
        runtimes = []
        tree_size = []
        accuracies = []
        for random_state in range(num_MC_runs):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=random_state)

            a = 12
            b = 2
            target = dt.TreeTarget(a, b)
            initialProposal = dt.TreeInitialProposal(X_train, y_train)
            
            dtSMC = DiscreteVariableSMC(dt.Tree, target, initialProposal)

            MPI.COMM_WORLD.Barrier()
            start = MPI.Wtime()

            treeSMCSamples = dtSMC.sample(T, N)
 
        
            smcLabels = [dt.stats(x, X_test).predict(X_test) for x in treeSMCSamples]
            smcAccuracy = [dt.accuracy(y_test, x) for x in smcLabels]



            accuracy = np.zeros(1, 'd')
            MPI.COMM_WORLD.Allreduce(sendbuf=[np.sum(smcAccuracy), MPI.DOUBLE], recvbuf=[accuracy, MPI.DOUBLE], op=MPI.SUM)
            accuracies.append(accuracy / N)

            MPI.COMM_WORLD.Barrier()
            end = MPI.Wtime()

            if MPI.COMM_WORLD.Get_rank() == 0:
                runtimes.append(end - start)
                print("SMC mean accuracy: ", accuracies[-1])
                print("SMC run-time: ", runtimes[-1])
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("SMC mean of mean accuracies: ", np.mean(accuracies))
            print("SMC median runtime: ", np.median(runtimes))
            print("MCMC standard deviation of accuracies: ", np.std(accuracies))
            print("MCMC standard deviation of runtimes: ", np.std(runtimes))
    except ZeroDivisionError:
        print("SMC sampling failed due to division by zero")

if DF:
    try:
        runtimes = []
        accuracies = []
        for random_state in range(num_MC_runs):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)

            MPI.COMM_WORLD.Barrier()
            start = MPI.Wtime()
            P = MPI.COMM_WORLD.Get_size()

            fitting = decision_forest(X_train, y_train, num_trees=int(N * T / P))

            # Predict the response for test dataset
            dfLabels = [tree.predict(X_test) for tree in fitting]
            dfAccuracy = np.array([metrics.accuracy_score(y_test, x) for x in dfLabels])

            accuracy = np.zeros(1, dtype=dfAccuracy.dtype)
            MPI.COMM_WORLD.Allreduce(sendbuf=[np.sum(dfAccuracy), MPI.DOUBLE], recvbuf=[accuracy, MPI.DOUBLE], op=MPI.SUM)
            accuracies.append(accuracy / (N*T))

            MPI.COMM_WORLD.Barrier()
            end = MPI.Wtime()

            if MPI.COMM_WORLD.Get_rank() == 0:
                runtimes.append(end - start)
                print("DF mean accuracy: ", accuracies[-1])
                print("DF run-time: ", runtimes[-1])
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("DF mean of mean accuracies: ", np.mean(accuracies))
            print("DF median runtime: ", np.median(runtimes))
    except ZeroDivisionError:
        print("Decision Forest failed due to division by zero")
