import math
import numpy as np
from ...base import types
from .metrics import calculate_leaf_occurences


class TreeTarget(types.DiscreteVariableTarget):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def eval(self, x):
        # call test tree to calculate Π(Y_i|T,theta,x_i)
        target1, leafs_possibilities_for_prediction = calculate_leaf_occurences(x)
        # call test tree to calculate  (theta|T)
        target2 = self.features_and_threshold_probabilities(x)
        # p(T)
        target3 = self.evaluatePrior(x)
        return (target1+target2+target3)

    # (theta|T)
    def features_and_threshold_probabilities(self, x):
        # this need to change
        logprobabilities = []

        '''
            the likelihood of choosing the specific feature and threshold must
            be computed. We need to find out the probabilty of selecting the
            specific feature multiplied by 1/the margins. it should be
            (1/number of features) * (1/(upper bound-lower bound))
        '''
        for node in x.tree:
            logprobabilities.append(-math.log(len(x.X_train[0]))
            - math.log(max(x.X_train[:, node[3]]) - min(x.X_train[:, node[3]]))
               
            )

            # probabilities.append(math.log( 1/len(X_train[0]) *
            # 1/len(X_train[:]) ))
            
            #-math.log(len(x.X_train[0]))
            #- math.log(max(x.X_train[:, node[3]]) - min(x.X_train[:, node[3]]))

        logprobability = np.sum(logprobabilities)
        return (logprobability)
    
    
    def evaluatePrior(self, x):#poisson
        #print(len(x.tree))
        lam = self.a
        k = len(x.leafs)
        logprior = math.log(math.pow(lam, k) / ((math.exp(lam)-1) * math.factorial(k)))

        #print(math.exp(logprior))
        return logprior
    
    
    # def evaluatePrior(self, x):#chipman prior
    #     def p_node(a, b, d):
    #         return math.log(a / math.pow(1 + d, b))
        
    #     def p_leaf(a, b, d):
    #         return math.log(1 - math.exp(p_node(a, b, d)))
        
    #     logprior = 0
    #     for node in x.tree:
    #         logprior += p_node(self.a, self.b, node[5])
        
    #     for leaf in x.leafs:
    #         d = x.depth_of_leaf(leaf)
    #         logprior += p_leaf(self.a, self.b, d)
    #     return logprior
        
    
    
    













