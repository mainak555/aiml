#%%
import numpy as np
class Activation:
    def __init__(self):
        self.function = {
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'tanh': lambda x: np.tanh(x),
            'arctan': lambda x: np.arctan(x),
            'softmax': lambda x: (lambda exps= np.exp(x): exps / float(sum(exps)))(),
            'softplus': lambda x: np.log(1 + np.exp(x)),
            'swish': lambda x: x / (1 + np.exp(-x)),
            'identity': lambda x: x,
            'relu': lambda x: [item if item >= 0 else 0 for item in x],
            'prelu': lambda x, alpha: [item if item >= 0 else alpha * item for item in x],
            'elu': lambda x, alpha: [item if item >= 0 else np.dot(alpha, np.exp(item) - 1) for item in x], #Leaky Relu
            'arctanh': lambda x: np.arctanh(x) 
            }            
        
        self.derivative = {
            'sigmoid': lambda x: (lambda fx= self.function['sigmoid'](x): fx * (1 - fx))(),
            'tanh': lambda x: (lambda fx = self.function['tanh'](x): 1 - fx**2)(),
            'arctan': lambda x: 1 / (1 + x**2),
            'softmax': lambda x: (lambda fx= self.function['softmax'](x): fx * (1 - fx))(),
            'softplus': lambda x: self.function['sigmoid'](x),
            'swish': lambda x: (lambda x= x, fx= x * self.function['sigmoid'](x): fx + (1 - fx) * self.function['sigmoid'](x))(),
            'identity': lambda x: np.ones(x.shape[0]),
            'relu': lambda x: [1 if item >= 0 else 0 for item in x],
            'prelu': lambda x, alpha: [1 if item >= 0 else alpha for item in x],
            'elu': lambda x, alpha: [1 if item >= 0 else sum(self.function['elu']([item], alpha), alpha) for item in x],
            'arctanh': lambda x: 1 / (1- x**2)
        }