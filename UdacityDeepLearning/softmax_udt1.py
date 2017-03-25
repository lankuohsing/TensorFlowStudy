# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:33:32 2017
softmax
@author: lankuohsing
"""
scores=[3.0, 1.0, 0.2]
import numpy as np

def softmax(x):
    """ compute softmax values for each sets of scores in x."""
    pass #TODO: Compute and return softmax(x)
    return np.exp(x)/np.sum(np.exp(x), axis=0)
    
print(softmax(scores))


#plot softmax curves
import matplotlib.pyplot as plt
x=np.arange(-2.0, 6.0, 0.1)
print(x.shape)
scores=np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)] )

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()