# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 22:15:10 2021

@author: lankuohsing
"""
import time
import numpy as np
# In[]
#list_shape=[3,2,4,5]
#list_index=[2,1,3,1]
rank=1000
list_shape=(np.ones((rank))*10).tolist()
list_index=np.random.randint(0,10,rank).tolist()


# In[]
start=time.time()
linear_index=list_index[0]
for i in range(1,len(list_shape)):
    linear_index=linear_index*list_shape[i]+list_index[i]
end=time.time()
print(end-start)
print(linear_index)
# In[]
start=time.time()
linear_index=0
for i in range(0,len(list_shape)):
    tmp=list_index[len(list_shape)-1-i]
    for j in range(0,i):
        tmp=tmp*list_shape[len(list_shape)-1-j]
    linear_index=linear_index+tmp
end=time.time()
print(end-start)
print(linear_index)