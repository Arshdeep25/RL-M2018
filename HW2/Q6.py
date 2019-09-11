#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np


# ## Policy Iteration

# In[30]:


stateValueMatrix = np.zeros(16)
expectedRewardMatrix = np.zeros(16)-1
expectedRewardMatrix[0] = 0
expectedRewardMatrix[15] = 0
policyMatrix = np.zeros((16,16))
theta = 1e-50
gamma = 0.9
for i in range(4):
    for j in range(4):
        index = i*4+j
        if (i == 0 and j == 0) or (i==3 and j==3):
            policyMatrix[index,index] = 1
        elif i == 0 and j == 3:
            policyMatrix[index,index] = 2/4
            policyMatrix[index, (i+1)*4+(j)] = 1/4
            policyMatrix[index, (i)*4+(j-1)] = 1/4
        elif i == 3 and j == 0:
            policyMatrix[index,index] = 2/4
            policyMatrix[index, (i-1)*4+(j)] = 1/4
            policyMatrix[index, (i)*4+(j+1)] = 1/4
        elif i == 0:
            policyMatrix[index,index] = 1/4
            policyMatrix[index, (i+1)*4+(j)] = 1/4
            policyMatrix[index, (i)*4+(j-1)] = 1/4
            policyMatrix[index, (i)*4+(j+1)] = 1/4
        elif j == 0:
            policyMatrix[index,index] = 1/4
            policyMatrix[index, (i+1)*4+(j)] = 1/4
            policyMatrix[index, (i-1)*4+(j)] = 1/4
            policyMatrix[index, (i)*4+(j+1)] = 1/4
        elif i == 3:
            policyMatrix[index,index] = 1/4
            policyMatrix[index, (i-1)*4+(j)] = 1/4
            policyMatrix[index, (i)*4+(j-1)] = 1/4
            policyMatrix[index, (i)*4+(j+1)] = 1/4
        elif j== 3:
            policyMatrix[index,index] = 1/4
            policyMatrix[index, (i+1)*4+(j)] = 1/4
            policyMatrix[index, (i-1)*4+(j)] = 1/4
            policyMatrix[index, (i)*4+(j-1)] = 1/4
        else:
            policyMatrix[index, (i+1)*4+(j)] = 1/4
            policyMatrix[index, (i-1)*4+(j)] = 1/4
            policyMatrix[index, (i)*4+(j+1)] = 1/4
            policyMatrix[index, (i)*4+(j-1)] = 1/4


# In[31]:


while True:
    print(policyMatrix)
    oldPolicyStateValue = np.copy(stateValueMatrix)
    while True:
        maxDifference = 0
        oldStateValueMatrix = np.copy(stateValueMatrix)
        stateValueMatrix = expectedRewardMatrix + np.matmul(policyMatrix, stateValueMatrix)
        stateValueMatrix[0] = 0
        stateValueMatrix[15] = 0
        stateValueMatrix = np.round(stateValueMatrix, decimals=1)
        difference = abs(stateValueMatrix - oldStateValueMatrix)
        if np.max(difference) < theta:
            break
    for row in range(4):
        for col in range(4):
            print(stateValueMatrix[row*4+col], end=" ")
        print()
    if np.max(abs(oldPolicyStateValue - stateValueMatrix)) < theta:
        break
    for i in range(16):
        indices = []
        value = []
        for j in range(16):
            if policyMatrix[i,j] > 0:
                value.append(policyMatrix[i,j] * stateValueMatrix[j])
                indices.append(j)
        maxIndices = np.argwhere(value == np.max(value)).flatten()
        numMaxIndices = len(maxIndices)
        policyMatrix[i] = 0
        for j in range(len(maxIndices)):
            policyMatrix[i, indices[maxIndices[j]]] = 1/numMaxIndices
print(policyMatrix)        


# ## Value Interation

# In[32]:


stateValueMatrix = np.zeros(16)
expectedRewardMatrix = np.zeros(16)-1
expectedRewardMatrix[0] = 0
expectedRewardMatrix[15] = 0
policyMatrix = np.zeros((16,16))
theta = 1e-50
gamma = 0.9
for i in range(4):
    for j in range(4):
        index = i*4+j
        if (i == 0 and j == 0) or (i==3 and j==3):
            policyMatrix[index,index] = 1
        elif i == 0 and j == 3:
            policyMatrix[index,index] = 2/4
            policyMatrix[index, (i+1)*4+(j)] = 1/4
            policyMatrix[index, (i)*4+(j-1)] = 1/4
        elif i == 3 and j == 0:
            policyMatrix[index,index] = 2/4
            policyMatrix[index, (i-1)*4+(j)] = 1/4
            policyMatrix[index, (i)*4+(j+1)] = 1/4
        elif i == 0:
            policyMatrix[index,index] = 1/4
            policyMatrix[index, (i+1)*4+(j)] = 1/4
            policyMatrix[index, (i)*4+(j-1)] = 1/4
            policyMatrix[index, (i)*4+(j+1)] = 1/4
        elif j == 0:
            policyMatrix[index,index] = 1/4
            policyMatrix[index, (i+1)*4+(j)] = 1/4
            policyMatrix[index, (i-1)*4+(j)] = 1/4
            policyMatrix[index, (i)*4+(j+1)] = 1/4
        elif i == 3:
            policyMatrix[index,index] = 1/4
            policyMatrix[index, (i-1)*4+(j)] = 1/4
            policyMatrix[index, (i)*4+(j-1)] = 1/4
            policyMatrix[index, (i)*4+(j+1)] = 1/4
        elif j== 3:
            policyMatrix[index,index] = 1/4
            policyMatrix[index, (i+1)*4+(j)] = 1/4
            policyMatrix[index, (i-1)*4+(j)] = 1/4
            policyMatrix[index, (i)*4+(j-1)] = 1/4
        else:
            policyMatrix[index, (i+1)*4+(j)] = 1/4
            policyMatrix[index, (i-1)*4+(j)] = 1/4
            policyMatrix[index, (i)*4+(j+1)] = 1/4
            policyMatrix[index, (i)*4+(j-1)] = 1/4


# In[33]:


print(policyMatrix)
while True:
    maxDifference = 0
    oldStateValueMatrix = np.copy(stateValueMatrix)
    for i in range(1,15):
        value = []
        for j in range(16):
            if policyMatrix[i,j] > 0:
                value.append(policyMatrix[i,j]*expectedRewardMatrix[i] + policyMatrix[i,j]*stateValueMatrix[j])
        stateValueMatrix[i] = max(value)       
    stateValueMatrix[0] = 0
    stateValueMatrix[15] = 0
    stateValueMatrix = np.round(stateValueMatrix, decimals=2)
    for row in range(4):
        for col in range(4):
            print(stateValueMatrix[row*4+col], end=" ")
        print()
    print("\n")
    difference = abs(stateValueMatrix - oldStateValueMatrix)
    if np.max(difference) < theta:
        break
        
for i in range(16):
    indices = []
    value = []
    for j in range(16):
        if policyMatrix[i,j] > 0:
            value.append(policyMatrix[i,j] * stateValueMatrix[j])
            indices.append(j)
    maxIndices = np.argwhere(value == np.max(value)).flatten()
    numMaxIndices = len(maxIndices)
    policyMatrix[i] = 0
    for j in range(len(maxIndices)):
        policyMatrix[i, indices[maxIndices[j]]] = 1/numMaxIndices

print(policyMatrix)

