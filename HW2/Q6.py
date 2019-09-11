#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# ## Policy Iteration

# In[ ]:



'''
stateValueMatrix - Represents the state value function for the 16 states (5X5) with i,j state stored in i*4+j index
expectedRewardMatrix - Represents the expected reward given a state
policyMatrix - Represents the probability of going from one state to another
'''

stateValueMatrix = np.zeros(16)
expectedRewardMatrix = np.zeros(16)-1
expectedRewardMatrix[0] = 0
expectedRewardMatrix[15] = 0
policyMatrix = np.zeros((16,16))
theta = 1e-50
gamma = 1


'''
This loop initialises the Policy Matrix, 
Example - 
If i==0 and j==0 or i==3 and j==3 i.e. the terminal state then it is only possible to go to the same state
For corner positions it possible to stay in same state with 1/2 probability and go into other two states with 1/4 probability
Similarly for others
'''

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


# In[ ]:


'''
Performing Policy Iteration to solve the Non Linear Equation 

The Policy Evalutation Step

v(s) = Σ(a|s)Σp(s',r|s,a)[r+ gamma*v(s')] , for all s'. The outer summation is over a, the inner summation is over r and s'
This Equation can be simplified to

v(s) = Σrp(r|s) + gamma*(Σp(s'|s)v(s'))

The Policy Improvement Step

π'(s) = argmax (Σrp(r,a|s) + gamma*(Σp(s',a|s)v(s')))


'''

#Loop for the whole policy iteration
while True:
    
    #Printing Policy
    print(policyMatrix)
    
    oldPolicyStateValue = np.copy(stateValueMatrix)
    
    #Loop for Policy Evaluation
    while True:
        maxDifference = 0
        oldStateValueMatrix = np.copy(stateValueMatrix)
        #Update Step
        stateValueMatrix = expectedRewardMatrix + np.matmul(policyMatrix, stateValueMatrix)
        stateValueMatrix[0] = 0
        stateValueMatrix[15] = 0
        stateValueMatrix = np.round(stateValueMatrix, decimals=1)
        difference = abs(stateValueMatrix - oldStateValueMatrix)
        if np.max(difference) < theta:
            break
    
    #Printing Value Matrix
    for row in range(4):
        for col in range(4):
            print(stateValueMatrix[row*4+col], end=" ")
        print()
    
    #Breaking Out of Policy Loop if Value Function using previous policy is same
    if np.max(abs(oldPolicyStateValue - stateValueMatrix)) < theta:
        break
    
    #Updating Policy
    for i in range(16):
        indices = []
        value = []
        for j in range(16):
            if policyMatrix[i,j] > 0:
                expectedReward = 0
                if i != 0 and i != 15:
                    expectedReward = -1
                #Storing Value for Each action
                value.append(policyMatrix[i,j]*expectedReward + policyMatrix[i,j] * stateValueMatrix[j])
                indices.append(j)
        #Taking the max of actions
        value = np.round(value, decimals=1)
        maxIndices = np.argwhere(value == np.max(value)).flatten()
        numMaxIndices = len(maxIndices)
        policyMatrix[i] = 0
        #Updating with the max values
        for j in range(len(maxIndices)):
            policyMatrix[i, indices[maxIndices[j]]] = 1/numMaxIndices

print(policyMatrix)        


# ## Value Interation

# In[ ]:


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


# In[ ]:


print(policyMatrix)
while True:
    maxDifference = 0
    oldStateValueMatrix = np.copy(stateValueMatrix)
    
    #Rather than updating the state value with the expected value over actions, taking the max over all actions
    for i in range(1,15):
        value = []
        for j in range(16):
            if policyMatrix[i,j] > 0:
                value.append(policyMatrix[i,j]*expectedRewardMatrix[i] + policyMatrix[i,j]*stateValueMatrix[j])
        #Updating with the max value
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

#Updating Policy
for i in range(16):
    indices = []
    value = []
    for j in range(16):
        if policyMatrix[i,j] > 0:
            expectedReward = 0
            if i != 0 and i != 15:
                expectedReward = -1
            #Storing Value for Each action
            value.append(policyMatrix[i,j]*expectedReward + policyMatrix[i,j] * stateValueMatrix[j])
            indices.append(j)
    maxIndices = np.argwhere(value == np.max(value)).flatten()
    numMaxIndices = len(maxIndices)
    policyMatrix[i] = 0
    for j in range(len(maxIndices)):
        policyMatrix[i, indices[maxIndices[j]]] = 1/numMaxIndices

print(policyMatrix)

