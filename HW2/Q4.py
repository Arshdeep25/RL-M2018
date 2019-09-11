#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np


# In[27]:


'''
stateValueMatrix - Represents the state value function for the 25 states (5X5) with i,j state stored in i*5+j index
expectedRewardMatrix - Represents the expected reward given a state
transitionProbMatrix - Represents the probability of going from one state to another
'''
stateValueMatrix = np.zeros(25)
expectedRewardMatrix = np.zeros(25)
transitionProbMatrix = np.zeros((25,25))
gamma = 0.9
theta = 1e-10


# In[28]:


'''
This loop initialises the Expected Reward Matrix, 
For all the corner positions two actions result in reward -1 and rest in 0
For the two special positions 0,1 and 0,3, the expected reward is 10 and 5
For rows and columns on the boundary, there is a reward of -1 in one case and 0 in rest
In all other coordinates, the reward is 0

'''
for i in range(5):
    for j in range(5):
        
        if (i == 0 and j == 0) or (i == 4 and j == 4) or (i == 0 and j == 4) or (i == 4 and j == 0):
            expectedRewardMatrix[i*5+j] = -2*1/4
        elif i == 0 and j == 1:
            expectedRewardMatrix[i*5+j] = 10
        elif i == 0 and j == 3:
            expectedRewardMatrix[i*5+j] = 5
        elif i==0 or i == 4 or j == 0 or j == 4:
            expectedRewardMatrix[i*5+j] = -1*1/4
        else:
            expectedRewardMatrix[i*5+j] = 0
        


# In[29]:


'''
Initialising the transition Probability matrix picking states with equal probability

Example
For state 0,0 we can go right with 1/4, bottom with 1/4 and out of the grid with 2/4 that results in the same state

'''
for i in range(5):
    for j in range(5):
        index = i*5+j
        if i == 0 and j == 0:
            transitionProbMatrix[index,(i+1)*5+j] = 1/4
            transitionProbMatrix[index,i*5+j+1] = 1/4
            transitionProbMatrix[index, index] = 2/4
        elif i == 0 and j == 4:
            transitionProbMatrix[index,(i+1)*5+j] = 1/4
            transitionProbMatrix[index,i*5+j-1] = 1/4
            transitionProbMatrix[index,index] = 2/4
        elif i == 4 and j == 0:
            transitionProbMatrix[index,(i-1)*5+j] = 1/4
            transitionProbMatrix[index,i*5+j+1] = 1/4
            transitionProbMatrix[index,index] = 2/4
        elif i == 4 and j == 4:
            transitionProbMatrix[index,(i-1)*5+j] = 1/4
            transitionProbMatrix[index,i*5+j-1] = 1/4
            transitionProbMatrix[index,index] = 2/4
        elif i == 0 and j == 1:
            transitionProbMatrix[index,(i+4)*5+j] = 1
        elif i == 0 and j == 3:
            transitionProbMatrix[index,(i+2)*5+j] = 1
        elif i == 0:
            transitionProbMatrix[index,(i+1)*5+j] = 1/4
            transitionProbMatrix[index,i*5+j+1] = 1/4
            transitionProbMatrix[index,i*5+j-1] = 1/4
            transitionProbMatrix[index,index] = 1/4
        elif i == 4:
            transitionProbMatrix[index,(i-1)*5+j] = 1/4
            transitionProbMatrix[index,i*5+j+1] = 1/4
            transitionProbMatrix[index,i*5+j-1] = 1/4
            transitionProbMatrix[index,index] = 1/4
        elif j == 0:
            transitionProbMatrix[index,(i+1)*5+j] = 1/4
            transitionProbMatrix[index,(i-1)*5+j] = 1/4
            transitionProbMatrix[index,i*5+j+1] = 1/4
            transitionProbMatrix[index,index] = 1/4
        elif j == 4:
            transitionProbMatrix[index,(i+1)*5+j] = 1/4
            transitionProbMatrix[index,(i-1)*5+j] = 1/4
            transitionProbMatrix[index,i*5+j-1] = 1/4
            transitionProbMatrix[index,index] = 1/4
        else:
            transitionProbMatrix[index,(i+1)*5+j] = 1/4
            transitionProbMatrix[index,(i-1)*5+j] = 1/4
            transitionProbMatrix[index,i*5+j+1] = 1/4
            transitionProbMatrix[index,i*5+j-1] = 1/4


# In[30]:


while True:
    print(transitionProbMatrix)
    oldPolicyStateValue = np.copy(stateValueMatrix)
    while True:
        maxDifference = 0
        oldStateValueMatrix = np.copy(stateValueMatrix)
        stateValueMatrix = expectedRewardMatrix + gamma*np.matmul(transitionProbMatrix, stateValueMatrix)
        difference = abs(stateValueMatrix - oldStateValueMatrix)
        if np.max(difference) < theta:
            break
    stateValueMatrix = np.around(stateValueMatrix, decimals=2)
    for row in range(5):
        for col in range(5):
            print(stateValueMatrix[row*5+col], end=" ")
        print()
    if np.max(abs(oldPolicyStateValue - stateValueMatrix)) < theta:
        break
    for i in range(25):
        indices = []
        value = []
        for j in range(25):
            if transitionProbMatrix[i,j] > 0:
                value.append(transitionProbMatrix[i,j] * stateValueMatrix[j])
                indices.append(j)
        maxIndices = np.argwhere(value == np.max(value)).flatten()
        numMaxIndices = len(maxIndices)
        transitionProbMatrix[i] = 0
        for j in range(len(maxIndices)):
            transitionProbMatrix[i, indices[maxIndices[j]]] = 1/numMaxIndices
    
    for i in range(25):
        expectedReward = 0
        for j in range(25):
            if transitionProbMatrix[i,j] > 0:
                if i == j:
                    expectedReward += -1*transitionProbMatrix[i,j]
                elif i == 1 and j == 21:
                    expectedReward += 10*transitionProbMatrix[i,j]
                elif i == 3 and j == 13:
                    expectedReward += 5*transitionProbMatrix[i,j]
        expectedRewardMatrix[i] = expectedReward
        
print(transitionProbMatrix)


# In[ ]:




