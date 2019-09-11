#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


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


# In[ ]:


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
        


# In[ ]:


'''
Initialising the transition Probability matrix picking states with equal probability

Example
For state 0,0 we can go right with 1/4, bottom with 1/4 and out of the grid with 2/4 that results in the same state
Similarly for other states
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
    for row in range(25):
        print(row, transitionProbMatrix[row])
    
    oldPolicyStateValue = np.copy(stateValueMatrix)
    
    #Loop for Policy Evaluation
    while True:
        maxDifference = 0
        oldStateValueMatrix = np.copy(stateValueMatrix)
        #Update Step
        stateValueMatrix = expectedRewardMatrix + gamma*np.matmul(transitionProbMatrix, stateValueMatrix)
        stateValueMatrix = np.round(stateValueMatrix, decimals=1)
        difference = abs(stateValueMatrix - oldStateValueMatrix)
        if np.max(difference) < theta:
            break
    
    #Printing Value Matrix
    for row in range(5):
        for col in range(5):
            print(stateValueMatrix[row*5+col], end=" ")
        print()
    
    #Breaking Out of Policy Loop if Value Function using previous policy is same
    if np.max(abs(oldPolicyStateValue - stateValueMatrix)) < theta:
        break
    
    #Updating Policy
    for i in range(25):
        indices = []
        value = []
        for j in range(25):
            expectedReward = 0
            if transitionProbMatrix[i,j] > 0:
                if i == j:
                    expectedReward += -1*transitionProbMatrix[i,j]
                elif i == 1 and j == 21:
                    expectedReward += 10*transitionProbMatrix[i,j]
                elif i == 3 and j == 13:
                    expectedReward += 5*transitionProbMatrix[i,j]
                #Storing Value for Each action
                value.append(expectedReward + gamma*transitionProbMatrix[i,j] * stateValueMatrix[j])
                indices.append(j)
        value = np.round(value, decimals=1)
        #Taking the max of actions
        maxIndices = np.argwhere(value == np.max(value)).flatten()
        numMaxIndices = len(maxIndices)
        transitionProbMatrix[i] = 0
        #Updating with the max values
        for j in range(len(maxIndices)):
            transitionProbMatrix[i, indices[maxIndices[j]]] = 1/numMaxIndices
    
    #Updating the expected reward due to new policy
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
        
for row in range(25):
    print(row, transitionProbMatrix[row])

