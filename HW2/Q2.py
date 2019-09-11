#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


'''
Our goal is to solve the following system of linear equations
v(s) = Σ rP[r|s] + gamma*Σv(s')P[s'|s]
The first summation is over R and second is over S

It can be simplified to get,
v(s) = np.matmul(np.inverse((I - gamma*P[s'|s])) , Σ rP[r|s])

where v(s) is 25 X 1 matrix
Σ rP[r|s] is 25 X 1 matrix
I - gamma*P[s'|s] is 25X25 matrix
'''


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


identitMatrix = np.identity(25)
stateValueMatrix = np.matmul(np.linalg.inv(identitMatrix - gamma*transitionProbMatrix), expectedRewardMatrix)


# In[ ]:


stateValueMatrix = np.around(stateValueMatrix, decimals=1)


# In[ ]:


for i in range(5):
    for j in range(5):
        print(stateValueMatrix[i*5+j], end=" ")
    print()


# In[ ]:




