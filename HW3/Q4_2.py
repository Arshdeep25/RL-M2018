#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# In[2]:


actionSpaceValue = {}
actionSpaceCount = {}
policyFunction = {}
numberEpisodes = 100000
for currentSum in range(12,22):
    for dealerCard in range(1,11):
        for usableAce in range(0,2):
            if currentSum >= 20:
                policyFunction[(currentSum, dealerCard, usableAce)] = {0:1,1:0}
            else:
                policyFunction[(currentSum, dealerCard, usableAce)] = {0:0,1:1}
            actionSpaceValue[(currentSum, dealerCard, usableAce,0)] = 0
            actionSpaceValue[(currentSum, dealerCard, usableAce,1)] = 0
            actionSpaceCount[(currentSum, dealerCard, usableAce,0)] = 0
            actionSpaceCount[(currentSum, dealerCard, usableAce,1)] = 0


# In[3]:


def pickCard():
    
    card = random.randint(1,13)
    if card >= 11:
        card = 10
    return card


# In[4]:


def playPlayer(firstState, listCurrentStates):
    
    currentState = list(firstState)
    
    while True:
        
            randomCard = pickCard()

            if currentState[2] == 1:
                if randomCard + currentState[0] > 21:
                    currentState[0] -= 10
                    currentState[2] = 0
                currentState[0] += randomCard
            else:
                if randomCard == 1:
                    if currentState[0] + 11 <= 21:
                        currentState[0] += 11
                        currentState[2] = 1
                    else:
                        currentState[0] += 1
                else:
                    currentState[0] += randomCard

            if currentState[0] > 21:
                return -1

            stickProbability = policyFunction[tuple(currentState)][0]
            hitProbability = policyFunction[tuple(currentState)][1]
            
            if hitProbability > stickProbability:
                action = 1
            elif hitProbability < stickProbability:
                action = 0
            else:
                action = random.randint(0,1)
                
            listCurrentStates.append(tuple(currentState)+(action,))

            if action == 0:
                return playDealer(currentState)


# In[5]:


def playDealer(currentState):
    
    currentCard = currentState[1]
    hiddenCard = pickCard()
    currentSum = 0
    useableAce = False
    if currentCard or hiddenCard == 1:
        if currentCard == 1:
            currentSum += 11 + hiddenCard
        else:
            currentSum += 11 + currentCard
        usableAce = True
    else:
        currentSum += currentCard + hiddenCard
    bust = False
    
    if currentSum < 17:
    
        while True:

                randomCard = pickCard()

                if useableAce:
                    if randomCard + currentSum > 21:
                        useableAce = False
                        currentSum -= 10
                    currentSum += randomCard
                else:
                    if randomCard == 1:
                        if currentSum + 11 <= 21:
                            currentSum += 11
                            useableAce = True
                        else:
                            currentSum += 1
                    else:
                        currentSum += randomCard

                if currentSum > 21:
                    bust = True
                    break
                elif currentSum >= 17:
                    break
    
    if bust == True or currentSum < currentState[0]:
        return 1
    elif currentSum > currentState[0]:
        return -1
    else: 
        return 0


# In[6]:


def updateStateValue(result, listCurrentStates):
    
    for state in listCurrentStates:
        
        actionSpaceValue[state] = (actionSpaceValue[state]*actionSpaceCount[state] + result)/(actionSpaceCount[state]+1)
        actionSpaceCount[state] += 1
    


# In[7]:


def updatePolicy():
    
    for state in policyFunction:
        hitValue = actionSpaceValue[state+(1,)]
        stickValue = actionSpaceValue[state+(0,)]
        if hitValue > stickValue:
            policyFunction[state][0] = 0
            policyFunction[state][1] = 1
        elif hitValue < stickValue:
            policyFunction[state][0] = 1
            policyFunction[state][1] = 0
        else:
            policyFunction[state][0] = 0.5
            policyFunction[state][1] = 0.5


# In[8]:


def firstRandomState():
        
        playerSum = 0
        useableAce = False
        while playerSum < 12:
            card = pickCard()
            if card == 1:
                if playerSum + 11 <= 21:
                    playerSum += 11
                    useableAce = True
                else:
                    playerSum += 1
            else:
                playerSum += card
        
        dealerSum = pickCard()
        return (playerSum, dealerSum, useableAce)


# In[9]:


for episodes in range(numberEpisodes):
    
    listCurrentStates = []
    
    randomFirstState = firstRandomState()
    
    if policyFunction[randomFirstState][0] == 1:
        listCurrentStates.append(randomFirstState+(0,))
        result = playDealer(randomFirstState)
    else:
        listCurrentStates.append(randomFirstState+(1,))
        result = playPlayer(randomFirstState, listCurrentStates)
    
    updateStateValue(result, listCurrentStates)
    updatePolicy()


# In[10]:


fig = plt.figure(figsize=[15, 6])

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

usableAce_x = []
usableAce_y = []
usableAce_P = []

noUsableAce_x = []
noUsableAce_y = []
noUsableAce_P = []

for k,v in policyFunction.items():
    
    if k[2] == 0:
        usableAce_x.append(k[1])
        usableAce_y.append(k[0])
        usableAce_P.append(0 if v[0] == 1 else 1)
    else:
        noUsableAce_x.append(k[1])
        noUsableAce_y.append(k[0])
        noUsableAce_P.append(0 if v[0] == 1 else 1)

usableAce_x = np.array(usableAce_x)
usableAce_y = np.array(usableAce_y)
usableAce_P = np.array(usableAce_P)

noUsableAce_x = np.array(noUsableAce_x)
noUsableAce_y = np.array(noUsableAce_y)
noUsableAce_P = np.array(noUsableAce_P)

ax1.scatter(usableAce_x[usableAce_P == 0], usableAce_y[usableAce_P == 0], c='b', label='stick')
ax1.scatter(usableAce_x[usableAce_P == 1], usableAce_y[usableAce_P == 1], c='g', label='hit')
ax1.set_title("usable ace")
ax1.legend()

ax2.scatter(noUsableAce_x[noUsableAce_P == 0], noUsableAce_y[noUsableAce_P == 0], c='b', label='stick')
ax2.scatter(noUsableAce_x[noUsableAce_P == 1], noUsableAce_y[noUsableAce_P == 1], c='g', label='hit')
ax2.set_title("No usable ace")
ax2.legend()


# In[11]:


stateSpaceValue = {}
for state in policyFunction:
    
    hitValue = actionSpaceValue[state+(1,)]
    stickValue = actionSpaceValue[state+(0,)]
    
    stateSpaceValue[state] = hitValue if hitValue > stickValue else stickValue


# In[12]:


usable_ace = {}
nonusable_ace = {}

for k, v in stateSpaceValue.items():
    if k[2]:
        usable_ace[k] = v
    else:
        nonusable_ace[k] = v
                 
fig = plt.figure(figsize=[15, 6])

ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

x1 = [k[1] for k in usable_ace.keys()]
y1 = [k[0] for k in usable_ace.keys()]
z1 = [v for v in usable_ace.values()]

x1 = np.array(x1)
y1 = np.array(y1)
z1 = np.array(z1)
x1 = x1.reshape(10,10)
y1 = y1.reshape(10,10)
z1 = z1.reshape(10,10)
ax1.set_zlim(-1,1)
ax1.set_xlim(1,10)
ax1.set_ylim(12,21)
ax1.plot_surface(x1, y1, z1)

ax1.set_title("usable ace")
ax1.set_xlabel("dealer showing")
ax1.set_ylabel("player sum")
ax1.set_zlabel("reward")

x2 = [k[1] for k in nonusable_ace.keys()]
y2 = [k[0] for k in nonusable_ace.keys()]
z2 = [v for v in nonusable_ace.values()]
x2 = np.array(x2)
y2 = np.array(y2)
z2 = np.array(z2)
x2 = x2.reshape(10,10)
y2 = y2.reshape(10,10)
z2 = z2.reshape(10,10)
ax2.set_zlim(-1,1)
ax2.set_xlim(1,10)
ax2.set_ylim(12,21)
ax2.plot_surface(x2, y2, z2)

ax2.set_title("non-usable ace")
ax2.set_xlabel("dealer showing")
ax2.set_ylabel("player sum")
ax2.set_zlabel("reward")

plt.show()


# In[ ]:




