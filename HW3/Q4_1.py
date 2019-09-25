#!/usr/bin/env python
# coding: utf-8

# In[28]:


import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# In[37]:


stateSpaceValue = {}
stateSpaceCount = {}
numberEpisodes = 500000
for currentSum in range(12,22):
    for dealerCard in range(1,11):
        for usableAce in range(0,2):
            stateSpaceValue[(currentSum, dealerCard, usableAce)] = 0
            stateSpaceCount[(currentSum, dealerCard, usableAce)] = 0


# In[38]:


def pickCard():
    
    card = random.randint(1,13)
    if card >= 11:
        card = 10
    return card
    


# In[39]:


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
            
            listCurrentStates.append(tuple(currentState))
            
            if currentState[0] >= 20:
                return playDealer(tuple(currentState), listCurrentStates)
    


# In[40]:


def playDealer(currentState, listCurrentStates):
    
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
    
    


# In[41]:


def updateStateValue(result, listCurrentStates):
    
    for state in listCurrentStates:
        stateSpaceValue[state] = (stateSpaceValue[state]*stateSpaceCount[state] + result)/(stateSpaceCount[state]+1)
        stateSpaceCount[state] += 1
    


# In[42]:


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
                


# In[43]:


for episodes in range(numberEpisodes):
    
    listCurrentStates = []
    
    randomFirstState = firstRandomState()
        
    listCurrentStates.append(randomFirstState)
    
    if randomFirstState[0] >= 20:
        result = playDealer(randomFirstState, listCurrentStates)
    else:
        result = playPlayer(randomFirstState, listCurrentStates)
    
    updateStateValue(result, listCurrentStates)
    


# In[44]:


# Graph Implementation Seen from - https://github.com/MJeremy2017/Reinforcement-Learning-Implementation/blob/master/BlackJack/blackjack_mc.py
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





# In[ ]:





# In[ ]:




