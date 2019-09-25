#!/usr/bin/env python
# coding: utf-8

# In[103]:


import random
import matplotlib.pyplot as plt
import numpy as np


# In[104]:


alpha = 0.5
epsilon = 0.1
actionValuesSarsa = {}
actionValuesQLearning = {}
rewardStateTransition = {}
episodeNumber = 500
runs = 500


# In[105]:


def initialiseState():
    # 36 - Start
    # 37 - End
    # Action : 0 - Top, 1 - Left, 2 - Bottom, 3 - Right
    for i in range(3):
        for j in range(12):
            state = i*12+j
            actionValuesQLearning[state] = {
                0:0,
                1:0,
                2:0,
                3:0
            }
            actionValuesSarsa[state] = {
                0:0,
                1:0,
                2:0,
                3:0
            }
            if i == 0 and j == 0:
                rewardStateTransition[state] = {
                    0 : (i*12+j,-1),
                    1 : (i*12+j,-1),
                    2 : ((i+1)*12+j,-1),
                    3 : (i*12+j+1,-1)
                }

            elif i == 0 and j == 11:
                rewardStateTransition[state] = {
                    0 : (i*12+j,-1),
                    1 : (i*12+(j-1),-1),
                    2 : ((i+1)*12+j,-1),
                    3 : (i*12+j,-1)
                }

            elif i == 2 and j == 0:
                rewardStateTransition[state] = {
                    0 : ((i-1)*12+j, -1),
                    1 : (i*12+j, -1),
                    2 : (36, -1),
                    3 : (i*12+j+1, -1)
                }

            elif i == 2 and j == 11:
                rewardStateTransition[state] = {
                    0 : ((i-1)*12+j, -1),
                    1 : (i*12+j-1, -1),
                    2 : (37, -1),
                    3 : (i*12+j, -1)
                }

            elif i == 0:
                rewardStateTransition[state] = {
                    0 : (i*12+j, -1),
                    1 : (i*12+j-1, -1),
                    2 : ((i+1)*12+j, -1),
                    3: (i*12+j+1, -1)
                }

            elif i == 2:
                rewardStateTransition[state] = {
                    0 : ((i-1)*12+j, -1),
                    1 : (i*12+j-1, -1),
                    2 : (36, -100),
                    3: (i*12+j+1, -1)
                }
            elif j == 0:
                rewardStateTransition[state] = {
                    0 : ((i-1)*12+j, -1),
                    1 : (i*12+j, -1),
                    2 : ((i+1)*12+j, -1),
                    3: (i*12+j+1, -1)
                }
            elif j == 11:
                rewardStateTransition[state] = {
                    0 : ((i-1)*12+j, -1),
                    1 : (i*12+j-1, -1),
                    2 : ((i+1)*12+j, -1),
                    3: (i*12+j, -1)
                }
            else:
                rewardStateTransition[state] = {
                    0 : ((i-1)*12+j, -1),
                    1 : (i*12+j-1, -1),
                    2 : ((i+1)*12+j, -1),
                    3: (i*12+j+1, -1)
                }

    rewardStateTransition[36] = {
        0 : (24, -1),
        1: (36, -1),
        2 : (36, -1),
        3: (36, -100)
    }
    actionValuesSarsa[36] = {
        0 : 0,
        2 : 0,
        1 : 0,
        3 : 0
    }
    actionValuesQLearning[36] = {
        0 : 0,
        2 : 0,
        1 : 0,
        3 : 0
    }
    actionValuesSarsa[37] = {
        0 : 0,
        2 : 0,
        1 : 0,
        3 : 0
    }
    actionValuesQLearning[37] = {
        0 : 0,
        2 : 0,
        1 : 0,
        3 : 0
    }


# In[106]:


def greedyState(state, mode):
    
    maxAction = []
    if mode == 'Sarsa':
        maxActionValue = max(actionValuesSarsa[state].values())
        for k,v in actionValuesSarsa[state].items():
            if v == maxActionValue:
                maxAction.append(k)
    else:
        maxActionValue = max(actionValuesQLearning[state].values())
        for k,v in actionValuesQLearning[state].items():
            if v == maxActionValue:
                maxAction.append(k)
    return random.choice(maxAction)
    


# In[107]:


def epsilonGreedyState(state, mode):
    epsilonProbability = random.uniform(0,1)
    if epsilonProbability <= epsilon:
        randomAction = random.randint(0,3)
        return randomAction
    return greedyState(state, mode)


# # Q Learning

# In[108]:


rewardMeanRunQLearning = []
run = 0
for run in range(runs):
    rewardEpisodeListQLearning = []
    initialiseState()
    epsiode = 0
    for episode in range(episodeNumber):
        state = 36
        rewardEpisode = 0
        while True:
            action = epsilonGreedyState(state, 'QLearning')
            nextState, reward = rewardStateTransition[state][action]
            rewardEpisode += reward
            nextAction = greedyState(nextState, 'QLearning')
            actionValuesQLearning[state][action] = actionValuesQLearning[state][action] + alpha*(reward + actionValuesQLearning[nextState][nextAction] - actionValuesQLearning[state][action])
            state = nextState
            if state == 37:
                break
        rewardEpisodeListQLearning.append(rewardEpisode)
    rewardMeanRunQLearning.append(rewardEpisodeListQLearning)
rewardMeanRunQLearning = np.array(rewardMeanRunQLearning)
rewardMeanRunQLearning = np.mean(rewardMeanRunQLearning, axis=0)


# # Sarsa

# In[109]:


rewardMeanRunSarsa = []
run = 0
for run in range(runs):
    rewardEpisodeListSarsa = []
    initialiseState()
    episode = 0
    for episode in range(episodeNumber):
        state = 36
        rewardEpisode = 0
        action = epsilonGreedyState(state, 'Sarsa')
        while True:
            nextState, reward = rewardStateTransition[state][action]
            rewardEpisode += reward
            nextAction = epsilonGreedyState(nextState, 'Sarsa')
            actionValuesSarsa[state][action] = actionValuesSarsa[state][action] + alpha*(reward + actionValuesSarsa[nextState][nextAction] - actionValuesSarsa[state][action])
            state = nextState
            action = nextAction
            if state == 37:
                break
        rewardEpisodeListSarsa.append(rewardEpisode)
    rewardMeanRunSarsa.append(rewardEpisodeListSarsa)
rewardMeanRunSarsa = np.array(rewardMeanRunSarsa)
rewardMeanRunSarsa = np.mean(rewardMeanRunSarsa, axis=0)


# In[110]:


plt.figure()
plt.plot(rewardMeanRunQLearning, c='b', label='Q Learning')
plt.plot(rewardMeanRunSarsa, c='g', label='Sarsa')
plt.ylim(-100,-20)
plt.legend()
plt.xlabel('Epsiodes')
plt.ylabel('Reward')
plt.show()


# In[ ]:




