# POLICY EVALUATION
### NAME : CHAITANYA P S
### REG.NO : 212222230024
## AIM
To develop a Python program to evaluate the given policy by maximizing its cumulative reward while dealing with slippery terrain.

## PROBLEM STATEMENT
The Bandit Slippery Walk problem is a Reinforcement Learning (RL) problem in which the agent must learn to navigate a slippery environment to reach the goal state.

1. we are tasked with creating an RL agent to solve the "Bandit Slippery Walk" problem.

2. The environment consists of Seven states representing discrete positions the agent can occupy.

3. The agent must learn to navigate this environment while dealing with the challenge of slippery terrain.

4. Slippery terrain introduces stochasticity in the agent's actions, making it difficult to predict the outcomes of its actions accurately.

## STATE
The environment has 7 states:

Two Terminal States: G: The goal state & H: A hole state.Five Transition states / Non-terminal States including S: The starting state.

## Actions
The agent can take two actions: R (move right) and L (move left). 

The transition probabilities for each action are as follows:

50% chance that the agent moves in the intended direction.
33.33% chance that the agent stays in its current state.
16.66% chance that the agent moves in the opposite direction.

## REWARD
The agent receives a reward of +1 for reaching the goal state and a reward of 0 for all other states.

## GRAPHICAL REPRESENTATION
<img src='https://github.com/user-attachments/assets/a3751c64-eb01-4cbc-87ad-971fe212be2e' width=50%>

## FORMULA
<img src='https://github.com/user-attachments/assets/bc41a5f0-04b5-40d5-8e9a-5db82ee5626e' width=50%>


## POLICY EVALUATION FUNCTION
```python
pip install git+https://github.com/mimoralea/gym-walk#egg=gym-walk
import warnings ; warnings.filterwarnings('ignore')
import gym, gym_walk
import numpy as np

import random
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123)

# Reference https://github.com/mimoralea/gym-walk
def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")
def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")
def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)
def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)
env = gym.make('SlipperyWalkFive-v0')
P = env.env.P
init_state = env.reset()
goal_state = 6
LEFT, RIGHT = range(2)
P
init_state
state, reward, done, info = env.step(RIGHT)
print("state:{0} - reward:{1} - done:{2} - info:{3}".format(state, reward, done, info))
# First Policy
pi_1 = lambda s: {
    0:LEFT, 1:LEFT, 2:LEFT, 3:LEFT, 4:LEFT, 5:LEFT, 6:LEFT
}[s]
print_policy(pi_1, P, action_symbols=('<', '>'), n_cols=7)
# Find the probability of success and the mean return of the first policy
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_1, goal_state=goal_state)*100,
    mean_return(env, pi_1)))
# Create your own policy

pi_2 = lambda s: {
    0:LEFT, 1:LEFT, 2:LEFT, 3:RIGHT, 4:RIGHT, 5:RIGHT, 6:RIGHT
}[s]

print_policy(pi_2, P, action_symbols=('<', '>'), n_cols=7)

## Find the probability of success and the mean return of you your policy
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
      probability_success(env, pi_2, goal_state=goal_state)*100,
      mean_return(env, pi_2)))

# Calculate the success probability and mean return for both policies
success_prob_pi_1 = probability_success(env, pi_1, goal_state=goal_state)
mean_return_pi_1 = mean_return(env, pi_1)

success_prob_pi_2 = probability_success(env, pi_2, goal_state=goal_state)
mean_return_pi_2 = mean_return(env, pi_2)
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    # Write your code here to evaluate the given policy
    while True:
      V=np.zeros(len(P),dtype=np.float64)
      for s in range(len(P)):
        for prob, next_state, reward, done in P[s][pi(s)]:
          V[s]+=prob*(reward+gamma*prev_V[next_state]*(not done))
      if np.max(np.abs(prev_V-V))<theta:
        break
      prev_V=V.copy()
      return V
# Code to evaluate the first policy
V1 = policy_evaluation(pi_1, P)
print_state_value_function(V1, P, n_cols=7, prec=5)
# Code to evaluate the second policy
# Write your code here
V2 = policy_evaluation(pi_2, P)
print_state_value_function(V2, P, n_cols=7, prec=5)
# Comparing the two policies
if(np.sum(V1>=V2)==7):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")
# Compare the two policies based on the value function using the above equation and find the best policy
V1
print_state_value_function(V1, P, n_cols=7, prec=5)
V2
print_state_value_function(V2, P, n_cols=7, prec=5)
V1>=V2
if(np.sum(V1>=V2)==7):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")

```
## OUTPUT:
### POLICY 1
<img src ='https://github.com/user-attachments/assets/a23bcc2c-4928-4fbb-b01d-fc6d837e9bd0' width=50%>
<img src ='https://github.com/user-attachments/assets/e637d739-a107-488c-a6e9-c7580b8f015a' width=50%>
<img src ='https://github.com/user-attachments/assets/f6f213c9-7861-4ceb-bce9-f4c6a1afba7a' width=50%>

### POLICY 2
<img src ='https://github.com/user-attachments/assets/ea993d87-933d-4775-bf4d-d362b80cbabe' width=50%>
<img src ='https://github.com/user-attachments/assets/68f57bc0-381a-493b-aef3-5266818c36a4' width=50%>
<img src ='https://github.com/user-attachments/assets/b49d1e11-14ae-412b-8914-9975b3972dd4' width=50%>

### COMPARISON
<img src ='https://github.com/user-attachments/assets/dccb724e-509e-40d9-b24e-54eadd255531' width=50%>

### CONCLUSION
<img src ='https://github.com/user-attachments/assets/fe676bea-230b-4a35-b5d2-72e53822d227' width=50%>

## RESULT:
Thus, This program will evaluate the given policy in the Bandit Slippery Walk environment and predict the expected reward of the policy.
