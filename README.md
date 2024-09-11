# POLICY EVALUATION
### NAME : CHAITANYA P S
### REG.NO : 212222230024
## AIM :
To develop a Python program to evaluate the given policy by maximizing its cumulative reward while dealing with slippery terrain.
<br>
## PROBLEM STATEMENT : 

We are assigned with the task of creating an RL agent to solve the "Bandit Slippery Walk" problem. 
The environment consists of Seven states representing discrete positions the agent can occupy.
The agent must learn to navigate this environment while dealing with the challenge of slippery terrain.
Slippery terrain introduces stochasticity in the agent's actions, making it difficult to predict the outcomes of its actions accurately.
<br>

### States :

The environment has 7 states :
* Two Terminal States: **G**: The goal state & **H**: A hole state.
* Five Transition states / Non-terminal States including  **S**: The starting state.
<br>

### Actions :

The agent can take two actions:
* R -> Move right.
* L -> Move left.
<br>

### Transition Probabilities :

The transition probabilities for each action are as follows:
* **50%** chance that the agent moves in the intended direction.
* **33.33%** chance that the agent stays in its current state.
* **16.66%** chance that the agent moves in the opposite direction.
<br>

### Rewards :

* The agent receives a reward of +1 for reaching the goal state (G). 
* The agent receives a reward of 0 for all other states.
<br>

### Graphical Representation:
<img src="https://github.com/anto-richard/rl-policy-evaluation/assets/93427534/74eea05b-fd7a-4e0b-a9de-a3a124d7607a" width=50%>
<br><br>

## POLICY EVALUATION FUNCTION :
<br>

### Formula :

![out2](https://github.com/anto-richard/rl-policy-evaluation/assets/93427534/0fb0fe63-3a14-416e-b7fc-fdf3bcb495ba)


```python

pip install git+https://github.com/mimoralea/gym-walk#egg=gym-walk

```

```python

import warnings ; warnings.filterwarnings('ignore')

import gym, gym_walk
import numpy as np

import random
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123)

```

```python

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
        if (s + 1) % n_cols ==0:print("|")

```

```python

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

```

```python

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

```

```python

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

```

## Slippery Walk Five MDP:

```python

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

```


## Policy Evaluation:

```python

def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    # Write your code here to evaluate the given policy
    while True:
      V = np.zeros(len(P))
      for s in range(len(P)):
        for prob, next_state, reward, done in P[s][pi(s)]:
          V[s] += prob * (reward + gamma *  prev_V[next_state] * (not done))
      if np.max(np.abs(prev_V - V)) < theta:
        break
      prev_V = V.copy()
    return V


# Code to evaluate the first policy
V1 = policy_evaluation(pi_1, P)
print_state_value_function(V1, P, n_cols=7, prec=5)



# Code to evaluate the second policy
V2 = policy_evaluation(pi_2, P)
print_state_value_function(V2, P, n_cols=7, prec=5)

```



## Comparing the two policies

```python
# Compare the two policies based on the value function using the above equation and find the best policy

V1

print_state_value_function(V1, P, n_cols=7, prec=5)

V2

print_state_value_function(V2, P, n_cols=7, prec=5)

V1>=V2

if(np.sum(V1>=V2)==7):
  print("The first policy has the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy has the better policy")
else:
  print("Both policies have their merits.")

```

## OUTPUT :

### Policy-1 : 

<img src="https://github.com/user-attachments/assets/81640a2e-62d3-4d57-8c43-7df26c3de6f5" width=50%>

### Policy-1 State-value function :
<img src="https://github.com/user-attachments/assets/623f8b29-02f2-45ba-9461-fc582094830d" width=50%>

### the probability of success and the mean return of the first policy : 

<img src="https://github.com/user-attachments/assets/d99f4a04-edf0-4d06-b3e2-18f8a705d835" width=50%>

### Policy-2 :
<img src="https://github.com/user-attachments/assets/7282da02-63eb-4854-b9e5-bbf6f2263c44" width=50%>

### Policy-2 State-value function :
<img src="https://github.com/user-attachments/assets/70da3d5f-88f4-42a1-999f-5be4b8a29caf" width=50%>

### the probability of success and the mean return of the second policy :
<img src="https://github.com/user-attachments/assets/0240f01a-65e4-4b5f-8bc0-ccc5f136a0f1" width=50%>

### Comparing the two policies:
<img src="https://github.com/user-attachments/assets/1cccaa4b-3852-4da3-b3fe-ccc789f638a7" width=50%>

## RESULT :

Thus, the Given Policy has been Evaluated and Optimal Policy has been Computed using Python Programming and executed successfully.

