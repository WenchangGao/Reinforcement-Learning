import gym
import numpy as np
from math import fabs
gamma = 0.9
theta = 1e-20
env = gym.make('FrozenLake-v0')


def policy_evaluation(value_table, policy):
    index = 0
    while True:
        delta = 0.00
        index += 1
        for state in range(env.observation_space.n):
            old_value = value_table[state]
            action = policy[state]
            value_table[state] = sum(trans_prob * (reward + gamma * (value_table[next_state])) for
                                     trans_prob, next_state, reward, done in env.P[state][action])
            delta = max(delta, fabs(old_value - value_table[state]))
        if delta <= theta:
            print('policy evaluation converged after %d iterations' % index)
            break
    return value_table


def policy_improvement(value_table, policy):
    while True:
        policy_stable = True
        for state in range(env.observation_space.n):
            old_action = policy[state]
            Qtable = np.zeros(env.action_space.n)
            for action in range(env.action_space.n):
                Qtable[action] += sum(trans_prob * (reward + gamma * value_table[next_state]) for
                                  trans_prob, next_state, reward, done in env.P[state][action])
            policy[state] = np.argmax(Qtable)
            if not policy[state] == old_action:
                policy_stable = False
        if not policy_stable:
            value_table = policy_evaluation(value_table, policy)
        else:
            break
    return policy


if __name__ == '__main__':
    value_table = np.zeros(env.observation_space.n)
    policy = np.zeros(env.observation_space.n)
    policy = policy_improvement(value_table, policy)
    print(policy)
