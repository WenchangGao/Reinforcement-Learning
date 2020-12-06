import numpy as np
import gym

theta = 1e-20
gamma = 0.9
T = 100000
env = gym.make('FrozenLake-v0')


def value_iteration():
    value_table = np.zeros(env.observation_space.n)
    index = 0
    while True:
        old_value_table = np.copy(value_table)
        for state in range(env.observation_space.n):
            Qtable = []
            for action in range(env.action_space.n):
                next_state_reward = []
                for trans_prob, next_state, reward, done in env.P[state][action]:
                    # print(trans_prob, next_state, reward, done)
                    next_state_reward.append(trans_prob * (reward + gamma * old_value_table[next_state]))
                Qtable.append(np.sum(next_state_reward))
            value_table[state] = max(Qtable)
        index += 1
        if np.sum(np.fabs(value_table - old_value_table)) <= theta:
            print('value converged at %d iteration.' % index)
            break
    return value_table


def extract_policy(value_table):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        Qtable = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for trans_prob, next_state, reward, done in env.P[state][action]:
                Qtable[action] += trans_prob * (reward + gamma * value_table[next_state])
        policy[state] = np.argmax(Qtable)
    return policy


if __name__ == '__main__':
    env.reset()
    optimal_value = value_iteration()
    optimal_policy = extract_policy(optimal_value)
    print(optimal_policy)
    env.render()
    state = 0
    action = optimal_policy[state]
    done = False
    index = 1
    for trans_prob, next_state, reward, done in env.P[state][action]:
        print(index)
        index += 1
        env.render()
        print(reward)
        print(done)
        if done:
            break
        state = next_state
