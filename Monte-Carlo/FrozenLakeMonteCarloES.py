import numpy as np
import gym
import random
import math


class FrozenLakeMCES:
    env = gym.make('FrozenLake-v0')
    gamma = 0.9

    def __init__(self, gamma=0.9, episodes=10000000):
        self.gamma = gamma
        self.policy = []
        self.episodes = episodes
        self.statespace = self.env.observation_space.n
        self.actionspace = self.env.action_space.n
        self.returns = []
        self.Q = np.zeros([self.statespace, self.actionspace])
        for i in range(self.statespace):
            self.returns.append([])
            for j in range(self.actionspace):
                self.returns[i].append([])
            self.policy.append(random.randint(0, self.actionspace - 1))
        print('state space :', self.statespace)
        print('action space :', self.actionspace)
        print(self.policy)

    def train(self):
        for episode in range(self.episodes):
            for i in self.policy:
                self.policy[i] = random.randint(0, self.actionspace - 1)
            self.env.reset()
            if episode % 1000 == 0:
                print('%d episodes finished' % episode)
            state = random.randint(0, self.statespace - 1)
            action = random.randint(0, self.actionspace - 1)
            generated_episode = []
            # generate an episode from state, action
            for trans_prob, next_state, reward, done in self.env.P[state][action]:
                generated_episode.append((state, action, reward))
                if done:
                    break
                action = self.policy[state]
                state = next_state
            G = 0
            for step in reversed(generated_episode):
                current_state = step[0]
                current_action = step[1]
                G = self.gamma * G + step[2]
                # check if s,a appears in steps before
                appeared = False
                for before in generated_episode:
                    if before == step:
                        break
                    if before[0] == step[0] and before[1] == step[1]:
                        appeared = True
                        break
                if not appeared:
                    self.returns[current_state][current_action].append(G)
                    returns = self.returns[current_state][current_action]
                    self.Q[current_state][current_action] = sum(returns)/len(returns)
                    self.policy[current_state] = np.argmax(self.Q[current_state])


if __name__ == '__main__':
    mces = FrozenLakeMCES()
    mces.train()
    print(mces.Q)
    print(mces.policy)
