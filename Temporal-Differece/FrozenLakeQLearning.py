import numpy as np
import gym
import random


class FrozenLakeQLearning:
    env = gym.make('FrozenLake-v0')

    def __init__(self, gamma=0.95, episodes=2000, epsilon=0.1, alpha=0.8):
        self.statespace = self.env.observation_space.n
        self.actionspace = self.env.action_space.n
        self.gamma = gamma
        self.episodes = episodes
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = np.zeros([self.statespace, self.actionspace])

    def train(self):
        for episode in range(self.episodes):
            state = self.env.reset()
            n = 0
            done = False
            while not done:
                action = np.argmax(self.Q[state, :] + np.random.randn(1, self.actionspace) * (1. / (episode + 1)))
                next_state, reward, done, _ = self.env.step(action)
                n += 1
                if n > 100:
                   break
                self.Q[state, action] = self.Q[state, action] + self.alpha * (reward +
                                                        self.gamma * max(self.Q[next_state, :]) - self.Q[state, action])
                action = np.argmax(self.Q[state, :] + np.random.randn(1, self.actionspace) * (1. / (episode + 1)))
                state = next_state
                next_state, reward, done, _ = self.env.step(action)

    # load model, to be continued
    def load(self):
        pass

    # save model, to be continued
    def save(self):
        pass


if __name__ == '__main__':
    fl = FrozenLakeQLearning()
    fl.train()
    print(fl.Q)

