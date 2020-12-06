import gym
import numpy as np
import random

env = gym.make('FrozenLake-v0')
Q_all = np.zeros([env.observation_space.n, env.action_space.n])
alpha = 0.8
gamma = 0.95
num_episodes = 2000
rList = []
for i in range(num_episodes):
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    while j < 99:
        j += 1
        # 贪婪动作选择，含嗓声干扰
        a = np.argmax(Q_all[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        # 从环境中得到新的状态和回报
        s1, r, d, _ = env.step(a)
        # 更新Q表
        Q_all[s, a] = Q_all[s, a] + alpha * (r + gamma * np.max(Q_all[s1, :]) - Q_all[s, a])
        # 累加回报
        rAll += r
        # 更新状态
        s = s1
        # Game Over
        if d == True:
            break
    rList.append(rAll)

print("Score over time：" + str(sum(rList) / num_episodes))
print(Q_all)
