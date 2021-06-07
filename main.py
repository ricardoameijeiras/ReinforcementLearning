from random import random
import gym
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

gym.envs.register(
    id='MountainCarMyEasyVersion-v1',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=1000,  # MountainCar-v0 uses 200
)


def Qlearn(env, learning, discount, epsilon, episodes):
    num_states = (env.observation_space.high - env.observation_space.low) * \
                 np.array([10, 100])
    num_states = np.round(num_states, 0).astype(int) + 1
    # Initialize Q table
    Q = np.random.uniform(low=-1, high=1,
                          size=(num_states[0], num_states[1],
                                env.action_space.n))
    reward_list = []
    ave_reward_list = []

    reduction = epsilon/episodes

    for i in range(episodes):
        observation = env.reset()
        done = False
        timesteps = 0
        total_rew = 0

        state_adj = (observation - env.observation_space.low) * np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)
        while not done:
            if i >= episodes-5:
                env.render()
            if random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0],state_adj[1]])
            else:
                action = env.action_space.sample()  # your agent here (this takes random actions)
            observation, reward, done, info = env.step(action)

            state2_adj = (observation - env.observation_space.low) * np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)
            if observation[0] >= 0.5:
                Q[state_adj[0], state_adj[1], action] = reward
            else:
                delta = learning * (reward +
                                   discount * np.max(Q[state2_adj[0],
                                                       state2_adj[1]]) -
                                   Q[state_adj[0], state_adj[1], action])
                Q[state_adj[0], state_adj[1], action] += delta

            total_rew += reward
            state_adj = state2_adj
            epsilon -= reduction
            reward_list.append(total_rew)

            if (i + 1) % 100 == 0:
                ave_reward = np.mean(reward_list)
                reward_list = []
                if done:
                    print('Episode {} Average Reward: {}'.format(i + 1, ave_reward))
                    ave_reward_list.append(ave_reward)
            timesteps += 1
    env.close()
    return ave_reward_list, Q


def main():
    env = gym.make('MountainCarMyEasyVersion-v1')
    episodes = 1000
    rewards, Q = Qlearn(env, learning=0.15, discount=0.9, epsilon=0.9, episodes= episodes)
    xvalues = np.arange(-0.07, 0.07, 0.01)
    yvalues = np.arange(-1.2, 0.6, 0.1)
    xvalues = ['{:3.2f}'.format(x) for x in xvalues]
    yvalues = ['{:3.1f}'.format(x) for x in yvalues]
    Q = np.moveaxis(Q, 2, 0)
    ax = sns.heatmap(Q[0], xticklabels=xvalues, yticklabels=yvalues)

    plt.title("Heatmap")
    plt.savefig("heatmap.jpg")
    plt.show()
    plt.close()

    plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig("rewards.jpg")
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()