# Example from library Stable Baselines 3
# https://stable-baselines3.readthedocs.io/en/master/
# https://github.com/DLR-RM/stable-baselines3
# We load gym environment CartPole-v1
# It is solved with Policy Gradient algorithm PPO 
# The policy is approximated with a Multi-Layer Perceptron (MLP) Neural Network
#
# install:
# pip install "stable-baselines3[extra]"

import sys
import gym

from stable_baselines3 import PPO, DQN

env_names = ["FrozenLake-v0", "CartPole-v1", "LunarLanderContinuous-v2", "LunarLander-v2"]

env_name = "CartPole-v1" 

if len(sys.argv) > 1:
    i = int(sys.argv[1])
    if i >= 0:
        env_name = env_names[i] 
    else:
        env_name = sys.argv[1]
 
env = gym.make(env_name) if env_name not in ["FrozenLake-v0", "FrozenLake-v1"] else gym.make(env_name, is_slippery=False)


n_episodes = 10000
if len(sys.argv) > 2:
    i = int(sys.argv[2])
    if i >= 0:
        n_episodes = i


model = PPO("MlpPolicy", env, verbose=1)
#model = DQN("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=n_episodes)
model.save(env_name)


total_reward = 0
trial_reward = 0
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    #env.render()

    trial_reward += reward
    if done:
      print("trial_reward", trial_reward)
      total_reward += trial_reward
      trial_reward = 0
      obs = env.reset()

print("----- total_reward", total_reward)
env.close()