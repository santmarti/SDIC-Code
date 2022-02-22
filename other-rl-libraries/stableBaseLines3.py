# Example from library Stable Baselines 3
# https://stable-baselines3.readthedocs.io/en/master/
# https://github.com/DLR-RM/stable-baselines3
# We load gym environment CartPole-v1
# It is solved with Policy Gradient algorithm PPO 
# The policy is approximated with a Multi-Layer Perceptron (MLP) Neural Network
#
# install:
# pip install "stable-baselines3[extra]"

import gym

from stable_baselines3 import PPO

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()