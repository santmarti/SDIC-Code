# Safety Gym
# is a benchmark environments suit
# https://openai.com/blog/safety-gym/
# https://github.com/openai/safety-gym
#
# it links with several simulators as mujoco
#
# install:
# git clone https://github.com/openai/safety-gym.git
# cd safety-gym
# 
# Here we learn in one custom environment with stablebaselines3 (see the provided example)
# install:
# pip install "stable-baselines3[extra]"

import sys
import numpy as np  
import gym
from gym.envs.registration import register

import safety_gym  
from safety_gym.envs.engine import Engine

from stable_baselines3 import PPO


model = None

# Walls TMaze shape
w = [[-1,y] for y in np.arange(-2.6,0,.2)]
w += [[1,y] for y in np.arange(-2.6,0,.2)]
w += [[x,.2] for x in np.arange(-1,-3,-.2)]
w += [[x,.2] for x in np.arange(1,3,.2)]
w += [[-3,y] for y in np.arange(0.2,3,.2)]
w += [[3,y] for y in np.arange(0.2,3,.2)]
w += [[x,3] for x in np.arange(-3,3.2,.2)]
w += [[x,-3] for x in np.arange(-1,1.2,.2)]


config = {
    'robot_base': 'xmls/car.xml',
    'task': 'goal',                    # goal, button, push, x, z, circle, or none (for screenshots)
    'lidar_max_dist': 10,
    'lidar_num_bins': 4,

    'observe_walls': True,
    'walls_num': len(w),
    'walls_size': .2,
    'walls_locations': w,

    'hazards_num': 1,
    'hazards_locations': [[0,.5]],  # Fixed locations to override placements
    'hazards_keepout': 0.4,  # Radius of hazard keepout for placement
    'hazards_size': 0.3,  # Radius of hazards
    'hazards_cost': 0,

    'goal_placements': [(-2.5, 1.2, -1.5, 1.8), (1.5, 1.2, 2.5, 1.8)],
    'goal_keepout': 0.2,
     #'goal_locations': [(2, 1.5), (-2, 1.5)],
    'robot_locations': [(0, -2.4)],
    'robot_rot': np.pi,

    'observe_sensors': False,
    'observe_goal_lidar': True,

    'observation_flatten': False,

    'reward_distance': 0.0
}




def reactive(obs):
    walls = np.array(obs['walls_lidar']) / 20
    wl,wr = walls[3], walls[2]
    dl = wl - wr

    lidar = np.array(obs['goal_lidar']) / 60      # [b_left b_right f_right, f_left] 0..1
    l,r = lidar[3], lidar[2]
    act = [r - l/2 +dl, l - r/2 - dl]  # left right motor                     

    act.reverse()  # act is [right, left]  0..1
    return act



def run(env):
    global model

    obs = env.reset()
    done = False
    ep_ret = 0
    ep_cost = 0
    while True:
        if done:
            print('Episode Return: %.3f \t Episode Cost: %.3f'%(ep_ret, ep_cost))
            ep_ret, ep_cost = 0, 0
            obs = env.reset()

        #print(obs)
        #print(env.robot_pos)

        if model == None:
             action = reactive(obs)
        else:
             action, _ = model.predict(obs, deterministic=True)
             
        obs, reward, done, info = env.step(action)

        ep_ret += reward
        ep_cost += info.get('cost', 0)
        env.render()


def main(bLearn=False):
    global model, config

    if bLearn:
        config['observation_flatten'] = True
        config['reward_distance'] = 1.0


    env = Engine(config)
    register(id='SafexpTestEnvironment-v0',
             entry_point='safety_gym.envs.mujoco:Engine',
             kwargs={'config': config})

    if bLearn:
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000)

    run(env)



if __name__ == '__main__':
    bLearn = True if len(sys.argv) > 1 else False
    main(bLearn=bLearn)
