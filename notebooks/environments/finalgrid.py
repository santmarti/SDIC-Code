import sys
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import itertools
from collections import OrderedDict

#from numba import jit

from Plotting import draw_layers, draw_obs
from Environment import ACT_MODE, OBS_MODE, ACT_PLUS, OBS_PLUS, OBS_SHAPE, OBS_MEM


class MULTIAGENT_ORDER(Enum):
    RANDOM = 0
    ROTATING = 1
    FIXED = 2

def log_rand():
    return 1 - np.log(np.random.uniform(low=1, high=np.e))


class FinalGrid():

    def __init__(self, conf={}):
        self.conf = conf
        self.layers = OrderedDict()

        self.fig = None
        self.num_agents = 1 if "num_agents" not in conf else conf["num_agents"]
        self.moves = (np.array([0, -1]),np.array([0, 1]), np.array([-1, 0]), np.array([1, 0]))
        self.enemy_database = []

        self.multiagent_order = MULTIAGENT_ORDER.RANDOM
        if "multiagent_order" in conf:
            if conf["multiagent_mode"] == "fixed":
                self.multiagent_order = MULTIAGENT_ORDER.FIXED
            elif conf["multiagent_mode"] in ["rotating", "rotatory", "rolling"]:
                self.multiagent_order = MULTIAGENT_ORDER.ROTATING

        if "n" in conf:
            self.n = conf["n"]
            rows_cols = np.array([self.n, self.n])
            conf["rows"], conf["cols"] = self.n, self.n
        elif "rows" in conf and "cols" in conf:
            rows_cols = [conf["rows"], conf["cols"]]

        self.rows, self.cols = rows_cols
        self.rows_cols = np.array([self.rows, self.cols])
        self.n = self.rows

        self.set_obs()
        self.set_actions()

        self.reward = {"step": -0.01, "food": 1, "exit": 0} if "reward" not in conf else conf["reward"]
        self.torus = False if "torus" not in conf else conf["torus"]

        self.agent_ini_pos = None if "agent_ini_pos" not in conf else conf["agent_ini_pos"]

        self.term_mode = {}
        if "term_mode" in conf:
            self.term_mode = conf["term_mode"]
            if isinstance(self.term_mode, str):
                self.term_mode = {self.term_mode}

        if "update" in conf and conf["update"] == "forest_fire":
            if not "fire" in self.reward:
                self.reward["fire"] = -10
            if not "tree" in self.reward:
                self.reward["tree"] = 0.5

        if "walls" in conf:
            if isinstance(conf["walls"], str):
                conf["walls"] = [conf["walls"]]

        if "floor" in conf:
            if isinstance(conf["floor"], str):
                conf["floor"] = [conf["floor"]]
        else:
            self.floor = None

        if "drop_action" in conf:
            if isinstance(conf["drop_action"], str):
                conf["drop_action"] = {conf["drop_action"]}

        self.nepisode = 0
        self.actions = [0] * self.num_agents
        self.reset()


    def set_obs(self):
        conf = self.conf
        self.obs_mode = OBS_MODE.GLOBAL
        self.obs_plus = OBS_PLUS.SAME
        if "obs_plus" in conf:
            self.obs_plus = conf["obs_plus"]

        if "obs_mode" not in conf:
            conf["obs_mode"] = OBS_MODE.GLOBAL

        self.obs_mode = conf["obs_mode"]

        if "obs_radius" in conf:
            self.obs_radius = min(conf["obs_radius"], int(self.rows/2))
            conf["obs_radius"] = self.obs_radius

        elif self.obs_mode not in OBS_MODE.global_family():
            self.obs_radius = 1
            conf["obs_radius"] = 1

        if "obs_radius_plus" in conf:
            self.obs_radius_plus = conf["obs_radius_plus"]


        if self.obs_mode in [OBS_MODE.LOCAL_ONE_HOT, OBS_MODE.LOCAL_SUM_ONE_HOT, OBS_MODE.LOCAL_ID]:
            self.state_map = {}   # for one hot encoding 

    def set_actions(self):
        conf = self.conf
        self.action_mode = ACT_MODE.ALLOCENTRIC
        self.action_plus = ACT_PLUS.SAME
        if "action_mode" in conf: self.action_mode = conf["action_mode"]
        if "action_plus" in conf: self.action_plus = conf["action_plus"]
        self.nactions = 4
        if self.action_mode is ACT_MODE.EGOCENTRIC:
            self.nactions = 3
        if "objects" in conf or "keys" in conf or "boxes" in conf:
            self.nactions += 1
        if "drop_action" in conf and "automatic" in conf["drop_action"]:
            self.nactions -= 1
        if self.action_plus in [ACT_PLUS.NOTHING_RANDOM, ACT_PLUS.NOTHING_RANDOM_HARVEST]:
            self.nactions = 2
            if self.action_plus == ACT_PLUS.NOTHING_RANDOM_HARVEST:
                self.nactions += 1
        if self.action_plus in [ACT_PLUS.FOREST_FIRE, ACT_PLUS.NOTHING_OBS_RADIUS]:
            self.nactions += 1

    def obs_astype(self, s):
        if self.obs_mode != OBS_MODE.LOCAL_ONION:
            return s.astype(int)
        else:
            return s

    def reset_gol(self):
        conf, ini_cells = self.conf, 10
        if "update_params" in conf:
            if "ini_cells" in conf["update_params"]:
                ini_cells = conf["update_params"]["ini_cells"]
        self.add_food(ini_cells)

    def reset_ff(self):
        conf = self.conf
        conf["floor"] = []
        self.add_floor([])
        self.layers["floor"] = self.floor
        self.grid_sum = np.zeros([3, self.rows, self.cols])

        self.sample_probs, self.prob_tree_ini, self.prob_fire, self.prob_tree, self.prob_harvest = False, 0.0, 0.0, 0.0, 0.5

        if "update_params" in conf:
            update_params = conf["update_params"]
            if "prob_fire" in update_params:
                self.prob_fire = update_params["prob_fire"]
            if "prob_tree" in update_params:
                self.prob_tree = update_params["prob_tree"]
            if "prob_harvest" in update_params:
                self.prob_harvest = update_params["prob_harvest"]
            if "prob_fruit" in update_params:
                self.prob_fruit = update_params["prob_fruit"]
                if "food" not in self.reward:
                    self.reward["food"] = 1
                if "food" not in conf:
                    conf["food"] = 0

            if "prob_tree_ini" in update_params:
                self.prob_tree_ini = update_params["prob_tree_ini"]

            if "sample_probs" in update_params:
                self.sample_probs = update_params["sample_probs"]
                if self.sample_probs:
                    self.prob_tree_ini = log_rand() * update_params["prob_tree_ini"]
                    self.prob_tree = log_rand() * update_params["prob_tree"]
                    self.prob_fire = log_rand() * update_params["prob_fire"]
                    print("> Sampled: p_tree_ini", self.prob_tree_ini, "p_tree", self.prob_tree, "p_fire", self.prob_fire)
                    if hasattr(self, "stats") and "episode" in self.stats:
                        stats_epi = self.stats["episode"]
                        stats_epi["sample_probs"] = {"prob_tree_ini": self.prob_tree_ini, "prob_fire": self.prob_fire, "prob_tree": self.prob_tree}

            for i in range(self.rows):
                for j in range(self.cols):
                    if np.random.rand() <= self.prob_tree_ini:
                        self.floor[i][j] = 1

            if "fire_steps" in conf["update_params"]:
                self.fire_steps = conf["update_params"]["fire_steps"]
            if "no_fire_steps" in conf["update_params"]:
                self.no_fire_steps = conf["update_params"]["fire_steps"]

        if hasattr(self, "stats"):
            self.stats["current_num_trees"] = 0
            self.stats["current_num_fires"] = 0


    def reset(self, params={}):
        # Environment can be reseted during a conf["run"] and stats may be wanted run["stats"]
        self.layers = OrderedDict()

        conf = self.conf
        if "run" in conf:
            if "stats" in conf["run"]:
                self.stats = conf["run"]["stats"]
        elif "stats" in conf:
            self.stats = conf["stats"]

        if "floor" in conf:  self.add_floor(conf["floor"])
        if "walls" in conf:  self.add_walls(conf["walls"])

        if "food_pos" in conf: self.add_food(conf["food"], conf["food_pos"])
        elif "food" in conf:  self.add_food(conf["food"])

        if "enemies" in conf: self.add_enemies(conf["enemies"])
        if "objects" in conf:  self.add_objects(conf["objects"])

        if "blocks" in self.term_mode:
            good = self.layers["floor"] == self.layers["objects"]
            self.prev_reward = np.sum(good)

        if "tolman" in self.term_mode:

            self.prev_reward = 0

            # Adding the food state to the food information dict
            if "food_info" in self.conf:
                for food in self.conf["food_info"]:
                    rew_state = int(self.cols * food["coor"][0] + food["coor"][1])
                    food["s"] = rew_state

            # Reserving reward states
            max_num_rewards = 5

            # Creating a terminal state
            self.tolman_terminal_state = self.cols * self.rows + max_num_rewards + 1
            self.tolman_reward_triggered = False

        if "update" in conf:
            if conf["update"] == "game_of_life":
                self.reset_gol()
            if conf["update"] == "forest_fire":
                self.reset_ff()

        if "goal_sequence" in self.term_mode:
            self.current_goal = 1

        for i in range(self.num_agents):
            self.ini_agent(i, params=params)

        self.agent_xy = self.agents_xy[0]
        self.epi_elpased_time = time.time()
        self.nepisode += 1

        if self.num_agents == 1:
            # Special case for one agent NO multi_agent_mode: step is used
            # and obs is directly the observation and not a list of observations
            if not hasattr(self, "multi_agent_mode") or not self.multi_agent_mode:
                # define state that is other wise done in step
                layers_names = self.get_layers_names()
                self.state = np.stack([self.layers[k] for k in layers_names])  # was giving problems:  self.state = np.stack(list(self.layers.values()))  
                # special return for one agent when NO multi_agent_mode
                return self.obs_astype(self.generate_obs())

        return self.obs_astype(self.generate_multiobs())


    def drop_action(self):
        x, y = self.agent_xy
        reward, done = 0, False
        if "objects" not in self.layers: return reward, done
        if self.objects[x,y] == 0: return reward, done         # to drop we need to carry something

        if self.action_mode == ACT_MODE.ALLOCENTRIC:
            # implement
            return reward, done

        if "drop_action" in self.conf:
            if "automatic" in self.conf["drop_action"]:
                return reward, done

        xn, yn = self.agent_xy + self.agent_dir
        if not self.free_cell([xn, yn]): return reward, done         # free cell so drop

        self.layers["objects"][xn,yn] = self.layers["objects"][x,y]
        self.layers["objects"][x,y] = 0

        if "floor" in self.conf:
            ifloor = self.floor[xn, yn]
            obj = self.objects[xn, yn] if hasattr(self, "objects") else None
            if obj == ifloor:
                if "blocks" in self.term_mode and "empty" in self.term_mode:
                    self.objects[xn, yn] = 0   # remove object
                if "object_areas" in self.conf["floor"]:          # ants world
                    reward += 1
                    self.objects[xn,yn] += 1
                    if self.objects[xn,yn] == self.conf["floor"]["object_areas"]:  # reached end area
                        self.objects[xn,yn] = 0     # remove object

            if obj != ifloor:
                if "blocks" in self.term_mode or "object_areas" in self.conf["floor"]:
                    reward -= 0.05


        return reward, done

    def is_move_valid(self, move):
        x, y = self.agent_xy
        xn, yn = self.agent_xy + move
        if self.torus:
            xn, yn = xn % self.rows, yn % self.cols
        valid_move = xn >= 0 and yn >= 0 and xn < self.rows and yn < self.cols
        if not valid_move:
            return False

        valid_move = valid_move and self.layers["agents"][xn, yn] == 0
        if "walls" in self.layers:
            valid_move = valid_move and self.layers["walls"][xn, yn] == 0
        if "objects" in self.layers:
            valid_obj = self.layers["objects"][xn, yn] > 0 and self.layers["objects"][x, y] == 0
            valid_obj = valid_obj or self.layers["objects"][xn, yn] == 0
            valid_move = valid_move and valid_obj
        return valid_move

    def move_enemies(self):
        enemy_reward = 0
        done = False

        for enemy in self.enemy_database:
            #First move each enemy:
            if enemy["mobility"] == "random":
                stay_in_position = False
                new_move = self.moves[np.random.randint(0,3)]
                x,y = enemy["position"]
                nx, ny = enemy["position"]+new_move

                if self.torus: nx,ny = nx %self.rows, ny %self.cols
                elif nx == self.rows or ny == self.cols or nx < 0 or ny < 0:
                    nx,ny = x,y
                    stay_in_position = True

                if not stay_in_position:
                    self.layers["enemies"][nx, ny] = self.layers["enemies"][x, y]
                    self.layers["enemies"][x,y] = 0
                    if enemy["eats_food"] and self.layers["food"][nx,ny]==1: self.layers["food"][nx, ny] = 0

                enemy["position"] = [nx,ny]

            if enemy["attack_range"] == 0 and enemy["position"] == list(self.agent_xy):
                enemy_reward -= enemy["attack_damage"]
                if enemy["lethal"]: done = True

            if enemy["attack_range"] > 0:
                #Does not work with torus!!!
                x_min = max(enemy["position"][0] - enemy["attack_range"], 0)
                x_max = min(enemy["position"][0] + enemy["attack_range"], self.n-1)
                y_min = max(enemy["position"][1] - enemy["attack_range"], 0)
                y_max = min(enemy["position"][1] + enemy["attack_range"], self.n-1)
                x_range = list(range(x_min, x_max+1))
                y_range = list(range(y_min, y_max+1))
                danger_zone = list(itertools.product(x_range, y_range))

                if tuple(self.agent_xy) in danger_zone:
                    enemy_reward -= enemy["attack_damage"]
                    if enemy["lethal"]: done = True

        return enemy_reward, done



    def move_agent(self, move):
        reward, done = 0.0, False
        x, y = self.agent_xy
        xn, yn = self.agent_xy + move
        if self.torus:
            xn, yn = xn % self.rows, yn % self.cols

        self.layers["agents"][xn, yn] = self.layers["agents"][x, y]
        self.layers["agents"][x, y] = 0
        self.agent_xy = np.array([xn, yn])

        if "objects" in self.layers:
            obj_old_pos = self.layers["objects"][x, y]
            obj_new_pos = self.layers["objects"][xn, yn]
            if obj_old_pos == 0 and obj_new_pos > 0:
                reward += 10 * abs(self.reward["step"])
            if obj_old_pos > 0:
                self.layers["objects"][x, y] = 0
                self.layers["objects"][xn, yn] = obj_old_pos
                if "drop_action" in self.conf and "automatic" in self.conf["drop_action"]:
                    if obj_old_pos == self.floor[xn, yn]:
                        self.layers["objects"][xn, yn] = 0

        if "food" in self.layers:
            bLogReward = hasattr(self, "stats")
            if "food_info" in self.conf:
                for f in self.conf["food_info"]:
                    if (f["coor"][0], f["coor"][1]) == (xn, yn):
                        reward += f["rew"]
            else:
                reward += self.layers["food"][xn, yn]
                if bLogReward: self.stats["episode"]["each_agent"][self.agent_i]["rewards"]["fruit"] += 1

            self.layers["food"][xn, yn] = 0

        return reward, done

    def pick_action(self):
        x,y = self.agent_xy
        return 0, False

    def harvest_at_pos(self,x,y,rx=0,ry=0):
        floor = self.floor
        floor_old = self.floor_old

        hx, hy = (x + rx) % self.rows, (y + ry) % self.cols
        if floor[hx, hy] == 1 or floor_old[hx, hy] == 1:
            floor[hx, hy] = 0
            if hasattr(self, "food"):
                self.food[hx, hy] = 0

    def harvest_in_front(self):
        xn, yn = self.agent_xy + self.agent_dir
        self.harvest_at_pos(xn,yn)  # harvest single cell in front
        #dx, dy = self.agent_dir
        #self.harvest_at_pos(xn,yn,-dy,dx)
        #self.harvest_at_pos(xn,yn,dy,-dx)


    def harvest(self):
        x,y = self.agent_xy
        r = self.obs_radius
        my_range = np.arange(-r, r + 1, 1)
        for rx, ry in itertools.product(my_range, my_range):
            if rx != 0 or ry != 0:
                if np.random.rand() <= self.prob_harvest:
                    self.harvest_at_pos(x, y, rx, ry)

    def harvest_line(self):
        x,y = self.agent_xy
        r = self.obs_radius
        my_range = np.arange(-r, r + 1, 1)
        for rx in my_range:
            self.harvest_at_pos(x, y, rx, my_range[0])
            self.harvest_at_pos(x, y, rx, my_range[-1])
        for ry in my_range:
            self.harvest_at_pos(x, y, my_range[0], ry)
            self.harvest_at_pos(x, y, my_range[-1], ry)

    def harvest_line_within(self):
        x,y = self.agent_xy
        r = self.obs_radius
        #my_range = np.arange(-r+1, r, 1)
        my_range = np.arange(-1, 2, 1)
        for rx in my_range:
            self.harvest_at_pos(x, y, rx, my_range[0])
            self.harvest_at_pos(x, y, rx, my_range[-1])
        for ry in my_range:
            self.harvest_at_pos(x, y, my_range[0], ry)
            self.harvest_at_pos(x, y, my_range[-1], ry)


    def change_floor(self):
        if "update" in self.conf and self.conf["update"] == "forest_fire":   # harvest
            if self.action_mode == ACT_MODE.EGOCENTRIC:
                self.harvest_in_front()
            else:
                #self.harvest()
                #self.harvest_line()
                self.harvest_line_within()

        return 0, False


    def move_UpDownLeftRight(self, a):
        reward, done = 0.0, False
        if a < 4:
            move = self.moves[a]        # [0, -1],[0, 1],[-1, 0],[1, 0])
            if self.is_move_valid(move):
                reward, done = self.move_agent(move)
        return reward, done

    def move_FwdTurn(self, a):
        reward, done = 0.0, False
        x, y = self.agent_xy
        dx, dy = self.agent_dir
        if a in [0,1]:
            if a == 0:
                self.agent_dir = np.array([-dy, dx])  # order: 1:[0,1] 2:[-1,0] 3:[0,-1] 4:[1,0]
                self.agents[x, y] += 1
                if self.agents[x, y] >= 5:
                    self.agents[x, y] = 1
            elif a == 1:
                self.agent_dir = np.array([dy, -dx])  
                self.agents[x, y] -= 1
                if (self.agents[x, y] == 0):
                    self.agents[x, y] = 4
        elif a == 2:
            move = self.agent_dir
            if self.is_move_valid(move):
                reward, done = self.move_agent(move)
                if "move" in self.reward:
                    reward += self.reward["move"]
        return reward, done

    def act_plus(self, a):
        reward, done = 0.0, False
        if self.action_mode == ACT_MODE.ALLOCENTRIC:
            if a == 4:
                if self.action_plus == ACT_PLUS.FOREST_FIRE:
                    reward, done = self.change_floor()
                else:
                    reward, done = self.pick_action()
            elif a == 5:
                reward, done = self.drop_action()

        if self.action_mode == ACT_MODE.EGOCENTRIC:
            if a == 3:
                if self.action_plus == ACT_PLUS.FOREST_FIRE:
                    reward, done = self.change_floor()
                else:
                    reward, done = self.drop_action()
        return reward, done


    def generate_state_array(self, layers_names):
        layers_list = []
        for k in layers_names:   # state order in layers names
            if k == "agents" and self.action_mode == ACT_MODE.EGOCENTRIC:
                d = self.get_orientation()
                layers_list.append(np.array(self.agents_orientation[d-1]))
            else:
                layers_list.append(self.layers[k])

        return np.stack(layers_list)


    def get_layers_names(self):
        layers_names = list(self.layers.keys())
        bOneAgentNotGlobal = self.obs_mode not in OBS_MODE.global_family() and self.num_agents <= 1
        if self.obs_plus in [OBS_PLUS.NO_AGENTS] or bOneAgentNotGlobal:
            if "agents" in layers_names:
                layers_names.remove("agents")
        if hasattr(self, "obs_radius") and self.obs_radius < 1:   # we cannot be in a wall
            layers_names.remove("walls")
        return layers_names


    # Precomputed Ego Agents Layers
    def set_ego_layers(self):
        if self.action_mode != ACT_MODE.EGOCENTRIC or self.num_agents < 2:
            return
        self.agents_orientation = []
        for d in [1,2,3,4]:
            a = np.array(self.layers["agents"])
            xs, ys = np.nonzero(a)
            for x,y in zip(xs,ys):
                a[x][y] = (5 - d + a[x][y])
                if a[x][y] > 4:
                    a[x][y] -= 4
            self.agents_orientation.append(a)

    # Ego Agent Layer computed on the fly
    def set_ego_agents(self, obs):
        layers_names = self.get_layers_names()
        if "agents" not in layers_names or self.action_mode != ACT_MODE.EGOCENTRIC or self.num_agents < 2:
            return obs

        i_agent_layer = layers_names.index("agents")

        ox, oy = obs[0].shape
        ox, oy = int(ox/2), int(oy/2)
        d = int(obs[i_agent_layer][ox][oy])
        obs[i_agent_layer][ox][oy] = 0        # agent is obviously in the center

        xs, ys = np.nonzero(obs[i_agent_layer])
        for x,y in zip(xs,ys):
            obs[i_agent_layer][x][y] = (5 - d + obs[i_agent_layer][x][y])
            if obs[i_agent_layer][x][y] > 4:
                obs[i_agent_layer][x][y] -= 4

        return obs


    def sum_radius(self, obs, r=1):
        sumv = []
        for i,o in enumerate(obs):
            c = int(o.shape[0]/2)
            sumv += [int(o[c,c])]
        if r>0:
            for i,o in enumerate(obs):
                for v in [1,2]:
                    x = 0 if o[c,c] != v else 1
                    s = np.sum(o == v) - x
                    sumv += [s]

        return np.array(sumv)

    def sum_radius_moore(self, obs, r=1):
        sumv = []
        c = int(obs[0].shape[0] / 2)  # obs shape is always squared (nxn) and odd n%2 == 1
        for i,o in enumerate(obs):
            sumv += [o[c,c]]
        if r>0:
            sumv += [o[c+1,c]]
            sumv += [o[c,c+1]]
            sumv += [o[c+1,c+1]]
            sumv += [o[c+1,c-1]]
            sumv += [o[c-1,c+1]]
            sumv += [o[c-1,c]]
            sumv += [o[c,c-1]]
            sumv += [o[c-1,c-1]]

        if r>1:
            for i,o in enumerate(obs):
                for v in [1]:
                    s1 = np.sum(o[c+1:,:]==v)
                    s2 = np.sum(o[:c,:]==v)
                    s3 = np.sum(o[:,c+1:]==v)
                    s4 = np.sum(o[:,:c]==v)
                    sumv += [s1,s2,s3,s4]

        return np.array(sumv).astype(int)


    def sum_radius_onion(self, obs, r=1):
        c = int(obs[0].shape[0] / 2)  # needs to be the agent one: obs shape is always squared (nxn) and odd n%2 == 1
        values = [1, 2]  # values to track in the observation / for forest fire 1 (tree) and 2 (fire)
        l = len(values)
        obs_sum = [0] * l
        sub_sum = np.array(obs_sum)
        for o in obs:
            for i in range(r + 1):  # i index for radius 0..r
                r_sum = []
                ii = r - i
                for v in values:
                    if i == 0:
                        r_sum += [np.sum(o[c, c] == v)]
                    elif i == r:
                        r_sum += [int(np.sum(o == v))]
                    else:
                        r_sum += [int(np.sum(o[ii:-ii, ii:-ii] == v))]

                sub_sum += np.array(obs_sum[-l:])
                obs_sum += list(np.array(r_sum) - sub_sum)

        obs_sum = np.array(obs_sum[l:]).astype(int)
        return obs_sum


    def radius_onion(self, s):
        r = self.rows-1-self.obs_radius
        subs = s.astype(float)
        gamma = 0.999
        expg = gamma ** r
        for _ in range(r):
            subrs = []
            for sub in subs:
                subr = sub[1:-1, 1:-1]
                cr = subr.shape[0]
                for _ in range(4):
                    for t in range(cr):
                        subr[t, 0] += expg * np.mean(sub[t:t + 3, 0])
                    subr = np.rot90(subr)
                    sub = np.rot90(sub)
                subrs.append(subr)
            expg /= gamma
            subs = subrs
        return np.array(subs)


    def one_hot_map(self, obs, r=1, max_states=100):
        o = tuple(obs.flatten())
        if o not in self.state_map:
            if len(self.state_map) >= max_states-1:
                print("ONE_HOT: Reached", max_states, "stored states -------------------------------------------------")
            self.state_map[o] = len(self.state_map) % max_states
        return o


    def one_hot_radius(self, obs, r=1, max_states=100):
        o = self.one_hot_map(obs, r, max_states)
        obs = np.zeros(max_states)
        obs[ self.state_map[o] ] = 1
        return obs

    def one_hot_id(self, obs, r=1, max_states=100):
        o = self.one_hot_map(obs, r, max_states)
        return np.array([self.state_map[o]])

    def one_hot_sum_radius(self, obs, r=1, max_states=100):
        obs = self.sum_radius(obs, r=self.obs_radius)
        return self.one_hot_radius(obs, r=r, max_states=max_states)

    def set_agent_position(self, xy_new, orientation=None, incr_radius=None):
        x,y = self.agent_xy
        x_n, y_n = xy_new

        if not self.free_cell(xy_new, check_layers=["walls", "food", "objects"]):
            return None

        self.agents[x, y] = 0
        self.agents[x_n, y_n] = 1
        self.agent_xy = xy_new
        if self.action_mode is ACT_MODE.EGOCENTRIC:
            dirs = [[0, 1], [-1, 0], [0, -1], [1, 0]]
            if orientation is not None and orientation in [1, 2, 3, 4]:
                self.agents[x_n, y_n] = orientation
                self.agent_dir = dirs[orientation-1]
            else:
                self.agent_dir = dirs[np.random.randint(4)]

        if incr_radius is not None:
            self.obs_radius += incr_radius

        layers_names = self.get_layers_names()
        self.state = np.stack([self.layers[k] for k in layers_names])
        obs = self.obs_astype(self.generate_obs())

        if incr_radius is not None:
            self.obs_radius -= incr_radius

        return obs

    def set_agent_local(self, i):
        self.agent_i = i
        self.agent_xy = self.agents_xy[i]
        self.agent_xy_old = self.agent_xy.copy()
        self.action = self.actions[i]
        if self.action_mode is ACT_MODE.EGOCENTRIC:
            self.agent_dir = self.agents_dir[i]

    def get_agent_local(self, i):
        self.agents_xy[i] = self.agent_xy
        if self.action_mode is ACT_MODE.EGOCENTRIC:
            self.agents_dir[i] = self.agent_dir

    def actions_do(self, a):
        r_plus = 1 if not hasattr(self, "obs_radius_plus") else self.obs_radius_plus
        if self.action_plus in [ACT_PLUS.NOTHING_OBS_RADIUS]:
            if a == 4 and self.action_mode == ACT_MODE.ALLOCENTRIC:
                self.obs_radius += r_plus
            if a == 3 and self.action_mode == ACT_MODE.EGOCENTRIC:
                self.obs_radius += r_plus

    def actions_undo(self, a):
        r_plus = 1 if not hasattr(self, "obs_radius_plus") else self.obs_radius_plus
        if self.action_plus in [ACT_PLUS.NOTHING_OBS_RADIUS]:
            if a == 4 and self.action_mode == ACT_MODE.ALLOCENTRIC:
                self.obs_radius -= r_plus
            if a == 3 and self.action_mode == ACT_MODE.EGOCENTRIC:
                self.obs_radius -= r_plus

    def add_history(self, i, s, a, r):            # obs of an agent
        """ s may be reversed (obs_r,obs_r,ch) or not (ch, obs_r, obs_r)
        """
        if not hasattr(self, "obs_mem"):
            return s

        if len(s.shape) == 2:              # when third layer (agents or additional layer) is not there, we add one
            s = np.expand_dims(s, axis=2)

        bReverse = True if hasattr(self, "obs_shape") and self.obs_shape is OBS_SHAPE.REVERSE else False

        obs_r = s.shape[1]
        states_list = [h[0] for h in self.history[i]]
        actions_list = [h[1] * np.ones_like(s)  for h in self.history[i]]
        rew_list = [h[2] * np.ones_like(s) for h in self.history[i]]

        for _ in range(self.obs_mem_h - len(self.history[0])):
            if self.obs_mem in [OBS_MEM.STATE_MEM, OBS_MEM.STATE_MEAN, OBS_MEM.STATE_ACTION_MEM, OBS_MEM.SAR_MEM]:
                states_list += [np.zeros_like(s)]

            if self.obs_mem in [OBS_MEM.ACTION_MEAN, OBS_MEM.ACTION_MEM, OBS_MEM.STATE_ACTION_MEM, OBS_MEM.SAR_MEM]:
                actions_list += [np.zeros_like(s)]

        mem_list = [s]

        axis = 0   # obs will have shape: (ch+h*ch, rows,cols)
        if bReverse:
            axis = 2  # obs will have shape: (rows, cols, ch + h*ch)

        if self.obs_mem in [OBS_MEM.STATE_MEM, OBS_MEM.STATE_ACTION_MEM, OBS_MEM.SAR_MEM]:
            mem_list += states_list

        if self.obs_mem in [OBS_MEM.ACTION_MEM, OBS_MEM.STATE_ACTION_MEM, OBS_MEM.SAR_MEM]:
            #actions_list += [a * np.ones((obs_r, obs_r))]
            mem_list += actions_list

        if self.obs_mem in [OBS_MEM.STATE_MEAN]:
            states_sum = np.sum( np.sum(states_list, axis=0), axis=axis)
            states_sum = np.expand_dims(states_sum, axis=axis)
            states_sum = [list(states_sum)]
            mem_list += states_sum

        if self.obs_mem in [OBS_MEM.ACTION_MEAN]:
            actions_sum = [list(np.sum(actions_list, axis=0))]
            mem_list += actions_sum

        obs = np.concatenate(mem_list, axis=axis)  # stack along the relevant dimension (depends on reversing)
        return obs

    def reverse_obs(self, obs):
        if not hasattr(self,"obs_shape"):
            return obs

        if self.obs_shape is OBS_SHAPE.REVERSE:
            sh = obs.shape
            obs = np.rollaxis(obs, 0, len(sh))   # roll axis 0 to position 3
        return obs


    def make_agents_obs_layer(self):
        if self.obs_plus not in [OBS_PLUS.AGENTS_OBS] or self.num_agents <= 1:
            return

        self.layers["agents_obs"] = np.zeros([self.rows, self.cols])
        agents_obs = self.layers["agents_obs"]

        for xy in self.agents_xy:
            _, _, cx, cy, shift = pad_shift(xy, self.rows_cols)
            agents_obs = np.roll(agents_obs, shift, axis=(0, 1))
            v = 0.1
            r = self.obs_radius
            l = [int(cx - r), int(cx + r + 1), int(cy - r), int(cy + r + 1)]
            agents_obs[l[0]:l[1], l[2]:l[3]] += v
            agents_obs = np.roll(agents_obs, -shift, axis=(0, 1))

        self.layers["agents_obs"] = agents_obs



    def generate_multiobs(self, a_vector=None):
        self.make_agents_obs_layer()

        # Pre-computation of ego agent layers for every orientation
        # self.set_ego_layers()
        # state = self.generate_state_array(layers_names)

        layers_names = self.get_layers_names()
        self.state = np.stack([self.layers[k] for k in layers_names])

        if a_vector is None:
            a_vector = np.zeros(self.num_agents)

        states = []
        for i, a in enumerate(a_vector):
            self.set_agent_local(i)
            self.actions_do(a)        # for special actions that need undo

            obs = self.generate_obs()
            obs = self.reverse_obs(obs)  # maybe reversed it for keras, or...

            states.append(obs)
            self.actions_undo(a)

        self.obs = np.array(states)
        return self.obs


    def get_orientation(self):
        x,y = self.agent_xy
        return int(self.agents[x, y])

    def get_global_id(self):
        return int(self.cols * self.agent_xy[0] + self.agent_xy[1])

    def get_global_coordinate(self):
        c = []
        for layer in self.state:
            c.append(np.array(layer.nonzero())) # Append all non-zero/feature coordinates
            d = np.concatenate(c, axis=0).flatten()
        return d


    def get_position_from_global(self, id):
        return int(id / self.cols), id % self.cols

    def get_global_id_ego(self):
        return [self.get_global_id(),  self.get_orientation()]

    def get_state_map(self):
        return self.layers

    def add_channels(self, l, obs_ch, n_values):
        for i in range(n_values):
            new_ch = np.zeros(obs_ch.shape)
            new_ch[ obs_ch == i+1 ] = 1
            l.append(new_ch)

    def add_greater_zero(self, l, obs_ch):
        new_ch = np.zeros(obs_ch.shape)
        new_ch[ obs_ch > 0 ] = 1
        l.append(new_ch)


    def undo_one_hot(self, obs):
        # TO DO
        layers_names = self.get_layers_names()
        ch_list = []
        return np.stack(ch_list)


    def set_one_hot(self, obs):
        layers_names = self.get_layers_names()
        ch_list = []
        for i, l_name in enumerate(layers_names):  # state order in layers names
            obs_ch = obs[i,:,:]
            if l_name == "floor":
                self.add_channels(ch_list, obs_ch, 2)   # adding 2 channels || ch_list: current ch list / obs_ch: current channel
            elif l_name == "agents":
                self.add_greater_zero(ch_list, obs_ch)
                self.add_channels(ch_list, obs_ch, 4)   # adding 4 channels
            else:
                ch_list.append(obs_ch)
        return np.stack(ch_list)

    def generate_obs(self):
        if self.obs_mode is OBS_MODE.GLOBAL:                  # first serve the GLOBAL variations
            return self.state
        elif self.obs_mode is OBS_MODE.GLOBAL_ID:
            return np.array([self.get_global_id()])
        elif self.obs_mode is OBS_MODE.GLOBAL_COORDINATE:
            return np.array([self.get_global_coordinate()])
        elif self.obs_mode is OBS_MODE.GLOBAL_ID_EGO:
            return np.array(self.get_global_id_ego())
        elif self.obs_mode is OBS_MODE.GLOBAL_CENTER_PAD:
            return gridworld_center_view(self.state, self.agent_xy, self.obs_mode, padding=True, torus=self.torus)
        elif self.obs_mode is OBS_MODE.GLOBAL_CENTER_WRAP:
            return gridworld_center_view(self.state, self.agent_xy, self.obs_mode, padding=False, torus=self.torus)

        # **************************  Lets continue with LOCAL observations
        dir = None
        if self.action_mode == ACT_MODE.EGOCENTRIC:
            dir = int(self.layers["agents"][self.agent_xy[0], self.agent_xy[1]])

        (i,j),r  = self.agent_xy, self.obs_radius

        if self.obs_mode == OBS_MODE.LOCAL_ONION:
            obs = center_crop(i, j, self.rows-1, O=self.state, dir=dir, padding=True, torus=self.torus)
            return self.radius_onion(obs)

        obs = center_crop(i, j, r, O=self.state, dir=dir, torus=self.torus)

        if self.obs_plus not in [OBS_PLUS.NO_ORIENTATION]:
            obs = self.set_ego_agents(obs)                    # on the fly agent orientation calculation

        if self.obs_plus in [OBS_PLUS.ONE_HOT_BOTH]:          # one hot encoding of both floor, agents and orientation
            obs = self.set_one_hot(obs)

        if self.obs_mode == OBS_MODE.LOCAL:
            return obs

        elif self.obs_mode == OBS_MODE.LOCAL_SUM:
            return self.sum_radius(obs, r=self.obs_radius)

        elif self.obs_mode == OBS_MODE.LOCAL_ONE_HOT:
            max_states = 100 * (self.obs_radius * self.obs_radius + 1)
            return self.one_hot_radius(obs, r=self.obs_radius, max_states=max_states)

        elif self.obs_mode == OBS_MODE.LOCAL_SUM_ONE_HOT:
            max_states = 100 * (self.obs_radius * self.obs_radius + 1)
            return self.one_hot_sum_radius(obs, r=self.obs_radius, max_states=max_states)

        elif self.obs_mode == OBS_MODE.LOCAL_ID:
            max_states = 2 * 100 * (self.obs_radius * self.obs_radius + 1)
            return self.one_hot_id(obs, r=self.obs_radius, max_states=max_states)

        elif self.obs_mode == OBS_MODE.LOCAL_SUM_MOORE:
            return self.sum_radius_moore(obs, r=self.obs_radius)

        elif self.obs_mode == OBS_MODE.LOCAL_SUM_ONION:
            return self.sum_radius_onion(obs, r=self.obs_radius)


    def check_term(self, reward=0, done=False):
        if hasattr(self, "agent_xy"):
            x, y = self.agent_xy  # think it as rows,cols: x is in the y axis

        if "goal_sequence" in self.term_mode:
            if self.layers["floor"][x,y] == self.current_goal:
                self.layers["floor"][x,y] = 0
                self.current_goal += 1
                reward += 1
            done = done or self.current_goal == 4
            if self.current_goal == 4:
                reward += 9

        if "empty" in self.term_mode:
            if "objects" in self.layers:
                done = done or np.count_nonzero(self.layers["objects"]) == 0
            if "food" in self.layers:
                done = done or np.count_nonzero(self.layers["food"]) == 0

        if "survival" in self.term_mode:
            if "food" in self.layers:
                done = done or np.count_nonzero(self.layers["food"]) == 0
            if "enemy" in self.layers:
                pass

        if "timeout" in self.term_mode:
            # Done flag only when time runs out
            pass

        if "tolman" in self.term_mode:
            # A flag informing the environment that tolman tasks are running.
            # Triggers some adjustments/additions to various functions
            pass

        if "floor_exit" in self.term_mode:
            done = done or self.layers["floor"][x,y] == self.floor_max
            if(done):
                reward += self.reward["exit"]

        if "single_step" in self.term_mode:
            done = True

        if "gid_states" in self.term_mode:
            gid_states = self.term_mode["gid_states"]
            gid = self.get_global_id_ego()
            if tuple(gid) in gid_states:
                done = True
                reward += gid_states[tuple(gid)]

        if "update" in self.conf and self.conf["update"] == "forest_fire" and self.num_agents > 0:
            bLogReward = hasattr(self, "stats")

            x_old, y_old = self.agent_xy_old
            if self.layers["floor"][x, y] == 2 or (self.floor[x_old, y_old] == 2 and self.floor_old[x, y] == 2):
                reward += self.reward["fire"]
                if bLogReward: self.stats["episode"]["each_agent"][self.agent_i]["rewards"]["fire"] += 1

            if self.layers["floor"][x, y] in [1,3]:
                reward += self.reward["tree"]
                if bLogReward: self.stats["episode"]["each_agent"][self.agent_i]["rewards"]["tree"] += 1

        if "ratio_trees" in self.term_mode and hasattr(self, "current_num_trees"):
            p = self.term_mode["ratio_trees"] * self.rows * self.cols
            done = done or self.current_num_trees <= p


        if "counting" in self.term_mode:
            i = np.sum(self.layers["floor"][:, 0])
            done = done or (y == 1 and x == i-1)
            if done:
                reward += 1
            elif y == 1:
                reward -= 1

        if "blocks" in self.term_mode:
            good = self.layers["floor"] == self.layers["objects"]
            done = done or good.all()
            r = np.sum(good)
            if r > self.prev_reward:
                reward += r - self.prev_reward
                self.prev_reward = r

        if "harlow" in self.term_mode:
            discrimination_mode = 0 # Object_Quality or Right_Position

            self.nepisode % 10 < 5
            if self.floor[x,y] == 4:
                reward -= 3

            elif self.floor[x, y] == 1:
                self.floor[x, y] = 0
                self.layers["floor"] = self.floor
                if bReward:
                    reward += 10

            elif self.floor[x,y] == 3:
                self.floor[x, y] = 0
                self.layers["floor"] = self.floor
                if not bReward:
                    reward += 10

            done = done or self.floor[x,y] == 2

        return reward, done


    def process_action(self,a):
        reward, done = 0, False
        if self.action_plus in [ACT_PLUS.NOTHING_OBS_RADIUS]:       # increase obs radius: needs check
            if a == 4 and self.action_mode == ACT_MODE.ALLOCENTRIC:
                a = -1
            if a == 3 and self.action_mode == ACT_MODE.EGOCENTRIC:
                a = -1
        if self.action_plus in [ACT_PLUS.NOTHING_RANDOM, ACT_PLUS.NOTHING_RANDOM_HARVEST]:
            if a == 1:  # do random action
                if self.action_mode == ACT_MODE.ALLOCENTRIC:
                    a = np.random.randint(4)
                elif self.action_mode == ACT_MODE.EGOCENTRIC:
                    a = np.random.randint(3)
            else:
                if a == 2:  self.change_floor() # harvest
                a = -1         # do nothing for ACT_MODE: harvest or do nothing a=0

        if a >= 0 and self.action_mode == ACT_MODE.ALLOCENTRIC:
            reward, done = self.move_UpDownLeftRight(a) if a < 4 else self.act_plus(a)
        if a >= 0 and self.action_mode == ACT_MODE.EGOCENTRIC:
            reward, done = self.move_FwdTurn(a) if a < 3 else self.act_plus(a)

        if "enemies" in self.conf:
            enemy_reward, enemy_done = self.move_enemies()
            reward += enemy_reward
            done = done or enemy_done

        return reward, done

    def step(self, a):
        # Normal Simple - step for one single agent
        # Special case for one agent NO multi_agent_mode
        # and obs is directly the observation and not a list of observations

        if self.num_agents > 1:
            print("FinalGrid WARNING: step called with more than one agent")

        self.actions = [a]
        self.agent_xy_old = self.agent_xy.copy()
        self.update()

        reward, done = self.process_action(a)
        reward, done = self.check_term(reward, done)
        reward += self.reward["step"]  # usually a negative reward or 0

        self.actions_do(a)

        layers_names = self.get_layers_names()
        self.state = np.stack([self.layers[k] for k in layers_names])

        obs = self.obs_astype(self.generate_obs())
        self.actions_undo(a)

        self.obs = obs
        obs, reward, done, infos = self.tolman_check(obs, reward, done)
        return obs, reward, done, infos


    def tolman_check(self, obs, reward, done, infos=[]):
        if "tolman" not in self.conf["term_mode"]:
            return obs, reward, done, infos

        # Tolman specific features/requirements:
        # Adding extra reward transition state for tolman tasks
        if self.tolman_reward_triggered:
            # If reward is triggered last step, reward is given now & forced to terminal state
            done = True
            reward = self.prev_reward
            obs = self.tolman_terminal_state
        else:
            # If reward is triggered, check which reward state to transition by checking reward ids
            if reward != 0: # If there is a reward
                done = False # Do not terminate the episode
                self.prev_reward = reward # Save the reward
                for f in self.conf["food_info"]: # Check which reward the agent found
                    if f["coor"] == tuple(self.agent_xy):
                        # Transfer the agent to the corresponding reward's own state instead of the grid position:
                        obs = self.rows * self.cols + f["id"]
                # Remember that the previous steps were executed, so next step the reward is given and the episode terminated
                self.tolman_reward_triggered = True
                reward = 0

        # Passing on reward position information to correct plotting
        if "food_info" in self.conf:
            for data in self.conf["food_info"]:
                infos.append(data)

        return obs, reward, done, infos

    def step_one_multi(self, a):
        reward, done = self.process_action(a)
        reward, done = self.check_term(reward, done)
        reward += self.reward["step"]  # usually a negative reward or 0

        if hasattr(self, "stats") and hasattr(self, "floor"):  # update (forest gol) has been done before step_one_multi
            xy = self.agent_xy
            if self.floor:
                c = int(self.floor[xy[0]][xy[1]])
                stats_epi = self.stats["episode"]
                stats_epi["floor_occupation"][c] += 1
                stats_epi["each_agent"][self.agent_i]["floor_occupation"][c] += 1
                #print("agent_i", self.agent_i, "    ", stats_epi["each_agent"][self.agent_i]["floor_occupation"])

        return None, reward, done, None

    def step_zero_multi(self):
        if self.num_agents != 0:
            print("FinalGrid WARNING: step_zero called with agents")
        self.update()
        _, done = self.check_term()
        return None, None, done, None

    def generate_agents_list(self):
        if self.multiagent_order == MULTIAGENT_ORDER.RANDOM:
            agents_list = np.random.permutation(self.num_agents)
        elif self.multiagent_order == MULTIAGENT_ORDER.FIXED:
            agents_list = np.arange(self.num_agents)
        elif self.multiagent_order == MULTIAGENT_ORDER.ROTATING:
            if not hasattr(self, "agents_list"):
                self.agents_list = np.arange(self.num_agents)
            else:
                self.agents_list = np.roll(self.agents_list, 1)
                agents_list = self.agents_list
        return agents_list


    def steps(self, a_vector):
        # Multi Agent Mode - step for one single agent is "step_one_multi"
        # An observation is a list of observations of every agent
        # we include the 0 and 1 agent case

        if self.num_agents == 0:
            return self.step_zero_multi()

        self.agents_xy_old = self.agents_xy.copy()
        self.actions = a_vector.copy()

        self.update()

        done, rews = False, np.zeros(self.num_agents)

        agents_list = self.generate_agents_list()
        for i in agents_list:
            self.set_agent_local(i)
            _, r, d, _ = self.step_one_multi(a_vector[i])
            self.get_agent_local(i)
            done = done or d
            rews[i] = r

        states = self.generate_multiobs(a_vector)      # consider conversion depending of state type
        return states, rews, done, []

    def free_cell(self, xy, check_layers = ["agents", "walls", "food", "objects", "enemies"]):
        x, y = xy  
        free = x >= 0 and y >= 0 and x < self.rows and y < self.cols
        if not free: return False
        for lstr in check_layers:
                if lstr in self.layers:
                    free = free and self.layers[lstr][x, y] == 0
        return free

    def find_free_cell(self, check_layers=["agents", "walls", "food", "objects", "floor", "enemies"], params={}):
        if "set_ini_pos" in params:
            if self.num_agents > 1:
                print("WARNING set_ini_pos called with", self.num_agents, "agents")

            floor_array = np.array(self.layers["floor"]) if "floor" in self.layers else np.zeros([self.rows, self.cols])
            walls_array = np.array(self.layers["walls"]) if "walls" in self.layers else np.zeros([self.rows, self.cols])
            xs = np.where(floor_array + walls_array == 0)[0]
            ys = np.where(floor_array + walls_array == 0)[1]

            p = params["set_ini_pos"]
            if type(p) in [np.float64, float]:
                one = 1 - int(np.random.rand() < p)
                x, y = xs[one], ys[one]
            elif type(p) in [int, np.int64, np.int32]:
                x, y = xs[p], ys[p]
            elif type(p) in [np.ndarray, list]:
                x, y = p

            return x, y

        valid = False
        while not valid:
            x = np.random.randint(self.rows)
            y = np.random.randint(self.cols)
            valid = True
            for lstr in check_layers:
                if (lstr in self.layers):
                    valid = valid and self.layers[lstr][x, y] == 0

                if valid and "enemies" in self.layers: #Prevents agent spawning in the attack range of enemies
                    for enemy in self.enemy_database:
                        if enemy["attack_range"] > 0:
                            x_min = max(enemy["position"][0]-enemy["attack_range"], 0)
                            x_max = min(enemy["position"][0]+enemy["attack_range"], self.n-1)
                            y_min = max(enemy["position"][1]-enemy["attack_range"], 0)
                            y_max = min(enemy["position"][1]+enemy["attack_range"], self.n-1)
                            x_range = list(range(x_min,x_max+1))
                            y_range = list(range(y_min,y_max+1))
                            danger_zone = list(itertools.product(x_range,y_range))

                            if len(danger_zone) >= self.n**2: self.reset() #Resets the environment if there is no suitable position
                            valid = valid and tuple([x,y]) not in danger_zone
        return x, y

    def ini_agent(self, i=0, params={}):
        if "agents" not in self.layers:
            self.agents = np.zeros([self.rows, self.cols])
            self.agents_xy = self.num_agents * [0]
            if self.action_mode is ACT_MODE.EGOCENTRIC:
                self.agents_dir = self.num_agents*[1,0]
            self.layers["agents"] = self.agents

        check_layers = ["agents", "walls", "food", "objects", "floor", "enemies"]
        if "update" in self.conf and "forest_fire" in self.conf["update"]:
            check_layers = ["agents", "walls", "food", "objects"]

        if self.rows * self.cols <= 9:
            if "blocks" in self.term_mode:
                check_layers = ["agents", "objects"]

        x, y = self.find_free_cell(check_layers=check_layers, params=params)
        if "counting" in self.term_mode:
            x,y = 0,0

        if self.agent_ini_pos is not None:
            x,y = self.agent_ini_pos[0], self.agent_ini_pos[1]

        self.agents_xy[i] = np.array([x, y])
        if self.action_mode is ACT_MODE.EGOCENTRIC:
            dirs = [[0, 1], [-1, 0], [0, -1], [1, 0]]     # agent initialization special for forest fire: Random
            orient = 0
            if "update" in self.conf and self.conf["update"] == "forest_fire":
                orient = np.random.randint(4)
            dir =  dirs[orient]
            self.agent_dir = dir
            self.agents_dir[i] = dir
            self.agents[x, y] = orient+1
        else:
            self.agents[x, y] = 1


    def add_floor(self, mode):
        self.floor = np.zeros([self.rows, self.cols])
        self.layers["floor"] = self.floor

        if "two_areas" in mode:
            c = mode["two_areas"]
            self.floor[:,c:] = 1

        if "areas" in mode:
            c_list = mode["areas"]
            c_i, fvalue = 1,0
            for c_f in c_list:
                self.floor[:,c_i:c_f] = fvalue
                fvalue += 1
                c_i = c_f

        if "hotel" in mode:
            self.floor[:,:] = 1
            self.floor[int(self.rows/2),-1] = 2
            self.floor[int(self.rows/2),0] = 0

        if "exit" in mode:
            if mode["exit"] == "dynaq":
                self.floor[1, -2] += 1
                self.floor[3, 1] = 0
            else:
                self.floor[int(self.rows/2), -2] += 1

        if "object_areas" in mode:
            a = mode["object_areas"]
            if (self.n % a == 0):
                self.floor[int(self.n / a):int(2 * self.n / a), :] = 1
                self.floor[int(2 * self.n / 3):, :] = 2
            elif (self.n % 2 == 0):
                self.floor[int(self.n / 2):, :] = 1

        if "goal_sequence" in mode:
            ng = mode["goal_sequence"]
            for g in range(ng):
                x, y = np.random.randint(self.n, size=2)
                x, y = x%self.rows, y%self.cols
                while self.floor[x, y] > 0:
                    x, y = np.random.randint(self.n, size=2)
                    x, y = x%self.rows, y%self.cols
                self.floor[x, y] = g+1

        if "counting" in mode:
            i = np.random.randint(self.rows)+1
            for _ in range(i):
                j = np.random.randint(self.rows)
                while self.floor[j, 0] > 0:
                    j = np.random.randint(self.rows)
                self.floor[j, 0] = 1

        if "blocks" in mode:
            if "objects" not in self.conf:
                print("Blocks World termination mode needs key objects:num")
                sys.exit(0)
            nobj = self.conf["objects"]
            ncolors = self.rows if "ncolors" not in self.conf else self.conf["ncolors"]
            for _ in range(nobj):
                i = self.cols-1
                while not any(self.floor[:, i] == 0):
                    i -= 1
                j = np.random.randint(self.rows)
                while self.floor[j, i] > 0:
                    j = np.random.randint(self.rows)

                self.floor[j, i] = np.random.randint(ncolors)+1

        if "harlow" in mode:
            self.floor += 4
            self.floor[-1,int(self.cols / 2)] = 0
            self.floor[-1,0] = 1
            self.floor[-1,-1] = 2
            self.floor[0,int(self.cols / 2)] = 3

        if hasattr(self, "walls"):
            self.floor[self.walls == 1] = 0   # We add 0 to all the walls

        self.floor_max = np.max(self.floor)

    def add_objects(self, num):
        self.nobjects = num
        self.objects = np.zeros([self.rows, self.cols])
        self.layers["objects"] = self.objects
        for _ in range(num):
            check_layers = ["agents", "walls", "food", "objects", "floor"]
            if "blocks" in self.term_mode:
                if self.cols < 4:
                    check_layers.remove("floor")
            i, j = self.find_free_cell(check_layers=check_layers)
            self.objects[i, j] = 1

        if "blocks" in self.term_mode:
            l = list(map(int,self.floor.flatten()))
            while 0 in l:
                l.remove(0)

            iobjs,jobjs = np.nonzero(self.objects)
            for i,j in zip(iobjs,jobjs):
                self.objects[i,j] = l.pop()


    def add_food(self, num, pos=[]):
        self.food = np.zeros([self.rows, self.cols])
        self.layers["food"] = self.food

        if "food_info" in self.conf:
            for f in self.conf["food_info"]:
                self.food[f["coor"][0],f["coor"][1]] = 1
        else:
            if pos == []:
                for _ in range(num):
                    valid = False
                    while not valid:
                        x, y = np.random.randint(self.n, size=2)
                        valid = self.food[x, y] == 0
                        if "walls" in self.layers:
                            valid = valid and self.walls[x, y] == 0
                    self.food[x, y] = 1
            else:
                for p in pos:
                    self.food[p[0], p[1]] = 1

    def add_enemies(self, enemy_list):
        self.enemies = np.zeros([self.rows, self.cols])
        self.enemy_database = []
        tried_position = False

        for enemy in enemy_list:

            if "attack_range" not in enemy: enemy["attack_range"] = 0
            if "attack_damage" not in enemy: enemy["attack_damage"] = 1
            if "lethal" not in enemy: enemy["lethal"] = False
            if "mobility" not in enemy: enemy["mobility"] = "stationary"
            if "count" not in enemy: enemy["count"] = 1
            if "eats_food" not in enemy: enemy["eats_food"] = False

            for i in range(enemy["count"]):

                valid = False
                while not valid:

                    if "start_position" in enemy and not tried_position:
                        x = enemy["start_position"][0]
                        y = enemy["start_position"][1]
                        tried_position = True
                    else:
                        x,y = np.random.randint(self.n, size=2)

                    valid = self.enemies[x,y] == 0
                    if "walls" in self.layers:
                        valid = valid and self.walls[x,y] == 0
                    if "object" in self.layers:
                        valid = valid and self.objects[x,y] == 0
                    if "food" in self.layers:
                        valid = valid and self.food[x,y] == 0
                    if "floor" in self.layers:
                        valid = valid and self.floor[x,y] == 0

                enemy["position"] = [x,y]
                self.enemies[x,y] = 1
                self.enemy_database.append(copy.deepcopy(enemy))

        self.layers["enemies"] = self.enemies


    def add_walls(self, mode):
        self.walls = np.zeros([self.rows, self.cols])

        if "hotel" in mode:
            self.walls[int(self.rows/2), int(self.cols/2)] = 1
        else:
            self.walls = np.ones([self.rows, self.cols])
            self.walls[1:-1, 1:-1] = 0

        if "two_tunnels" in mode:
            rd2 = int((self.rows - 1) / 2)
            cd2 = int((self.cols - 1) / 2)

            if self.cols <= 7:
                self.walls[2, 0] = 0  # agent initial position
                self.walls[-3, 0] = 0  # agent initial position
                self.walls[3:-3, 2:-2] = 1  # middle walls for elevator
                self.walls[1, 1:-2] = 1  # middle walls for elevator
                self.walls[-2, 1:-2] = 1  # middle walls for elevator

                #self.walls[rd2-1:rd2+2,1] = 1  # middle walls for elevator
                if "tunnel_closed" in mode:
                    self.walls[-3, 2] = 1
                if "one_start" in mode:
                    self.walls[-3, 0] = 1  # agent initial position

            elif self.rows == 13:
                self.walls[3, 0] = 0  # agent initial position
                self.walls[-4, 0] = 0  # agent initial position
                self.walls[1, 1:-2] = 1    # top walls before elevator
                self.walls[2, 2:-2] = 1
                self.walls[-2, 1:-2] = 1   # bottom walls before elevator
                self.walls[-3, 2:-2] = 1
                h = 1
                self.walls[rd2 - h:rd2 + 1 + h, 1] = 1  # middle walls before elevator
                self.walls[rd2 - 1 - h:rd2 + 2 + h, 3:-2] = 1  # middle walls for elevator
                if "tunnel_closed" in mode:
                    self.walls[-4, 3] = 1
                if "one_start" in mode:
                    self.walls[-4, 0] = 1  # agent initial position

            else:
                self.walls[2, 0] = 0  # agent initial position
                self.walls[-3, 0] = 0  # agent initial position
                self.walls[1, 2:-2] = 1    # top walls before elevator
                self.walls[-2, 2:-2] = 1   # bottom walls before elevator
                self.walls[rd2:rd2 + 1, 1] = 1  # middle walls before elevator
                self.walls[3:-3, 3:-2] = 1  # middle walls for elevator
                if "tunnel_closed" in mode:
                    self.walls[-4, 3] = 1
                if "one_start" in mode:
                    self.walls[-3, 0] = 1  # agent initial position

        if "dynaq" in mode:
            #self.walls[3:-4,1] = 1
            #self.walls[3:-2,2] = 1

            self.walls[2:5,3] = 1
            self.walls[5,6] = 1
            self.walls[1:4,8] = 1
            #self.walls[4:-1,4] = 1

            #self.walls[2:-2,6] = 1
            #self.walls[2,7] = 1

        if "tolman_detour" in mode:
            self.walls[0:2,0] = 1
            self.walls[3:-1,0] = 1
            self.walls[1,2:5] = 1
            self.walls[3:6,2:5] = 1
            self.walls[0:1,-1] = 1
            self.walls[4:-1,-1] = 1

            self.walls[0:2,-2] = 1
            self.walls[3:-1,-2] = 1
            self.walls[-1,:] = 1

        if "tolman_detour_2" in mode:
            self.walls[0:2, 0] = 1
            self.walls[3:-1, 0] = 1
            self.walls[1, 2:5] = 1
            self.walls[3:6, 2:5] = 1
            self.walls[0:1, -1] = 1
            self.walls[4:-1, -1] = 1

            self.walls[2,4] = 1

            self.walls[0:2, -2] = 1
            self.walls[3:-1, -2] = 1
            self.walls[-1, :] = 1

        if "tolman_labyrinth" in mode:
            self.walls[0,0] = 1
            self.walls[3:7,0] = 1
            self.walls[9,0] = 1

            self.walls[0,2:4] = 1
            self.walls[3:10,2] = 1
            self.walls[3:5,3] = 1
            self.walls[8:10,3] = 1

            self.walls[0,5:7] = 1
            self.walls[1:3,6] = 1
            self.walls[3,5:7] = 1

            self.walls[6,5:7] = 1
            self.walls[7:9,5] = 1
            self.walls[9,5:7] = 1

            self.walls[2:8,8:10] = 1


        self.layers["walls"] = self.walls


    def update(self):
        if "update" not in self.conf:
            return

        bUpdate, step_freq, steps = True, 1, 1
        if "update_params" in self.conf:
            params = self.conf["update_params"]
            if "step_freq" in params:
                step_freq = params["step_freq"]

        if self.conf["update"] == "forest_fire":
            if self.num_agents > 0:
                x_old, y_old = self.agent_xy
                self.floor_xy_old = self.floor[x_old, y_old]

        bUpdate = True if step_freq <=0 else (self.t % step_freq) == 0
        if bUpdate:
            if step_freq <= 0:
                steps = abs(step_freq)+1
            for _ in range(steps):
                if self.conf["update"] == "game_of_life":
                    self.update_gol()
                if self.conf["update"] == "forest_fire":
                    self.update_forest()


    def update_gol(self):
        grid = self.layers["food"]
        new_grid = self.layers["food"].copy()
        N, M = self.rows, self.cols
        for i in range(N):
            for j in range(M):
                total = grid[i, (j-1)%N] + grid[i, (j+1)%N] + grid[(i-1)%N, j] + grid[(i+1)%N, j] + grid[(i-1)%N, (j-1)%N] + grid[(i-1)%N, (j+1)%N] + grid[(i+1)%N, (j-1)%N] + grid[(i+1)%N, (j+1)%N]
                if grid[i, j] == 1:
                    if (total < 2) or (total > 3):
                        new_grid[i, j] = 0
                else:
                    if total == 3:
                        if self.free_cell([i, j]):
                            new_grid[i, j] = 1
        grid[:] = new_grid[:] 

    def get_layer(self, name):
        if name in self.layers:
            return self.layers[name]
        else:
            print("Error Layer does not exist")
            return None


    def update_forest(self):
        grid = self.floor
        grid_old = grid.copy()
        self.floor_old = grid_old
        N, M = self.rows, self.cols
        agents = self.agents

        self.current_num_trees = 0
        if hasattr(self, "stats"):
            self.stats["current_num_trees"] = 0
            self.stats["current_num_fires"] = 0

        for i in range(N):
            for j in range(M):
                if grid_old[i, j] == 2:
                    grid[i, j] = 0
                elif grid_old[i, j] == 0:
                    grid[i, j] = 0
                    if np.random.rand() <= self.prob_tree:
                        grid[i, j] = 1
                elif grid_old[i, j] == 1:
                    grid[i, j] = 1
                    if self.agents[i, j] == 0 and np.random.rand() <= self.prob_fire:
                        if "only_bottom" in self.conf["update_params"]:
                            if i==N-1:
                                grid[i, j] = 2
                        elif "only_corner_cell" in self.conf["update_params"]:
                            if i == N-1 and j == M-1:
                                grid[i, j] = 2
                        elif "only_corner" in self.conf["update_params"]:
                            if i >= N-2 and j >= M-2:
                                grid[i, j] = 2
                        else:
                            grid[i, j] = 2
                    elif neighbour_fire(i, j, grid_old, torus=self.torus):  # fire propagation
                        grid[i, j] = 2

                if "update_params" in self.conf:
                    if "fire_steps" in self.conf["update_params"]:
                        t_fire = self.conf["update_params"]["fire_steps"]
                        if self.t % t_fire == 0:
                            trees = np.where(grid[-1, :] == 1)
                            grid[-1, trees] = 2

                if hasattr(self, "prob_fruit"):
                    if grid[i, j] == 1:
                        if np.random.rand() <= self.prob_fruit:
                            if self.num_agents == 0 or self.agents[i,j] == 0:
                                self.food[i,j] = self.reward["food"]
                    else:
                        self.food[i, j] = 0


                if grid[i,j] == 1:
                    self.grid_sum[0, i, j] += 1
                    self.current_num_trees += 1      # extra variable to keep track if there are no trees
                elif grid[i,j] == 2:
                    self.grid_sum[1, i, j] += 1
                if self.agents[i,j] == 1:
                    self.grid_sum[2, i, j] += 1

                if hasattr(self, "stats"):
                    if grid[i,j] == 1:
                        self.stats["current_num_trees"] += 1
                    elif grid[i,j] == 2:
                        self.stats["current_num_fires"] += 1
        if N == 3:
            grid[1, 1] = 0

        self.layers["floor"] = grid

    def mean_dist(self, agents_xy):
        # Calculating Center of Mass in an Unbounded 2D Environment
        Xi,Xj = [],[]
        im, jm = self.rows, self.cols
        ri = im/(2*np.pi)
        rj = jm/(2*np.pi)        
        for i,j in agents_xy:
            ti = (2*np.pi)*i/float(im)
            tj = (2*np.pi)*j/float(jm)
            Xi.append(np.array([ri*np.cos(ti), j, ri*np.sin(ti)]))
            Xj.append(np.array([i, rj*np.cos(tj), rj*np.sin(tj)]))

        xi,yi,zi = np.sum(Xi,axis=0) / len(Xi)
        xj,yj,zj = np.sum(Xj,axis=0) / len(Xj)
        i_ = (np.arctan2(-zi,-xi)+np.pi)*im/(2*np.pi)
        j_ = (np.arctan2(-zj,-yj)+np.pi)*jm/(2*np.pi)

        ti = (2*np.pi)*i_/float(im)
        tj = (2*np.pi)*j_/float(jm)
        ui = np.array([ri*np.cos(ti), j, ri*np.sin(ti)])
        uj = np.array([i, rj*np.cos(tj), rj*np.sin(tj)])

        dists = [] 
        for i in range(len(agents_xy)):
            d = min(np.linalg.norm(ui-Xi[i]), np.linalg.norm(uj-Xj[i]))
            dists.append(d)

        return  np.sum(dists), np.std(dists)


    def state_summary_stats(self, S, agents, each):
        maxv = 3 if self.obs_plus is not OBS_PLUS.ONE_HOT_BOTH else 1
        stats_epi = self.stats["episode"]

        # we inspect integer values from 1, 2,...,maxv-1
        # last maxv is reserved for values greater than 0
        if stats_epi["sum_range"] is None:
            stats_epi["sum_range"] = [np.zeros_like(self.state) for _ in range(maxv+1)]

        stats_epi["n_sums"] += 1
        for v in np.arange(maxv):
            stats_epi["sum_range"][v] += (self.state >= v+1) & (self.state < v+2)
        stats_epi["sum_range"][maxv] += (self.state > 0)

        for i,o in enumerate(S):
            if agents["obs"]["sum_range"] is None:   # agents obs sum_range (values, rows, cols, ch)
                agents["obs"]["sum_range"] = [np.zeros_like(o) for _ in range(maxv+1)]
            if each[i]["obs"]["sum_range"] is None:
                each[i]["obs"]["sum_range"] = [np.zeros_like(o) for _ in range(maxv+1)]

            # shape is (2*obs_radius + 1, 2*obs_radius + 1, ch)
            if agents["obs"]["sum_range"][0].shape != o.shape:  # added for ACT_PLUS.NOTHING_OBS_RADIUS, obs have different sizes
                continue

            agents["obs"]["n_sums"] += 1
            each[i]["obs"]["n_sums"] += 1
            for v in np.arange(maxv):
                agents["obs"]["sum_range"][v] += (o >= v+1) & (o < v+2)
                each[i]["obs"]["sum_range"][v] += (o >= v+1) & (o < v+2)

            agents["obs"]["sum_range"][maxv] += (o > 0)
            each[i]["obs"]["sum_range"][maxv] += (o > 0)

        #print("agents[obs][sum_range]-----------")        
        #print(np.array(agents["obs"]["sum_range"])[0,:,:,0])
        #print(np.sum(agents["obs"]["sum_range"][0]))
        #print("n_sums",stats_epi["n_sums"])
        #print("[sum_range][0][0]", stats_epi["sum_range"][0][0])
        #print("[sum_range][1][0]", stats_epi["sum_range"][1][0])
        #print("[sum_range][1][1]", stats_epi["sum_range"][0][1])
        #print(np.sum(stats_epi["sum_range"][0]))
        #print(stats_epi["trees_list"])
        #time.sleep(4)

    def stats_ini(self):
        """ self.stats["episode"] = { 
                "sum_range": (values,ch,rows,cols)
                "n_sums": how many times did we sum

                "agents": { 
                    "mean_dist_sum": 0,
                    "std_dist_sum": 0,

                    "obs": {
                        "sum_range": [],
                        "n_sums": 0
                    }
                },

                "each_agent": [{}, ..., {
                        "action_count":[0,0,0,0],
                        "obs": {
                            "sum_range": [],
                            "n_sums": 0
                        }
                    }
                ],

                "trees_list" : [],
                "fires_list" : [],
            }
        """
        stats_epi = self.stats["episode"]
        _agents, _each = stats_epi["agents"], stats_epi["each_agent"]
 
        stats_epi["sum_range"] = None
        stats_epi["n_sums"] = -1 
        _agents["mean_dist_sum"] = 0
        _agents["std_dist_sum"] = 0
        _agents["obs"]["sum_range"] = None
        _agents["obs"]["n_sums"] = -1 
        for i in range(self.num_agents):
            _each[i]["obs"]["sum_range"] = None   # each agents count
            _each[i]["obs"]["n_sums"] = -1
            _each[i]["rewards"] = {"tree": 0, "fire": 0, "fruit": 0}

        if "update" in self.conf:
            if self.conf["update"] == "forest_fire":
                stats_epi["trees_list"] = []  # global CA trees counts 
                stats_epi["fires_list"] = []  # global CA fires counts

    def stats_update(self, S, A, r, done=False):
        stats, stats_epi = self.stats, self.stats["episode"]
        _agents, _each = stats_epi["agents"], stats_epi["each_agent"]

        if self.num_agents == 1:
            self.agents_xy = [self.agent_xy]
            S, A = [S], [A]
        else:
            dmean, dstd = self.mean_dist(self.agents_xy)
            _agents["mean_dist_sum"] += dmean
            _agents["std_dist_sum"] += dstd

        self.state_summary_stats(S, _agents, _each)

        if "update" in self.conf:
            if self.conf["update"] == "forest_fire":
                stats_epi["trees_list"] += [stats["current_num_trees"]]
                stats_epi["fires_list"] += [stats["current_num_fires"]]
                if done:
                    stats_epi["grid_sum"] = self.grid_sum
                    if "run_trees_mean_list" in stats:
                        trees_list, fires_list = stats_epi["trees_list"], stats_epi["fires_list"]
                        stats["run_trees_mean_list"] += [np.mean(trees_list)]
                        stats["run_trees_std_list"] += [np.std(trees_list)]
                        stats["run_fires_mean_list"] += [np.mean(fires_list)]
                        stats["run_fires_std_list"] += [np.std(fires_list)]

    def plot_ascii(self):
        print("shape of state:", np.array(list(self.layers.values())).shape, "\n")
        for k, layer in self.layers.items():
            print("----------------------", k, ":")
            print(layer)
            print(" ")

    def get_agents_xy(self):
        agents_xy = []
        if hasattr(self, "agents_xy"):
            agents_xy = self.agents_xy if self.num_agents > 1 else [self.agent_xy]
        return agents_xy


    def clear_plt(self):
        plt.clf()
        ax = plt.axes()
        ax.set_aspect('equal')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.xticks([])
        plt.yticks([])
        return ax

    def render_twofig(self, params={}):
        ax = self.clear_plt()
        ax = plt.subplot(1, 2, 1)
        draw_layers(self.conf, self.layers, agents_xy=self.agents_xy, actions=self.actions, ax=ax, enemy_info=self.enemy_database)

        ax = plt.subplot(1, 2, 2)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        if hasattr(self, "obs"):
            draw_obs(self, self.obs[0], ax=ax)
        plt.show()
        plt.pause(0.001)

    def render_fig(self, fig_param, params={}, bForce=False):
        agents_xy = self.get_agents_xy()
        new_fig_size = fig_param.get_size_inches() if not bForce else [1000]
        if min(new_fig_size) > 1:
            draw_layers(self.conf, self.layers, agents_xy=agents_xy, enemy_info=self.enemy_database, actions=self.actions, params=params)
        if min(new_fig_size) > 1 or self.t % 500 == 0 or bForce:   # self.t % 500 is for having window focus enabled
            fig_param.canvas.draw()
            plt.pause(0.001)

    def render(self, fig_param=None, params={}):
        agents_xy = self.get_agents_xy()
        if fig_param is None:
            draw_layers(self.conf, self.layers, agents_xy=agents_xy, actions=self.actions,  enemy_info=self.enemy_database, params=params)
            return
        if hasattr(fig_param, "canvas"):    # is figure
            self.render_fig(fig_param)
            #self.render_twofig()
        else:                               # is axes
            draw_layers(self.conf, self.layers, agents_xy=agents_xy, actions=self.actions, ax=fig_param, enemy_info=self.enemy_database, params=params)

    def make_id(self):
        conf = self.conf
        run = conf["run"] if "run" in conf else {}        
        str_all = ""
        if "run" in conf and "title" in run:
            str_all += run["title"]
        elif conf["num_agents"] > 1:
            str_all += "%02dx%02d-%02da"%(conf["rows"], conf["cols"], conf["num_agents"])

        if "action_mode" in conf:
            str_all += "-allo" if conf["action_mode"] in [ACT_MODE.ALLOCENTRIC] else "-ego"

        if "action_plus" in conf:
            str_all += '-' + ACT_PLUS.id(conf["action_plus"])

        str_all += '-' + OBS_MODE.id(conf["obs_mode"])

        if "obs_plus" in conf:
            str_all += '-' + OBS_PLUS.id(conf["obs_plus"])

        if conf["obs_mode"] not in OBS_MODE.global_family():
            str_all += "-%drad"%conf["obs_radius"] # maybe part

        #ff = conf["update_params"]
        #str_ff = "h-%.1f-t-%.4f-f-%.4f" % (ff["prob_harvest"], ff["prob_tree"], ff["prob_fire"])
        return str_all


    def close(self):
        pass



def center_crop(i, j, r, O, dir=None, torus=True, padding=False):
    if not torus or padding:
        pad = ((0, 0), (r+1, r+1), (r+1, r+1))  # No padding for the first dimension / feature channels
        O = np.pad(O, pad_width=pad, mode="constant", constant_values=1) # changed for two tunnels
        i,j = i+r+1, j+r+1

    o = O.take(np.arange(i-r,i+r+1), axis=1, mode='wrap')
    o = o.take(np.arange(j-r,j+r+1), axis=2, mode='wrap')
    if dir is not None:
        o = np.rot90(o, (5 - dir) % 4, axes=(1, 2))
    return o


#@jit(nopython=True)
def add_row_col_zero(ex, ey, z, state):
    if ex == 1 and ey == 1:
        z[:,:-1,:-1] = state
    elif ex == 1 and ey == 0:
        z[:,:-1,:] = state
    elif ex == 0 and ey == 1:
        z[:,:,:-1] = state
    return z

#@jit(nopython=True)
def add_row_col_roll(ex, ey, z, state):
    if ex == 1:
        z[:,-1,:] = state[:,0,:]
    if ey == 1:
        z[:,:,-1] = state[:,:,0]
    return z

#@jit(nopython=True)
def pad_shift(agent_xy, rows_cols):
    ex, ey = (rows_cols+1) % 2   # odd row, odd col
    cx, cy = (rows_cols - np.array([1, 1]) + np.array([ex, ey])) / 2.0
    shift = np.array([int(cx), int(cy)]) - agent_xy
    return ex, ey, int(cx), int(cy), shift


#@jit(nopython=True)
def gridworld_center_view(state, agent_xy, obs_mode, torus=False, padding=False):
    rows_cols = np.array(state.shape[1:])
    ex, ey, cx, cy, shift = pad_shift(agent_xy, rows_cols)
    if padding:
        r = max(cx,cy)
        pad = ((0,0), (r,r), (r,r))    # No padding for the first dimension / feature channels
        state = np.pad(state, pad_width=pad, mode="constant", constant_values=0)

    #state = np.roll(state, (0,0,shift[0]), axis=(0,0,1))   # rows shift of shift[0] rows
    #state = np.roll(state, (0,0,shift[1]), axis=(0,0,-1))  # cols shift

    centeredview = np.zeros(state.shape)
    for i in range(state.shape[0]):    # if there is enough padding it shifts normally, otherwise it wraps around/rolls over
        centered = np.roll(state[i], shift, axis=(0,1))
        centeredview[i,:,:] = centered
    state = centeredview

    if np.any([ex,ey]) and obs_mode in [OBS_MODE.GLOBAL_CENTER_PAD, OBS_MODE.GLOBAL_CENTER_WRAP]:
        z = np.zeros(np.array(state.shape) + [0, ex, ey])
        state = add_row_col_zero(ex, ey, z, state)
        if torus:
            state = add_row_col_roll(ex, ey, z, state)

    return state




#@jit(nopython=True)
def neighbour_fire(i, j, grid, torus=False, moore=False):
    N, M = grid.shape
    neigh_fire = False

    if moore:
        neigh_fire = np.any(grid[max(0,i-1):i+2,max(0,j-1):j+2].flatten() == 2)  # Moore neighbourhood
    else:
        if i > 0:
            neigh_fire = neigh_fire or grid[i - 1, j] == 2
        if i < N - 1:
            neigh_fire = neigh_fire or grid[i + 1, j] == 2
        if j > 0:
            neigh_fire = neigh_fire or grid[i, j - 1] == 2
        if j < M - 1:
            neigh_fire = neigh_fire or grid[i, j + 1] == 2

    if torus and not neigh_fire:
        if moore:
            if i == 0:
                neigh_fire = neigh_fire or np.any(grid[-1:, max(0, j - 1):j + 2].flatten() == 2)
            elif i == N - 1:
                neigh_fire = neigh_fire or np.any(grid[:1, max(0, j - 1):j + 2].flatten() == 2)
            if j == 0:
                neigh_fire = neigh_fire or np.any(grid[max(0, i - 1):i + 2, -1:].flatten() == 2)
            elif j == M - 1:
                neigh_fire = neigh_fire or np.any(grid[max(0, i - 1):i + 2, :1].flatten() == 2)
        else:
            if i == 0:
                neigh_fire = neigh_fire or grid[N - 1, j] == 2
            elif i == N - 1:
                neigh_fire = neigh_fire or grid[0, j] == 2
            if j == 0:
                neigh_fire = neigh_fire or grid[i, M - 1] == 2
            elif j == M - 1:
                neigh_fire = neigh_fire or grid[i, 0] == 2

    return neigh_fire
