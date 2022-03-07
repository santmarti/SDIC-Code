""" course: System Design Integration and Control SDIC, CSIM, UPF
    contact: marti.sanchez@upf.edu
    Environment.py : Wrapper class for OpenAI-Gym python.environments
    and other defined here;
    "BlackJack" : Sutton and Barto Book
    "CliffWalking" : Sutton and Barto Book
    "GridworldSutton" : Sutton and Barto Book
    "WindyGridworld" : Sutton and Barto Book
    "SuttonSimplest" : Simple MDP, Sutton talk
    "BattleExes" : multi agent grid version of Battle of the Sexes
    "FruitCollection" : SuperMini,Mini,Small,Large
    "JackCarRental" : Sutton and Barto Book
    "BimanualRobot" : PyBox2d simulation (not an environment yet)
    "KeyCollection" : Hierarchical key collection
    "Wisconsin" : Wisconsin card test
"""
import datetime
import os
import sys
from ast import literal_eval
import time

import gym
import numpy as np
from gym.envs.toy_text import discrete
from gym.spaces import Discrete, Box

import pickle

from enum import Enum

class OBS_MODE(Enum):
    GLOBAL = 0
    GLOBAL_ID = 1
    GLOBAL_ID_EGO = 2
    GLOBAL_CENTER_PAD = 3
    GLOBAL_CENTER_WRAP = 4
    LOCAL = 5
    LOCAL_SUM = 6
    LOCAL_ONE_HOT = 7
    LOCAL_SUM_ONE_HOT = 8
    LOCAL_ID = 9
    LOCAL_SUM_MOORE = 10
    LOCAL_SUM_ONION = 11
    LOCAL_ONION = 12
    GLOBAL_COORDINATE = 13

    def id(om):
        if om is OBS_MODE.GLOBAL: return 'g'
        elif om is OBS_MODE.GLOBAL_ID: return 'gid'
        elif om is OBS_MODE.GLOBAL_ID_EGO: return 'gidego'
        elif om is OBS_MODE.GLOBAL_CENTER_PAD: return 'gcp'
        elif om is OBS_MODE.GLOBAL_CENTER_WRAP: return 'gcw'
        elif om is OBS_MODE.LOCAL: return 'local'
        elif om is OBS_MODE.LOCAL_SUM: return 'lsum'
        elif om is OBS_MODE.LOCAL_ONE_HOT: return 'lhot'
        elif om is OBS_MODE.LOCAL_SUM_ONE_HOT: return 'lsumhot'
        elif om is OBS_MODE.LOCAL_ID: return 'lid'
        elif om is OBS_MODE.LOCAL_SUM_MOORE: return 'lsmoore'
        elif om is OBS_MODE.LOCAL_SUM_ONION: return 'lsonion'
        elif om is OBS_MODE.LOCAL_ONION: return 'lonion'
        elif om is OBS_MODE.GLOBAL_COORDINATE: return "gcor"
        return "Not Defined"

    def global_family():
        return [OBS_MODE.GLOBAL_ID, OBS_MODE.GLOBAL_ID_EGO, OBS_MODE.GLOBAL, OBS_MODE.GLOBAL_CENTER_PAD, OBS_MODE.GLOBAL_CENTER_WRAP, OBS_MODE.GLOBAL_COORDINATE]

    def need_state_map():
        return [OBS_MODE.LOCAL_ID, OBS_MODE.LOCAL_ONE_HOT, OBS_MODE.LOCAL_SUM_ONE_HOT]

class OBS_PLUS(Enum):
    SAME = 0
    NO_AGENTS = 1
    AGENTS_OBS = 2
    NO_ORIENTATION = 3
    ONE_HOT_BOTH = 4

    def id(op):
        if op is OBS_PLUS.NO_AGENTS: return 'noag'
        elif op is OBS_PLUS.AGENTS_OBS: return 'agobs'
        elif op is OBS_PLUS.NO_ORIENTATION: return 'norien'
        elif op is OBS_PLUS.ONE_HOT_BOTH: return 'onehot'
        return "ag"

class OBS_SHAPE(Enum):
    SAME = 0
    REVERSE = 1


class OBS_MEM(Enum):
    SAME = 0
    STATE_MEM = 1
    ACTION_MEM = 2
    STATE_ACTION_MEM = 3
    SAR_MEM = 4
    STATE_MEAN = 5
    ACTION_MEAN = 6


class ACT_MODE(Enum):
    ALLOCENTRIC = 0
    EGOCENTRIC = 1

    def id(act):
        if act is ACT_MODE.ALLOCENTRIC: return 'allo'
        elif act is ACT_MODE.EGOCENTRIC: return 'ego'
        return "Not Defined"

class ACT_PLUS(Enum):
    SAME = 0
    PLUS_DO_NOTHING = 2
    NOTHING_RANDOM = 3           # do nothing or move random according to ACT_MODE
    NOTHING_RANDOM_HARVEST = 4   # not move, random, harvest
    FOREST_FIRE = 5              # add harvest to ACT_MODE
    NOTHING_OBS_RADIUS = 6       # do nothing but get higher radius observation

    def id(act):
        if act is ACT_PLUS.PLUS_DO_NOTHING : return 'no'
        elif act is ACT_PLUS.NOTHING_RANDOM: return 'norand'
        elif act is ACT_PLUS.NOTHING_RANDOM_HARVEST: return 'noraha'
        elif act is ACT_PLUS.FOREST_FIRE: return 'ha'
        elif act is ACT_PLUS.NOTHING_OBS_RADIUS: return 'incrad'
        return "Not Defined"


def open_file_save_copy(fname, mode='w', bMakeCopy=False):
    import os
    import shutil

    if os.path.isfile(fname):
        fname_new = fname
        while os.path.isfile(fname_new):

            j = fname_new.rfind('.')
            fsplit = [fname_new[:j], fname_new[j+1:]] if j > 0 else [fname_new]

            i = fsplit[-2].rfind('-v')
            fsplit_ = [fsplit[-2][:i], fsplit[-2][i+2:]] if i > 0 else [fsplit[-2]]

            number = int(fsplit_[-1]) if len(fsplit_) > 1 else 0
            part_1 = fsplit[-2] + '-v1'
            if len(fsplit_) > 1:
                part_1 = fsplit_[-2] + '-v' + str(number+1)
            fname_new = ".".join([part_1, fsplit[-1]])

        if not bMakeCopy:
            print("Renaming file: ", fname)
            os.rename(fname, fname_new)
        else:
            print("Copy file: ", fname)
            shutil.copy(fname, fname_new)

    return open(fname, mode)


def seed_all(run, ini=None):
    if "seed" in run:
        ini = run["seed"]
    elif ini is None:
        run["seed"] = ini

    run["seed"] = ini
    np.random.seed(ini)
    rseed = np.random.get_state()
    run["random_state"] = np.random.get_state()
    print("> Seeding numpy, gym, tensorflow with seed ", ini, " - RandomState stored")
    try:
        from gym.spaces import prng
        prng.seed(rseed)
    except:
        pass
    return ini



class Environment():

    def __init__(self, NameOrParams):
        if not self.filter_params(NameOrParams):
            return

        params = None if len(self.name.split('-')) == 1 else self.name.split('-')[1:]
        name,basename = self.name,self.basename
        self.fig = None

        envids = [spec.id for spec in gym.envs.registry.all()]
        if name in envids:
            print(name, "is an openai gym registered environment")
            self.my_env = gym.make(name)
        else:
            self.create_non_gym_env(basename, params)

        self.action_space = self.my_env.action_space
        self.observation_space = self.my_env.observation_space

        self.check_discretize_action()
        self.set_gym_attribs()

        self.freq_start_time = time.time()  # nade to be initialized when created

        self.reset()

    def filter_params(self, NameOrParams):
        self.envs_names = ["BlackJack","CliffWalking","GridworldSutton","WindyGridworld","SuttonSimplest","BattleExes","BimanualRobot"]
        self.conf = None
        self.max_steps = 100000
        self.discretize_action = 0
        self.multi_agent_mode = False
        self.num_agents = 1

        if isinstance(NameOrParams,dict):
            in_conf = NameOrParams 
            if(not "name" in in_conf):
                print("Error: Environment config dictionary must include env name: {\"name\":\"....\"")
                return False
            if(not "max_steps" in in_conf):
                print("Setting default max_steps per episode: %d"%self.max_steps)
                in_conf["max_steps"] = self.max_steps
            if(not "num_agents" in in_conf):
                print("Default number of agents: %d"%self.num_agents)
                in_conf["num_agents"] = self.num_agents
            elif in_conf["num_agents"] > 1:
                self.multi_agent_mode = True

            self.__dict__.update(**in_conf)   # translate dictionnary into class attributes 
            self.basename = self.name

        if isinstance(NameOrParams,str):
            print("Setting default max_steps per episode: %d"%self.max_steps)
            self.name = NameOrParams
            self.basename = self.name.split('-')[0]
            in_conf = { "title": self.basename }
            
        self.conf = in_conf
        return True

    def check_discretize_action(self):
        if not isinstance(self.my_env.action_space,Box):
            print(self.make_id() + " Discrete Action Space with", self.my_env.action_space)            
            return

        low = self.my_env.action_space.low
        high = self.my_env.action_space.high
        print(self.make_id() + " Continuous Action Space:", self.my_env.action_space, "Low", low, "High", high)
        if self.discretize_action > 0:
            self.action_space = Discrete(self.discretize_action)
            for i in range(len(low)):
                self.action_values = np.linspace(low[i], high[i], num=self.discretize_action)
            print("Discretizing actions to",self.action_values)

    def set_gym_attribs(self):
        if isinstance(self.my_env,discrete.DiscreteEnv):
            self.nS = self.my_env.nS
            self.nA = self.my_env.nA
            self.nactions = self.nA
            self.P = self.my_env.P
            if hasattr(self.my_env,"grid_shape"):
                self.grid_shape = self.my_env.grid_shape

        if hasattr(self,"my_env"):
            if hasattr(self.my_env,"nactions"):
                self.nactions = self.my_env.nactions
            if hasattr(self.my_env,"num_agents"):
                self.num_agents = self.my_env.num_agents

    def create_non_gym_env(self, basename, params=None):
        # The environment is not a gym environment
        if(basename == "BlackJack"):
            from environments.blackjack import BlackjackEnv
            self.my_env = BlackjackEnv()

        elif(basename == "CliffWalking"):
            from environments.cliff_walking import CliffWalkingEnv
            self.my_env = CliffWalkingEnv()

        elif(basename == "GridworldSutton"):
            from environments.gridworld_sutton import GridworldSuttonEnv
            self.my_env = GridworldSuttonEnv()

        elif(basename == "WindyGridworld"):
            from environments.windy_gridworld import WindyGridworldEnv
            self.my_env = WindyGridworldEnv()

        elif(basename == "SuttonSimplest"):
            from environments.suttonsimplest import SuttonSimplestEnv
            self.max_steps = 1000
            self.my_env = SuttonSimplestEnv()

        elif(basename == "RockScissorsPaper"):
            from environments.rockscissorspaper import RockScissorsPaper
            self.my_env = RockScissorsPaper()

        elif(basename == "BattleExes"):
            from environments.battle_of_exes import BattleOfExesEnv
            self.my_env = BattleOfExesEnv()

        elif(basename == "BattleExesMin"):
            from environments.battle_of_exes import BattleOfExesMin
            self.my_env = BattleOfExesMin(self.conf)
            obs = self.my_env.reset()
            self.my_env.observation_space = Box(low=np.inf, high=np.inf, shape = obs.shape, dtype=np.int)
            self.my_env.action_space= Discrete(2)

        elif(basename == "FruitCollection"):
            from environments.fruit_collection import FruitCollectionSuperMini, FruitCollectionMini, FruitCollectionSmall, FruitCollectionLarge
            bRender = True
            bFruit, bGhost = True, False
            if(len(params) == 2):
                bFruit = literal_eval(params[0])
                bGhost = literal_eval(params[1])
            if(len(params) == 3):
                bFruit = literal_eval(params[1])
                bGhost = literal_eval(params[2])

            if(len(params) in [1, 3]):
                    if(params[0] == "SuperMini"): self.my_env = FruitCollectionSuperMini(lives=1, rendering=bRender, is_fruit=bFruit,is_ghost=bGhost)
                    if(params[0] == "Mini"): self.my_env = FruitCollectionMini(lives=1, rendering=bRender, is_fruit=bFruit,is_ghost=bGhost)
                    if(params[0] == "Small"): self.my_env = FruitCollectionSmall(lives=1, rendering=bRender, is_fruit=bFruit,is_ghost=bGhost)
                    if(params[0] == "Large"): self.my_env = FruitCollectionLarge(lives=1, rendering=bRender, is_fruit=bFruit,is_ghost=bGhost)
            else:
                self.my_env = FruitCollectionSmall(lives=1, rendering=bRender, is_fruit=bFruit,is_ghost=bGhost)

            self.my_env.reset()
            obs = self.my_env.get_state()
            self.my_env.observation_space = Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)
            self.my_env.action_space = Discrete(4)
            print("FruitCollection ", self.my_env.scr_w,"x",self.my_env.scr_h, " created. With fruits:", self.my_env.is_fruit,"  With ghosts:",self.my_env.is_ghost)

            if(bRender): self.my_env.render()

        elif (basename == "KeyCollection"):
            from environments.key_collect import KeyCollection
            bRender = True
            bKeys = 3
            bSize = 5
            if (len(params) == 2):
                bKeys = literal_eval(params[0])
                bSize = literal_eval(params[1])
            self.my_env = KeyCollection(rendering=bRender, num_keys=bKeys, size=bSize)
            self.my_env.reset()
            obs = self.my_env.get_state()
            self.my_env.observation_space = Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)
            self.my_env.action_space = Discrete(4)
            print("KeyCollection ", self.my_env.scr_w, "x", self.my_env.scr_h, " created with ", self.my_env.num_keys, " keys.")
            if (bRender): self.my_env.render()

        elif(basename == "JackCarRental"):
            from environments.jackcar import JackCarRentalEnv
            if params:
                param = int(self.name.split('-')[1])
                if len(params) > 1:
                    trent = literal_eval(params[0])
                    tret = literal_eval(params[1])
                    self.my_env = JackCarRentalEnv(max_cars = param, rents_per_day = trent, returns_per_day = tret)
                else:
                    self.my_env = JackCarRentalEnv(max_cars = param)
            else:
                self.my_env = JackCarRentalEnv()

        elif(basename == "TwoArms"):
            from environments import box2d_biarms
            self.my_env = box2d_biarms.TwoArmsEnv()

        elif(basename == "DualCart"):
            from environments import box2d_dualcart
            self.my_env = box2d_dualcart.DualCartSimEnv(multi_agent_mode=self.multi_agent_mode)

        elif(basename == "BimanualRobot"):
            from environments import box2d_bimanualsim
            self.my_env = box2d_bimanualsim.BimanualSimEnv()

        elif(basename == "Wisconsin"):
            from environments.Wisconsin import Wisconsin
            render = True
            game_len=50
            self.my_env = Wisconsin(rendering=render, gamelen=game_len)
            self.my_env.reset()
            obs = self.my_env.get_state()
            self.my_env.observation_space = Box(low=np.inf, high=np.inf, shape = obs.shape, dtype=np.int)
            self.my_env.action_space= Discrete(4)
            print("Wisconsin Card Test started with ", game_len, " trials.")
            if(render == True): self.my_env.render()

        elif(self.name == "FinalGrid"):
            from environments.finalgrid import FinalGrid
            conf = self.conf
            self.my_env = FinalGrid(conf=conf)
            self.my_env.multi_agent_mode = self.multi_agent_mode
            obs = self.my_env.reset()
            self.my_env.grid_shape = obs.shape[1:] # We eliminate the channel dimension for final grid
            self.my_env.observation_space = Box(low=np.inf, high=np.inf, shape=obs.shape, dtype=np.float)
            self.my_env.action_space = Discrete(self.my_env.nactions)
            self.my_env.agent_num = 1 if "num_agents" not in conf else conf["num_agents"]
            self.nA = self.my_env.nactions
            self.nS = conf["rows"]*conf["cols"]

        elif(self.name == "Continuous1D"):
            from environments.continuous1D import Continuous1D
            self.my_env = Continuous1D()
            self.my_env.observation_space = Box(low=np.inf, high=np.inf, shape=(1,), dtype=np.float)
            self.my_env.action_space = Box(low=-0.1, high=0.1, shape=(1,), dtype=np.float)
        else:
            print("No environment found", basename)



    def stats_ini(self):
        conf, stats = self.conf, self.stats
        self.my_env.stats = self.stats

        if not "reward_sum" in stats:
            stats["step_sum"] = 0
            stats["reward_sum"] = 0 if self.num_agents == 1 else np.zeros(self.num_agents)
            stats["action_sum"] = np.zeros(self.nactions)
            stats["time_sum"] = 0
            self.epi_start_time = time.time()

        self.stats_freq = 1   # refers to stats episode freq: record / print every .. episodes
        if "run" in conf:  # if conf has "run" key means we are ready for stats
            n_epi = self.conf["run"]["num_episodes"] / 100
            self.stats_freq = stats["stats_freq"] if "stats_freq" in stats else int(n_epi)

        self.reset_epi_dict()
        if hasattr(self.my_env, "stats_ini"):   # we call environment stats_ini at reset                    
            self.my_env.stats_ini()

    def reset_epi_dict(self):
        """ self.stats["episode"] = {
                "reward_sum": [rew_i,...],
                "action_count":[0,0,0,0],
                "each_agent": [{"obs": {}}, ..., {"obs": {}}],
            }
        """
        stats = self.stats
        stats["episode"] = {}   # dictionnary to store in database
        stats_epi = stats["episode"]

        stats_epi["reward_sum"] = 0 if self.num_agents == 1 else np.zeros(self.num_agents)
        stats_epi["action_count"] = np.zeros(self.nactions)
        stats_epi["agents"] = { "obs": {} }
        dict_ini = {"obs": {}, "action_count": np.zeros(self.nactions)}
        stats_epi["each_agent"] = [ dict_ini.copy() for _ in range(self.num_agents)]

    def reset(self, bStats=True, params={}):
        if bStats and hasattr(self, "i_episode"):
            self.i_episode += 1

        conf = self.conf
        if bStats and conf is not None:
            if "stats" in conf:
                self.stats = conf["stats"]
                stats = self.stats
            if "run" in conf and "stats" in conf["run"]:
                self.stats = conf["run"]["stats"]
                stats = self.stats
            if bStats and hasattr(self, "stats"):
                self.stats_ini()        # at every episode

        if len(params) == 0:
            state = self.my_env.reset()
        else:
            state = self.my_env.reset(params=params)

        if(self.basename in ["TwoArms", "FruitCollection", "KeyCollection"]):
            state = self.my_env.get_state() 

        self.t = 0
        self.my_env.t = self.t
        state = self.add_history(state, [], [])  # carefull with multiagent / single agent

        return state

    def create_folder(self, path, root_folder=None):
        if not os.path.isdir(path):
            print("Creating New Folder", path)
            if root_folder is not None:
                if not os.path.isdir(root_folder):
                    os.mkdir(root_folder)
            os.mkdir(path)
            print("Created lmdb ", path)


    def save_folder(self, return_folder=True, bCreate=True):
        run = self.conf["run"]
        #if "simulation_folder" not in run:
        #    run["simulation_folder"] = "./simulation_" + datetime.datetime.now().strftime("%Y_%m_%d-time-%H_%M_%S")

        if "simulation_folder" in run:
            if return_folder:
                if run["simulation_folder"][-1] != "/":
                    run["simulation_folder"] += "/"

            root_folder = "./" if not "root_folder" in run else run["root_folder"]
            path = root_folder + run["simulation_folder"]
            if bCreate:
                self.create_folder(path, root_folder)
            return path

        return None


    def set_run(self, run, root_folder="./", stats_freq=10, iSeed=None, build_database=True, post_fix=None, new_files=True):
        conf = self.conf
        conf["run"] = run  # when we reset() environment conf will have "run" key

        if stats_freq >= 0:
            run["stats"] = {"stats_freq": stats_freq,  # stats contain all info of the running that is updated
                            "str_log_list": [],
                            "freq_reward_list": [],
                            "freq_action_list": [],
                            "freq_step_list": [],
                            "freq_time_list": []}

        if build_database:
            run["episode_db"] = True

        run["root_folder"] = root_folder
        if iSeed is None:
            bExists = True
            while bExists:
                iSeed = np.random.randint(100)
                run["simulation_folder"] = self.make_id(run, iSeed=iSeed, post_fix=post_fix)
                bExists = os.path.isdir(run["root_folder"] + run["simulation_folder"])
        else:
            run["simulation_folder"] = self.make_id(run, iSeed=iSeed, post_fix=post_fix)
            bExists = os.path.isdir(run["root_folder"] + run["simulation_folder"])

        run["seed"] = iSeed
        if new_files:
            open_file_save_copy(self.save_folder() + "conf_freq.pickle")

        return bExists


    def save_freq_file(self):
        fname = self.save_folder() + "conf_freq.pickle"
        pickle.dump(self.conf, open(fname, "wb"))

    def save_freq(self, i_episode):
        if (i_episode + 1) % self.stats_freq > 0:  return     # number of episodes for updating
        stats = self.stats

        str_log = str(i_episode + 1)
        for t in ["reward", "action", "step", "time"]:            # we update all stats
            freq_str, sum_str = "freq_"+t+"_list", t+"_sum"
            if freq_str in stats:
                s = stats[sum_str] / self.stats_freq
                if t in ["action"]: s = s.astype(float)
                if t in ["step", "time"]:
                    s = np.round(s,1) if type(s) is np.ndarray else round(s,1)
                if t in ["reward"] and self.num_agents == 1:
                    s = np.round(s,1) if type(s) is np.ndarray else round(s,1)

                stats[freq_str] += [s]

                str_log += " " + t + ": " 
                if t == "reward" and self.num_agents > 3:
                    str_log += str(round(min(s),1)) + " / " + str(round(np.mean(s), 1)) + " / " + str(round(max(s),1))
                else:
                    str_log += str(s)

        str_log += " total time: " + str(round(time.time() - self.freq_start_time,2))
        self.freq_start_time = time.time()
        print(self.make_id() + " " + str_log)

        if "str_log_list" in stats:
            stats["str_log_list"] += [str_log]

        self.save_freq_file()

        stats["step_sum"] = 0
        stats["reward_sum"] = 0 if self.num_agents == 1 else np.zeros(self.num_agents)
        stats["action_sum"] = np.zeros(self.nactions)
        stats["time_sum"] = 0


    def save_db(self, str_entry, data):
        with self.db.begin(write=True) as dbw:
            p = pickle.dumps(data)
            dbw.put(str_entry.encode('ascii'), p)

    def save_episode_db(self, i_episode): # Save episode in DataBase File
        run= self.conf["run"]
        if not "episode_db" in run: return        
        if i_episode % run["episode_db"] != 0: return

        stats, stats_epi = self.stats, self.stats["episode"]
        stats_epi["random_state"] = np.random.get_state()

        if not hasattr(self,"db"):
            self.load_db_ini()

        if i_episode == 0:  # create
            self.save_db("conf", self.conf)

        self.save_db("%05d"%i_episode, stats["episode"])

    def load_db_ini(self, bCreateFolder=True):
        import lmdb
        db_name = self.save_folder(return_folder=False, bCreate=False)   # db name is a folder in lmdb framework

        if bCreateFolder:
            self.create_folder(db_name)

        self.db = lmdb.open(db_name, map_size=int(1e10))

        if self.obs_mode in OBS_MODE.need_state_map():
            with self.db.begin() as dbr:
                p_data = dbr.get(b'state_map')
                if p_data is not None:
                    self.my_env.state_map = pickle.loads(p_data)
                    print("Loaded state_map data from db: %d entries"%len(self.my_env.state_map))

    def save_db_end(self):
        if self.obs_mode in [OBS_MODE.LOCAL_ONE_HOT, OBS_MODE.LOCAL_ID]:
            self.save_db("state_map", self.my_env.state_map)


    def stats_update(self, s, actions, r, done):
        if not hasattr(self,"stats"): return

        if self.num_agents == 1:
            actions = [actions]

        stats, stats_epi = self.stats, self.stats["episode"]

        stats_epi["reward_sum"] += r            # put to 0 in reset()
        stats_epi["steps"] = self.t
        for i, a in enumerate(actions):
            stats_epi["action_count"][a] += 1     # put to 0 in reset()
            iagent = stats_epi["each_agent"][i]
            iagent["action_count"][a] += 1

        if hasattr(self.my_env,"stats_update"):
            self.my_env.stats_update(s, a, r, done)

        if done:
            stats["reward_sum"] += stats_epi["reward_sum"]
            stats["action_sum"] += stats_epi["action_count"]
            stats["step_sum"] += self.t

            if not hasattr(self, "epi_start_time"):
                self.epi_start_time = 0

            stats["time_sum"] += np.round(time.time() - self.epi_start_time, 2)

            if "i_episode" in stats:
                i_episode = stats["i_episode"]
            elif hasattr(self, "i_episode"):
                i_episode = self.i_episode
            else:
                i_episode = 0

            self.save_episode_db(i_episode)  # episode data saved in file data base
            self.save_freq(i_episode)
            self.reset_epi_dict()
            self.epi_start_time = time.time()

    def update_history(self, state, actions, reward):
        """ Updates history and returns stacked state with history
        """
        if not "obs_mem" in self.conf:
            return state

        if self.obs_mem is OBS_MEM.SAME:
            return state

        hist = self.my_env.history   # we store in history a raw state (without history)

        state_h = self.add_history(state, actions, reward)

        for i, (s,a,r) in enumerate(zip(state, actions, reward)):
            hist[i].append([s,a,r])
            if len(hist[i]) > self.obs_mem_h:
                hist[i].pop(0)

        # we return with hist, shape with history: (rows,cols, ch + h*ch)
        return state_h


    def add_history(self, states, actions, rewards):
        if not "obs_mem" in self.conf:
            return states

        bReset = len(actions) == 0
        if bReset:
            self.my_env.history = [[] for _ in range(self.num_agents)]
            actions = [0]*self.num_agents
            rewards = [0.0]*self.num_agents
            if self.num_agents == 1:          # consider one agent as list only at reset
                states = [states]             # because at step it is already called like that

        states_h = []
        for i, (s,a,r) in enumerate(zip(states, actions, rewards)):
            s = self.my_env.add_history(i,s,a,r)
            states_h.append(s)

        states = states_h

        #if self.num_agents == 1:  # undo:  consider one agent as list
        #    states = states[0]

        return np.array(states)


    def incr_t(self):
        self.t += 1
        self.my_env.t = self.t


    def steps(self, actions):
        self.incr_t()
        state, reward, done, info = self.my_env.steps(actions)
        done = done or self.t == self.max_steps
        self.stats_update(state, actions, reward, done)
        state = self.update_history(state, actions, reward)    # if history is active, update and build state:
        return state, reward, done, info


    def step(self, action):
        if self.num_agents == 1:
            self.incr_t()

        state, reward, done, info = self.my_env.step(action)
        done = done or self.t == self.max_steps
        self.stats_update(state, [action], reward, done)
        state_array = self.update_history([state], [action], [reward])

        return state_array[0], reward, done, info   #  Agent Dimension : the first is eliminated



    def set_figure(self, figure, params={}):
        self.fig = figure

    def render(self, fig_param=None, params={}):
        if(fig_param is not None):
            self.my_env.render(fig_param=fig_param, params=params)
        else:
            if self.fig is not None:
                self.my_env.render(fig_param=self.fig, params=params)
            else:
                if len(params) > 0:
                    self.my_env.render(params=params)
                else:
                    return self.my_env.render()

    def close(self):
        self.my_env.close()



    def make_id(self, run=None, iSeed=None, bFolder=False, post_fix=None):
        conf = self.conf
        str_all = conf["title"]+"-" if "title" in conf else ""
        if hasattr(self.my_env, "make_id"):
            str_all += self.my_env.make_id()

        str_all += "-%dag"%self.num_agents

        if "obs_mem" in conf and conf["obs_mem"] is not OBS_MEM.SAME:
            str_all += "-%dh"%conf["obs_mem_h"]  

        if run is not None:
            run = conf["run"]
            #str_all += "-%de" % run["num_episodes"]
            str_alg = ["q", "dynaq", "rforce", "a2c"]
            if "algorithm" in run:
                str_all += "-%s"%str_alg[int(run["algorithm"])]
            if "net_version" in run:
                str_all += "-net_%s"%run["net_version"]

        if post_fix is not None:
            str_all += '-' + post_fix

        if iSeed is not None:
            str_all += "-seed%02d"%iSeed

        if bFolder:
            str_all += '/'

        return str_all



    def ini_observation_seed(self, seed=None, debug=False):
        # for seeding / history / keras needs reversed obs / i_episode counting = 0
        # environment may have been created without reversion
        if "run" not in self.conf:
            self.conf["run"] = {}            

        conf, run = self.conf, self.conf["run"]
        num_agents = conf["num_agents"] if "num_agents" in conf else 1

        if seed == None:
            seed = np.random.randint(100)
            if "seed" in run:
                del run["seed"]
            if "RandomState" in run:
                del run["RandomState"]

        obs = self.my_env.reset()
        if debug:
            print("Initializse Observation Shape Before: ", obs.shape, "   (ch, obs_rows, obs_cols),    obs_row = 1 + 2*obs_radius")


        if "obs_shape" in run:
            conf["obs_shape"] = run["obs_shape"]
        if "obs_shape" in conf:
            run["obs_shape"] = conf["obs_shape"]

        bReverse = False
        if "obs_shape" in conf:
            self.obs_shape = conf["obs_shape"]
            self.my_env.obs_shape = conf["obs_shape"]
            print(" -Obs Reversing applied: ", self.obs_shape)
            bReverse = True

        if "obs_mem" in conf:
            run["obs_mem"] = conf["obs_mem"]
            self.obs_mem = conf["obs_mem"]
            self.obs_mem_h = conf["obs_mem_h"] if "obs_mem_h" in self.conf else 1
            self.my_env.obs_mem = self.obs_mem
            self.my_env.obs_mem_h = self.obs_mem_h
            self.my_env.history = [[] for _ in range(self.num_agents)]
            print(" -Obs Memory applied: ", self.obs_mem, "memory history:", self.obs_mem_h)

        obs = self.reset()   # We need to call Environment reset and not self.my_env.reset
        self.my_env.observation_space = Box(low=np.inf, high=np.inf, shape=obs.shape, dtype=np.float)
        self.observation_space = self.my_env.observation_space

        if debug:
            str_shape = "  (num_agents, obs_rows, obs_cols, ch)" if bReverse else "  (num_agents, ch, obs_rows, obs_cols)"
            print("----------- Observation Shape After:", obs.shape,  str_shape)

        self.i_episode = -1
        seed_all(run, seed)


    #  ------------------------------
    #  Dirty Access functions section

    def index2state(self, index):
        return np.unravel_index(index, self.my_env.grid_shape)

    def state2index(self, state):
        return np.ravel_multi_index(state, self.my_env.grid_shape)

    def getGridSize(self):
        if(self.basename in ["FruitCollection"]):
            return (self.my_env.scr_w, self.my_env.scr_h)
        if(self.basename in ["FinalGrid"]):
            return (self.my_env.rows, self.my_env.cols)
        return None

    def getAgentPos(self):
        if(self.basename in ["FruitCollection"]):
            return (self.my_env.player_pos_x, self.my_env.player_pos_y)
        if(self.basename in ["FinalGrid"]):
            return self.my_env.agent_xy
        return None

    def get_env(self):
        return self.my_env

    def get_actions(self):                        # only works for FinalGrid
        return self.my_env.actions

    def get_global_id(self):                      # only works for FinalGrid
        return self.my_env.get_global_id()

    def get_position_from_global(self, id):
        return self.my_env.get_position_from_global(id)

    def get_global_id_ego(self):                  # only works for FinalGrid
        return self.my_env.get_global_id_ego()

    def get_state_map(self):                      # only works for FinalGrid
        return self.my_env.layers
