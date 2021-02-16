import os
from copy import deepcopy
import pygame
import numpy as np
import random
import collections
import click


# RGB colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
GREEN = (0, 255, 100)
WALL = (80, 80, 80)

class KeyCollection(object):
    def __init__(self, game_length = 50, size = 5, is_key=True, num_keys=5, rng=None, rendering=False):

        self.game_length = game_length
        self.is_key = is_key
        self.num_keys = num_keys
        if rng is None:
            self.rng= np.random.RandomState(1234)
        else:
            self.rng = rng
        self.lives=1
        self.legal_actions = [0, 1, 2, 3]
        self.action_meanings = ['up', 'down', 'left', 'right']
        self.num_actions = len(self.legal_actions)

        self.player_pos_x = int()
        self.player_pos_y = int()
        self.agent_init_pos = None
        self.step_reward = -0.01
        self.state_mode = "features"
        self.possible_keys = []

        self.scr_w = size
        self.scr_h = size

        self.rendering_scale = 100
        #self.init_with_mode()
        self.walls = []
        self.num_non_wall = self.scr_w * self.scr_h - len(self.walls)

        self._rendering = rendering

        if rendering:
            self._init_pygame()

        self.active_targets = None  # boolean list
        self.active_keys = None

        self.state_shape = None
        self.state = None
        self.step_id = 0
        self.game_over = False
        self.reset()

    def close(self):
        if self.rendering:
            pygame.quit()

    def _init_pygame(self):
        pygame.init()
        size = [self.rendering_scale * self.scr_w, self.rendering_scale * self.scr_h]
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("Key Collection")

    @property
    def rendering(self):
        return self._rendering

    @rendering.setter
    def rendering(self, flag):
        if flag is True:
            if self._rendering is False:
                self._init_pygame()
                self._rendering = True
        else:
            self.close()
            self._rendering = False

    def get_state(self):

        if self.state_mode == 'pixel':
            return self.get_state_pixel()

        elif self.state_mode == 'flat':
            return self.get_state_flat()

        elif self.state_mode =="features":
            return self.get_state_features()

    def get_state_features(self):
        # multiple binary heads: player, walls, keys*num_keys
        features = []

        #for player pos:
        player_state = np.zeros((self.scr_w, self.scr_h), dtype=np.int8)
        player_state[self.player_pos_x,self.player_pos_y] = 1
        #print("PLayer state: ", self.player_pos_x, self.player_pos_y, player_state)
        features.append(player_state)

        #for keys:
        for key_id in self.all_keys:
            key_state = np.zeros((self.scr_w, self.scr_h), dtype=np.int8)
            if key_id in self.active_keys:
                #print(self.pos_dict[key_id])
                key_state[self.pos_dict[key_id][0],self.pos_dict[key_id][1]] = 1
                #print("Key state: ", key_state)
                features.append(key_state)
            else:
                features.append(np.zeros((self.scr_w, self.scr_h), dtype=np.int8))


        #for walls:
        #features.append(np.zeros((self.scr_w, self.scr_h), dtype=np.int8))

        state = np.stack(features, axis=0)

        #fstate = np.asarray(state, dtype=np.uint8).reshape(self.scr_h, self.scr_w , self.num_keys+1)
        return state

    def reset(self):
        self.pos_dict = {}
        self.col_dict = {}
        self.active_keys = collections.deque()
        self.all_keys = []
        self.game_over = False
        self.step_id = 0

        for i in range(self.num_keys):
            key_x = random.randint(0,self.scr_w-1)
            key_y = random.randint(0,self.scr_h-1)
            key_pos = [key_x, key_y]

            while key_pos in self.pos_dict.values():
                key_x = random.randint(0, self.scr_w - 1)
                key_y = random.randint(0, self.scr_h - 1)
                key_pos = [key_x, key_y]

            key_id = "key" + str(i)
            key_hue = (i+1) * (240/(self.num_keys+1))
            key_color = (0,0,key_hue)
            self.active_keys.append(key_id)
            self.all_keys.append(key_id)
            self.pos_dict[key_id] = key_pos
            self.col_dict[key_id] = key_color

        position = [self.player_pos_x,self.player_pos_y]

        while position in self.pos_dict.values():
            self.player_pos_x = random.randint(0,self.scr_w-1)
            self.player_pos_y = random.randint(0,self.scr_h-1)
            position = [self.player_pos_x, self.player_pos_y]

        if self.state_mode == 'pixel':
            self.state_shape = [3, self.scr_w, self.scr_h]
        elif self.state_mode == 'flat':
            layers = self.num_keys + 2 #keys + player_pos + walls
            self.state_shape = [layers * self.scr_w * self.scr_h]
        elif self.state_mode == "features":
            layers = self.num_keys + 2 #keys + player_pos + walls
            self.state_shape = [layers, self.scr_w, self.scr_h]

        self.get_state()

    def _move_player(self, action):
        assert action in self.legal_actions, 'Illegal action.'

        if action == 3:  # right
            if [self.player_pos_x + 1, self.player_pos_y] not in self.walls and self.player_pos_x < self.scr_w - 1:
                self.player_pos_x += 1
            else:
                self.player_pos_x += 0

        elif action == 2:  # left
            if [self.player_pos_x - 1, self.player_pos_y] not in self.walls and self.player_pos_x > 0:
                self.player_pos_x -= 1
            else:
                self.player_pos_x -= 0

        elif action == 1:  # down
            if [self.player_pos_x, self.player_pos_y + 1] not in self.walls and self.player_pos_y < self.scr_h - 1:
                self.player_pos_y += 1
            else:
                self.player_pos_y += 0

        elif action == 0:  # up
            if [self.player_pos_x, self.player_pos_y - 1] not in self.walls and self.player_pos_y > 0:
                self.player_pos_y -= 1
            else:
                self.player_pos_y -= 0

    def _check_key(self):

        on_key = False
        key_id = None

        for id, pos in self.pos_dict.items():
            if pos == [self.player_pos_x, self.player_pos_y]:
                on_key = True
                key_id = id

        return on_key, key_id

    def _grab_key(self, key_id):
        #print(self.active_keys)
        #print("POS: ", self.player_pos_x, self.player_pos_y, " KEY: ", key_id, "POS: ", self.pos_dict[key_id])
        self.active_keys.popleft()
        #print(self.active_keys)
        del self.pos_dict[key_id]
        del self.col_dict[key_id]


    def step(self, action):
        # actions: [0, 1, 2, 3] == [up, down, left, right]
        if self.game_over:
            raise ValueError('Environment has already been terminated.')

        if self.step_id >= self.game_length - 1:
            self.game_over = True

            return self.get_state(), 0., self.game_over, None

        self._move_player(action)

        key_reward = 0

        on_key, key_id = self._check_key()

        if on_key == True:
            if key_id == self.active_keys[0]:
                key_reward = (self.num_keys - len(self.active_keys)) + 1
                self._grab_key(key_id)

        if not self.active_keys:
            self.game_over = True

        if self.lives == 0:
            self.game_over = True

        self.step_id += 1

        return self.get_state(), key_reward, self.game_over, None

    def render(self):

        if not self.rendering:
            return

        pygame.event.pump()
        self.screen.fill(BLACK)
        size = [self.rendering_scale, self.rendering_scale]

        for idx, pos in self.pos_dict.items():

            p = [self.rendering_scale * pos[0], self.rendering_scale * pos[1]]
            gl = pygame.Rect(p[0], p[1], size[0], size[1])
            pygame.draw.rect(self.screen, self.col_dict[idx], gl)

        player = pygame.Rect(self.rendering_scale * self.player_pos_x, self.rendering_scale * self.player_pos_y,
                             size[0], size[1])
        player.inflate_ip(-1,-1)
        pygame.draw.ellipse(self.screen, WHITE, player)


        for wall_pos in self.walls:
            p = [self.rendering_scale * wall_pos[0], self.rendering_scale * wall_pos[1]]
            wall = pygame.Rect(p[0], p[1], size[0], size[1])
            pygame.draw.rect(self.screen, WALL, wall, width=1)

        pygame.display.flip()
