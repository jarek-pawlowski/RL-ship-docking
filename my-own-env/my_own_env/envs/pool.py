import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

import skimage

class Poll2DEnv(gym.Env):
    """
    Poll2DEnv contains SIDE_LENGTH+2 x SIDE_LENGTH+2 lattice
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, L: int = 10, max_steps: int = 10):
        """Initialization of the gym environment"""
        # lattice L x L
        self.L = L
        self.max_steps = max_steps
        self.step_no = 1
        # initialize pool environment
        self.environment, self.destination = self.generate_pool_env()
        # ship possition
        self.position_space = spaces.MultiDiscrete([self.L, self.L])
        self.position = self.position_space.sample()+1
        # measurements
        self.observation_space = spaces.MultiDiscrete([self.L, self.L, self.L, self.L, 2*self.L+1, 2*self.L+1, 4])   # change to composite (two last should be box-type)
        self.state = self.observation_space.sample()
        self.init_distance = self.distance_to_destiantion()
        self.update_state() 
        # actions = movements:
        self.action_space = spaces.Discrete(4)  # four possible movements: up/down/right/left

    def generate_pool_env(self):
        """
        simple pool environement: 
        - single obstracle in form of a line with size no longer than 2/3 of the pool size
        - and destination randomly located on one of the edges
        """
        pool = np.zeros((self.L+2, self.L+2), dtype=np.int8)
        # boundary
        pool[0,:]  = 1  
        pool[-1,:] = 1
        pool[:,0]  = 1
        pool[:,-1] = 1
        obstracle_length = self.L
        while obstracle_length > 0.6*self.L:
            coords = np.random.randint(self.L, size=4)
            obstracle_length = np.sqrt((coords[2]-coords[0])**2+(coords[3]-coords[1])**2)
        obstracle = skimage.draw.line(*coords)
        pool[obstracle] = 1
        # destination
        select_edge = np.random.randint(4)
        if select_edge == 0:
            destination = np.array([self.L+1, 2+np.random.randint(self.L-1)]).astype(np.int8)
        elif select_edge == 1:
            destination = np.array([0, 2+np.random.randint(self.L-1)]).astype(np.int8)
        elif select_edge == 2:
            destination = np.array([2+np.random.randint(self.L-1), 0]).astype(np.int8)
        else:
            destination = np.array([2+np.random.randint(self.L-1), self.L+1]).astype(np.int8)
        #destination = np.array([0,5]).astype(np.int8)
        pool[destination[0], destination[1]] = 2 
        return pool, destination

    def measure_distance(self):
        # calculate 4 distances to nearby obstracle
        [x,y] = self.position
        top_sonar = self.environment[:x,y][::-1]
        t_d = np.argwhere(top_sonar > 0)[0]
        bottom_sonar = self.environment[x+1:,y]
        b_d = np.argwhere(bottom_sonar > 0)[0]
        left_sonar = self.environment[x,:y][::-1]
        l_d = np.argwhere(left_sonar > 0)[0]
        right_sonar = self.environment[x,y+1:]
        r_d = np.argwhere(right_sonar > 0)[0]        
        return [t_d, b_d, l_d, r_d]
  
    def direction_to_destiantion(self):
        # calulate direction towards destination point
        return self.position - self.destination + self.L
    
    def distance_to_destiantion(self):
        # calulate distance to destination point
        return np.sqrt((self.destination[0]-self.position[0])**2+(self.destination[1]-self.position[1])**2)
    
    def update_state(self, action=None):
        measure = self.measure_distance()
        self.state[:4] = measure
        direction = self.direction_to_destiantion()
        self.state[4:6] = direction
        if action is not None:
            self.state[6] = action
    
    def render_pool(self):
        pool = self.environment.copy()
        pool[self.position[0], self.position[1]] = 3
        return pool

    def step(self, action):
        info = {}
        if action == 0:
            self.position[1] += 1  # right
        elif action == 1:
            self.position[1] -= 1  # left
        elif action == 2:
            self.position[0] -= 1  # up
        else:
            self.position[0] += 1  # down

        if self.environment[self.position[0], self.position[1]] == 1:  # ship reach boundary or obstacle
            done = True
            reward = -(self.max_steps - self.step_no) * 0.5 - self.distance_to_destiantion()
            reward += np.exp(5. - self.L * self.distance_to_destiantion() / self.init_distance)
        elif self.environment[self.position[0], self.position[1]] == 2:  # ship reach destination
            done = True
            reward = 1000.
        else:
            self.update_state(action)
            done = False
            reward = 0.

        self.step_no += 1
        if self.step_no > self.max_steps:
            done = True
            reward = -self.distance_to_destiantion()
            reward += np.exp(5. - self.L * self.distance_to_destiantion() / self.init_distance)

        return self.state.copy(), float(reward), bool(done), info

    def reset(self):
        self.environment, self.destination = self.generate_pool_env()
        self.position = self.position_space.sample()+1
        self.init_distance = self.distance_to_destiantion()
        self.update_state() 
        self.step_no = 1
        return self.state

    def render(self, mode="human"):
        print(f"{self.render_pool()}")
