import math
import gym
from enum import IntEnum
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from .rendering import *
from .window import Window
import numpy as np
import copy
import pickle

#
ENCODE_DIM = 6

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32


agent_0_impaired = True


# Map of color names to RGB values
COLORS = {
    'red': np.array([255, 0, 0]),
    'green': np.array([0, 255, 0]),
    'grey': np.array([100, 100, 100])
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {
    'red': 0,
    'green': 1,
    'grey': 2,
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
# evaluate whther it makes sense that each agent has its own "type encoding" to maintain 3-dim encoding structure
OBJECT_TO_IDX = {
    'wall': 0,
    'door': 1,
    'box': 2,
    'goal': 3,
    'agent': 4
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))


# Map of state names to integers
STATE_TO_IDX = {
    'open': 0,
    'closed': 1,
    'locked': 2,
}

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

def three_digit_encoding(type, color, specification):

    encoding = type*100 + color*10 + specification

    return int(encoding)




class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None


    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self):
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return three_digit_encoding(OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError


class Goal(WorldObj):
    def __init__(self, index, reward=1, color=None):
        if color is None:
            super().__init__('goal', IDX_TO_COLOR[index])
        else:
            super().__init__('goal', IDX_TO_COLOR[color])
        self.index = index
        self.reward = reward

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Wall(WorldObj):
    def __init__(self, color='grey'):
        super().__init__('wall', color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Box(WorldObj):
    def __init__(self, color, contains=None, strength=2):
        super(Box, self).__init__('box', color)
        self.contains = contains
        self.strength = strength

        if self.strength == 0:
            self.color = "green"

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""

        assert self.strength >= 0

        if self.strength == 0:
            return True
        else:
            return False

    def can_pickup(self):
        return False

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self):
        # decrease strength of box
        if self.strength > 0:
            self.strength -= 1

        if self.strength == 0:
            self.color = "green"

        return True
    
    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""

        return three_digit_encoding(OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], self.strength)


class Agent(WorldObj):
    def __init__(self, index):
        super(Agent, self).__init__('agent', IDX_TO_COLOR[index])
        self.pos = None
        self.dir = None
        self.index = index

    def render(self, img):
        c = COLORS[self.color]
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )
        # Rotate the agent based on its direction
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.dir)
        fill_coords(img, tri_fn, c)

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return three_digit_encoding(OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], self.dir)

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert self.dir >= 0 and self.dir < 4
        return DIR_TO_VEC[self.dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.pos + self.dir_vec

    def can_overlap(self):
        """
        can the agent overlap?
        """
        return True


# TODO create goal object that is agent specific

class GridCell:
    value = None

    def __init__(self):
        self.value = []

    def set(self, v):
        if v is None:
            self.value = []
        else:
            assert isinstance(v, list)
            self.value = copy.deepcopy(v)

    def add(self, obj):
        assert not isinstance(obj, list)

        # check if object to be added is a wall, if wall is already given, return
        if isinstance(obj, Wall):
            for item in self.value:
                if isinstance(item, Wall):
                    return

        assert self.can_overlap()

        self.value.append(obj)

        assert len(self.value) <= 2

    def can_overlap(self):
        for item in self.value:
            if not item.can_overlap():
                return False
        return True

    def remove(self, obj):
        assert obj in self.value
        self.value.remove(obj)

    def get(self):
        return self.value if self.value != [] else None

    def isGoal(self):
        for item in self.value:
            if isinstance(item, Goal):
                return True
        return False

    def toggle(self):
        for item in self.value:
            item.toggle()


    def encode(self, current_agent=None):

        # TODO simplify and speed up the encoding process

        assert len(self.value) <= 2

        encoding = np.ones((ENCODE_DIM//3,), dtype=np.int64)*99999

        for i, obj in enumerate(self.value):

            if current_agent is not None and isinstance(obj, Agent) and obj.index != current_agent:
                continue

            encoding[i] = obj.encode()

        encoding = np.sort(encoding)
        encoding[np.argwhere(encoding == 99999)] = 0

        final_encoding = []

        for i in range(encoding.shape[0]):
            nstring = str(encoding[i]).zfill(3)
            nstrings = [int(nstring[i]) for i in range(len(nstring))]
            final_encoding += nstrings

        return tuple(final_encoding)


class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height

        self.grid = [GridCell() for _ in range(width*height)]

    def copy(self):
        return copy.deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i].set(v)

    def add(self, i, j, obj):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i].add(obj)

    def remove(self, i, j, obj):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i].remove(obj)

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i].get()

    def get_cell(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

    def encode_cell(self, i, j, current_agent=None):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i].encode(current_agent)

    def horz_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.add(x + i, y, obj_type())

    def vert_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.add(x, y + j, obj_type())

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y + h - 1, w)
        self.vert_wall(x, y+1, h-2)
        self.vert_wall(x + w - 1, y+1, h-2)

    @classmethod
    def render_tile(
            cls,
            cell,
            tile_size=TILE_PIXELS,
            subdivs=3
    ):
        """
        Render a tile and cache the result
        """

        key = cell.encode()

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if cell.get():
            for itm in cell.get():
                itm.render(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(
            self,
            tile_size
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get_cell(i, j)

                tile_img = Grid.render_tile(
                    cell,
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def encode(self, current_agent=None):
        """
        Produce a compact numpy encoding of the grid
        """

        array = np.zeros((self.width, self.height, ENCODE_DIM), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                array[i, j, :] = self.encode_cell(i, j, current_agent=current_agent)

        return array


class Actions:

    available=['still', 'left', 'right', 'forward', 'toggle']

    still = 0
    left = 1
    right = 2
    forward = 3
    toggle = 4


class MultiGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10
    }

    def __init__(
            self,
            width=None,
            height=None,
            max_steps=20,
            seed=2,
            agents=None,
            actions_set=Actions
    ):
        self.agents = agents

        self.use_special_reward = False

        # Action enumeration for this environment
        self.actions = actions_set

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions.available))

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(width, height, ENCODE_DIM),
            dtype='uint8' # TODO revert this?
        )

        self.ob_dim = np.prod(self.observation_space.shape)
        self.ac_dim = self.action_space.n

        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps

        # Initialize the RNG
        self.seed(seed=seed)

        # Initialize the state
        self.reset()

        self.special_rewards = self.read_special_rewards()

    def read_special_rewards(self):
        fname = "/Users/tim/Code/blocks/rl-starter-files/storage/box4/v2-s9-ppo_box4g9/mean_values.pickle"
        with open(fname, "rb") as output_file:
            mean_values = pickle.load(output_file)

        return mean_values

    def make_observation(self):
        # Return first observation
        obs = [self.grid.encode(current_agent=agent.index) for agent in self.agents]

        if len(obs) == 1:
            return obs[0]
        else:
            return obs


    def reset(self, configuration=None):

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(configuration)

        # These fields should be defined by _gen_grid
        for a in self.agents:
            assert a.pos is not None
            assert a.dir is not None

        # Step count since episode start
        self.step_count = 0

        return self.make_observation()


    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def _gen_grid(self, width, height):
        assert False, "_gen_grid needs to be implemented by each environment"

    def _reward(self):
        """
        Compute the reward to be given upon success
        """

        return 1 - 0.9 * (self.step_count / self.max_steps)

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """

        return self.np_random.randint(low, high)

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]


    def position_is_possible(self, pos, reject_fn=None):
        # Don't place the object on top of another object
        # TODO modify this to allow multiple objects/agent in the same place
        if self.grid.get(*pos) != None:
            return False

        # Check if there is a filtering criterion
        elif reject_fn and reject_fn(self, pos):
            return False

        else:
            return True

    def get_possible_location(self, top=None, size=None, reject_fn=None, max_tries=1000):

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
            ))

            if self.position_is_possible(pos, reject_fn):
                break

        return pos

    def place_obj(self,
        obj,
        top=None,
        size=None,
        reject_fn=None
    ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        pos = self.get_possible_location(top, size, reject_fn)

        self.grid.add(*pos, obj)

        return pos

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.add(i, j, obj)

    def place_agent(
            self,
            agent,
            top=None,
            size=None,
            rand_dir=True,
            max_tries=math.inf
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """

        agent.pos = None
        pos = self.place_obj(agent, top, size)
        agent.pos = pos

        if rand_dir:
            agent.dir = self._rand_int(0, 4)

        agent.init_dir = agent.dir

        return pos


    def step(self, actions):

        if not isinstance(actions, list):
            actions = [actions]

        self.step_count += 1

        order = np.random.permutation(len(actions))

        rewards = np.zeros(len(actions))
        done = False

        for i in order:

            # Get the position in front of the agent
            fwd_pos = self.agents[i].front_pos

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get_cell(*fwd_pos)

            # Rotate left
            if actions[i] == self.actions.left:
                self.agents[i].dir -= 1
                if self.agents[i].dir < 0:
                    self.agents[i].dir += 4

            # Rotate right
            elif actions[i] == self.actions.right:
                self.agents[i].dir = (self.agents[i].dir + 1) % 4

            # Move forward
            elif actions[i] == self.actions.forward:
                if fwd_cell is not None:
                    if fwd_cell.isGoal():
                        done = True
                        rewards[i] = self._reward()
                    elif fwd_cell.can_overlap():
                        self.grid.add(*fwd_pos, self.agents[i])
                        self.grid.remove(*self.agents[i].pos, self.agents[i])
                        self.agents[i].pos = fwd_pos
                elif fwd_cell.get() is None:
                    self.grid.add(*fwd_pos, self.agents[i])
                    self.grid.remove(*self.agents[i].pos, self.agents[i]) # TODO define a move agent function instead
                    self.agents[i].pos = fwd_pos

            # Toggle/activate an object
            elif actions[i] == self.actions.toggle:
                fwd_cell.toggle()


        if self.step_count >= self.max_steps:
            done = True

        obs = [self.grid.encode(current_agent=agent.index) for agent in self.agents]

        if self.use_special_reward:
            rewards[i] = self.special_rewards[self.agents[i].pos[1], self.agents[i].pos[0], self.box_obj.strength]

        if len(obs) == 1:
            assert len(rewards) == 1

            return obs[0], rewards[0], done, {}
        else:
            return obs, rewards, done, {}


    def get_obs_render(self, obs, tile_size=TILE_PIXELS // 2):
        """
        Render an agent observation for visualization
        """

        grid = Grid.decode(obs)

        # Render the whole grid
        img = grid.render(
            tile_size
        )

        return img

    def render(self, mode='human', close=False, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            self.window = Window('gym_multigrid')
            self.window.show(block=False)

        # Render the whole grid
        img = self.grid.render(
            tile_size
        )

        if mode == 'human':
            self.window.show_img(img)

        return img

    def render_blank_image(self):
        if self.window:
            white_image = np.ones((self.grid.height * TILE_PIXELS, self.grid.width * TILE_PIXELS, 3))
            self.window.show_img(white_image)

    def close(self):
        if self.window:
            self.window.close()
        return
