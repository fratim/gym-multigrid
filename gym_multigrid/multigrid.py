import math
import gym
from enum import IntEnum
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from .rendering import *
from .window import Window
import numpy as np

#
ENCODE_DIM = 3

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    'red': np.array([255, 0, 0]),
    'green': np.array([0, 255, 0]),
    'blue': np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey': np.array([100, 100, 100])
}

COLOR_NAMES = sorted(list(COLORS.keys()))

class World:

    # Used to map colors to integers
    COLOR_TO_IDX = {
        'red': 0,
        'green': 1,
        'blue': 2,
        'purple': 3,
        'yellow': 4,
        'grey': 5
    }

    IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

    # Map of object type to integers
    # evaluate whther it makes sense that each agent has its own "type encoding" to maintain 3-dim encoding structure
    OBJECT_TO_IDX = {
        'unseen': 0,
        'empty': 1,
        'wall': 2,
        'floor': 3,
        'door': 4,
        'key': 5,
        'ball': 6,
        'box': 7,
        'goal': 8,
        'lava': 9,
        'objgoal': 10,
        'switch': 11,
        'agent': 12
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


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, world, type, color):
        assert type in world.OBJECT_TO_IDX, type
        assert color in world.COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

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

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self, world):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], 0)

    @staticmethod
    def decode(type_idx, color_idx, state):
        assert False, "not implemented"

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError


class Goal(WorldObj):
    def __init__(self, world, index, reward=1, color=None):
        if color is None:
            super().__init__(world, 'goal', world.IDX_TO_COLOR[index])
        else:
            super().__init__(world, 'goal', world.IDX_TO_COLOR[color])
        self.index = index
        self.reward = reward

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Wall(WorldObj):
    def __init__(self, world, color='grey'):
        super().__init__(world, 'wall', color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Box(WorldObj):
    def __init__(self, world, color, contains=None, strength=2):
        super(Box, self).__init__(world,'box', color)
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

    def toggle(self, env, pos):
        # decrease strength of box
        if self.strength > 0:
            self.strength -= 1

        if self.strength == 0:
            self.color = "green"

        return True
    
    def encode(self, world):
        """Encode the a description of this object as a 3-tuple of integers"""

        return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], self.strength)


class Agent(WorldObj):
    def __init__(self, world, index=0, view_size=7):
        super(Agent, self).__init__(world, 'agent', world.IDX_TO_COLOR[index])
        self.pos = None
        self.dir = None
        self.index = index
        self.view_size = view_size
        self.carrying = None
        self.terminated = False
        self.started = True
        self.paused = False

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

    def encode(self, world):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (world.OBJECT_TO_IDX[self.type],
                world.COLOR_TO_IDX[self.color],
                self.dir)

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

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.view_size
        hs = self.view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = (rx * lx + ry * ly)
        vy = -(dx * lx + dy * ly)

        return vx, vy

    def get_view_exts(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """

        # Facing right
        if self.dir == 0:
            topX = self.pos[0]
            topY = self.pos[1] - self.view_size // 2
        # Facing down
        elif self.dir == 1:
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1]
        # Facing left
        elif self.dir == 2:
            topX = self.pos[0] - self.view_size + 1
            topY = self.pos[1] - self.view_size // 2
        # Facing up
        elif self.dir == 3:
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1] - self.view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + self.view_size
        botY = topY + self.view_size

        return (topX, topY, botX, botY)

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.view_size or vy >= self.view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None


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

        self.grid = [None] * width * height

    def __contains__(self, key):
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other):
        grid1 = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

    def horz_wall(self, world, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type(world))

    def vert_wall(self, world, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type(world))

    def wall_rect(self, world, x, y, w, h):
        self.horz_wall(world, x, y, w)
        self.horz_wall(world, x, y + h - 1, w)
        self.vert_wall(world, x, y, h)
        self.vert_wall(world, x + w - 1, y, h)

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, world, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and \
                        y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall(world)

                grid.set(i, j, v)

        return grid

    @classmethod
    def render_tile(
            cls,
            world,
            obj,
            tile_size=TILE_PIXELS,
            subdivs=3
    ):
        """
        Render a tile and cache the result
        """

        key = obj.encode(world) if obj else (0, 0, 0)

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj != None:
            obj.render(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(
            self,
            world,
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
                cell = self.get(i, j)

                # agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = Grid.render_tile(
                    world,
                    cell,
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def encode(self, world):
        """
        Produce a compact numpy encoding of the grid
        """

        array = np.zeros((self.width, self.height, ENCODE_DIM), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)

                if v is None:
                    array[i, j, 0] = world.OBJECT_TO_IDX['empty']
                    array[i, j, 1] = 0
                    array[i, j, 2] = 0

                else:
                    array[i, j, :] = v.encode(world)

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

    # Enumeration of possible actions

    def __init__(
            self,
            grid_size=None,
            width=None,
            height=None,
            max_steps=100,
            seed=2,
            agents=None,
            actions_set=Actions,
            objects_set = World
    ):
        self.agents = agents

        # Can't set both grid_size and width/height
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size

        # Action enumeration for this environment
        self.actions = actions_set

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions.available))

        self.objects = objects_set

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(width, height, ENCODE_DIM),
            dtype='uint8'
        )

        self.ob_dim = np.prod(self.observation_space.shape)
        self.ac_dim = self.action_space.n

        # Range of possible rewards
        self.reward_range = (0, 1)

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

    def reset(self, configuration=None):

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(configuration)

        # These fields should be defined by _gen_grid
        for a in self.agents:
            assert a.pos is not None
            assert a.dir is not None

        # Item picked up, being carried, initially nothing
        for a in self.agents:
            a.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Return first observation

        obs = [self.grid.encode(self.objects) for i in range(len(self.agents))]

        if len(obs)==1:
            return obs[0]
        else:
            return obs

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            'wall': 'W',
            'floor': 'F',
            'door': 'D',
            'key': 'K',
            'ball': 'A',
            'box': 'B',
            'goal': 'G',
            'lava': 'V',
        }

        # Short string for opened door
        OPENDED_DOOR_IDS = '_'

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {
            0: '>',
            1: 'V',
            2: '<',
            3: '^'
        }

        str = ''

        for j in range(self.grid.height):

            for i in range(self.grid.width):
                if i == self.agent_pos[0] and j == self.agent_pos[1]:
                    str += 2 * AGENT_DIR_TO_STR[self.agent_dir]
                    continue

                c = self.grid.get(i, j)

                if c == None:
                    str += '  '
                    continue

                if c.type == 'door':
                    if c.is_open:
                        str += '__'
                    elif c.is_locked:
                        str += 'L' + c.color[0].upper()
                    else:
                        str += 'D' + c.color[0].upper()
                    continue

                str += OBJECT_TO_STR[c.type] + c.color[0].upper()

            if j < self.grid.height - 1:
                str += '\n'

        return str

    def _gen_grid(self, width, height):
        assert False, "_gen_grid needs to be implemented by each environment"

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def _handle_build(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def _handle_special_moves(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def _handle_switch(self, i, rewards, fwd_pos, fwd_cell):
        pass

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

    def _rand_float(self, low, high):
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """

        return (self.np_random.randint(0, 2) == 0)

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable, num_elems):
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self):
        """
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(self, xLow, xHigh, yLow, yHigh):
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.randint(xLow, xHigh),
            self.np_random.randint(yLow, yHigh)
        )

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

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

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
        agent.init_pos = pos

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

            if self.agents[i].terminated or self.agents[i].paused or not self.agents[i].started or actions[i] == self.actions.still:
                continue

            # Get the position in front of the agent
            fwd_pos = self.agents[i].front_pos

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)

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
                    if fwd_cell.type == 'goal':
                        done = True
                        rewards[i] = self._reward()
                    elif fwd_cell.can_overlap():
                        self.grid.set(*fwd_pos, self.agents[i])
                        self.grid.set(*self.agents[i].pos, None)
                        self.agents[i].pos = fwd_pos
                    elif fwd_cell.type == 'switch':
                        raise NotImplementedError
                        #self._handle_switch(i, rewards, fwd_pos, fwd_cell)
                elif fwd_cell is None:
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.grid.set(*self.agents[i].pos, None)
                    self.agents[i].pos = fwd_pos

            # Toggle/activate an object
            elif actions[i] == self.actions.toggle:
                if fwd_cell:
                    fwd_cell.toggle(self, fwd_pos)

        if self.step_count >= self.max_steps:
            done = True

        obs = [self.grid.encode(self.world) for i in range(len(self.agents))]

        if len(obs) == 1:
            assert len(rewards) == 1

            return obs[0], rewards[0], done, {}
        else:
            return obs, rewards, done, {}

    # def gen_obs_grid(self):
    #     """
    #     Generate the sub-grid observed by the agents.
    #     This method also outputs a visibility mask telling us which grid
    #     cells the agents can actually see.
    #     """
    #
    #     grids = []
    #
    #     for a in self.agents:
    #
    #         topX, topY, botX, botY = a.get_view_exts()
    #
    #         grid = self.grid.slice(self.objects, topX, topY, a.view_size, a.view_size)
    #
    #         for i in range(a.dir + 1):
    #             grid = grid.rotate_left()
    #
    #         grids.append(grid)
    #
    #     return grids
    #
    # def gen_obs(self):
    #     """
    #     Generate the agent's view (partially observable, low-resolution encoding)
    #     """
    #
    #     grids = self.gen_obs_grid()
    #
    #     # Encode the partially observable view into a numpy array
    #     obs = [grid.encode_for_agents(self.objects, [grid.width // 2, grid.height - 1]) for grid in zip(grids)]
    #
    #     return obs

    def get_obs_render(self, obs, tile_size=TILE_PIXELS // 2):
        """
        Render an agent observation for visualization
        """

        grid = Grid.decode(obs)

        # Render the whole grid
        img = grid.render(
            self.objects,
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
            self.objects,
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
