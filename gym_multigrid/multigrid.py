import gym
from gym import spaces
from gym.utils import seeding
from .rendering import *
from .window import Window
import numpy as np
import pickle

#
ENCODE_DIM = 6


# TODO create goal object that is agent specific


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
