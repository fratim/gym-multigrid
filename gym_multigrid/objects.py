import numpy as np
from .rendering import *

# Map of object type to integers
# evaluate whther it makes sense that each agent has its own "type encoding" to maintain 3-dim encoding structure
OBJECT_TO_IDX = {
    'wall': 0,
    'door': 1,
    'box': 2,
    'goal': 3,
    'agent': 4
}


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


IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

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

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
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
    def __init__(self, index, reward=1, color=None, for_agents_with_ind=None):
        if color is None:
            super().__init__('goal', IDX_TO_COLOR[index])
        else:
            super().__init__('goal', IDX_TO_COLOR[color])
        self.index = index
        self.reward = reward
        self.for_agents_with_ind = for_agents_with_ind

    def for_agent(self, ag):
        if self.for_agents_with_ind is None:
            return True
        elif ag.index in self.for_agents_with_ind:
            return True
        else:
            return False

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
    def __init__(self, color, strength, agents_that_can_toggle=None):
        super(Box, self).__init__('box', color)
        self.strength = strength
        self.agents_that_can_toggle = agents_that_can_toggle

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

    def toggle(self, ag):

        if self.agents_that_can_toggle is not None and ag.index not in self.agents_that_can_toggle:
            return False

        else:
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