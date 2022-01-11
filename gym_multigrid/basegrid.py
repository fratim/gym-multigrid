from .rendering import *
import numpy as np
import copy

from .objects import Goal, Wall, Agent

ENCODE_DIM = 6

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

    def move_object(self, obj, desired_pos):
        # add object at new position
        self.add(*desired_pos, obj)
        # remove object at old position
        self.remove(*obj.pos, obj)
        # update position of object
        obj.pos = desired_pos

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

