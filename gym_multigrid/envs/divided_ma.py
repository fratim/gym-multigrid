from gym_multigrid.multigrid import *
from gym_multigrid.register import register

from gym_multigrid.basegrid import Grid
from gym_multigrid.objects import Agent, Box, Goal

class MADivided(MultiGridEnv):
    """
    Environment in which the agents have to fetch the balls and drop them in their respective goals
    """

    def __init__(
        self,
        agents_index,
        width=5,
        height=5,
        random_goals=True,
        max_steps=20,
        max_box_strength=4
    ):

        self.random_goals = random_goals
        self.max_box_strength = max_box_strength
        self.box_obj = None

        agents = []
        for i in agents_index:
            agents.append(Agent(i))

        super().__init__(
            width=width,
            height=height,
            max_steps=max_steps,
            agents=agents
        )

    def _gen_grid(self, configuration):
        #generate grid
        self.grid = Grid(self.width, self.height)

        # Generate the surrounding walls
        self.set_up_walls()

        # place box or leave middle cell empty
        box_strength = self.place_box(configuration)

        # determine and fix goal position
        if not self.use_special_reward:
            goal_pos = self.set_up_goal(configuration)

        # determine and fix agent position
        for ag in self.agents:
            self.set_up_agent(ag, configuration)

        # create mission statement
        self.mission = f"get to the goal by destroying the box of strength {box_strength}"

    def place_box(self, configuration):

        if configuration and configuration.box_strength is not None:
            box_strength = configuration.box_strength
        else:
            box_strength = np.random.randint(0, self.max_box_strength+1)

        self.grid.set(int(self.width / 2), int(self.height / 2), None)
        box_obj = Box(color="red", strength=box_strength)
        self.grid.add(int(self.width / 2), int(self.height / 2), box_obj)
        self.box_obj = box_obj

        return box_strength

    def set_up_goal(self, configuration):
        # Place a goal goal
        if configuration and configuration.goal_pos is not None:
            goal_pos = configuration.goal_pos
            assert self.position_is_possible(goal_pos)
        else:
            if self.random_goals:
                goal_pos = self.get_possible_location()
            else:
                goal_pos = (self.width - 2, self.height - 2)

        self.grid.add(*goal_pos, Goal(1))
        self.goal_pos = goal_pos

        return goal_pos

    def set_up_agent(self, ag, configuration):
        if configuration and configuration.agent_pos[ag.index] is not None:
            assert configuration.agent_dir[ag.index] is not None
            self.place_agent(agent=ag, pos=configuration.agent_pos[ag.index], dir=configuration.agent_dir[ag.index])
        else:
            self.place_agent(ag)

    def set_up_walls(self):
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, self.width, self.height)

        # Create a vertical splitting wall
        self.grid.vert_wall(int(self.width / 2), 0)

    def step(self, actions):
        obs, rewards, done, info = MultiGridEnv.step(self, actions)
        return obs, rewards, done, info


class SingleDivided_Index0_5x5(MADivided):
    def __init__(self):
        super().__init__(agents_index=[0])

class SingleDivided_Index1_5x5(MADivided):
    def __init__(self):
        super().__init__(agents_index=[1])

class MultiDivided5x5(MADivided):
    def __init__(self):
        super().__init__(agents_index=[0, 1])

register(
            id='multigrid-1agents-index0-divided-v0',
            entry_point='gym_multigrid.envs:SingleDivided_Index0_5x5',
        )

register(
            id='multigrid-1agents-index1-divided-v0',
            entry_point='gym_multigrid.envs:SingleDivided_Index1_5x5',
        )

register(
            id='multigrid-2agents-divided-v0',
            entry_point='gym_multigrid.envs:MultiDivided5x5',
        )
