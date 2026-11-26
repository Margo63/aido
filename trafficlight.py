from gym_duckietown.tasks.task_solution import TaskSolution
from gym_duckietown.objects import TrafficLightObj
import cv2
import numpy as np


class TrafficlightTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)

    def nearest_light(self, env, x):
        lights = [obj for obj in env.objects if isinstance(obj, TrafficLightObj)]

        if not lights:
            return None

        ahead = [l for l in lights if l.pos[0] >= x - 1e-3]

        if not ahead:
            return None

        return min(ahead, key=lambda l: l.pos[0] - x)

    def can_drive(self, env, position_x):
        light = self.nearest_light(env, position_x)

        if light is None:
            return True

        dist = light.pos[0] - position_x

        if dist > 1.1:
            return True

        return light.pattern == 0

    def solve(self):
        env = self.generated_task['env']

        simulator = env.unwrapped
        simulator.cur_pos = np.array([0.8, 0.0, 0.2925])
        simulator.cur_angle = 0.0

        angular_velocity = 0.0

        obs, _, _, _ = env.step([0.0, angular_velocity])
        env.render()

        TILE = 0.585
        start_x = 0.2925
        target_x = 20 * TILE + start_x

        while True:
            if simulator.cur_pos[0] >= target_x:
                break

            if self.can_drive(env, simulator.cur_pos[0]):
                linear_velocity = 0.5
            else:
                linear_velocity = 0.0

            obs, _, done, _ = env.step([linear_velocity, angular_velocity])
            env.render()

            if done:
                break


if __name__ == "__main__":
    from gym_duckietown.tasks.default.task_generator import DefaultTaskGenerator

    task_generator = DefaultTaskGenerator()
    env = task_generator.generate_task(map_name='trafficmap.yaml')

    solution = TrafficlightTaskSolution(task_generator.generated_task)
    solution.solve()
