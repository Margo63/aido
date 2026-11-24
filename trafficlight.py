from gym_duckietown.tasks.task_solution import TaskSolution
from gym_duckietown.objects import TrafficLightObj
import cv2
import numpy as np


class TrafficlightTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)

    def can_drive(self, env):
        traffic_lights = [
            obj for obj in env.unwrapped.objects
            if isinstance(obj, TrafficLightObj)
        ]
        green = any(light.pattern == 1 for light in traffic_lights)

        return green

    def solve(self):
        env = self.generated_task['env']
        simulator = env.unwrapped

        simulator.cur_pos = np.array([0.8, 0.0, 0.2925])
        simulator.cur_angle = 0.0

        linear_velocity = 0.2
        angular_velocity = 0
        obs, _, _, _ = env.step([linear_velocity, angular_velocity])
        env.render()

        while True:
            if not self.can_drive(env):
                linear_velocity = 0.5
            else:
                linear_velocity = 0
            obs, _, _, _ = env.step([linear_velocity, angular_velocity])
            env.render()

if __name__ == "__main__":
    from gym_duckietown.tasks.default.task_generator import DefaultTaskGenerator

    task_generator = DefaultTaskGenerator()
    env = task_generator.generate_task(map_name='trafficmap.yaml')

    solution = TrafficlightTaskSolution(task_generator.generated_task)
    solution.solve()
