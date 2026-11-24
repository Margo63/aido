from gym_duckietown.tasks.task_solution import TaskSolution
import cv2
import numpy as np

class DontCrushDuckieTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)
        self.step = 0

    def calculate_yellow_percentage(self, obs):
        img = np.ascontiguousarray(obs)

        mask_yellow = cv2.inRange(img, np.array([150, 150, 0]), np.array([255, 250, 180]))

        amount_yellow = cv2.countNonZero(mask_yellow)
        total = img.shape[0] * img.shape[1]
        print((amount_yellow / total) * 100)
        return (amount_yellow / total) * 100


    def solve(self):
        env = self.generated_task['env']

        linear_velocity = 0
        angular_velocity = 0
        obs, _, _, _ = env.step([linear_velocity, angular_velocity])
        env.render()

        done = False
        while not done:
            done = self.calculate_yellow_percentage(obs) > 10
            if done:
                linear_velocity = 0
            else:
                linear_velocity = 1

            obs, _, _, _ = env.step([linear_velocity, angular_velocity])
            env.render()



if __name__ == "__main__":
    from gym_duckietown.tasks.default.task_generator import DefaultTaskGenerator

    task_generator = DefaultTaskGenerator()
    task_generator.generate_task(map_name='duckie.yaml', initial_pose=(0,0))
    solution = DontCrushDuckieTaskSolution(task_generator.generated_task)
    solution.solve()