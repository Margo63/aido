from gym_duckietown.tasks.task_solution import TaskSolution
import numpy as np
import cv2

class DontCrushPedestrianTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)

    def calculate_yellow_percentage(self, obs):
        img = np.ascontiguousarray(obs)
        h, w = img.shape[:2]

        y1 = int(h * 0.35)
        y2 = int(h * 0.55)
        x1 = int(w * 0.25)
        x2 = int(w * 0.75)

        img = img[y1:y2, x1:x2]

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
            if self.calculate_yellow_percentage(obs) > 10:
                linear_velocity = 0
            else:
                linear_velocity = 1

            done = env.unwrapped.cur_pos[0] > self.generated_task['target_coordinates'][-1][0]
            obs, _, _, _ = env.step([linear_velocity, angular_velocity])
            env.render()

if __name__ == "__main__":
    from gym_duckietown.tasks.default.task_generator import DefaultTaskGenerator

    task_generator = DefaultTaskGenerator()
    task_generator.generate_task(map_name='duckie.yaml', initial_pose=(0, 0))
    task_generator.generated_task['target_coordinates'] = np.array([0.8, 0.0, 1.2925])
    solution = DontCrushPedestrianTaskSolution(task_generator.generated_task)
    solution.solve()