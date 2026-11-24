from gym_duckietown.tasks.task_solution import TaskSolution
import cv2
import numpy as np

class LaneCenterTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)
        self.step = 0

    def calculate_yellow_pixels(self, obs):
        img = np.ascontiguousarray(obs)

        mask_yellow = cv2.inRange(img, np.array([165, 155, 0]), np.array([255, 240, 110]))

        amount_yellow = cv2.countNonZero(mask_yellow)
        print("amount_yellow: ", amount_yellow)
        vis = img.copy()
        vis[mask_yellow > 0] = [0, 255, 0]
        #cv2.imwrite(f"no_masked_green{self.step}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #cv2.imwrite(f"masked_green{self.step}.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        self.step += 1
        return amount_yellow


    def solve(self):
        env = self.generated_task['env']
        linear_velocity = 0
        angular_velocity = 0
        obs, _, _, _ = env.step([linear_velocity, angular_velocity])
        env.render()

        done = False
        while not done:
            done = self.calculate_yellow_pixels(obs) > 8700
            if done:
                angular_velocity = 0
                linear_velocity = 1
            else:
                angular_velocity = 1
                linear_velocity = 0

            obs, _, _, _ = env.step([linear_velocity, angular_velocity])
            env.render()



if __name__ == "__main__":
    from gym_duckietown.tasks.default.task_generator import DefaultTaskGenerator

    task_generator = DefaultTaskGenerator()
    task_generator.generate_task(map_name='straight_road_no_start_tile')
    solution = LaneCenterTaskSolution(task_generator.generated_task)
    solution.solve()