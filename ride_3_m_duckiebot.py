from gym_duckietown.tasks.task_solution import TaskSolution
import cv2
import numpy as np

class Ride3MDuckiebotTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)
        self.count = 0
        self.step = 0

    def calculate_pixels(self, obs, left_side_color, right_side_color):
        img = np.ascontiguousarray(obs)

        mask = cv2.inRange(img, left_side_color, right_side_color)

        amount = cv2.countNonZero(mask)
        print("amount:", amount)

        self.step += 1
        return amount

#суть стоим пока видим много красного(можно было бы и синий) иначе едем, 75 шагов для 3 метров со скоростью 1
    def solve(self):
        env = self.generated_task['env']

        linear_velocity = 0.0
        angular_velocity = 0.0
        obs, _, _, _ = env.step([linear_velocity, angular_velocity])
        env.render()

        while self.count < 75:
            find = self.calculate_pixels(obs, np.array([150, 0, 0]), np.array([255, 80, 80])) > 7000
            if not find:
                angular_velocity = 0
                linear_velocity = 1
                self.count += 1
                print(self.count)
            else:
                linear_velocity = 0.0



            obs, _, _, _ = env.step([linear_velocity, angular_velocity])
            env.render()

if __name__ == "__main__":
    from gym_duckietown.tasks.default.task_generator import DefaultTaskGenerator

    task_generator = DefaultTaskGenerator()
    task_generator.generate_task(map_name='duckiebots.yaml')
    solution = Ride3MDuckiebotTaskSolution(task_generator.generated_task)
    solution.solve()
