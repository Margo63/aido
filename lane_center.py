from gym_duckietown.tasks.task_solution import TaskSolution
import cv2
import numpy as np

class LaneCenterTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)
        self.step = 0

    def calculate(self, obs):
        img = np.ascontiguousarray(obs)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        mask = cv2.inRange(
            hsv,
            np.array([15, 50, 70]),
            np.array([40, 255, 255])
        )

        height, width = mask.shape
        total = height * width
        yellow_pct = cv2.countNonZero(mask) / total * 100

        if yellow_pct < 0.3:
            return None, yellow_pct

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        def center_of_yellow(y1, y2):
            """Центр жёлтой массы по X в полосе [y1:y2]."""
            band = mask[y1:y2, :]
            ys, xs = np.nonzero(band)
            if len(xs) < 30:
                return None
            return float(np.mean(xs))

        # ближняя полоса (низ кадра) и дальняя (верх)
        y_near_1, y_near_2 = int(height * 0.65), int(height * 0.85)
        y_far_1, y_far_2 = int(height * 0.40), int(height * 0.60)

        center_near = center_of_yellow(y_near_1, y_near_2)
        center_far = center_of_yellow(y_far_1, y_far_2)

        if center_near is None or center_far is None:
            return None, yellow_pct

        position_error = (center_near - width / 2) / (width / 2)
        heading_error = (center_near - center_far) / (width / 2)

        return (position_error, heading_error), yellow_pct

    def control(self, position_error, heading_error, yellow_pct):
        K_POSITION = 0.5
        K_HEADING = 0.9
        TURN_SIGN = 1.0

        angular_velocity = TURN_SIGN * (
                K_POSITION * position_error +
                K_HEADING * heading_error
        )
        angular_velocity = np.clip(angular_velocity, -0.8, 0.8)

        # «преследование»: чем больше жёлтого — тем быстрее едем
        if yellow_pct > 8:
            base_speed = 0.12
        elif yellow_pct > 3:
            base_speed = 0.08
        else:
            base_speed = 0.04

        # при большом рассогласовании — медленнее, чтобы успеть выровняться
        err = abs(position_error) + abs(heading_error)
        if err > 0.4:
            linear_velocity = base_speed * 0.5
        elif err > 0.15:
            linear_velocity = base_speed * 0.8
        else:
            linear_velocity = base_speed

        return linear_velocity, angular_velocity

    def calculate_yellow_percentage(self, obs):
        img = np.ascontiguousarray(obs)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        mask_yellow = cv2.inRange(
            hsv,
            np.array([15, 50, 70]),
            np.array([40, 255, 255])
        )
        #mask_yellow = cv2.inRange(img, np.array([150, 150, 0]), np.array([255, 250, 180]))

        amount_yellow = cv2.countNonZero(mask_yellow)
        total = img.shape[0] * img.shape[1]
        print("yellow:", (amount_yellow / total) * 100)
        return (amount_yellow / total) * 100

    def solve(self):
        env = self.generated_task['env']
        obs, _, _, _ = env.step([0.0, 0.0])
        env.render()

        done = False
        while not done:
            result, yellow_pct = self.calculate(obs)

            if result is None:
                # жёлтого мало/не видно — крутимся, ищем
                linear_velocity = 0.0
                angular_velocity = 0.4
            else:
                position_error, heading_error = result
                linear_velocity, angular_velocity = self.control(
                    position_error, heading_error, yellow_pct
                )

            obs, _, _, _ = env.step([linear_velocity, angular_velocity])
            env.render()



if __name__ == "__main__":
    from gym_duckietown.tasks.default.task_generator import DefaultTaskGenerator

    task_generator = DefaultTaskGenerator()
    task_generator.generate_task(map_name='straight_road_no_start_tile')
    solution = LaneCenterTaskSolution(task_generator.generated_task)
    solution.solve()