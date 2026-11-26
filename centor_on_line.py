from gym_duckietown.tasks.task_solution import TaskSolution
import cv2
import numpy as np

class LaneCenterTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)
        self.step = 0

    def control(self, position_error, heading_error):

        K_POSITION = 0.5
        K_HEADING = 0.8

        TURN_SIGN = 1.0

        angular_velocity = TURN_SIGN * (
                K_POSITION * position_error +
                K_HEADING * heading_error
        )

        angular_velocity = np.clip(
            angular_velocity,
            -0.8,
            0.8
        )

        # Уменьшаем скорость при больших ошибках
        error = abs(position_error) + abs(heading_error)

        if error > 0.5:
            linear_velocity = 0.04
        elif error > 0.2:
            linear_velocity = 0.07
        else:
            linear_velocity = 0.1

        return linear_velocity, angular_velocity

    def calculate(self, obs):
        img = np.ascontiguousarray(obs)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        mask_yellow = cv2.inRange(
            hsv,
            np.array([15, 50, 70]),
            np.array([40, 255, 255])
        )

        amount_yellow = cv2.countNonZero(mask_yellow)


        if amount_yellow < 250:
            print("amount yellow:", amount_yellow)
            return None


        kernel = np.ones((3, 3), np.uint8)

        mask_yellow = cv2.morphologyEx(
            mask_yellow,
            cv2.MORPH_CLOSE,
            kernel
        )

        height, width = mask_yellow.shape

        def get_yellow_x(y, band=25):

            y1 = max(0, y - band)
            y2 = min(height, y + band)

            projection = np.count_nonzero(
                mask_yellow[y1:y2, :],
                axis=0
            )

            xs = np.where(projection >= 2)[0]

            if len(xs) < 2:
                return None

            groups = np.split(
                xs,
                np.where(np.diff(xs) > 10)[0] + 1
            )

            groups = [g for g in groups if 2 <= len(g)]

            if not groups:
                return None

            group = min(groups, key=lambda g: abs(np.mean(g) - width / 2))

            return float(np.mean(group))

        y_near = int(height * 0.72)
        y_far = int(height * 0.52)

        yellow_near = get_yellow_x(y_near)
        yellow_far = get_yellow_x(y_far)

        if yellow_near is None or yellow_far is None:
            return None

        if yellow_near > width / 2:
            return {
                "direction_ok": False,
                "position_error": None,
                "heading_error": None,
                "yellow_near": yellow_near
            }

        # Желаемая траектория справа от жёлтой линии
        OFFSET_NEAR = 100
        OFFSET_FAR = 35

        target_near = yellow_near + OFFSET_NEAR
        target_far = yellow_far + OFFSET_FAR

        # Ошибка положения
        position_error = (target_near - width / 2) / (width / 2)

        # Ошибка ориентации
        heading_error = (target_near - target_far) / (width / 2)

        print("Yellow near:", yellow_near)
        print("Yellow far:", yellow_far)
        print("Position error:", position_error)
        print("Heading error:", heading_error)

        vis = img.copy()

        cv2.circle(
            vis,
            (int(yellow_near), y_near),
            7, (255, 0, 0), -1
        )

        cv2.circle(
            vis,
            (int(yellow_far), y_far),
            7, (255, 0, 0), -1
        )

        cv2.circle(
            vis,
            (int(target_near), y_near),
            7, (0, 255, 0), -1
        )

        cv2.circle(
            vis,
            (int(target_far), y_far),
            7, (0, 255, 0), -1
        )

        cv2.line(
            vis,
            (width // 2, 0),
            (width // 2, height),
            (255, 0, 0), 2
        )
        cv2.line(
            vis,
            (int(target_near), y_near),
            (int(target_far), y_far),
            (0, 255, 0),  # зелёный
            2
        )

        cv2.imwrite(
            f"debug_{self.step}.png",
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        )

        self.step += 1
        if position_error is None or heading_error is None:
            return None

        return {
            "position_error": position_error,
            "heading_error": heading_error
        }





    def solve(self):
        env = self.generated_task['env']

        obs, _, _, _ = env.step([0.0, 0.0])
        env.render()

        done = False
        while not done:

            result = self.calculate(obs)

            if result is None  or result["position_error"] is None or result["heading_error"] is None:
                linear_velocity = 0.0
                angular_velocity = 0.3

            else:
                position_error, heading_error = result["position_error"], result["heading_error"]

                linear_velocity, angular_velocity = self.control(
                    position_error,
                    heading_error
                )

                done = abs(position_error) < 0.7 and abs(heading_error) < 0.05

            obs, _, _, _ = env.step([
                linear_velocity,
                angular_velocity
            ])

            env.render()



if __name__ == "__main__":
    from gym_duckietown.tasks.default.task_generator import DefaultTaskGenerator

    task_generator = DefaultTaskGenerator()
    task_generator.generate_task(map_name='straight_road_no_start_tile')
    solution = LaneCenterTaskSolution(task_generator.generated_task)
    solution.solve()