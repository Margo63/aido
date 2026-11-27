from gym_duckietown.tasks.task_solution import TaskSolution
import numpy as np

class RoundtripTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)

    def get_robot_pose(self, simulator):
        """Позиция (3,) и угол робота. Подстройте под API среды."""
        pos = np.asarray(simulator.cur_pos, dtype=float)
        angle = float(simulator.cur_angle)
        return pos, angle

    def angle_error(self, pos, angle, target):
        """Ошибка угла: на сколько нужно повернуться, чтобы смотреть на target."""
        d = np.asarray(target[:2], dtype=float) - np.asarray(pos[:2], dtype=float)
        desired = np.arctan2(d[1], d[0])
        return (desired - angle + np.pi) % (2 * np.pi) - np.pi

    def reached(self, pos, target, tol=0.2):
        return np.linalg.norm(np.asarray(pos[:2]) - np.asarray(target[:2])) < tol

    def solve(self):
        env = self.generated_task['env']
        simulator = env.unwrapped

        # 1. Стартовая поза — случайная, только читаем
        start_pos, start_angle = self.get_robot_pose(simulator)

        # 2. Цель
        target_coordinates = self.generated_task['target_coordinates'][-1]

        # 3. Параметры управления
        lin_speed = 0.5
        ang_gain = 1.5
        ang_max = 1.0
        pos_tol = 0.2
        heading_tol = 0.15  # допуск по углу, чтобы считать «смотрим на цель»
        max_steps = 8000

        # 4. Фазы: to_target -> turning -> return -> done
        phase = 'to_target'
        obs, _, _, _ = env.step([0.0, 0.0])
        env.render()

        for _ in range(max_steps):
            pos, angle = self.get_robot_pose(simulator)

            # ---- переходы фаз ----
            if phase == 'to_target' and self.reached(pos, target_coordinates, pos_tol):
                phase = 'turning'
            elif phase == 'turning':
                # развернулись, когда смотрим на старт с малой ошибкой
                if abs(self.angle_error(pos, angle, start_pos)) < heading_tol:
                    phase = 'return'
            elif phase == 'return' and self.reached(pos, start_pos, pos_tol):
                phase = 'done'
                break

            # ---- управление по фазе ----
            if phase == 'to_target':
                err = self.angle_error(pos, angle, target_coordinates)
                angular_velocity = float(np.clip(ang_gain * err, -ang_max, ang_max))
                linear_velocity = lin_speed if abs(err) < 0.3 else 0.0

            elif phase == 'turning':
                # разворот на месте: доворачиваемся на стартовую точку
                err = self.angle_error(pos, angle, start_pos)
                angular_velocity = float(np.clip(ang_gain * err, -ang_max, ang_max))
                linear_velocity = 0.0

            elif phase == 'return':
                err = self.angle_error(pos, angle, start_pos)
                angular_velocity = float(np.clip(ang_gain * err, -ang_max, ang_max))
                linear_velocity = lin_speed if abs(err) < 0.3 else 0.0

            else:  # done
                linear_velocity, angular_velocity = 0.0, 0.0

            obs, _, done, _ = env.step([linear_velocity, angular_velocity])
            env.render()
            if done:
                break

if __name__ == "__main__":
    from gym_duckietown.tasks.default.task_generator import DefaultTaskGenerator

    task_generator = DefaultTaskGenerator()
    task_generator.generate_task(map_name='straight_road', initial_pose=(0,0))
    task_generator.generated_task['target_coordinates'] = np.array([0.8, 0.0, 0.2925])
    solution = RoundtripTaskSolution(task_generator.generated_task)
    solution.solve()