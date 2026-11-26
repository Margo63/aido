from gym_duckietown.tasks.task_solution import TaskSolution
import math

class LaneCenterTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)

    def solve(self):
        env = self.generated_task['env']
        linear_velocity = 0
        angular_velocity = 0
        while True:
            lane_pos = env.get_lane_pos2(env.cur_pos, env.cur_angle)
            dist = lane_pos.dist
            ang_mismatch = lane_pos.angle_rad
            if dist>0.1:
                if abs(ang_mismatch)>0.01:
                    linear_velocity = 0
                    angular_velocity = 10*ang_mismatch/math.pi
                else:
                    linear_velocity = 0.1
                    angular_velocity = 0
            else:
                if abs(ang_mismatch)>0.01:
                    linear_velocity = 0
                    angular_velocity = 10*ang_mismatch/math.pi
                else:
                    linear_velocity = 0
                    angular_velocity = 0
                    break
            obs, _, _, _ = env.step([linear_velocity, angular_velocity])
            env.render()

if __name__ == "__main__":
    from gym_duckietown.tasks.default.task_generator import DefaultTaskGenerator

    task_generator = DefaultTaskGenerator()
    task_generator.generate_task(map_name='straight_road_no_start_tile')
    solution = LaneCenterTaskSolution(task_generator.generated_task)
    solution.solve()