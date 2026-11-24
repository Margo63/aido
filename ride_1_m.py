from gym_duckietown.tasks.task_solution import TaskSolution


class Ride1MTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)

    def solve(self):
        env = self.generated_task['env']
        linear_velocity = 1.0
        angular_velocity = 0.0
        for i in range(1,25):
            obs, _, _, _ = env.step([linear_velocity, angular_velocity])
            env.render()




if __name__ == "__main__":
    from gym_duckietown.tasks.default.task_generator import DefaultTaskGenerator

    task_generator = DefaultTaskGenerator()
    task_generator.generate_task(map_name='straight_road', initial_pose=(0,0))
    solution = Ride1MTaskSolution(task_generator.generated_task)
    solution.solve()
