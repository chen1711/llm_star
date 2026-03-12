import random
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from llmastar.env.search import env as env_search, plotting as plotting_search
import json
import os
import inquirer
import warnings
warnings.filterwarnings('ignore')  # 屏蔽matplotlib警告

class Dataset:
    def __init__(self):
        self.MAP = [(50, 30)]  # 论文标准50×30地图尺寸
        self.unique_env = 100   # 100张独立地图
        self.unique_sg = 10     # 每张地图10组起止点

    def generate_environment_Astar(self):
        """生成100张地图+每组10个起止点，输出JSON和可视化图片"""
        # 创建基础目录
        base_dir = "dataset/A*"
        os.makedirs(base_dir, exist_ok=True)
        
        for map_size in self.MAP:
            x_size, y_size = map_size
            x_range, y_range = (0, x_size + 1), (0, y_size + 1)
            json_path = os.path.join(base_dir, f'environment_{x_size}_{y_size}.json')
            maps_dir = os.path.join(base_dir, f'environment_{x_size}_{y_size}_maps')
            os.makedirs(maps_dir, exist_ok=True)

            # 初始化/读取JSON文件
            if os.path.exists(json_path):
                with open(json_path, 'r') as file:
                    environments = json.load(file)
            else:
                environments = []  # 空文件初始化

            # 生成100张独立地图
            for i in range(len(environments), self.unique_env):
                decision = False
                while not decision:
                    # 随机生成1-4个水平/垂直障碍
                    num_h = round(random.uniform(1, 4))
                    num_v = round(random.uniform(1, 4))
                    
                    # 生成障碍和起止点
                    data = {'id': i}
                    env_data = self._generate_random_obstacles_and_points_Astar(x_range, y_range, num_h, num_v)
                    data.update(env_data)
                    
                    # 可视化预览，人工筛选有效地图
                    self.plot_grid_Astar(
                        data['start_goal'][0][0], data['start_goal'][0][1],
                        data['range_x'], data['range_y'],
                        data['horizontal_barriers'], data['vertical_barriers'],
                        show=False
                    )
                    
                    # 人工确认地图有效性
                    action_planner = [
                        inquirer.List(
                            'approach',
                            message=f"Map {i} - Is this map valid? (Good/Bad)",
                            choices=[('Good', True), ('Bad', False)],
                            default=True
                        )
                    ]
                    decision = inquirer.prompt(action_planner)['approach']
                
                # 添加有效地图到列表
                environments.append(data)
                print(f"Generated valid map {i}/{self.unique_env}")

            # 保存所有地图数据到JSON
            with open(json_path, 'w') as f:
                json.dump(environments, f, indent=4)

            # 为每组起止点生成可视化图片
            for i in range(len(environments)):
                data = environments[i]
                map_dir = os.path.join(maps_dir, f"map_{i}")
                os.makedirs(map_dir, exist_ok=True)
                
                # 取当前地图的尺寸（修复变量作用域问题）
                x_range = data['range_x']
                y_range = data['range_y']

                # 为10组起止点生成图片
                for index, sg in enumerate(data['start_goal']):
                    self.plot_grid_Astar(
                        sg[0], sg[1],
                        data['range_x'], data['range_y'],
                        data['horizontal_barriers'], data['vertical_barriers'],
                        name=f"A* {i}-{index}",
                        path=os.path.join(map_dir, f"{index}.png"),
                        show=False
                    )
        
        print(f"Dataset generation completed!")
        print(f"- JSON file: {json_path}")
        print(f"- Maps directory: {maps_dir}")

    def _generate_random_obstacles_and_points_Astar(self, x_range, y_range, num_h_obstacles, num_v_obstacles):
        """核心函数：生成随机障碍和可行的起止点"""
        def generate_horizontal_obstacles(num_h, x_range, y_range, existing_obstacles):
            """生成水平障碍（y固定，x从start到end）"""
            horizontal_obstacles = []
            for _ in range(num_h):
                while True:
                    y = round(random.uniform(y_range[0], y_range[1]))
                    x_start = round(random.uniform(x_range[0], x_range[1]))
                    x_end = round(random.uniform(x_start, x_range[1]))
                    # 生成障碍线段
                    horizontal = LineString([(x_start, y), (x_end, y)])
                    horizontal_obstacles.append([y, x_start, x_end])
                    existing_obstacles.append(horizontal)
                    break
            return horizontal_obstacles

        def generate_vertical_obstacles(num_v, x_range, y_range, existing_obstacles):
            """生成垂直障碍（x固定，y从start到end）"""
            vertical_obstacles = []
            for _ in range(num_v):
                while True:
                    x = round(random.uniform(x_range[0], x_range[1]))
                    y_start = round(random.uniform(y_range[0], y_range[1]))
                    y_end = round(random.uniform(y_start, y_range[1]))
                    # 生成障碍线段
                    vertical = LineString([(x, y_start), (x, y_end)])
                    vertical_obstacles.append([x, y_start, y_end])
                    existing_obstacles.append(vertical)
                    break
            return vertical_obstacles

        def generate_random_point(x_range, y_range, existing_obstacles):
            """生成不与障碍重叠的随机点（起点/终点）"""
            while True:
                x = round(random.uniform(x_range[0], x_range[1] - 2))
                y = round(random.uniform(y_range[0], y_range[1] - 2))
                point = Point(x, y)
                # 确保点不在障碍上
                if not any(point.intersects(ob) for ob in existing_obstacles):
                    return [x, y]

        # 初始化障碍列表（先添加边界墙）
        existing_obstacles = []
        # 左右边界墙（x固定，覆盖整个y范围）
        for x in [x_range[0], x_range[1]]:
            existing_obstacles.append(LineString([(x, y_range[0]), (x, y_range[1])]))
        # 上下边界墙（y固定，覆盖整个x范围）
        for y in [y_range[0], y_range[1]]:
            existing_obstacles.append(LineString([(x_range[0], y), (x_range[1], y)]))

        # 生成水平/垂直障碍
        horizontal_barriers = generate_horizontal_obstacles(num_h_obstacles, x_range, y_range, existing_obstacles)
        vertical_barriers = generate_vertical_obstacles(num_v_obstacles, x_range, y_range, existing_obstacles)

        # 生成10组可行的起止点（修复逻辑：只添加不与障碍相交的路径）
        sg_list = []
        while len(sg_list) < self.unique_sg:
            start = generate_random_point(x_range, y_range, existing_obstacles)
            goal = generate_random_point(x_range, y_range, existing_obstacles)
            # 确保起止点连线不穿过障碍（核心修正！）
            if not any(LineString([start, goal]).intersects(ob) for ob in existing_obstacles):
                sg_list.append((start, goal))

        # 构造地图数据
        environment = {
            "range_x": x_range,
            "range_y": y_range,
            "horizontal_barriers": horizontal_barriers,
            "vertical_barriers": vertical_barriers,
            "start_goal": sg_list
        }
        return environment

    def add_query_Astar(self, filepath=None):
        """为每组起止点生成LLM查询提示词（适配论文的Prompt格式）"""
        if filepath is None:
            filepath = os.path.join("dataset/A*", f'environment_{self.MAP[0][0]}_{self.MAP[0][1]}.json')
        
        if not os.path.exists(filepath):
            print(f"Error: JSON file {filepath} not found!")
            return

        with open(filepath, 'r') as f:
            data = json.load(f)

        # 为每组起止点生成查询文本
        for environment in data:
            for sg_idx, sg in enumerate(environment['start_goal']):
                start, goal = sg[0], sg[1]
                x_size = environment['range_x'][1]
                y_size = environment['range_y'][1]
                horizontal_barriers = environment['horizontal_barriers']
                vertical_barriers = environment['vertical_barriers']
                
                # 论文标准Prompt格式
                query = f"""Design a path from [{start[0]}, {start[1]}] to [{goal[0]}, {goal[1]}] on a {x_size} by {y_size} grid that avoids horizontal barriers at {horizontal_barriers} and vertical barriers at {vertical_barriers}."""
                # 将查询词添加到起止点列表
                environment['start_goal'][sg_idx] = [start, goal, query]

        # 保存带查询词的JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Added LLM queries to {filepath}")

    def plot_grid_Astar(self, s_start, s_goal, range_x, range_y, horizontal_barriers, vertical_barriers, name='A*', path="temp.png", show=False):
        """可视化地图（修复Matplotlib无GUI问题，直接保存图片）"""
        try:
            Env = env_search.Env(range_x[1], range_y[1], horizontal_barriers, vertical_barriers)
            plot = plotting_search.Plotting(s_start, s_goal, Env)
            plot.plot_map(name, path, show)
        except Exception as e:
            print(f"Plotting error: {e}")
            # 降级方案：基础matplotlib绘图
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlim(range_x[0], range_x[1])
            ax.set_ylim(range_y[0], range_y[1])
            ax.grid(True)
            # 绘制起点/终点
            ax.scatter(s_start[0], s_start[1], c='green', s=100, label='Start')
            ax.scatter(s_goal[0], s_goal[1], c='red', s=100, label='Goal')
            # 绘制水平障碍
            for y, x1, x2 in horizontal_barriers:
                ax.plot([x1, x2], [y, y], 'b-', linewidth=3, label='Horizontal Barrier' if y == horizontal_barriers[0][0] else "")
            # 绘制垂直障碍
            for x, y1, y2 in vertical_barriers:
                ax.plot([x, x], [y1, y2], 'g-', linewidth=3, label='Vertical Barrier' if x == vertical_barriers[0][0] else "")
            ax.legend()
            ax.set_title(name)
            plt.savefig(path)
            plt.close()

# ====================== 快速运行入口 ======================
if __name__ == "__main__":
    # 初始化数据集生成器
    ds = Dataset()
    
    # 1. 生成100张地图+可视化图片
    ds.generate_environment_Astar()
    
    # 2. 为每组起止点添加LLM查询提示词
    ds.add_query_Astar()
    
    print("✅ Dataset generation finished!")
    print("📁 Output structure:")
    print("dataset/A*/")
    print("  ├─ environment_50_30.json (100 maps data)")
    print("  └─ environment_50_30_maps/ (100 map folders × 10 images)")