import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from utils.chineese_setting import set_plot_chineese
from utils.map_generate import generate_map_matrix
from utils.plot import visualize_map, visualize_final_route
from utils.optimizer_ga import BusRouteOptimizer_GA
# 支援中文顯示
set_plot_chineese()

#=== 參數設定 ===
# MAP尺寸大小
map_size=30
# 乘客人數
num_passengers = 100
# 地形難易程度 ['easy', 'medium', 'hard', 'extreme']
terrain_difficulty = 'medium'

# 定義地形成本
terrain_costs = {
    0: 1.0,             # 一般道路: 基本成本
    1: float('inf'),    # 禁區: 無法通過
    2: 1.3,             # 低成本地形
    3: 1.5,             # 中等成本地形
    4: 2.0,             # 高成本地形
    5: 2.5              # 極高成本地形
}
max_stations = 10
weight_route_cost = 1
weight_station_cost = 10
weight_walking_cost = 1


#=== 地圖生成 ===#
map_matrix, passengers, point_A, point_B = generate_map_matrix(
    map_size=map_size, 
    num_passengers=num_passengers, 
    terrain_difficulty=terrain_difficulty,
    seed=42
)

visualize_map(map_matrix, passengers, point_A, point_B)


#=== 建立優化器 ===#
optimizer_v4 = BusRouteOptimizer_GA(
    map_matrix, 
    passengers, 
    point_A, 
    point_B, 
    terrain_costs,
    max_stations=max_stations,
    weight_route_cost=weight_route_cost,
    weight_station_cost=weight_station_cost,
    weight_walking_cost=weight_walking_cost
)

#=== 執行優化 ===#
best_waypoints_v4, best_route_v4, best_stations_v4 = optimizer_v4.evolve(
    generations=300,    # 增加演化代數，給予更多時間收斂
    pop_size=80,        # 每代 50 個個體
    waypoints_num=10,   # 轉折點()
    mutation_rate=0.8,  # 初始突變率可以設高一些
    
)

#=== 輸出結果 ===#
total_cost_v4 = optimizer_v4.calculate_total_cost(best_route_v4, best_stations_v4)
print("\n--- BusRouteOptimizer_GA_V4 最佳解 ---")
print(f"轉折點: {best_waypoints_v4}")
print(f"車站位置: {best_stations_v4}")
print(f"總成本: {total_cost_v4:.2f}")

#=== 視覺化最終結果 ===#
visualize_final_route(
    map_matrix,
    passengers,
    point_A,
    point_B,
    best_route_v4,
    best_stations_v4,
    total_cost_v4
)