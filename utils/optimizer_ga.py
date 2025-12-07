import numpy as np

class BusRouteOptimizer_GA:
    """
    公車路線優化器 V4 - 引入錦標賽選擇與自適應突變率
    使用遺傳演算法來找出最佳的公車路線和站點配置
    """
    def __init__(self, map_matrix, passengers, point_A, point_B, 
                 terrain_costs, max_stations=5, weight_route_cost=1, weight_station_cost=10, weight_walking_cost=1):
        """
        初始化優化器
        
        參數說明:
        - map_matrix: 地圖矩陣 (0=一般道路, 1=禁區, 2-5=不同難度地形)
        - passengers: 乘客位置列表 [(x1,y1), (x2,y2), ...]
        - point_A: 起點座標 (x, y)
        - point_B: 終點座標 (x, y)
        - terrain_costs: 地形成本字典 {地形類型: 成本係數}
        - station_cost: 每個車站的建設成本 (預設10)
        - max_stations: 路線上最多可設置的車站數量 (預設5)
        """
        self.map_matrix = map_matrix          # 儲存地圖資訊
        self.passengers = passengers          # 儲存所有乘客位置
        self.point_A = point_A                # 起點
        self.point_B = point_B                # 終點
        self.terrain_costs = terrain_costs    # 地形成本對照表
        self.station_cost = 1      # 單一車站成本
        self.max_stations = max_stations      # 車站數量上限
        self.map_size = map_matrix.shape[0]   # 地圖大小 (假設為正方形)
        self.weight_route_cost = weight_route_cost
        self.weight_station_cost = weight_station_cost
        self.weight_walking_cost = weight_walking_cost

    def calculate_route_cost(self, route):
        """
        計算公車路線的總成本 (包含距離和地形加成)
        """
        total_cost = 0
        for i in range(len(route) - 1):
            x1, y1 = route[i]
            x2, y2 = route[i + 1]
            distance = abs(x2 - x1) + abs(y2 - y1)
            terrain = self.map_matrix[x2, y2]
            terrain_multiplier = self.terrain_costs[terrain]
            total_cost += distance * terrain_multiplier
        return total_cost
    
    def calculate_passenger_walking_cost(self, stations):
        """
        計算所有乘客走到最近車站的總步行成本
        """
        if not stations:
            return float('inf')
        total_walking = 0
        for px, py in self.passengers:
            min_dist = min([abs(px - sx) + abs(py - sy) for sx, sy in stations])
            total_walking += min_dist
        return total_walking
    
    def calculate_station_cost(self, num_stations):
        """
        計算車站建設的總成本
        """
        return num_stations * self.station_cost
    
    def calculate_total_cost(self, route, stations):
        """
        計算方案的總成本 (三個成本加總)
        """
        route_cost = self.calculate_route_cost(route) * self.weight_route_cost
        walking_cost = self.calculate_passenger_walking_cost(stations) * self.weight_station_cost
        station_cost = self.calculate_station_cost(len(stations)) * self.weight_walking_cost
        return route_cost + walking_cost + station_cost
    
    def is_valid_route(self, route):
        """
        檢查路線是否合法
        """
        for x, y in route:
            if not (0 <= x < self.map_size and 0 <= y < self.map_size):
                return False
            if self.map_matrix[x, y] == 1:
                return False
        return True
    
    def find_path_between(self, start, end):
        """使用 A* 找到兩點之間的路徑"""
        from heapq import heappush, heappop
        
        frontier = [(0, start, [start])]
        visited = set()
        
        while frontier:
            cost, current, path = heappop(frontier)
            
            if current == end:
                return path
            
            if current in visited:
                continue
            visited.add(current)
            
            x, y = current
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.map_size and 0 <= ny < self.map_size 
                    and self.map_matrix[nx, ny] != 1):
                    new_cost = cost + self.terrain_costs[self.map_matrix[nx, ny]]
                    heuristic = abs(nx - end[0]) + abs(ny - end[1])
                    heappush(frontier, (new_cost + heuristic, (nx, ny), path + [(nx, ny)]))
    
        return None # 找不到路徑
    
    def generate_initial_population(self, pop_size, waypoints_num=6):
        population = []
        while len(population) < pop_size:
            num_waypoints = np.random.randint(2, waypoints_num)
            waypoints = [self.point_A]
            
            for _ in range(num_waypoints):
                while True:
                    x = np.random.randint(0, self.map_size)
                    y = np.random.randint(0, self.map_size)
                    if self.map_matrix[x, y] != 1:
                        waypoints.append((x, y))
                        break
            
            waypoints.append(self.point_B)
            
            full_route = self.rebuild_route(waypoints)
            if full_route is None: # 如果無法生成完整路徑，則跳過此個體
                continue

            num_stations = np.random.randint(2, self.max_stations + 1)
            station_indices = np.random.choice(len(full_route), num_stations, replace=False)
            stations = [full_route[i] for i in station_indices]
            
            population.append((waypoints, full_route, stations))
        return population

    def evolve(self, generations=100, pop_size=50, waypoints_num=6, mutation_rate=0.1):
        population = self.generate_initial_population(pop_size, waypoints_num)

        best_cost = float('inf')
        best_solution = None
        
        # 自適應突變率參數
        initial_mutation_rate = mutation_rate
        final_mutation_rate = 0.01

        for gen in range(generations):
            fitness = []
            for individual in population:
                waypoints, route, stations = individual
                if self.is_valid_route(route):
                    cost = self.calculate_total_cost(route, stations)
                    fitness.append((cost, individual))
                else:
                    fitness.append((float('inf'), individual))
            
            fitness.sort(key=lambda x: x[0])
            
            if fitness[0][0] < best_cost:
                best_cost = fitness[0][0]
                best_solution = fitness[0][1]

            # 計算當前的自適應突變率 (非線性衰減，例如使用指數衰減)
            decay_rate = 5.0
            current_mutation_rate = final_mutation_rate + (initial_mutation_rate - final_mutation_rate) * np.exp(-decay_rate * (gen / generations))
            
            # --- 優化後的選擇 (Selection) 策略 ---
            new_population = []

            # 1. 精英保留 (Elitism): 直接保留最好的 10% 個體
            elite_size = int(pop_size * 0.1)
            if elite_size > 0:
                new_population.extend([ind for _, ind in fitness[:elite_size]])

            # 2. 錦標賽選擇 (Tournament Selection) + 交配/突變 來產生剩餘的個體
            tournament_size = 3
            while len(new_population) < pop_size:
                # --- 透過錦標賽選擇父代1 ---
                tournament1_indices = np.random.choice(range(len(fitness)), tournament_size, replace=False)
                parent1 = min([fitness[i] for i in tournament1_indices], key=lambda x: x[0])[1]

                # --- 透過錦標賽選擇父代2 ---
                tournament2_indices = np.random.choice(range(len(fitness)), tournament_size, replace=False)
                parent2 = min([fitness[i] for i in tournament2_indices], key=lambda x: x[0])[1]
                
                child = self.crossover(parent1, parent2)
                if child is None: continue # 如果交配失敗，則跳過

                child = self.mutate(child, mutation_rate=current_mutation_rate, waypoints_num=waypoints_num)
                
                new_population.append(child)
            
            population = new_population
            
            if gen % 20 == 0:
                print(f"世代 {gen}: 最佳成本 = {best_cost:.2f}, 當前突變率 = {current_mutation_rate:.3f}")
        
        if best_solution is None:
            return [], [], []

        best_waypoints, best_route, best_stations = best_solution
        return best_waypoints, best_route, best_stations

    def crossover(self, parent1, parent2):
        waypoints1, _, stations1 = parent1
        waypoints2, _, stations2 = parent2
        
        mid_waypoints1 = waypoints1[1:-1]
        mid_waypoints2 = waypoints2[1:-1]
        
        min_len = min(len(mid_waypoints1), len(mid_waypoints2))
        if min_len > 0:
            cut_point = np.random.randint(1, min_len + 1)
            child_waypoints = [self.point_A] + mid_waypoints1[:cut_point] + mid_waypoints2[cut_point:] + [self.point_B]
        else:
            child_waypoints = waypoints1 if len(mid_waypoints1) > len(mid_waypoints2) else waypoints2
        
        child_route = self.rebuild_route(child_waypoints)
        if child_route is None:
            return None

        num_stations_p1 = len(stations1)
        num_stations_p2 = len(stations2)
        child_num_stations = np.random.randint(min(num_stations_p1, num_stations_p2, 2), max(num_stations_p1, num_stations_p2) + 1)
        
        if len(child_route) < child_num_stations:
            child_num_stations = len(child_route)

        station_indices = np.random.choice(len(child_route), child_num_stations, replace=False)
        child_stations = [child_route[i] for i in station_indices]
        
        return (child_waypoints, child_route, child_stations)
    
    def mutate(self, individual, mutation_rate=0.1, waypoints_num=6):
        waypoints, route, stations = individual
        
        if np.random.random() < mutation_rate:
            mutation_type = np.random.choice(['add_waypoint', 'remove_waypoint', 'move_waypoint', 'add_station', 'remove_station'])
            
            if mutation_type == 'add_waypoint' and len(waypoints) < waypoints_num + 2:
                while True:
                    x = np.random.randint(0, self.map_size)
                    y = np.random.randint(0, self.map_size)
                    if self.map_matrix[x, y] != 1:
                        insert_pos = np.random.randint(1, len(waypoints)) 
                        waypoints.insert(insert_pos, (x, y))
                        break
                rebuilt = self.rebuild_individual(waypoints, stations)
                if rebuilt:
                    new_route, new_stations = rebuilt
                    return (waypoints, new_route, new_stations)

            elif mutation_type == 'remove_waypoint' and len(waypoints) > 2:
                remove_idx = np.random.randint(1, len(waypoints) - 1)
                waypoints.pop(remove_idx)
                rebuilt = self.rebuild_individual(waypoints, stations)
                if rebuilt:
                    new_route, new_stations = rebuilt
                    return (waypoints, new_route, new_stations)

            elif mutation_type == 'move_waypoint' and len(waypoints) > 2:
                move_idx = np.random.randint(1, len(waypoints) - 1)
                while True:
                    x = np.random.randint(0, self.map_size)
                    y = np.random.randint(0, self.map_size)
                    if self.map_matrix[x, y] != 1:
                        waypoints[move_idx] = (x, y)
                        break
                rebuilt = self.rebuild_individual(waypoints, stations)
                if rebuilt:
                    new_route, new_stations = rebuilt
                    return (waypoints, new_route, new_stations)

            elif mutation_type == 'add_station' and len(stations) < self.max_stations:
                if route:
                    new_station = route[np.random.randint(0, len(route))]
                    if new_station not in stations:
                        stations.append(new_station)
                    
            elif mutation_type == 'remove_station' and len(stations) > 2:
                stations.pop(np.random.randint(0, len(stations)))
        
        return (waypoints, route, stations)
    
    def rebuild_route(self, waypoints):
        """根據轉折點重建完整路線"""
        route = []
        for i in range(len(waypoints) - 1):
            segment = self.find_path_between(waypoints[i], waypoints[i+1])
            if segment is None:
                return None # 如果任何一段路徑無法找到，則整個重建失敗
            route.extend(segment[:-1])
        route.append(self.point_B)
        return route

    def rebuild_individual(self, waypoints, old_stations):
        """當轉折點改變時，重建路線並重新選擇車站"""
        new_route = self.rebuild_route(waypoints)
        if new_route is None:
            return None # 重建失敗
        
        valid_old_stations = [s for s in old_stations if s in new_route]
        num_new_stations_needed = len(old_stations) - len(valid_old_stations)

        potential_new_stations = [p for p in new_route if p not in valid_old_stations]
        
        num_to_add = min(num_new_stations_needed, len(potential_new_stations))
        if num_to_add > 0:
            newly_added_indices = np.random.choice(len(potential_new_stations), num_to_add, replace=False)
            newly_added_stations = [potential_new_stations[i] for i in newly_added_indices]
        else:
            newly_added_stations = []

        new_stations = valid_old_stations + newly_added_stations
        if not new_stations and len(new_route) > 0: # 確保至少有一個車站
            new_stations.append(new_route[np.random.randint(0, len(new_route))])

        return new_route, new_stations