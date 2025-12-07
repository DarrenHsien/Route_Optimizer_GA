import numpy as np


def generate_map_matrix(map_size=10, num_passengers=15, terrain_difficulty='extreme', seed=42):
    """
    生成地圖矩陣
    
    參數:
    - map_size: 地圖大小 (預設 10x10)
    - num_passengers: 乘客數量 (預設 15)
    - terrain_difficulty: 地形嚴峻程度
        'easy': 較少障礙物和高成本地形 (10% 禁區, 20% 困難地形)
        'medium': 中等障礙物和高成本地形 (20% 禁區, 30% 困難地形)
        'hard': 較多障礙物和高成本地形 (30% 禁區, 40% 困難地形)
        'extreme': 極多障礙物和高成本地形 (40% 禁區, 50% 困難地形)
    - seed: 隨機種子 (預設 42)
    
    回傳:
    - map_matrix: 地圖矩陣
    - passengers: 乘客位置陣列
    - point_A: A點座標
    - point_B: B點座標
    """
    np.random.seed(seed)
    
    # 根據難度設定參數
    difficulty_params = {
        'easy': {'forbidden_ratio': 0.10, 'difficult_ratio': 0.20},
        'medium': {'forbidden_ratio': 0.20, 'difficult_ratio': 0.30},
        'hard': {'forbidden_ratio': 0.30, 'difficult_ratio': 0.40},
        'extreme': {'forbidden_ratio': 0.40, 'difficult_ratio': 0.50}
    }
    
    if terrain_difficulty not in difficulty_params:
        raise ValueError("terrain_difficulty 必須是 'easy', 'medium', 'hard', 或 'extreme'")
    
    params = difficulty_params[terrain_difficulty]
    
    # 初始化地圖 (全部為一般道路)
    map_matrix = np.zeros((map_size, map_size), dtype=int)
    
    # 定義地圖網格具體
    total_cells = map_size * map_size
    num_forbidden = int(total_cells * params['forbidden_ratio'])
    num_difficult = int(total_cells * params['difficult_ratio'])
    print(f"阻礙cell數 : {num_forbidden}")
    print(f"嚴峻地形cell數 : {num_difficult}")

    # 隨機生成禁區 (1)
    forbidden_positions = np.random.choice(total_cells, num_forbidden, replace=False)
    for pos in forbidden_positions:
        row, col = pos // map_size, pos % map_size
        # 確保 A 點和 B 點不是禁區
        if (row, col) != (0, 0) and (row, col) != (map_size-1, map_size-1):
            map_matrix[row, col] = 1
    
    # 隨機生成困難地形 (2-5)
    available_positions = []
    for i in range(total_cells):
        # 取得避掉禁區與A,B點的設置位置
        if i not in forbidden_positions:
            row, col = pos // map_size, pos % map_size
            if (row, col) != (0, 0) and (row, col) != (map_size-1, map_size-1):
                available_positions.append(i)
    
    # CELL抽樣
    difficult_positions = np.random.choice(available_positions, 
                                          min(num_difficult, len(available_positions)), 
                                          replace=False)
    for pos in difficult_positions:
        row, col = pos // map_size, pos % map_size
        # 隨機分配地形類型 (2: 低成本, 3: 中等, 4: 高成本, 5: 極高成本)
        map_matrix[row, col] = np.random.choice([2, 3, 4, 5], p=[0.3, 0.3, 0.25, 0.15])
    
    # A點和B點 (方格中心座標)
    point_A = (0, 0)
    point_B = (map_size - 1, map_size - 1)
    
    # 找出所有非禁區的位置來生成乘客
    valid_passenger_positions = np.argwhere(map_matrix != 1)
    
    # 確保有足夠的有效位置來放置乘客
    if len(valid_passenger_positions) < num_passengers:
        raise ValueError(f"可生成乘客的位置 ({len(valid_passenger_positions)}) 少於指定的乘客數量 ({num_passengers})。"
                         "請降低地形難度或乘客數量。")

    # 從有效位置中隨機選擇乘客位置 (確保不重複)
    passenger_indices = np.random.choice(len(valid_passenger_positions), num_passengers, replace=False)
    passengers = valid_passenger_positions[passenger_indices]
    
    return map_matrix, passengers, point_A, point_B