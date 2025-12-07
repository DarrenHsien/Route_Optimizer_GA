
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def visualize_map(map_matrix, passengers, point_A, point_B):
    """視覺化地圖"""
    map_size = map_matrix.shape[0]
    
    plt.figure(figsize=(10, 10))
    
    # 定義地形顏色 (0:一般, 1:禁區, 2-5:困難地形)
    # 0: 白色, 1: 黑色, 2-5: 淺綠到深綠
    colors = ['#FFFFFF',  # 0: 一般道路
              '#000000',  # 1: 禁區 (黑色)
              '#90EE90',  # 2: 困難地形 (淺綠)
              '#3CB371',  # 3: 困難地形 (中綠)
              '#2E8B57',  # 4: 困難地形 (深綠)
              '#006400']  # 5: 困難地形 (最深綠)
    
    # 根據地圖中的最大值決定使用多少顏色
    num_colors = map_matrix.max() + 1
    cmap = ListedColormap(colors[:num_colors])
    
    im = plt.imshow(map_matrix, cmap=cmap, alpha=0.7, extent=[-0.5, map_size-0.5, map_size-0.5, -0.5])
    
    # 標記A點和B點 (在方格中心)
    plt.plot(point_A[1], point_A[0], 'gs', markersize=20, label='A點 (起點)')
    plt.plot(point_B[1], point_B[0], 'rs', markersize=20, label='B點 (終點)')
    
    # 標記乘客位置 (在方格中心)
    plt.scatter(passengers[:, 1], passengers[:, 0], c='blue', s=100, 
                marker='o', edgecolors='black', linewidths=2, label='乘客')
    
    # 添加方格格線
    for i in range(map_size + 1):
        plt.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        plt.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
    
    plt.xticks(range(map_size))
    plt.yticks(range(map_size))
    
    plt.legend(loc='upper right', fontsize=12)
    plt.title(f'公車路線規劃地圖 ({map_size}x{map_size})\n乘客數: {len(passengers)}', fontsize=14)
    plt.colorbar(im, label='地形類型')
    
    plt.tight_layout()
    plt.show()

def visualize_final_route(map_matrix, passengers, point_A, point_B, route, stations, total_cost):
    """視覺化最終找到的最佳路線與車站位置"""
    map_size = map_matrix.shape[0]
    
    plt.figure(figsize=(12, 10))
    
    # --- 繪製地圖背景 (與 visualize_map 相同) ---
    colors = ['#FFFFFF', '#000000', '#90EE90', '#3CB371', '#2E8B57', '#006400']
    num_colors = map_matrix.max() + 1
    cmap = ListedColormap(colors[:num_colors])
    im = plt.imshow(map_matrix, cmap=cmap, alpha=0.7, extent=[-0.5, map_size-0.5, map_size-0.5, -0.5])
    
    # --- 繪製乘客、起點、終點 ---
    plt.scatter(passengers[:, 1], passengers[:, 0], c='blue', s=80, 
                marker='o', edgecolors='black', linewidths=1, label='乘客', alpha=0.7, zorder=5)
    plt.plot(point_A[1], point_A[0], 'gs', markersize=20, label='A點 (起點)', zorder=10)
    plt.plot(point_B[1], point_B[0], 'rs', markersize=20, label='B點 (終點)', zorder=10)

    # --- 繪製最佳路線 ---
    if route:
        import numpy as np
        route_np = np.array(route)
        plt.plot(route_np[:, 1], route_np[:, 0], color='darkorange', linewidth=4, label='最佳路線', alpha=0.8, zorder=8)

    # --- 繪製車站 ---
    if stations:
        import numpy as np
        stations_np = np.array(stations)
        plt.scatter(stations_np[:, 1], stations_np[:, 0], c='yellow', s=300, 
                    marker='*', edgecolors='black', linewidths=1.5, label='車站', zorder=12)

    # --- 繪製格線和標籤 ---
    for i in range(map_size + 1):
        plt.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        plt.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
    
    plt.xticks(range(map_size))
    plt.yticks(range(map_size))
    
    plt.legend(loc='upper right', fontsize=12, bbox_to_anchor=(1.3, 1.0))
    plt.title(f'最佳路線規劃結果\n總成本: {total_cost:.2f}', fontsize=16)
    plt.colorbar(im, label='地形類型', fraction=0.046, pad=0.04)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # 調整佈局以容納圖例
    plt.show()