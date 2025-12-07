# 路線優化遺傳演算法 (Route Optimizer GA)

本專案使用遺傳演算法 (Genetic Algorithm) 來解決路徑規劃問題，例如旅行推銷員問題 (Traveling Salesperson Problem, TSP)。

## 簡介

旅行推銷員問題 (TSP) 是一個經典的組合優化問題，目標是找到一條經過所有給定城市一次且僅一次後，回到起點的最短路徑。本專案透過模擬生物演化的遺傳演算法，來尋找這個問題的近似最佳解。

## 功能特色

- 使用 Python 實現遺傳演算法。
- 可自訂城市數量與座標。
- 視覺化呈現初始路徑與優化後的最佳路徑。
- 可調整遺傳演算法的各項參數 (如：族群大小、突變率、交叉率)。

## 環境建置與安裝


```bash
git clone https://github.com/DarrenHsien/Route_Optimizer_GA.git
cd route_optimizer_ga
```


## 如何使用

直接執行主程式 `main.py` 即可開始進行路線優化。

```bash
python main.py
```

## 授權 (License)

本專案採用 MIT License 授權。