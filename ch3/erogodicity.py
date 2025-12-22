import numpy as np
import matplotlib.pyplot as plt

# 模拟参数
N_people = 1000   # 1000个人 (系综)
T_time = 100      # 每人玩100轮 (时间)
initial_wealth = 100.0

# 游戏规则：50%概率赚50%，50%概率亏40%
# 算术期望 (Ensemble): 0.5*1.5 + 0.5*0.6 = 1.05 (+5% growth)
# 几何期望 (Time): sqrt(1.5 * 0.6) = sqrt(0.9) = 0.948 (-5% decay)

# 初始化数据结构 [人数, 时间]
wealth_paths = np.zeros((N_people, T_time))
wealth_paths[:, 0] = initial_wealth

np.random.seed(42)

for t in range(1, T_time):
    # 生成随机翻硬币结果 (0 或 1)
    flips = np.random.randint(0, 2, N_people)
    
    # 1 代表赢 (+50%), 0 代表输 (-40%)
    multipliers = np.where(flips == 1, 1.5, 0.6)
    
    # 更新财富 (乘法过程)
    wealth_paths[:, t] = wealth_paths[:, t-1] * multipliers

# 计算两种平均值
ensemble_average = np.mean(wealth_paths, axis=0) # 每一时刻，所有人的平均财富
median_person = np.median(wealth_paths, axis=0)  # 中位数人（典型个体）的财富

# 绘图
fig, ax = plt.subplots(figsize=(12, 6))

# 画出前20个人的个人路径（灰色背景）
for i in range(20):
    ax.plot(wealth_paths[i, :], color='gray', alpha=0.3, linewidth=1)

# 画出系综平均（蓝色）
ax.plot(ensemble_average, color='blue', linewidth=3, label='Ensemble Average (Expected Value)')

# 画出典型个体（红色）
ax.plot(median_person, color='red', linewidth=3, linestyle='--', label='Time Probability (Typical Individual)')

ax.set_title("Non-Ergodicity: Why The 'Average' is a Lie", fontsize=14)
ax.set_xlabel("Time Steps")
ax.set_ylabel("Wealth (Log Scale)")
ax.set_yscale('log') # 使用对数坐标看清楚衰减
ax.legend()
ax.grid(True, which="both", alpha=0.3)

plt.show()

# 打印最终结果
print(f"100轮后，系综平均财富: {ensemble_average[-1]:.2f} (看起来赚翻了)")
print(f"100轮后，典型个体财富: {median_person[-1]:.2f} (实际上亏惨了)")
print(f"100轮后，破产人数占比 (<1元): {np.sum(wealth_paths[:, -1] < 1) / N_people * 100:.1f}%")
