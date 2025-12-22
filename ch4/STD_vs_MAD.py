import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 设置随机种子
np.random.seed(42)

# 参数设置
alpha = 1.5  # 肥尾指数 (1 < alpha < 2)，意味着方差无穷大，但均值存在
N = 100000   # 样本数量足够大，模拟"无穷多"的过程

# 生成数据 (Pareto)
data = stats.pareto.rvs(b=alpha, size=N)

# 定义计算"累积"统计量的函数
def running_statistics(arr):
    n_range = np.arange(1, len(arr) + 1)
    # 累积均值
    running_mean = np.cumsum(arr) / n_range
    
    # 为了计算简便，我们用手动公式计算 Running MAD 和 Running STD
    # 注意：这里为了代码效率，使用了简化算法，主要展示趋势
    
    # Running MAD: sum(|x_i - mean_current|) / n 
    # (严谨计算需要每步重算均值，这里简化为相对于全局均值的偏差，足以说明稳定性)
    global_mean = np.mean(arr)
    running_mad = np.cumsum(np.abs(arr - global_mean)) / n_range
    
    # Running STD: sqrt( sum((x_i - mean)^2) / n )
    running_std = np.sqrt(np.cumsum((arr - global_mean)**2) / n_range)
    
    return running_mad, running_std

mad_seq, std_seq = running_statistics(data)

# 绘图对比
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 1. MAD 的表现
ax1.plot(mad_seq, color='green', label='Running MAD')
ax1.set_title(f"Mean Absolute Deviation (MAD) Convergence (alpha={alpha})")
ax1.set_ylabel("Value")
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. STD 的表现
ax2.plot(std_seq, color='red', label='Running STD')
ax2.set_title(f"Standard Deviation (STD) Divergence (alpha={alpha})")
ax2.set_ylabel("Value")
ax2.set_xlabel("Sample Size (Log Scale)")
ax2.set_xscale("log") # 使用对数坐标展示"无穷多"的过程
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()
