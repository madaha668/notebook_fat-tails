import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. 模拟现实：数据总是有限的，特别是尾部
np.random.seed(42)
N = 500  # 只有500个数据点
alpha = 1.5
data = stats.pareto.rvs(b=alpha, size=N)

# 2. 准备画图数据
x = np.linspace(1, 10, 100)

# PDF (直方图)
hist_vals, bin_edges = np.histogram(data, bins=30, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Survival Function (经验生存函数 ECDF)
sorted_data = np.sort(data)
y_survival = 1.0 - np.arange(len(sorted_data)) / len(sorted_data)

# 3. 绘图对比
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 左图：PDF (看起来很糟糕)
ax1.plot(bin_centers, hist_vals, 'o-', label='Empirical PDF (Noisy)')
# 画出理论曲线
ax1.plot(x, stats.pareto.pdf(x, b=alpha), 'r--', label='Theoretical PDF')
ax1.set_title("PDF: Noisy and Hard to Read in Tail")
ax1.set_xlabel("Value")
ax1.set_ylabel("Density")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 右图：生存函数 Log-Log (非常清晰)
ax2.loglog(sorted_data, y_survival, 'b-', linewidth=2, label='Survival Function (Smooth)')
# 画出理论曲线
ax2.loglog(x, stats.pareto.sf(x, b=alpha), 'r--', label='Theoretical Survival')

ax2.set_title("Survival Function (Log-Log): The Straight Line Truth")
ax2.set_xlabel("Value (Log)")
ax2.set_ylabel("P(X > x) (Log)")
ax2.legend()
ax2.grid(True, which="both", alpha=0.3)

plt.show()
