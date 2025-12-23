import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

np.random.seed(42)
N = 1000

# 1. 生成数据
# 极端斯坦：帕累托分布 (alpha = 1.16, 典型的80/20法则参数)
alpha = 1.16
pareto_data = stats.pareto.rvs(b=alpha, size=N)

# 平庸斯坦：正态分布 (取绝对值以模拟大小)
gaussian_data = np.abs(np.random.normal(0, 1, N))

# 2. 准备齐普夫图数据 (排序)
# 从大到小排序
pareto_sorted = np.sort(pareto_data)[::-1]
gaussian_sorted = np.sort(gaussian_data)[::-1]

# 生成排名 (1, 2, ..., N)
ranks = np.arange(1, N + 1)

# 3. 绘图
fig, ax = plt.subplots(figsize=(10, 7))

# 绘制帕累托 (齐普夫定律)
ax.loglog(ranks, pareto_sorted, 'r-', linewidth=2, label=f'Pareto (Fat Tail, alpha={alpha})')

# 绘制高斯
ax.loglog(ranks, gaussian_sorted, 'b-', linewidth=2, label='Gaussian (Thin Tail)')

# 4. 添加辅助直线来验证线性趋势
# 理论斜率应该是 -1/alpha
# 我们在帕累托曲线上画一条拟合直线
fit_slope = -1.0 / alpha
# 找一个截距点让线对齐 (比如通过第10个点)
intercept = np.log(pareto_sorted[9]) - fit_slope * np.log(ranks[9])
y_fit = np.exp(intercept + fit_slope * np.log(ranks))

ax.loglog(ranks, y_fit, 'k--', linewidth=1.5, label=f'Theoretical Slope (-1/{alpha} ≈ {fit_slope:.2f})')

ax.set_title("Zipf Plot (Rank-Size Plot): Linearity Test", fontsize=14)
ax.set_xlabel("Rank (Log Scale)")
ax.set_ylabel("Size / Value (Log Scale)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

plt.show()
