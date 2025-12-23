import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

np.random.seed(42)
N = 10000

# 1. 生成基础数据
# 帕累托分布 (Fat Tail)
alpha = 1.5
pareto_data = stats.pareto.rvs(b=alpha, size=N)

# 2. 生成超级肥尾数据 (Log-Pareto)
# 公式: X = e^Y, 其中 Y 是帕累托分布
# 注意：为了避免数值溢出（e^100太大了），我们缩放一下 Y
y_scaled = pareto_data / 10.0 
super_fat_data = np.exp(y_scaled)

# 3. 准备绘图函数 (生存函数 Log-Log)
def get_survival_data(data):
    sorted_data = np.sort(data)
    y = 1.0 - np.arange(len(data)) / len(data)
    return sorted_data, y

x_pareto, y_pareto = get_survival_data(pareto_data)
x_super, y_super = get_survival_data(super_fat_data)

# 4. 绘图
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制帕累托 (直线)
ax.loglog(x_pareto, y_pareto, 'b-', linewidth=2, label='Pareto (Fat Tail) -> Straight Line')

# 绘制超级肥尾 (曲线)
ax.loglog(x_super, y_super, 'r-', linewidth=2, label='Log-Pareto (Super Fat) -> Curving Up')

ax.set_title("Fat Tail vs. Super Fat Tail (Log-Log Plot)", fontsize=14)
ax.set_xlabel("Value (Log Scale)")
ax.set_ylabel("Probability P(X>x) (Log Scale)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

plt.show()
