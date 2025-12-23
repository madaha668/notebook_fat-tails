import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm

# 1. 设置 x 轴范围
x = np.linspace(-6, 6, 1000)

# 2. 生成不同自由度 (df) 的 t-分布 和 标准正态分布
# df (degrees of freedom) 是 t-分布的核心参数，控制它有多"胖"
y_norm = norm.pdf(x)
y_t1 = t.pdf(x, df=1)   # 柯西分布 (极端肥尾)
y_t5 = t.pdf(x, df=5)   # 典型的金融收益分布 (中等肥尾)
y_t30 = t.pdf(x, df=30) # 接近正态分布 (几乎薄尾)

# 3. 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- 左图：线性坐标 (常规视角) ---
ax1.plot(x, y_norm, 'k--', linewidth=2, label='Gaussian (Normal)')
ax1.plot(x, y_t5, 'b-', label='Student-t (df=5)')
ax1.plot(x, y_t1, 'r-', alpha=0.6, label='Student-t (df=1, Cauchy)')

ax1.set_title("Linear Scale: The 'Lower Peak'", fontsize=14)
ax1.fill_between(x, y_t5, y_norm, where=(np.abs(x)>2), color='blue', alpha=0.1, label="The Fat Tail Region")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- 右图：对数坐标 (塔勒布视角) ---
ax2.plot(x, y_norm, 'k--', linewidth=2, label='Gaussian')
ax2.plot(x, y_t5, 'b-', label='Student-t (df=5)')
ax2.plot(x, y_t1, 'r-', alpha=0.6, label='Student-t (df=1)')

ax2.set_yscale('log')
ax2.set_ylim(0.0001, 1)
ax2.set_title("Log Scale: The 'Fatter Tail'", fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.show()
