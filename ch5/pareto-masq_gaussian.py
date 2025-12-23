import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

np.random.seed(42)

# 1. 设定环境：肥尾，但参数恒定 (i.i.d)
# alpha = 1.5 (方差无穷大区间，典型的金融数据特征)
alpha = 1.5
N = 2000

# 生成帕累托数据 (模拟收益率变化，有正有负)
# 帕累托只产生正数，我们随机给它正负号来模拟市场涨跌
raw_data = stats.pareto.rvs(b=alpha, size=N)
signs = np.random.choice([-1, 1], size=N)
returns = raw_data * signs

# 2. 计量经济学家的视角：计算滚动波动率 (Rolling Volatility)
window_size = 50
df = pd.DataFrame({'returns': returns})
# 计算滚动标准差
df['rolling_vol'] = df['returns'].rolling(window=window_size).std()

# 3. 绘图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 上图：原始数据
ax1.plot(df['returns'], color='blue', alpha=0.6, linewidth=0.8)
ax1.set_title(f"Raw Process (i.i.d. Pareto, alpha={alpha}) - Constant Parameters", fontsize=12)
ax1.set_ylabel("Returns")
ax1.grid(True, alpha=0.3)

# 下图：伪装的异方差
ax2.plot(df['rolling_vol'], color='red', linewidth=1.5)
ax2.set_title(f"The Masquerade: Rolling Volatility (Window={window_size})", fontsize=12)
ax2.set_ylabel("Measured Volatility")
ax2.set_xlabel("Time")
ax2.grid(True, alpha=0.3)

# 标记一个"伪造的"高波动区间
ax2.axvspan(800, 1000, color='yellow', alpha=0.2, label='Looks like "High Volatility Regime"')
ax2.legend()

plt.tight_layout()
plt.show()
