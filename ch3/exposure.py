import numpy as np
import matplotlib.pyplot as plt

# 模拟市场变化：从 -50% (崩盘) 到 +50% (暴涨)
market_change = np.linspace(-0.50, 0.50, 100)

# 1. 线性敞口 (持有股票)
# P&L = 敞口 * 变化
linear_pnl = 1.0 * market_change 

# 2. 凹敞口 (Concave / 脆弱) - 比如做空波动率，或者高杠杆爆仓机制
# 这是一个简化的二次函数模型：P&L = 变化 - c * 变化^2
# 在塔勒布看来，这代表如果你在极端行情下，不仅亏钱，还会因为流动性枯竭亏得更多
concave_pnl = market_change - 5.0 * (market_change**2)

# 3. 凸敞口 (Convex / 反脆弱) - 比如持有虚值看跌期权
# 在平静时亏一点权利金，大跌时赚大钱
convex_pnl = -0.05 + 5.0 * (market_change**2) # -0.05 是成本

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(market_change, linear_pnl, label="Linear Exposure (Stock)", linestyle='--')
ax.plot(market_change, concave_pnl, label="Concave Exposure (Fragile)", color='red', linewidth=2)
ax.plot(market_change, convex_pnl, label="Convex Exposure (Antifragile)", color='green', linewidth=2)

# 标记“黑天鹅”区域
ax.axvspan(-0.5, -0.2, color='gray', alpha=0.2, label="Fat Tail Event (Crash)")

ax.set_title("Payoff Profiles: Why Exposure Shape Matters More Than Prediction")
ax.set_xlabel("Market Move (%)")
ax.set_ylabel("Profit / Loss")
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.legend()
ax.grid(True, alpha=0.3)

plt.show()
