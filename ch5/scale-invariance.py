import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. 准备数据
N = 1000000
# 帕累托 (尺度不变)
alpha = 1.5
pareto_data = stats.pareto.rvs(b=alpha, size=N)
# 高斯 (尺度依赖) - 取绝对值
gaussian_data = np.abs(np.random.normal(0, 1, N))

# 2. 定义验证函数：计算 P(X > k*x | X > x)
def calculate_conditional_prob_ratio(data, k=2.0, steps=50):
    # 设定 x 的扫描范围 (从 10% 分位 到 99% 分位)
    thresholds = np.logspace(np.log10(np.percentile(data, 10)), 
                             np.log10(np.percentile(data, 99)), 
                             steps)
    
    ratios = []
    valid_thresholds = []
    
    for x in thresholds:
        # 分母：有多少数据 > x
        count_x = np.sum(data > x)
        if count_x < 10: continue # 数据太少就不算了
        
        # 分子：在 >x 的这些数据里，有多少 > k*x
        count_kx = np.sum(data > (k * x))
        
        # 计算条件概率
        prob = count_kx / count_x
        ratios.append(prob)
        valid_thresholds.append(x)
        
    return valid_thresholds, ratios

# 3. 计算
k_factor = 2.0 # 我们看"翻倍"的概率
x_par, y_par = calculate_conditional_prob_ratio(pareto_data, k=k_factor)
x_gau, y_gau = calculate_conditional_prob_ratio(gaussian_data, k=k_factor)

# 4. 绘图
fig, ax = plt.subplots(figsize=(10, 6))

# 帕累托结果
ax.plot(x_par, y_par, 'r-', linewidth=3, label=f'Pareto (Scale Invariant): P(>2x | >x)')
# 理论值: (2x)^-a / x^-a = 2^-a
theoretical_prob = k_factor**(-alpha)
ax.axhline(theoretical_prob, color='black', linestyle='--', label=f'Theoretical Constant ({theoretical_prob:.2f})')

# 高斯结果
ax.plot(x_gau, y_gau, 'b-', linewidth=3, label='Gaussian (Scale Dependent): P(>2x | >x)')

ax.set_title(f"Visualizing Scale Invariance (k={k_factor})")
ax.set_xlabel("Threshold x (Log Scale)")
ax.set_ylabel("Conditional Probability (Probability of Doubling)")
ax.set_xscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

plt.show()
