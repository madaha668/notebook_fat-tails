import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 设置随机种子以便复现
np.random.seed(42)

# 1. 生成数据
N = 10000  # 样本数量

# 平庸斯坦：正态分布 (均值=0, 标准差=1)
gaussian_data = np.random.normal(0, 1, N)

# 极端斯坦：帕累托分布 (Pareto Distribution)
# Scipy中的 'b' 参数就是塔勒布书中的 alpha (α)
# 当 alpha = 1.16 时，这就是著名的 "80/20 法则" (80%的财富掌握在20%人手中)
alpha_tail = 1.16 
pareto_data = stats.pareto.rvs(b=alpha_tail, size=N)

# 打印极值对比
print(f"高斯分布最大值: {np.max(gaussian_data):.2f} (约为均值的 {np.max(gaussian_data)/np.std(gaussian_data):.2f} 倍标准差)")
print(f"帕累托分布最大值: {np.max(pareto_data):.2f} (注意看这个数值有多巨大)")

def plot_survival_function(data, label, color, ax):
    # 1. 对数据进行排序
    sorted_data = np.sort(data)
    # 2. 计算生存概率 P(X > x)
    # y轴：大于当前数值的数据占比
    y = 1.0 - np.arange(len(data)) / len(data)

    # 3. 绘制双对数图
    ax.loglog(sorted_data, y, label=label, color=color, linewidth=2)

fig, ax = plt.subplots(figsize=(10, 6))

# 只取正数部分画图 (Log坐标不能处理负数)
plot_survival_function(gaussian_data[gaussian_data>0], "Gaussian (Mediocristan)", "blue", ax)
plot_survival_function(pareto_data, f"Pareto (Extremistan, alpha={alpha_tail})", "red", ax)

ax.set_title("Survival Function: Log-Log Plot (The Signature of Fat Tails)", fontsize=14)
ax.set_xlabel("Value (x) - Log Scale")
ax.set_ylabel("Probability P(X > x) - Log Scale")
ax.grid(True, which="both", ls="--", alpha=0.5)
ax.legend()

plt.show()

from sklearn.linear_model import LinearRegression

def estimate_alpha(data, start_percentile=0.9):
    """
    通过尾部数据的对数回归来估算 Alpha。
    注意：这是一种简化的直觉性算法 (Hill Estimator是更严谨的方法，但这个更直观)
    我们只看数据最大的 10% (即尾部)
    """
    # 1. 截取尾部数据
    threshold = np.quantile(data, start_percentile)
    tail_data = data[data > threshold]
    tail_data = np.sort(tail_data)

    # 2. 准备 X 和 Y
    # X = log(数值)
    # Y = log(生存概率)
    n = len(tail_data)
    y_probs = 1.0 - np.arange(n) / n
    # 去掉最后一个点避免 log(0)
    X = np.log(tail_data[:-1]).reshape(-1, 1)
    Y = np.log(y_probs[:-1])

    # 3. 线性回归拟合斜率
    model = LinearRegression()
    model.fit(X, Y)

    # 斜率是负的 alpha
    estimated_alpha = -model.coef_[0]
    return estimated_alpha

# 计算并验证
calc_alpha = estimate_alpha(pareto_data)
print(f"我们设定的真实 Alpha: {alpha_tail}")
print(f"从数据中反向估算的 Alpha: {calc_alpha:.4f}")

if calc_alpha < 2:
    print("结论: 这是一个肥尾分布，方差是无穷大的，标准差失效。")
else:
    print("结论: 这是一个薄尾分布，可以使用标准差。")

 
def plot_max_to_sum(data, name, ax):
   # 计算累计和 (Partial Sums)
    cumsum = np.cumsum(data)
    # 计算累计最大值 (Running Max)
    running_max = np.maximum.accumulate(data)
    # 计算比例
    ratio = running_max / cumsum

    ax.plot(ratio, label=name, linewidth=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 高斯数据的 Max/Sum
# 我们取绝对值，因为高斯有负数，便于比较量级
plot_max_to_sum(np.abs(gaussian_data), "Gaussian Abs Data", ax1)
ax1.set_title("Max-to-Sum Ratio: Gaussian")
ax1.set_ylim(0, 1)
ax1.set_xlabel("Number of Samples")

# 帕累托数据的 Max/Sum
plot_max_to_sum(pareto_data, "Pareto Data", ax2)
ax2.set_title(f"Max-to-Sum Ratio: Fat Tail (alpha={alpha_tail})")
ax2.set_ylim(0, 1)
ax2.set_xlabel("Number of Samples")

plt.show()
