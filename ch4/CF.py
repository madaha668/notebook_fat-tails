import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy

# 1. 定义空间 (x) 和 频率 (t)
x = np.linspace(-20, 20, 1000)
dx = x[1] - x[0]
t_values = np.linspace(-5, 5, 200)

# 2. 获取 PDF 信息
pdf_gauss = norm.pdf(x)
pdf_cauchy = cauchy.pdf(x)

# 3. 推导特征函数 (CF) - 数值积分方法
# CF(t) = Integral( e^{i*t*x} * f(x) dx )
def compute_cf(t_vals, x_vals, pdf_vals, dx):
    cf_vals = []
    for t in t_vals:
        # e^{itx} = cos(tx) + i*sin(tx)
        # 我们这里主要看实部 (对于对称分布，虚部为0)
        integrand = np.cos(t * x_vals) * pdf_vals
        val = np.sum(integrand) * dx
        cf_vals.append(val)
    return np.array(cf_vals)

cf_gauss = compute_cf(t_values, x, pdf_gauss, dx)
cf_cauchy = compute_cf(t_values, x, pdf_cauchy, dx)

# 4. 绘图对比
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# PDF 域
ax1.plot(x, pdf_gauss, label='Gaussian PDF', color='blue', alpha=0.6)
ax1.plot(x, pdf_cauchy, label='Cauchy PDF (Fat Tail)', color='red', alpha=0.6)
ax1.set_title("Probability Density Function (PDF)")
ax1.set_xlim(-10, 10)
ax1.legend()
ax1.grid(True, alpha=0.3)

# CF 域
ax2.plot(t_values, cf_gauss, label='Gaussian CF (e^{-t^2/2})', color='blue')
ax2.plot(t_values, cf_cauchy, label='Cauchy CF (e^{-|t|})', color='red')
# 理论上的 Cauchy CF 形状
ax2.plot(t_values, np.exp(-np.abs(t_values)), color='black', linestyle='--', alpha=0.3, label='Theoretical Cauchy CF')

ax2.set_title("Characteristic Function (Derived from PDF)")
ax2.set_xlabel("Frequency t")
ax2.set_ylabel("phi(t)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.show()
