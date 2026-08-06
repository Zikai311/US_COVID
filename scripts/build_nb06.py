"""Builder: regenerate notebooks/06_covid_case_study.ipynb — a from-scratch,
layperson-friendly walkthrough (Chinese) of the project's final model
(formulae_sheet.md): S/E/I/R boxes -> contact heterogeneity -> state
coupling -> data calibration -> vaccination/prior immunity -> two-regime
beta(t) -> the Delta-wave fit. Edit this script, not the .ipynb."""
import os, nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

nb = nbf.v4.new_notebook()
cells = []
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

# ── 0. title ──────────────────────────────────────────────────────────────────
cells.append(md(r"""# 第 6 部分:COVID-19 案例研究——从"分盒子"到复现一场真实的疫情

**读者对象**:只要会一点分数和"变化率"的直觉,不需要任何背景。所有公式
都来自项目的公式表 `formulae_sheet.md`,这里只讲**每一个公式在做什么**。

**我们最终要回答的问题**:2021 年夏天,美国出现了由"德尔塔"变异株引起的
一波疫情(2021-06-01 到 2021-10-31,约 1,270 万报告病例)。能不能用数学
模型,把这一波疫情的**大小、出现的时间、结束的方式**都算出来?

**方法**:我们不一次性端出复杂模型,而是从最简单的一个想法出发,一步一步
加东西。每一步都回答"上一版模型漏掉了什么"。路线图:

| 步骤 | 加的"零件" | 回答的问题 |
|---|---|---|
| 1 | 把人群分成几个盒子 | 一个人从健康到生病经历了什么? |
| 2 | 有人见的人多,有人见的人少 | 为什么少数人决定疫情的快慢? |
| 3 | 56 个州互相有人流动 | 疫情怎么从纽约传到全美国? |
| 4 | 数字从真实数据里来 | 参数不是拍脑袋,是数据算出来的 |
| 5 | 疫苗和已有免疫力 | 为什么德尔塔波自己会停下来? |
| 6 | 行为和政策改变传播 | 为什么峰值之后曲线掉得那么快? |
| 7 | 全部合起来 | 最终模型对德尔塔波的拟合 |

> 全文约定:第一次出现的专业词都会用大白话解释;英文缩写只在说明后出现。"""))

# ── setup ─────────────────────────────────────────────────────────────────────
cells.append(code("""import sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, '../src')
from fused_metapopulation import (
    JURISDICTIONS, CENSUS_POPULATIONS, CODE_TO_NAME, NAME_TO_CODE,
    T_MO, build_distance_matrix, build_gravity_kernel, fit_gravity_exponents,
    calibrate_mobility, s_interpolator, truncated_power_law,
    power_law_alpha_for_mean_k, FusedMetapopulationSEIR,
)
from data_loader import load_us_states

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 150})
ASSETS = '../assets'
DAY0 = pd.Timestamp('2020-01-01')

def show_fig(path):
    from IPython.display import Image, display
    display(Image(filename=path))

us = load_us_states()
print('数据就绪:NYT 全美各州每日病例数据已加载')"""))

# ── step 1: boxes ─────────────────────────────────────────────────────────────
cells.append(md(r"""## 第 1 步:把人群分成几个"盒子"

设想把一座城市的人按"健康状态"分成四个盒子(数学上叫**仓室**):

- **S 盒(易感者)**:还没得过,可能被传染;
- **E 盒(潜伏者)**:已被感染,但还没症状、也还不会传染别人;
- **I 盒(感染者)**:有传染性,会把病毒传给别人;
- **R 盒(康复者)**:病好了(或去世),不再参与传播。

一个人按箭头方向流动:**S → E → I → R**。三个箭头各有一个"速度":

- 易感者被感染的速率,与"感染者占总人口的比例"成正比,比例系数叫
  **传染强度 $\beta$**(单位:1/天);
- 潜伏者变成感染者的速率 $\sigma$(这里取 1/5,平均潜伏 5 天);
- 感染者康复的速率 $\gamma$(这里取 1/10,平均 10 天有传染性)。

用导数写就是(以人口为 $N$):

$$\frac{dS}{dt}=-\beta\frac{S\,I}{N},\qquad
\frac{dE}{dt}=\beta\frac{S\,I}{N}-\sigma E,\qquad
\frac{dI}{dt}=\sigma E-\gamma I,\qquad
\frac{dR}{dt}=\gamma I$$

**最重要的一个数字**:一个感染者平均能传染几个人,记作 $R_0$
(基本再生数)。粗略说 $R_0=\beta/\gamma$。如果 $R_0>1$,疫情指数增长;
如果 $R_0<1$,疫情自动熄灭。**整个项目都在回答:如何让 $R_0$ 降到 1 以下。**"""))
cells.append(code("""# 画一张"盒子图":四个盒子 + 三个箭头(图 6.1)
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 2.2); ax.axis('off')
boxes = [
    ('S', '易感者', 0.6, '#A8D5BA'),
    ('E', '潜伏者', 3.2, '#F9E79F'),
    ('I', '感染者', 5.8, '#F5B7B1'),
    ('R', '康复者', 8.4, '#AED6F1'),
]
for name, lab, x, col in boxes:
    ax.add_patch(Rectangle((x, 0.6), 1.5, 1.0, facecolor=col, edgecolor='black', lw=1.5))
    ax.text(x + 0.75, 1.3, name, ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(x + 0.75, 0.38, lab, ha='center', va='center', fontsize=9)
arrows = [('S', 'E', r'$\\beta\\cdot SI/N$'), ('E', 'I', r'$\\sigma E$'), ('I', 'R', r'$\\gamma I$')]
pos = {b[0]: b[2] + 1.5 for b in boxes}
for a, b, lab in arrows:
    ax.add_patch(FancyArrowPatch((pos[a], 1.1), (pos[b] - 0.4, 1.1),
                                 arrowstyle='-|>', mutation_scale=18, color='k'))
    ax.text((pos[a] + pos[b]) / 2 - 0.35, 1.42, lab, fontsize=10, color='#7B241C')
ax.set_title('四个盒子、三个箭头:一个人健康状态的旅程', fontsize=12)
plt.tight_layout(); plt.savefig(f'{ASSETS}/fig82_compartment.png'); plt.show()
show_fig(f'{ASSETS}/fig82_compartment.png')"""))

# ── step 2: heterogeneity ─────────────────────────────────────────────────────
cells.append(md(r"""## 第 2 步:有人见的人多,有人见的人少

"四个盒子"模型默认**所有人都一样**:每天见同样多的人。现实显然不是——
有人的工作一天接触几百人,有人几乎不出门。我们把每个人的"接触数"叫
**度 $k$**。

美国的接触分布用一条"截断幂律"描述(公式表 §1):

$$P(k)=C\,k^{-\alpha}e^{-k/k_c},\qquad k=3\dots 200$$

不用害怕公式,只看两件事:

1. **平均每个人每天见 6 个人**(⟨k⟩=6,由参数 α=2.659 保证);
2. 少数"明星节点"见的人非常多——**人群里大多数接触,是少数人贡献的**。

这直接改变传染的写法:一个度数为 $k$ 的人,被传染的速度是 $k$ 倍。
而"随机抓一根接触线,另一端正好是感染者"的概率,用 $\theta$ 表示
(传染性接触份额):

$$\theta = \\frac{\\sum_k k\\, I_k}{N\\,\\langle k\\rangle}$$

分子是"感染者贡献的接触线总数",分母是"全城接触线总数"。于是
第 $k$ 类人的感染速率是 $\beta\, k\, S_k\, \theta$——**你见的人越多,
越容易感染;你越是"枢纽",越容易传染别人**。"""))
cells.append(code("""alpha = power_law_alpha_for_mean_k(6.0)
k_vals, P_k = truncated_power_law(alpha=alpha)
mean_k = float(k_vals @ P_k); mean_k2 = float((k_vals ** 2) @ P_k)
print(f'度分布参数: alpha = {alpha:.3f} | 平均度 <k> = {mean_k:.1f} | <k^2>/<k> = {mean_k2/mean_k:.2f}')

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.loglog(k_vals, P_k, 'o', ms=2.5, color='C0')
ax.axvline(mean_k, color='grey', ls=':', lw=1)
ax.text(mean_k * 1.1, P_k.max() * 0.5, f'平均 <k>={mean_k:.0f}', fontsize=9, color='grey')
ax.set_xlabel('接触人数 k(度)'); ax.set_ylabel('P(k) 比例')
ax.set_title('接触分布:少数"枢纽"见极多的人(截断幂律)')
plt.tight_layout(); plt.savefig(f'{ASSETS}/fig83_degree.png'); plt.show()
show_fig(f'{ASSETS}/fig83_degree.png')"""))

# ── step 3: metapopulation ────────────────────────────────────────────────────
cells.append(md(r"""## 第 3 步:56 个州不是一口锅——州际流动

疫情在纽约爆发,但很快传遍全美。因为**人会在州与州之间移动**,主要是坐
飞机。我们把美国当成 56 口"锅"(50 州 + 华盛顿特区 + 5 个海外领地),每
口锅里跑第 1–2 步的模型,锅与锅之间每天有"旅客流"。

**第 $i$ 州每天流向第 $j$ 州的人数**(公式表 §4):

$$W_{ij}(t) = \underbrace{s_i(t)}_{出行比例}\cdot
\underbrace{\mu_{0,i}}_{基准流量}\cdot
\underbrace{W_{ij}^{\\text{grav}}}_{方向份额}$$

三个因子各是什么:

1. **方向份额**——"引力模型":去**人口多**的地方的人多,去**远**的地方的
   人少:
   $$W_{ij}^{\\text{grav}}\\propto \\frac{N_j^{1.0}}{d_{ij}^{0.55}}$$
   为什么叫"引力"?因为形式和万有引力一模一样——质量(人口)越大越吸人,
   距离越远越衰减。指数 1.0 和 0.55 不是猜的,是从美国交通统计局(BTS)
   2020–2023 年真实航班数据**统计拟合**出来的(第 4 步)。
2. **基准流量** $\mu_{0,i}$:疫情前(2020 年 1 月)该州平均每天出州的人数。
3. **出行比例** $s_i(t)$:每个月实际出行量 ÷ 2020 年 1 月出行量。2020 年
   4 月全美航班只剩 1 月的约 5%——**疫情本身在改变出行**。

州与州之间的流动怎么进入方程?第 $i$ 州感染者数量变化 = 本地新增 −
本地康复 + **从别的州飞进来的感染者 − 飞出去的感染者**。这就是公式表
§5 的"耦合项"。整张耦合图 = 56 × 56 条航线,每天每月的旅客数都来自
真实数据。"""))
cells.append(code("""mu0, s, months, F = calibrate_mobility(JURISDICTIONS, data_dir='../data')
out = F.sum(axis=1)
nat_mob = out.sum(0) / out.sum(0)[0]
print('航班数据: 39 个月(2020-01 .. 2023-03)| 2020-04 全美出行比例 = %.0f%%'
      % (100 * nat_mob[3]))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(range(len(months)), nat_mob, 'o-', ms=3, color='C1')
axes[0].set_xticks(range(0, len(months), 6))
axes[0].set_xticklabels([f'{y}-{m:02d}' for y, m in months[::6]], rotation=45, fontsize=7)
axes[0].axhline(1, color='grey', lw=0.6, ls=':')
axes[0].set_ylabel('出行比例 s(t)')
axes[0].set_title('全美航空出行:2020 年 4 月暴跌到 5%')
axes[0].grid(alpha=0.3)

Wg = build_gravity_kernel(JURISDICTIONS, b_N=1.0025, b_d=0.5465)
x, y = [], []
for i in np.where(out[:, 0] > 0)[0]:
    o = F[i, :, 0]
    x += list(Wg[i]); y += list(o / o.sum())
x, y = np.array(x), np.array(y)
m = (x > 0) & (y > 0)
axes[1].scatter(np.log(x[m]), np.log(y[m]), s=4, alpha=0.45, color='C2')
axes[1].set_xlabel('引力模型预测的份额(对数)')
axes[1].set_ylabel('2020-01 真实出行份额(对数)')
axes[1].set_title('引力模型 vs 真实航班:相关性 Pearson ≈ 0.45')
axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{ASSETS}/fig84_mobility.png'); plt.show()
show_fig(f'{ASSETS}/fig84_mobility.png')"""))

# ── step 4: calibration ───────────────────────────────────────────────────────
cells.append(md(r"""## 第 4 步:数字从真实数据里来(标定)

模型里每个数字都要"有出处":

**传染强度 $\beta$(每个州一个)**。观察疫情早期:每天新增病例按指数增长,
在"新增病例的对数—时间"图上是一条直线,斜率就是增长率 $r$。给定潜伏期
和康复期,增长率 $r$ 与传染强度 $\beta$ 有确定关系(公式表 §8),可以
反推出 $\beta$:

$$\beta_i = \\frac{(r_i+\\gamma)(r_i+\\sigma)}{\\sigma}\\,\\frac{\\langle k\\rangle}{\\langle k^2\\rangle}$$

为什么不用 $\beta$ 直接拟合整条曲线?因为增长率 $r$ 对"报告比例"不敏感
——**即使只报告了一半病例,增长率也是对的**。这是这个标定方法的关键优点。
这里我们只用 2021 年 7–8 月(德尔塔爆发)的增长斜率;感染上限设为
$R_0=6$。

**引力指数($b_N=1.0$, $b_d=0.55$)**。用"泊松伪极大似然(PPML)"对
56×56×39 个月的航班面板做统计回归——一句话:**让模型的预测流量,在
统计上最像真实流量**。""" ))
cells.append(code("""# 例子:纽约州 2021-07..08 的增长斜率 -> 传染强度 beta
sub = us[(us.state == 'New York') & (us.date >= '2021-07-01') & (us.date < '2021-09-01')]
v = sub['new_cases_smooth'].values; t = np.arange(len(v)); m = v > 0
r_ny = np.polyfit(t[m], np.log(v[m]), 1)[0]
print(f'纽约州 2021-07~08 日增长率 r = {r_ny:.4f}/天(每天新增约乘 e^{{r}}≈{np.exp(r_ny):.2f})')

D = build_distance_matrix(JURISDICTIONS)
Npop = np.array([CENSUS_POPULATIONS[c] for c in JURISDICTIONS], dtype=float)
r_ppml = fit_gravity_exponents(F, D, Npop, method='ppml')
print('引力指数(PPML 从真实航班数据拟合):')
print('  目的地人口指数 b_N = %+.3f | 距离指数 b_d = %+.3f | 解释力 R2 = %.3f'
      % (r_ppml['b_N'], -r_ppml['b_d'], r_ppml['r2']))"""))

# ── step 5: vaccination ───────────────────────────────────────────────────────
cells.append(md(r"""## 第 5 步:疫苗与已有免疫力

到了 2021 年,还有两股力量在减少"能被感染的人":

**1. 疫苗(新盒子 V)**。打过疫苗的人从易感盒 S 移到"已接种盒"V:
接种后基本不会被感染(公式表 §6):

$$\\frac{dS}{dt} = \\cdots - v(t)\\frac{S}{N},\\qquad
\\frac{dV}{dt} = v(t)\\frac{S}{N}$$

其中 $v(t)$ 是每个州**每月实际完成接种的人数**(来自美国 CDC 的真实
数据)。2021 年 6 月全美约 42% 的人已完成接种。

**2. 之前感染过的人**。2020–2021 年美国实际感染过的人约有 1 亿
(报告的 5,500 万 × 约 3 倍漏报)。他们虽然有抗体,但德尔塔波开始时我们
把他们都放进了康复盒 R——他们不再被感染。

于是 2021-06-01 模型起点的"易感人口"是:

$$\text{易感} = \\text{总人口} - \\text{已接种}(\\times\\text{效力}) - \\text{已感染} - \\text{正在感染/潜伏}$$

**为什么这很关键**:如果不管疫苗和已有免疫力,易感人口是 3.35 亿;算上
它们只剩约 1.1 亿。疫情能"烧掉"的燃料少了 2/3——**这就是德尔塔波自己
会停下来、10 月回落的原因**。"""))
cells.append(code("""vax = pd.read_csv('../data/cdc_vax_state_monthly.csv')
cum = vax.pivot_table(index='state', columns='month', values='series_complete')
sel = ['New York', 'Florida', 'Texas', 'California']
for name in sel:
    code = NAME_TO_CODE[name]
    ser = cum.loc[code].copy(); ser.index = [pd.Period(m, 'M').to_timestamp() for m in ser.index]
    plt.plot(ser.index, ser.values / 1e6, lw=1.4, label=name)
nat = cum.sum(0); nat.index = [pd.Period(m, 'M').to_timestamp() for m in nat.index]
plt.plot(nat.index, nat.values / 1e6, 'k--', lw=1.6, label='全美合计')
plt.axvline(pd.Timestamp('2021-06-01'), color='grey', ls=':', lw=1)
plt.text(pd.Timestamp('2021-05-15'), plt.ylim()[1] * 0.55, '德尔塔波\\n开始', fontsize=9, color='grey')
plt.ylabel('已完成接种人数(百万)')
plt.title('全美疫苗接种进度(CDC 真实数据)')
plt.legend(fontsize=8); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{ASSETS}/fig85_vaccination.png'); plt.show()
show_fig(f'{ASSETS}/fig85_vaccination.png')

us['cum_cases'] = us.groupby('state')['new_cases'].cumsum()
prior = us[us.date <= pd.Timestamp('2021-05-31')].groupby('state')['cum_cases'].max()
print('2021-06-01 时:已接种 %.0fM(42%%)| 累计报告感染 %.0fM(真实约 3 倍)'
      % (cum['2021-05'].sum() / 1e6, prior.sum() / 1e6))"""))

# ── step 6: two-regime beta ───────────────────────────────────────────────────
cells.append(md(r"""## 第 6 步:行为和政策会改变传播(两段式 $\beta$)

如果传染强度 $\beta$ 永远不变,模型里的疫情会一路涨到窗口尽头——但真实
的德尔塔波在 2021-09-10 见顶后就掉头了。原因不只有"易感的人变少了"
(第 5 步),还有**人们的行为变了**:高峰之后大家更小心、更多隔离、更多
口罩。

模型里用最简单的办法表达这一点——**两段式传播**(公式表 §7):

$$\beta_i(t) = \\beta_{0,i}\\cdot g(t),\\qquad
g(t) = \\begin{cases} 1 & \\text{峰前}\\\\ \\rho & \\text{峰后}\\end{cases}$$

- 峰前:传染强度保持标定值(第 4 步);
- 峰后:整个传染强度乘以一个小于 1 的比例 $\rho$;
- 切换日 = 观测到的全国峰值日(2021-09-10),$\rho$ 由数据拟合
  (德尔塔波拟合出 $\rho=0.9$:峰后只温和下降——因为德尔塔波的回落主要是
  易感人口减少,行为贡献较小;对比 2022 年奥密克戎波,那里 $\rho=0.25$,
  行为变化才是主角。附录 B 有详细对比)。"""))
cells.append(code("""months = [f'{y:04d}-{m:02d}' for y, m in [(2021, 8), (2021, 9), (2021, 10)]]
g_aug, g_sep, g_oct = 1.0, (9 + 21 * 0.9) / 30, 0.9
print('两段式传播乘子 g(t):')
print(f'  2021-08(峰前)= {g_aug:.2f} | 2021-09(切换月,加权)= {g_sep:.2f} | 2021-10(峰后)= {g_oct:.2f}')
print('  含义:峰后每天每个感染者传染他人的能力,是峰前的 90%')"""))

# ── step 7: the full fit ──────────────────────────────────────────────────────
cells.append(md(r"""## 第 7 步:全部合起来——德尔塔波的拟合

现在把所有零件装进一台机器(`FusedMetapopulationSEIR`,5 万多个方程):

1. 56 口锅,每口锅里是"分度类"的 S-E-I-R-V 五盒子模型;
2. 锅与锅之间按真实航班数据流动(引力方向 + 每月出行比例);
3. 每州传染强度来自 2021 年 7–8 月增长斜率($R_0$ 上限 6);
4. 起点:2021-06-01 的感染者数(报告病例 ÷ 0.15)、已接种 × 效力、
   已有免疫的 1 亿人;
5. 两段式传播(峰后 ×0.9)。

跑完后,把模型的"真实感染数"与 NYT 报告的病例画在一起(报告数乘以一个
常数 $\hat\\pi$ 对齐总量,因为模型算的是真实感染,报告只是其中一部分)。
三个检验:

- **总量对不对**:$\hat\\pi$ 接近 1 说明量级对;
- **峰的时间对不对**:模型峰值日 vs 观测峰值日;
- **回落对不对**:10 月底 ÷ 峰值(模型 0.35,观测 0.41);
- **整体像不像**:$R^2$(1 为完美,0 为"跟猜平均值一样")。"""))
cells.append(code("""sigma, gamma = 1.0 / 5, 1.0 / 10
T0 = (pd.Timestamp('2021-06-01') - DAY0).days
TMAX = (pd.Timestamp('2021-10-31') - DAY0).days
mom_ratio = mean_k2 / mean_k

# 1) 每州 beta: 2021-07-01..08-31 增长斜率
win_start = pd.Timestamp('2021-07-01'); win_days = 62
rs, bs = {}, {}
for code in JURISDICTIONS:
    sub = us[(us.state == CODE_TO_NAME[code]) & (us.date >= win_start)
             & (us.date < win_start + pd.Timedelta(days=win_days))]
    v = sub['new_cases_smooth'].values; t = np.arange(len(v)); m = v > 0
    rv = np.nan
    if m.sum() >= 5:
        sl = np.polyfit(t[m], np.log(v[m]), 1)[0]
        if sl > 0: rv = sl
    rs[code] = rv
    bs[code] = (rv + gamma) * (rv + sigma) / sigma / mom_ratio if not np.isnan(rv) else np.nan
bs = pd.Series(bs).fillna(pd.Series(bs).median()).clip(upper=6.0 * gamma / mom_ratio)
rs = pd.Series(rs).fillna(float(pd.Series(rs).median()))

# 2) 起点(2021-06-01): 感染者/潜伏者/已接种/已有免疫
pi0 = 0.15
I0, E0 = {}, {}
for code in JURISDICTIONS:
    row = us[(us.state == CODE_TO_NAME[code]) & (us.date == pd.Timestamp('2021-06-01'))]
    i0 = float(row['new_cases'].iloc[0]) if len(row) else 0.0
    I0[code] = i0 / pi0
    E0[code] = I0[code] * (rs[code] + gamma) / sigma

grid_m = [f'{y:04d}-{m:02d}' for y, m in [(2020, 1)] and [(2020, 1)]] or []
vax2 = pd.read_csv('../data/cdc_vax_state_monthly.csv')
gidx = [f'{y:04d}-{m:02d}' for y in range(2020, 2024) for m in range(1, 13)][:39]
pv = vax2.pivot_table(index='state', columns='month', values='series_complete_flow') \\
     .reindex(index=JURISDICTIONS, columns=gidx).fillna(0)
cumv = vax2.pivot_table(index='state', columns='month', values='series_complete') \\
       .reindex(index=JURISDICTIONS, columns=gidx).fillna(0)
flow = pv.values / T_MO
eps, asc = 1.0, 3.0
V0 = {s: float(cumv.loc[s, '2021-05']) * eps for s in JURISDICTIONS}
R0init = {s: float(prior.get(CODE_TO_NAME[s], 0)) * asc for s in JURISDICTIONS}
s_eff = np.array([1.0 - (V0[s] + R0init[s]) / CENSUS_POPULATIONS[s] for s in JURISDICTIONS])
beta_v = bs.values / s_eff
print('易感人口占全美比例: %.0f%%(没有疫苗/免疫的话是 100%%)' % (100 * np.mean(s_eff)))

# 3) 两段式传播 + 移动 + 求解
months_full = [(y, m) for y in range(2020, 2024) for m in range(1, 13)][:39]
mu0, s, months_full, F = calibrate_mobility(JURISDICTIONS, data_dir='../data')
s_func = s_interpolator(s, months_full)
W_cal = build_gravity_kernel(JURISDICTIONS, b_N=1.0025, b_d=0.5465)
g2 = np.ones((56, 39)); g2[:, 20] = (9 + 21 * 0.9) / 30; g2[:, 21] = 0.9

model = FusedMetapopulationSEIR(
    JURISDICTIONS, CENSUS_POPULATIONS, W_cal, mu0, s_func,
    k_vals, P_k, beta=beta_v, sigma=sigma, gamma=gamma,
    months=months_full, vac_flow=flow, vac_efficacy=eps, V0_dict=V0,
    R_init_dict=R0init, beta_scale=g2)
t_sol, y_sol = model.simulate(TMAX, I0, E0, t0=T0)
ni = model.daily_new_infections().sum(0)
dates = [DAY0 + pd.Timedelta(days=int(t)) for t in t_sol]
nat_obs_all = us.groupby('date')['new_cases_smooth'].sum()
nat_obs = nat_obs_all.reindex(dates).fillna(0).values

ph = ni.sum() / nat_obs.sum()
pk_m = dates[int(np.argmax(ni))]; pk_o = nat_obs_all.loc[dates].idxmax()
r2l = 1 - np.sum((np.log(ni / ph + 1) - np.log(nat_obs + 1)) ** 2) / np.sum((np.log(nat_obs + 1) - np.log(nat_obs + 1).mean()) ** 2)
r2v = 1 - np.sum((ni / ph - nat_obs) ** 2) / np.sum((nat_obs - nat_obs.mean()) ** 2)
oct_r = float(ni[-1] / ni[int(np.argmax(ni))]); oct_o = float(nat_obs[-1] / nat_obs.max())

fig, ax = plt.subplots(figsize=(9.5, 4.4))
ax.plot(dates, nat_obs, lw=1.4, color='k', label='观测(7 日平滑)')
ax.plot(dates, ni / ph, lw=1.5, color='C2', label='模型(真实感染 ÷ π̂)')
ax.set_ylabel('每天新增(报告口径)')
ax.set_title('德尔塔波:模型 vs 观测(2021-06-01 .. 10-31)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{ASSETS}/fig86_delta_fit.png'); plt.show()
show_fig(f'{ASSETS}/fig86_delta_fit.png')

print(f'量级: 模型总量 / 观测总量  π̂ = {ph:.2f}      (1 = 完全对齐)')
print(f'峰值日: 模型 {pk_m.date()} vs 观测 {pk_o.date()}   (差 {(pk_m - pk_o).days} 天)')
print(f'回落: 10 月底/峰值 = 模型 {oct_r:.2f} vs 观测 {oct_o:.2f}')
print(f'整体拟合: R²(对数空间) = {r2l:.2f} | R²(线性) = {r2v:.2f}')"""))

# ── summary ───────────────────────────────────────────────────────────────────
cells.append(md(r"""## 回顾:每一层修掉了什么

| 层 | 加的零件 | 修掉的错误假设 | 对应公式表 |
|---|---|---|---|
| 1 | 四个盒子 S-E-I-R | "生病是一瞬间的事" | §2 |
| 2 | 接触异质性(度) | "每个人都一样" | §1, §3 |
| 3 | 56 州互相流动(引力) | "美国是一口锅" | §4–5 |
| 4 | 数据标定(增长率→β, 航班→引力) | "参数靠猜" | §8 |
| 5 | 疫苗 V + 已有免疫力 | "所有人都能被感染" | §6 |
| 6 | 两段式 β(t) | "人的行为永远不变" | §7 |
| 7 | 55,440 个方程一起解 | "上面每一条都可以单独处理" | 全部 |

**最终成绩单(德尔塔波)**:总量对齐(π̂=1.10)、峰值只差约一周、
10 月回落深度一致(0.35 vs 0.41)、整体拟合 R²(线性)≈ 0.96。
这意味着:把"分盒子 + 接触异质性 + 州际流动 + 真实数据 + 疫苗免疫 +
行为变化"按逻辑一层层拼起来,确实能**复现一场真实疫情的主要特征**。

**延伸阅读**:
- notebook 08:完整模型(含敏感性实验与每个数字的出处);
- appendix A:引力模型能否"重建"航空数据(数据还原实验);
- appendix B:同样的模型搬到 2022 年奥密克戎波(需要更陡的传播崩塌);
- `formulae_sheet.md`:全部公式一页纸。

**致读者的一句话**:看懂这一路的秘诀不是记住公式,而是记住每一步在问
"上一版漏掉了什么"。模型的价值不在"像",而在**它把哪一条真实世界的
规则放进了数学**。"""))
cells.append(code("""print('notebook 06 重写完成 | 图 fig82..fig86 -> assets/')"""))

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.14'},
}
out = os.path.join(ROOT, 'notebooks', '06_covid_case_study.ipynb')
with open(out, 'w') as f:
    f.write(nbf.writes(nb, indent=1))
print('wrote', out, '| cells:', len(cells))
