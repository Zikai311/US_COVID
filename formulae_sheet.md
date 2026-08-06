# Formula Sheet — Fused Degree-Based SEIR × Gravity Metapopulation (SEIRV)

**2026-08-06 (final).** Each of 56 jurisdictions (50 states + DC + 5
territories) runs a **degree-class SEIR** (notebook 04); states are coupled
by a **temporally calibrated gravity model** (notebook 08); vaccination
depletes the susceptible pool (**SEIRV**, §6). Units: days, counts.

---

## 1. Degree distribution

$$P(k) = C\, k^{-\alpha}\, e^{-k/k_c}, \quad k = 3\dots 200,\ k_c = 1000$$
$$\langle k\rangle = \sum_k k\,P(k) = 6 \ \Rightarrow\ \alpha = 2.659, \qquad
\frac{\langle k^2\rangle}{\langle k\rangle} = 17.006$$
$$R_0 = \frac{\beta\,\langle k^2\rangle}{\gamma\,\langle k\rangle}$$
(全局截断幂律;密度→⟨k⟩ᵢ 缩放常量 ⟨k⟩₀、ξ 未定,fallback 用此全局分布)

## 2. Within-state dynamics (state i, degree class k)

$$\frac{dS_k}{dt} = -\beta\, k\, S_k\, \theta, \qquad
\frac{dE_k}{dt} = \beta\, k\, S_k\, \theta - \sigma E_k, \qquad
\frac{dI_k}{dt} = \sigma E_k - \gamma I_k, \qquad
\frac{dR_k}{dt} = \gamma I_k$$
(每个度类独立传播;仅 I 传染;E 潜伏不传播)

## 3. Force of infection — infectious-stub fraction

$$\theta_i = \frac{\sum_k k\, I_{ik}}{N_i\,\langle k\rangle}
\quad\Big(\equiv \frac{1}{\langle k\rangle}\sum_k k\,P(k)\, i_k,\quad
i_k = \frac{I_{ik}}{N_i P_i(k)}\Big)$$
**⚠ 勿把 P(k) 与计数混用**: $\frac{1}{\langle k\rangle}\sum_k k P(k) I_k/N$
的线性化矩是 $\sum k^2P^2\approx1$(低 k 主导)而非 ⟨k²⟩ → 亚临界,即使
R₀>1 也衰减(2026-08-06 notebook-08 修复的 bug)。

## 4. Inter-state coupling — temporal gravity

$$W_{ij}(t) = s_i(t)\cdot\mu_{0,i}\cdot W_{ij}^{\text{grav}} \qquad \text{[travellers/day]}$$
$$W_{ij}^{\text{grav}} \propto \frac{N_i^{\eta}\, N_j^{b_N}}{d_{ij}^{b_d}},\quad
\sum_j W^{\text{grav}}_{ij}=1$$
- $b_N = 1.003$、$b_d = 0.546$(PPML 校准);$\eta = 0.5$ 行归一后无关;d ≥ 100 km;对角=0
- $\mu_{0,i} = \dfrac{\sum_{j\ne i}F_{ij}(\text{Jan-2020})}{T_{\text{mo}}}$(每州 Jan-2020 日出流量)
- $s_i(t) = \dfrac{\sum_{j\ne i}F_{ij}(t)}{\sum_{j\ne i}F_{ij}(\text{Jan-2020})}$(每月移动比例,纯航空)
- 自洽:$\sum_j W_{ij}(t)$ = 观测日出流量(州内自环排除;无 Jan 基线 ⇒ μ₀=0,s≡0)

## 5. Coupled ODEs (state i, class k; X ∈ {S,E,I,R,V})

$$\frac{dX_{ik}}{dt} = \text{local}_X + \sum_j\Big[\,W_{ji}(t)\frac{X_{jk}}{N_j} - W_{ij}(t)\frac{X_{ik}}{N_i}\,\Big]$$
(双向非对称流量项,随机流动假设;全局人口守恒,州内总量随旅客移动;旅客保留度类)

## 6. SEIRV — vaccination depletes the susceptible pool

$$\frac{dS_{ik}}{dt} = -\beta k S_{ik}\theta_i - v_i(t)\,\frac{S_{ik}}{N_i} + w\,V_{ik},\qquad
\frac{dV_{ik}}{dt} = v_i(t)\,\frac{S_{ik}}{N_i} - w\,V_{ik}$$
- $v_i(t)$ = 月完全接种流(people/day)×ε;ε=1.0(wave-4 拟合,隐含吸收突破感染)
- $V_0$ = 窗口起点累计接种 × ε;$R_{\text{prior}}$ = 先验感染免疫(报告累计病例 × asc=3 ≈ 1 亿人,修复峰偏晚)
- $S_0 = N_i - I_0 - E_0 - V_0 - R_{\text{prior}}$;5·n·K = 55,440 ODE;`vac_flow=None` ⇒ 4 仓室逐位一致

## 7. Time-varying β(t) — NPI/behaviour channel

$$\beta_i(t) = \beta_{0,i}\, g_i(t)$$
- CMR(行为):$g_i(t) = 1 + \theta\, c_i(t)/100$(c = transit 月 % 变化 vs 2020-01 基线)
- OxCGRT(政策):$g_i(t) = e^{-\lambda\, s_i(t)/100}$(stringency 0–100)
- 校准月归一化:$g_i(\text{β₀ 校准月}) = 1$;缺数据 → 全国序列回退
- 教训:单一乘子无法同时匹配 4 月抑制与 6 月恢复(wave-1);Delta 的 10 月回落来自易感池耗竭(§6),非移动性

## 8. Calibration

**β from early growth** (增长窗: wave-1 2020-03-01→03-30;wave-4 2021-07-01→08-31):
$$r_i = \text{slope of } \log\big(\text{7d-smoothed new cases}\big),\qquad
\beta_i = \frac{(r_i+\gamma)(r_i+\sigma)}{\sigma}\,\frac{\langle k\rangle}{\langle k^2\rangle}, \qquad
\beta_i \leftarrow \min\big(\beta_i,\ R_0^{\max}\gamma\langle k\rangle/\langle k^2\rangle\big)$$
(R₀ cap: wave-1 4.0 / wave-4 6.0;斜率对报告率不变 ⇒ 无需 π;无可用窗 → 全国中位 β)

**Gravity exponents (PPML)**, origin + month fixed effects:
$$F_{ijt} \sim \text{Poisson}\big(\exp(\alpha_i + \gamma_t + b_N\ln N_j + b_d\ln d_{ij})\big)$$
(PPML 避免对数化 OLS 的 Jensen 偏差;OLS 仅供对比)

**Fixed parameters**: $\sigma = 1/5\ \mathrm{d}^{-1}$, $\gamma = 1/10\ \mathrm{d}^{-1}$,
$T_{\text{mo}} = 30.44\ \mathrm{d}$, IC 报告率 $\pi_0$: wave-1 0.1 / wave-4 0.3。

## 9. Variants (experiments, not defaults)

| 变体 | 公式 | 用途 / 结论 |
|---|---|---|
| pair-level 饱和 | $W_{ij}(t) = \mu_{0,ij}\, s_{ij}(t) = F_{ij}(t)/T_{\text{mo}}$ | 数据回放:R²(log)=1.000 平凡;敏感性对照(总量 ±20%,onset ±7–14 d) |
| 时不变 pair 因子 | $W_{ij}(t) = s_i(t)\,\mu_{0,ij}$ | 中间路线,R²(log) 0.247→0.908,可时间外推 |
| β(t) 驱动 | §7 | wave-1 π̂ 29.8→3.4、4 月回落重现,但 6 月过冲 17× |

## 10. Fit metrics

$$\hat\pi = \frac{\sum_t \text{model}(t)}{\sum_t \text{observed}(t)}, \qquad
R^2 = 1 - \frac{\sum (\hat y - y)^2}{\sum (y - \bar y)^2}$$
(模型输出真实感染,π̂ 为窗口总量比;R² 在 log(x+1) 空间与线性空间都报告)

**Wave-4 (Delta) final**: π̂ 3.37→**1.06**,R²(log) −0.29→**+0.85**,
R²(linear) −0.21→**+0.82**,峰 10-02→**09-12**(观测 09-10),10 月回落 0.52 vs 观测 0.41。
