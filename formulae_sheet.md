# Fused Degree-Based SEIR × Metapopulation Model — Formula Sheet

**PART II (2026-08-05).** Replaces the old "single-population SEIR +
gravity mobility" sheet. Each state/node is a **degree-based SEIR** model
(notebook 04) embedded in a **gravity metapopulation** (notebook 08);
a traveller keeps their degree when moving between states.

---

## 1. Degree distribution

The degree distribution is a **truncated / exponentially-decaying power
law** (high-*k* tail drops exponentially):

$$P(k) = C\, k^{-\alpha}\, e^{-k/k_{c}},\qquad k = k_{\min}, \dots, k_{\max},$$

with normalisation constant $C$ and mean degree

$$\langle k \rangle = \sum_k k\, P(k).$$

---

## 2. Within-state dynamics — degree-based SEIR

For each state and each degree class $k$:

$$\frac{dS_k}{dt} = -\beta\, k\, S_k\, \theta$$

$$\frac{dE_k}{dt} = \beta\, k\, S_k\, \theta - \sigma\, E_k$$

$$\frac{dI_k}{dt} = \sigma\, E_k - \gamma\, I_k$$

$$\frac{dR_k}{dt} = \gamma\, I_k$$

The edge-to-infection probability is

$$\theta = \frac{1}{\langle k \rangle}\sum_k k\, P(k)\, I_k.$$

- Only **infectious** individuals transmit; **exposed individuals do NOT**.
  This is a modelling assumption (latent period = not yet infectious);
  $\theta$ above depends **only on $I_k$**, never on $E_k$.
- $\theta$ is a **dimensionless** edge-to-infection probability in $[0,1]$
  (→ 1 only when the whole population is infectious; → 0 pre-pandemic).
  The pre-pandemic (full-mobility) reference is carried by
  $s_i(\text{Jan-2020}) = 1$ and the per-state $\mu_0$ (§3–§4), **not** by
  $\theta$.

---

## 3. Inter-state coupling — temporal gravity metapopulation

The number of travellers moving from state $i$ to state $j$ per **day**:

$$W_{ij}(t) = s_i(t)\cdot \mu_{0,i}\cdot W_{ij}^{\text{gravity}}$$

where:

- $W_{ij}^{\text{gravity}}$ — gravity kernel from **total population and
  distance** (Viboud 2006):
  $$W_{ij}^{\text{gravity}} \propto \frac{\sqrt{N_i N_j}}{d_{ij}^{\alpha}}$$
  (row-normalised so that $\sum_j W_{ij}^{\text{gravity}} = 1$).
- $s_i(t)$ — **per-state BTS-calibrated monthly scale factor**,
  $s_i(\text{Jan-2020}) = 1$.
- $\mu_{0,i}$ — **per-state pre-pandemic baseline mobility** (each state
  has its own constant; the "missing" factor that turned into one number
  before this split).
- **Previously $s_i(t)$ was used directly as $\mu_i(t)$; now the two are
  separated into two multipliers: $\mu_{0,i}$ (per-state constant) ×
  $s_i(t)$ (per-state time curve).**
- **Directional:** in general $W_{ij}(t) \ne W_{ji}(t)$.  The gravity kernel
  itself is symmetric ($W_{ij}^{\text{gravity}} = W_{ji}^{\text{gravity}}$),
  but the full weight multiplies in the **origin state's** $\mu_{0,i}$ and
  $s_i(t)$, and the observed flows are asymmetric (e.g. Jan-2020 TX→CA
  721,960 vs CA→TX 705,442).  Real BTS data: 889 of 1,485 state pairs are
  asymmetric.  Symmetrising (e.g. min / mean of the pair) is a **modelling
  choice**, not automatic.

Traveller composition: the E/S/I/R proportions leaving a state are
**random-linear** with the state's own frequencies, i.e. an outgoing
traveller is drawn as S, E, I or R with probability $S/N$, $E/N$, $I/N$,
$R/N$ (random-flux / proportional-flow assumption).

A traveller of degree $k$ in state A keeps degree $k$ in state B
("well-connected people stay well-connected").

### Coupled ODEs (state $i$, degree class $k$; non-symmetric two-term form)

$$\frac{dS_{ik}}{dt} = -\beta\, k\, S_{ik}\, \theta_i
+ \sum_j \Big[\, W_{ji}(t)\,\frac{S_{jk}}{N_j}
         - W_{ij}(t)\,\frac{S_{ik}}{N_i}\, \Big]$$

$$\frac{dE_{ik}}{dt} = \beta\, k\, S_{ik}\, \theta_i - \sigma\, E_{ik}
+ \sum_j \Big[\, W_{ji}(t)\,\frac{E_{jk}}{N_j}
         - W_{ij}(t)\,\frac{E_{ik}}{N_i}\, \Big]$$

$$\frac{dI_{ik}}{dt} = \sigma\, E_{ik} - \gamma\, I_{ik}
+ \sum_j \Big[\, W_{ji}(t)\,\frac{I_{jk}}{N_j}
         - W_{ij}(t)\,\frac{I_{ik}}{N_i}\, \Big]$$

$$\frac{dR_{ik}}{dt} = \gamma\, I_{ik}
+ \sum_j \Big[\, W_{ji}(t)\,\frac{R_{jk}}{N_j}
         - W_{ij}(t)\,\frac{R_{ik}}{N_i}\, \Big]$$

- **Inflow uses $W_{ji}$ (j→k), outflow uses $W_{ij}$ (k→j)** — the two
  directions are kept separate because $W$ is directional.
- If $W$ were symmetric and row-normalised, the two terms collapse to the
  compact diffusion form $\sum_j W_{kj}(f_j - f_k)$ (what
  `metapopulation.py:177,387` uses after symmetrising); with directional
  $W$ the two-term form is the correct one, and population conservation
  still holds under row normalisation.

---

## 4. Calibration of $\mu_0$ and $s_i(t)$ — "trust $s_i(t)$ × Gravity"

Decouple one state at a time, using the monthly state→state flows.

1. Build the gravity kernel from **total population and distance**:
   $W_{ij}^{\text{gravity}}$ (row-normalised, so $\sum_j W_{ij}^{\text{gravity}}=1$).
2. **Pre-pandemic month (Jan 2020):** set $s_i(\text{Jan-2020}) = 1$.  The
   per-state baseline mobility is the ratio of **total observed outflow** to
   the gravity row sum, expressed in **daily** units (the BTS flows $F_{ij}$
   are monthly; the model's time unit is one day, so divide by the
   month length $T_{\text{mo}} = 30.44$ d):

$$\mu_{0,i} = \frac{\sum_j F_{ij}(\text{Jan-2020})}{T_{\text{mo}}\sum_j W_{ij}^{\text{gravity}}}
           = \frac{\sum_j F_{ij}(\text{Jan-2020})}{T_{\text{mo}}},$$

i.e. each state's total Jan-2020 outbound passengers per day (equivalently:
the flow-weighted mean of the ratios $F_{ij}/W_{ij}^{\text{gravity}}$ with
weights $W_{ij}^{\text{gravity}}$; matches `bts_state_monthly.csv`
`outbound` ÷ 30.44).

3. **Later months:** with $\mu_{0,i}$ known, back out each state's scale
   evolution as its total outflow relative to its own baseline:

$$s_i(t) = \frac{\sum_j F_{ij}(t)}{\sum_j F_{ij}(\text{Jan-2020})}.$$

- **Self-consistent by construction:** the model's total outbound
  $\sum_j W_{ij}(t) = s_i(t)\,\mu_{0,i}$ exactly equals the observed total
  outbound **per day** in every month.
- $s_i(t)$ is the per-state monthly mobility relative to its own
  Jan-2020 baseline (dimensionless) — compatible with the per-state
  outbound ratios in `bts_state_monthly.csv` (the old `scale_out` carried
  the same number with $\mu_0$ folded in; **this model keeps the factors
  separate**).
- **Uniquely determined even for states with sparse traffic** (MT, DE):
  the total outflow is never zero where data exists.

**Units.** All rates are per day: $\beta,\sigma,\gamma$ in d⁻¹, $W_{ij}(t)$
in travellers/day, $F_{ij}(t)$ in travellers/month (convert with
$T_{\text{mo}}=30.44$ d when building $\mu_0$).

**Working hypothesis throughout:**
$$F_{ij}(t) \approx s_i(t)\cdot \mu_{0,i}\cdot W_{ij}^{\text{gravity}}.$$
We trust this structure and fit the best parameters (gravity exponent
$\alpha$, per-state $\mu_{0,i}$, per-month $s_i(t)$).

---

## 5. Notation

| Symbol | Meaning |
|--------|---------|
| $k$ | degree (number of contacts) |
| $P(k)$ | degree distribution (exponential-cutoff power law) |
| $\langle k \rangle$ | mean degree |
| $\theta$ | dimensionless edge-to-infection probability, $[0,1]$ (→1 iff whole pop. infectious) |
| $S_{ik}, E_{ik}, I_{ik}, R_{ik}$ | degree-class compartments, **counts** (per state $i$, class $k$) |
| $\beta$ | infection rate per contact (per day) |
| $\sigma$ | latent → infectious rate (per day) |
| $\gamma$ | recovery rate (per day) |
| $\xi$ | density-scaling exponent for ⟨k⟩ (dimensionless) |
| $N_i$ | total population of state $i$ |
| $d_{ij}$ | distance between states (or city hubs) |
| $W_{ij}^{\text{gravity}}$ | gravity kernel, pop × distance, row-normalised |
| $\mu_{0,i}$ | per-state baseline mobility (travellers/day) |
| $s_i(t)$ | per-state monthly mobility scale (dimensionless), $s_i(\text{Jan-2020})=1$ |
| $F_{ij}(t)$ | observed state→state passenger flow (travellers/month) |
| $T_{\text{mo}}$ | month length $= 30.44$ d (monthly→daily conversion) |

---

## 6. Decisions (2026-08-05) & open points

**Decided:**

1. **Degree distribution:** per-state via a **density proxy** — scale each
   state's mean degree from its census population density,
   $\langle k\rangle_i = \langle k\rangle_0\,(\text{dens}_i /
   \text{dens}_0)^{\xi}$, keeping one global shape
   $P(k)\propto k^{-\alpha}e^{-k/k_c}$ and adjusting it to match
   $\langle k\rangle_i$ (fallback while estimating: one constant global
   distribution $\langle k\rangle = 6$, $\alpha = 3$, $k_c$ large, matching
   the BA ensemble of notebook 04).
2. **Class coupling:** inflow into degree class $k$ of a state comes from
   **all other states' travellers who also belong to class $k$** — i.e. the
   coupling acts in parallel on each class, and a person's degree is
   conserved when they move (no re-mixing across classes).
3. **Mobility decomposition:** $s(t)$ is **per state** ($s_i(t)$); $\mu_{0,i}$
   is a **per-state constant**. Both are solved with the residue trick (§4):
   fix $s=1$ in Jan-2020 ⇒ $\mu_{0,i} = \sum_j F_{ij}(\text{Jan-2020})/T_{\text{mo}}$
   (the state's total Jan-2020 outbound **per day**, since gravity is
   row-normalised); later months
   $s_i(t) = \sum_j F_{ij}(t)/\sum_j F_{ij}(\text{Jan-2020})$.
4. **Node set:** run on the **56 NYT/BTS jurisdictions** (50 states + DC +
   PR + GU + VI + MP + AS) **if information is sufficient — no
   extrapolation**: a jurisdiction without real data is **excluded**, never
   filled by extrapolation.
5. **$\mu_{0,i}$ aggregation:** the per-state baseline mobility is the
   **total-outflow / gravity-row-sum** ratio converted to daily units (with
   row-normalised gravity, the state's total Jan-2020 outbound ÷
   $T_{\text{mo}}=30.44$ d), giving a unique and self-consistent constant:
   $$\mu_{0,i} = \frac{\sum_j F_{ij}(\text{Jan-2020})}{T_{\text{mo}}\sum_j W_{ij}^{\text{gravity}}},$$
   and $s_i(t) = \sum_j F_{ij}(t)/\sum_j F_{ij}(\text{Jan-2020})$.
   This makes the model's total outbound from $i$ exactly match the
   observed total **per day** in every month (§4).

**Still open (resolve during implementation):**

- Density-scaling constants: $\langle k\rangle_0$, $\xi$ (or a better
  density→mean-degree link; optionally refit $\alpha$, $k_c$ per state).

**Closed (2026-08-05):**

- **Density data:** `data/state_land_area_2020.csv` now holds land area and
  population density for **all 56 jurisdictions** (2020 Census; land areas
  cross-checked against the census tables — AK 1,518,800 km², DC 158 km²,
  AS 198 km², MP 472 km², GU 543 km², VI 348 km², PR 8,868 km²).  Loaded via
  `data_loader.load_state_land_area_2020()`.  The density→⟨k⟩ scaling needs
  only the two constants ⟨k⟩_0, γ (calibrated later, not a data issue).
- **The 56-node list is feasible** — 2020 Census populations verified for
  all five territories: PR 3,285,874, GU 153,836, VI 87,146, MP 47,329,
  AS 49,710.  NYT and BTS both cover all 56 jurisdictions, so no
  extrapolation is needed.

---

## 7. Full understanding (restatement, 2026-08-05)

**Architecture.** 56 nodes (50 states + DC + PR/GU/VI/MP/AS, matching the
NYT/BTS jurisdiction set). Each node runs a **degree-based SEIR**
internally; nodes are coupled by a **time-varying gravity
metapopulation**. Core state variable: per state $i$, per degree class
$k$, compartments $(S_{ik}, E_{ik}, I_{ik}, R_{ik})$.

**1. Within-state dynamics (per state $i$, per class $k$):**

$$\frac{dS_{ik}}{dt} = -\beta\, k\, S_{ik}\, \theta_i,\qquad
\frac{dE_{ik}}{dt} = \beta\, k\, S_{ik}\, \theta_i - \sigma\, E_{ik},\qquad
\frac{dI_{ik}}{dt} = \sigma\, E_{ik} - \gamma\, I_{ik},\qquad
\frac{dR_{ik}}{dt} = \gamma\, I_{ik}$$

$$\theta_i = \frac{1}{\langle k \rangle_i}\sum_k k\, P_i(k)\, \frac{I_{ik}}{N_i}$$

(the $I_{ik}$ here are **counts**; the within-state form in §2 uses the
equivalent fractional notation where $I_k$ is a per-class fraction — both
give the same dimensionless $\theta \in [0,1]$.)

- **Degree distribution** $P_i(k)$: per-state shape via a **density proxy**
  — scale $\langle k\rangle_i$ with census population density,
  $\langle k\rangle_i = \langle k\rangle_0\,(\text{dens}_i /
  \text{dens}_0)^{\xi}$, around the exponential-cutoff power law
  $P(k)\propto k^{-\alpha}e^{-k/k_c}$ from notebook 04; while the scaling
  constants are unset, use one constant global distribution
  ($\langle k\rangle = 6$, $\alpha = 3$, $k_c$ large, the notebook-04 BA
  ensemble).
- **Only infectious individuals transmit; exposed do not.** $\theta_i$ is a
  dimensionless edge-to-infection probability in $[0,1]$ (→ 1 only when the
  whole population is infectious; → 0 pre-pandemic).  The pre-pandemic
  reference is carried by $s_i(\text{Jan-2020}) = 1$, not by $\theta$.

**2. Inter-state coupling (per pair $i\to j$):**

$$W_{ij}(t) = s_i(t)\cdot \mu_{0,i}\cdot W_{ij}^{\text{gravity}},\qquad
W_{ij}^{\text{gravity}} \propto \frac{\sqrt{N_i N_j}}{d_{ij}^{\alpha}}
\ \text{(row-normalised)}$$

- **$\mu_{0,i}$**: per-state constant (pre-pandemic baseline mobility).
- **$s_i(t)$: per state** (not one national curve) — a separate multiplier
  from $\mu_{0,i}$ (previously $s(t)$ was used directly as $\mu(t)$).
- Outbound travellers are drawn **linearly at random** with the origin
  state's own compartment frequencies: $\Pr(\text{S,E,I,R}) = S/N, E/N,
  I/N, R/N$ (random-flux / proportional-flow).
- **Degree is conserved** per person: a degree-$k$ traveller from state $A$
  stays degree $k$ in state $B$.  Coupling acts **in parallel per class**:
  the inflow into class $k$ of a state comes from all other states'
  travellers who are also class $k$ (no re-mixing across classes).

**3. Coupled ODEs (per state $i$, per degree class $k$; non-symmetric
two-term mobility term, population-conserving):**

$$\frac{dI_{ik}}{dt} = \sigma E_{ik} - \gamma I_{ik}
+ \sum_j \Big[\, W_{ji}(t)\,\frac{I_{jk}}{N_j}
         - W_{ij}(t)\,\frac{I_{ik}}{N_i}\, \Big]$$

(S/E/R analogous.) **Inflow uses $W_{ji}$ (j→k), outflow uses $W_{ij}$
(k→j)** — kept separate because $W$ is directional; they collapse to the
diffusion form $\sum_j W_{kj}(f_j - f_k)$ only for symmetric, row-normalised
$W$ (as in `metapopulation.py:177,387`). Denominators are **state**
populations $N_j$ because outbound travellers are drawn with state-level
SEIR/N frequencies; inflow and outflow only touch the same degree class.

**4. Calibration — "trust $s_i(t)\times$ Gravity" (residue method).** Decouple
state by state, month by month.  Fix the pre-pandemic month $s=1$; the
per-state baseline mobility is the total-outflow / gravity-row-sum ratio in
**daily** units (equal to the total Jan-2020 outbound ÷ $T_{\text{mo}}$,
gravity being row-normalised):

$$\mu_{0,i} = \frac{\sum_j F_{ij}(\text{Jan-2020})}{T_{\text{mo}}\sum_j W_{ij}^{\text{gravity}}}
           = \frac{\sum_j F_{ij}(\text{Jan-2020})}{T_{\text{mo}}},$$

and each later month's scale is that state's total outflow relative to its
own baseline:

$$s_i(t) = \frac{\sum_j F_{ij}(t)}{\sum_j F_{ij}(\text{Jan-2020})}.$$

This makes the model's total outbound from each state exactly match the
observed total in every month (consistent with the per-state outbound
ratios in `bts_state_monthly.csv`).  Working hypothesis:
$F_{ij}(t) \approx s_i(t)\,\mu_{0,i}\,W_{ij}^{\text{gravity}}$; we only fit
the best parameters ($\alpha$, per-state $\mu_{0,i}$, per-month $s_i(t)$).

**5. Data requirements.** Real census populations for all 56 jurisdictions
(51 are already in `metapopulation.py`; census data for PR/GU/VI/MP/AS
must be added). **No extrapolation** — jurisdictions without real data are
excluded, never filled in. NYT and BTS both cover 56 jurisdictions and can
be aligned by abbreviation mapping (confirmed).