# From SIR to Network Epidemics

**Understanding Disease Spread Through Mathematical Modelling**

A computational project exploring epidemic modeling from basic SIR to agent-based models.

---

## Project Overview

This project progressively builds understanding of epidemic modelling through increasingly sophisticated models:

1. **Classical SIR Model** — Basic compartmental model with ODEs
2. **SEIR Model** — Adding latency/incubation period
3. **Network SIR** — Disease spread on contact networks
4. **Degree-Based SIR** — Heterogeneous connectivity
5. **Agent-Based Models** — Individual spatial simulations
6. **COVID-19 Case Study** — Real-world application (NYT data)
7. **Metapopulation Model** — Degree-based SEIR coupled across US states with
   time-varying BTS air-mobility via a calibrated gravity kernel

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
# Clone or download the project
cd Finial_Project

# Install dependencies (use virtual environment recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

---

## Project Structure

```
Finial_Project/
├── notebooks/           # Jupyter notebooks (educational sequence)
│   ├── 00_introduction.ipynb
│   ├── 01_SIR.ipynb
│   ├── 02_SEIR.ipynb
│   ├── 03_network_sir.ipynb
│   ├── 04_degree_based_sir.ipynb
│   ├── 05_abm.ipynb
│   ├── 06_covid_case_study.ipynb     # rewritten 2026-08-06: layperson walkthrough
│   │                              #   of the final SEIRV model (build_nb06.py)
│   ├── 07_data_analysis.ipynb        # data-only panels (no fitting)
│   ├── 08_metapopulation.ipynb       # rebuilt 2026-08-06 (build_nb08.py)
│   └── 09_conclusion.ipynb            # final synthesis → our COVID model
│   ├── appendix_A.ipynb              # gravity data-reconstruction report (build_appendix_a.py)
│   └── appendix_B.ipynb              # Omicron wave-5 run (build_appendix_b.py)
│
├── src/                # Python modules
│   ├── sir.py         # SIR model implementation
│   ├── seir.py        # SEIR model implementation
│   ├── network_sir.py # Network epidemic models
│   ├── degree_sir.py  # Degree-based models
│   ├── abm.py         # Agent-based model
│   ├── data_loader.py # NYT COVID + BTS/TSA aviation data loaders
│   ├── metapopulation.py          # legacy 51-state gravity SIR (notebook 07)
│   ├── fused_metapopulation.py   # active 56-jurisdiction degree-SEIR x mobility (08)
│   ├── visualisation.py  # Plotting utilities
│   ├── animation.py   # Animation tools
│   └── utils.py       # Helper functions
│
├── scripts/            # Reproducible notebook builders + validation
│   ├── build_nb06.py  # regenerates notebooks/06_covid_case_study.ipynb
│   ├── build_nb08.py  # regenerates notebooks/08_metapopulation.ipynb
│   ├── build_appendix_a.py  # regenerates notebooks/appendix_A.ipynb
│   ├── build_appendix_b.py  # regenerates notebooks/appendix_B.ipynb
│   ├── gravity_validation.py
│   ├── gravity_corr_states.csv
│   └── appendixA_state_corr.csv
│
├── assets/            # Figures (fig29–fig73, dpi=300)
│   ├── figures/      # (unused; notebooks write to assets/ root)
│   ├── gifs/
│   └── videos/
│
├── data/              # real data (gitignored: NYT COVID, BTS T-100, TSA daily)
├── requirements.txt
└── AGENTS.md
```

---

## Quick Start

### Run a Basic SIR Simulation

```python
import sys
sys.path.append('src')

from sir import simulate_sir, plot_sir

# Parameters
beta = 0.5    # transmission rate
gamma = 0.1   # recovery rate
S0, I0, R0 = 990, 10, 0  # initial conditions

# Simulate
solution = simulate_sir(beta, gamma, S0, I0, R0, t_span=(0, 160))

# Plot
plot_sir(solution, beta=beta, gamma=gamma, show_peak=True)
```

### Network Epidemic

```python
from network_sir import NetworkSIR, generate_scale_free

# Create network
graph = generate_scale_free(n=200, m=3)

# Initialize epidemic
sim = NetworkSIR(graph, beta=0.3, gamma=0.1, initial_infected=5)

# Simulate
history = sim.simulate(max_steps=100)

# Visualize
sim.plot_network()
```

### Agent-Based Model

```python
from abm import ABMSimulator

# Create simulation
sim = ABMSimulator(
    N=500,
    beta=0.3,
    gamma=0.1,
    speed=0.5,
    contact_radius=0.5,
    domain_size=10.0,
    initial_infected=5
)

# Run
history = sim.simulate(max_steps=150)

# Visualize
sim.plot_positions(show_contacts=True)
```

---

## Key Concepts

### SIR Model

The fundamental compartmental model:
- **S**: Susceptible (can catch the disease)
- **I**: Infected (has disease, can spread it)
- **R**: Recovered (immune)

**Equations:**
```
dS/dt = -β·S·I/N
dI/dt = β·S·I/N - γ·I
dR/dt = γ·I
```

**R₀ = β/γ** (basic reproduction number)

### SEIR Model

Adds latency period:
- **E**: Exposed (infected but not yet infectious)

**Flow:** S → E → I → R

### Network Models

Remove homogeneous mixing assumption:
- Disease spreads along edges of contact network
- Network structure affects epidemic dynamics
- Scale-free networks: hubs facilitate rapid spread

### Agent-Based Models

Individual-level simulation:
- Agents move in 2D space
- Infection through proximity
- Captures stochasticity and spatial effects

---

## Visualizations

The project generates:
- **58 figures in `assets/`** (currently fig29–fig86, dpi=300): epidemic
  dynamics, parameter sweeps, network structures, degree distributions,
  (notebook 08) gravity-kernel validation, per-state β estimates, national &
  state fits, spatial onset timing, mobility sensitivity, the β(t) NPI fit, and the SEIRV
  vaccination fit, plus (appendix A) the data-reconstruction figures fig68–fig74
- **3-5 animations** of spreading processes (abm.py/network_sir.py)
- Phase portraits, heatmaps, degree distributions

---

## Educational Philosophy

Each model arises from **limitations** of the previous:

```
Simple model
    ↓
Observe limitations
    ↓
New assumptions
    ↓
Richer model
    ↓
Repeat
```

The narrative builds understanding of **why** epidemic models evolved the way they did.

---

## Dependencies

Core packages:
- `numpy` — numerical computing
- `scipy` — ODE solvers
- `matplotlib` — plotting
- `networkx` — network analysis
- `pandas` — data handling

Optional:
- `ipywidgets` — interactive notebooks
- `tqdm` — progress bars

---

## Development Status

### Completed ✓
- [x] Core modules (SIR, SEIR, Network SIR, Degree-based, ABM)
- [x] Visualization utilities
- [x] Animation framework
- [x] Notebooks 00–06 (introduction → COVID-19 case study); 06 rewritten
      2026-08-06 as a layperson walkthrough of the final SEIRV model (fig82–86)
- [x] Notebook 07 conclusion **and** 07_data_analysis (data-only panels,
      2026-08-05: all SIR/SEIR fitting removed)
- [x] Notebook 08 metapopulation — **rebuilt 2026-08-06** with
      `src/fused_metapopulation.py` + `scripts/build_nb08.py`
      (56 jurisdictions, 2020-01→2023-03 BTS mobility, calibrated gravity
      kernel, per-state β with R₀≤4 cap, ~44k-ODE coupled degree-SEIR,
      §8.12 mobility sensitivity vs saturated per-pair flows,
      §8.13 time-varying β(t) from Google CMR / OxCGRT,
      §8.14 SEIRV vaccination from CDC data,
      §8.16 two-regime β(t) + fitted π₀ — R²(linear) +0.96)
- [x] Appendix B — Omicron (wave 5) run: immune-escape SEIRV grid
      + two-regime β(t) and fitted π₀ (R²(log) +0.74, peak ±6 days)
- [x] Appendix A — gravity data-reconstruction report
      (`notebooks/appendix_A.ipynb`, `scripts/build_appendix_a.py`):
      calibrated gravity model vs raw BTS T-100 flows, all validation tests
      of `scripts/gravity_validation.py` re-run on the PPML-calibrated kernel,
      plus the saturated per-pair mobility experiment (§A.9)
- [x] Gravity-kernel validation (`scripts/gravity_validation.py`)

### Estimated Completion
- Complete: all planned notebooks and modules build and execute
- Figures: 58 in `assets/` (fig29–fig86)

---

## Usage Notes

### Running Notebooks

Start from `00_introduction.ipynb` and proceed sequentially. Each notebook:
1. Motivates the need for a new model
2. Derives equations from first principles
3. Implements and simulates
4. Visualizes results
5. Identifies limitations

### Generating Animations

```python
from animation import animate_network_spread

# Requires completed simulation
anim = animate_network_spread(
    network_sim,
    filename='epidemic_spread',
    fps=5
)
```

Note: Animation export requires either `pillow` (GIF) or `ffmpeg` (MP4).

---

## Learning Objectives

After completing this project, you should understand:

**Conceptual:**
- Why mathematical models of disease exist
- How assumptions drive model complexity
- When to use each type of model

**Mathematical:**
- ODE formulation of compartmental models
- Mean-field approximations
- Network epidemic dynamics
- Degree distributions

**Computational:**
- Numerical ODE solving
- Network simulation
- Agent-based modeling
- Scientific visualization

---

## References

This project follows standard epidemic modeling theory:

1. Kermack & McKendrick (1927) — Original SIR model
2. Keeling & Rohani — "Modeling Infectious Diseases"
3. Pastor-Satorras & Vespignani — Network epidemiology
4. Ferguson et al. — COVID-19 Imperial College model

---

## Contributing

This is an educational project. Potential enhancements:

- [ ] Interactive parameter widgets
- [x] Real data fitting (COVID-19, notebooks 06 + 08)
- [ ] Vaccination models (SIRV)
- [ ] Age-structured models
- [x] Time-varying β (NPI) in the metapopulation (notebook 08 §8.13, Google
      CMR / OxCGRT drivers); next: two-regime or Rₜ-anchored β(t)
- [ ] GPU-accelerated ABM

---

## License

Educational project — free to use and modify.

---

## Contact

For questions about this implementation, see the notebooks (executable in
numbered order) and `src/` modules. The metapopulation entry point is
`notebooks/08_metapopulation.ipynb`, regenerable via `scripts/build_nb08.py`.

---

**Start exploring:** Open `notebooks/00_introduction.ipynb` →
