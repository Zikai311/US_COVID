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
6. **COVID-19 Case Study** — Real-world application

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
│   ├── 06_covid_case_study.ipynb
│   └── 07_conclusion.ipynb
│
├── src/                # Python modules
│   ├── sir.py         # SIR model implementation
│   ├── seir.py        # SEIR model implementation
│   ├── network_sir.py # Network epidemic models
│   ├── degree_sir.py  # Degree-based models
│   ├── abm.py         # Agent-based model
│   ├── visualisation.py  # Plotting utilities
│   ├── animation.py   # Animation tools
│   └── utils.py       # Helper functions
│
├── assets/            # Output directory
│   ├── figures/      # Generated plots
│   ├── gifs/         # Animations
│   └── videos/       # Video exports
│
├── requirements.txt   # Python dependencies
├── sepecification.md  # Project specification
└── implementation_plan.md  # Development plan
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
- **~28 figures** showing epidemic dynamics, parameter sweeps, network structures
- **3-5 animations** of spreading processes
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
- [x] Introduction notebook
- [x] SIR notebook (Part 1)
- [x] SEIR notebook (Part 2)

### In Progress
- [ ] Network SIR notebook (Part 3)
- [ ] Degree-based notebook (Part 4)
- [ ] ABM notebook (Part 5)
- [ ] COVID case study (Part 6)
- [ ] Conclusion notebook (Part 7)

### Estimated Completion
- **Code**: ~1,400 lines (target: 1,200-2,000)
- **Documentation**: ~4,000 words (target: 6,000-9,000)
- **Figures**: 10+ generated (target: 28+)

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
- [ ] Real data fitting (COVID-19)
- [ ] Vaccination models (SIRV)
- [ ] Age-structured models
- [ ] GPU-accelerated ABM

---

## License

Educational project — free to use and modify.

---

## Contact

For questions about this implementation, refer to the specification and implementation plan documents included in the project.

---

**Start exploring:** Open `notebooks/00_introduction.ipynb` →
