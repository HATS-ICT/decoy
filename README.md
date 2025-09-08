# DECOY

🎯 **A high-fidelity CS:GO simulation environment for strategic multi-agent planning research.** DECOY transforms complex 3D tactical gameplay into efficient discretized simulations while preserving environmental realism. Using neural models trained on real tournament data, it enables researchers to study strategic decision-making without the computational overhead of low-level game mechanics. Perfect for advancing multi-agent AI research in competitive scenarios.

![DECOY Framework](framework_diagram.jpg)

## Quick Start

```bash
# Setup environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Explore the simulation
python inspector_demo.py
```

## Features

- **Discretized Strategic Planning**: High-level tactical decisions without low-level mechanics
- **Real Data Integration**: Neural models trained on professional CS:GO tournament data  
- **Efficient Simulation**: Computationally lightweight while maintaining environmental fidelity
- **Research Ready**: Built for multi-agent planning and behavior generation research

## Roadmap

- [ ] MARL training examples
- [ ] Environment customization tools
- [ ] Interactive waypoint visualizer


# Citation

```bib
@inproceedings{wang2025csgo,
  author    = {Yunzhe Wang and Volkan Ustun and Chris McGroarty},
  title     = {A data-driven discretized {CS:GO} simulation environment to facilitate strategic multi-agent planning research},
  booktitle = {Proceedings of the 2025 Winter Simulation Conference (WSC)},
  year      = {2025},
  address   = {Los Angeles, CA, USA},
  publisher = {IEEE},
}
```