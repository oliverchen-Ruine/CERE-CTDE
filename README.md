# CERE-CTDE: Dual-Driven Optimization of Collaborative Multi-Agent via Case Learning and Curiosity

[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Framework for implementing CERE-CTDE**: A novel multi-agent reinforcement learning architecture integrating **Case-Based Reasoning (CBR)** with **Random Network Distillation (RND)** to address Exploration-Exploitation Trade-off in cooperative environments. Validated on StarCraft II (SMAC) and PressurePlate environments.

## Key Innovations
- **Combine design of Exploration-Oriented and Exploitation-Driven components in the dynamic tradeoff between exploration and exploitation**
- **CERE-CTDE improves the learning ability of Actor-Critic methods and Value-Decomposition methods**
- **Innovative measures for optimizing MAS learning efficiency**

## Environmental Support 
| Type          | Scenarios            | Verification Metrics               |
|---------------|----------------------|------------------------------------|
| StarCraft II  | SMAC v4.10 (27 maps) | Win Rate/Reward                    |
| PressurePlate | Static and dynamic 4p| Win Rate、Reward and Episode Length |


## Citation
When using this repository, please cite:
```bibtex
@ARTICLE{CHEN2025108083,
  title={Dual-Driven Optimization of Collaborative Multi-Agent via Case Learning and Curiosity},
  author={Ruizhu, Chen and Rong, Fei and Junhuai, Li and Aimin, Li and Yalin, Miao and Lili, Wu and Zhiming, Chen},
  journal = {Neural Networks},
  year={2025},
  issn = {0893-6080},
  doi = {https://doi.org/10.1016/j.neunet.2025.108083},
  url = {https://www.sciencedirect.com/science/article/pii/S0893608025009633}
}
```
## Reference
[1] [MARL-code-pytorch](https://github.com/Lizhi-sjtu/MARL-code-pytorch)<br />
[2] [EPyMARL](https://github.com/uoe-agents/epymarl)<br />
[3] [SMAC](https://github.com/oxwhirl/smac)<br />
[4] [PressurePlate](https://github.com/uoe-agents/pressureplate)<br />



