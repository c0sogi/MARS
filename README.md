# MARS: Modular Agent with Reflective Search

Automating AI research differs from general software engineering due to computationally expensive evaluation (e.g., model training) and opaque performance attribution. Current LLM-based agents struggle here, often generating monolithic scripts that ignore execution costs and causal factors.

We introduce **MARS** (**M**odular **A**gent with **R**eflective **S**earch), a framework optimized for autonomous AI research. MARS relies on three pillars:
1. **Budget-Aware Planning** via cost-constrained Monte Carlo Tree Search (MCTS) to explicitly balance performance with execution expense.
2. **Modular Construction**, employing a "Design-Decompose-Implement" pipeline to manage complex research repositories.
3. **Comparative Reflective Memory**, which addresses credit assignment by analyzing solution differences to distill high-signal insights.

MARS achieves state-of-the-art performance among open-source frameworks on **MLE-Bench** under comparable settings, maintaining competitiveness with the global leaderboard's top methods. Furthermore, the system exhibits qualitative "Aha!" moments, where 63% of all utilized lessons originate from cross-branch transfer, demonstrating that the agent effectively generalizes insights across search paths.

---

## Method

MARS reformulates the research process as a search for an optimal software repository. The framework explicitly balances performance maximization with execution expense and manages architectural complexity through a modular pipeline.

![MARS Overview](./figures/MARS_teaser.png)

### Core Components

1.  **Resource-Aware Planning**: A Budget-Aware MCTS strategically navigates the search space by selecting actions from *Draft new architecture*, *Debug runtime errors*, and *Improve a valid solution*. It optimizes an efficiency-guided reward that explicitly balances performance maximization with the penalty of high execution costs.
2.  **Modular Decomposition**: To replace fragile monolithic scripting, the system employs a "Design-Decompose-Implement" pipeline. Specialized *Idea*, *Modular*, and *Coding* agents architect the solution into independent, testable modules. This structure enables precise **Diff-Based Refinement**, allowing the agent to update specific logic blocks without regenerating the entire codebase.
3.  **Reflective Memory**: This module distills raw execution logs into structured **Debugging** and **Solution Lessons** to proactively prevent error repetition and accelerate convergence in later iterations.

---

## "Aha!" Moments in Exploration

MARS is capable of experiencing "Aha!" moments during long-horizon exploration, successfully navigating complex optimization landscapes where baselines fail.

![Aha Moment](./figures/mars-aha-moment.png)

*Figure: The "Aha!" moment of MARS on the challenging iMet-2020-FGVC7 task. The visualization tracks validation performance gains triggered by specific strategic lessons. While existing methods fail to reach medal-level performance, MARS progressively refines its strategy -- evolving from a lightweight residual network to model ensemble techniques -- to ultimately achieve a silver medal.*

---

## Repository Structure

This repository contains the official release of MARS generated code and trajectories.

- `exp-logs/`: Contains the agent trajectories and generated code.
- `figures/`: Figures illustrating the method and results.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{mars2026,
  title={MARS: Modular Agent with Reflective Search for Automated AI Research},
  author={Chen, Jiefeng and Mishra, Bhavana Dalvi and Nam, Jaehyun and Meng, Rui and Pfister, Tomas and Yoon, Jinsung},
  journal={arXiv preprint arXiv:2602.02660},
  year={2026}
}
```

## License

[MIT License](LICENSE)
