# RL Implementation by Minjae (for my use and it is on development)

Welcome to Minjae's RL Implementation repository! This project is a sophisticated reinforcement learning (RL) engine that uniquely combines both offline and online RL capabilities, setting it apart from other implementations that treat them separately.

## Overview

This repository is designed to provide a comprehensive solution for reinforcement learning tasks, incorporating state-of-the-art techniques for both offline and online learning. Here's a detailed overview of what you'll find in this implementation:

### Offline Reinforcement Learning

The majority of the offline RL components have been meticulously implemented. Offline RL is crucial for scenarios where interacting with the environment is costly or impractical, allowing the agent to learn from pre-collected datasets.

### Online Reinforcement Learning

For online RL, we have integrated two powerful algorithms:

- **Trust Region Policy Optimization (TRPO)**
- **Proximal Policy Optimization (PPO)**

These algorithms enable efficient and stable learning by directly interacting with the environment, adjusting the policy based on immediate feedback.

### Meta-Reinforcement Learning

In addition to standard RL techniques, this repository includes a Meta-RL implementation, specifically a multi-task PPO. This approach leverages embeddings to handle multiple tasks simultaneously, enhancing the agent's ability to generalize across different environments and tasks.

## Features

- **Combined Offline and Online RL:** A unified engine that allows seamless transition between offline and online learning phases.
- **Advanced Algorithms:** Implementation of TRPO and PPO for robust online learning.
- **Meta-RL Capabilities:** Multi-task PPO using embeddings to manage and learn from diverse tasks effectively.

## Getting Started

To get started with this implementation, please follow the instructions below:

1. **Clone the Repository:**
    ```sh
    git clone https://github.com/minjae-rl
    cd minjae-rl
    ```
2. **Create virtual environment:**
    ```sh
    conda create -n rlkit python==3.10.*
    ```
3. **Install d4rl and metaworld:**
    ```sh
    pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
    pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld
    ```
    
4. **Install Dependencies:**
    ```sh
    pip install e .
    ```


## Contributing

Contributions are welcome! Please fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

This implementation builds on the work of many researchers and developers in the field of reinforcement learning. Special thanks to the authors of the original repositories that inspired this project.
[OfflineRL-Kit ](https://github.com/yihaosun1124/OfflineRL-Kit)

---

Feel free to explore the repository and use the provided tools to build and experiment with various RL models. If you encounter any issues or have questions, please open an issue or reach out via the contact information provided.

Happy Reinforcement Learning!

---

Minjae
