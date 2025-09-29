# STORI: A Benchmark and Taxonomy for Stochastic Environments

[**TL;DR**] STORI (STOchastic-ataRI) is a benchmark that incorporates diverse stochastic effects and propose a taxonomy of stochasticity in RL environments.

<img width="1414" height="420" alt="image" src="https://github.com/user-attachments/assets/0901ceac-9d58-4a3c-9305-68af895d265a" />



### STORI supports the following Atari ALE games out of the box:

- **Breakout**
- **Boxing**
- **Gopher**
- **BankHeist**


## Quick install to play any stochasticity mode using miniconda:
```
# git clone stori repository
cd stori
conda create -n stori-env python=3.10
conda activate stori-env
pip install -r requirements.txt
pip install .
```

### Play any game in interactive mode: (All modes used in [Experiments](./Exp_configs.json) are directly accessible)

<img width="440" height="255" alt="image" src="https://github.com/user-attachments/assets/dc127155-ed5f-4de6-b18d-3f2c5634ad90" />

```
python play_game.py
```

### For usage details refer [Getting Started](./stori/docs/getting_started.md).

## Run STORI benchmarks on DreamerV3 and STORM

STORI provides ready-to-use configuration files and scripts to run DreamerV3 and STORM agents on all supported stochasticity types and modes.

### 1. Benchmarking with DreamerV3

- **Setup:**  
  Follow the installation instructions in the [DreamerV3 README](./dreamerv3-torch/README.md).

- **Running with STORI:**  
  Use the provided [Exp_configs.json](./Exp_configs.json) to select a stochasticity profile and update [run_dreamer.sh](./dreamerv3-torch/run_dreamer.sh) file.

  Example command:
  ```
  cd dreamerv3-torch
  bash run_dreamer.sh
  ```

- **Notes:**  
  - The source implementation and default parameters for Atari100K config used from the code repository (MIT license): https://github.com/NM512/dreamerv3-torch.
  - The source implementation was modified to support STORI, gymnasium and ale_py library.

### 2. Benchmarking with STORM

- **Setup:**  
  Follow the installation instructions in the [STORM README](./STORM/readme.md).

- **Running with STORI:**  
  Use the provided [Exp_configs.json](./Exp_configs.json) to select a stochasticity profile and update [run_storm.sh](./STORM/run_storm.sh) file.

  Example command:
  ```
  cd STORM
  bash run_storm.sh
  ```

- **Notes:**  
  - The source implementation and default parameters (except eval num\_episode was set to 100) used from the code repository: https://github.com/weipu-zhang/STORM
  - The source implementation was modified to support STORI library.

### 3. Customizing Stochasticity

- All available stochasticity types and modes are described in [Modes Directory](./stori/docs/modes_directory.md).
- To define your own stochasticity profile, edit or extend the configs from `Exp_configs.json` file.

### 4. Results and Reproducibility

- All experiments in the STORI paper can be reproduced using the provided configs and scripts using seeds 0, 7 and 13.
