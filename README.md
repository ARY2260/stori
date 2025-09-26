# STORI: A Benchmark and Taxonomy for Stochastic Environments

[**TL;DR**] STORI (STOchastic-ataRI) is a benchmark that incorporates diverse stochastic effects and propose a taxonomy of stochasticity in RL environments.

<img width="1414" height="420" alt="image" src="https://github.com/user-attachments/assets/0901ceac-9d58-4a3c-9305-68af895d265a" />



Quick install to play any stochasticity mode using miniconda:
```
# git clone stori repository
cd stori
conda create -n stori-env python=3.10
conda activate stori-env
pip install -r requirements.txt
```

Play any game in interactive mode: (All modes used in Experiments are directly accessible)
<img width="440" height="255" alt="image" src="https://github.com/user-attachments/assets/dc127155-ed5f-4de6-b18d-3f2c5634ad90" />

```
python play_game.py
```

### Additional Note
 - DreamerV3: The source implementation and default parameters for Atari100K config used from this code repository (MIT license): https://github.com/NM512/dreamerv3-torch
 - STORM: The source implementation and default parameters (except eval num\_episode was set to 100) used from this code repository: https://github.com/weipu-zhang/STORM
