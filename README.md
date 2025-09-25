# STORI: A Benchmark and Taxonomy for Stochastic Environments

[**TL;DR**] STORI (STOchastic-ataRI) is a benchmark that incorporates diverse stochastic effects and propose a taxonomy of stochasticity in RL environments.

![STORI Overview](https://raw.githubusercontent.com/stori/stori/main/docs/stori_overview.png)


Quick install to play any stochasticity mode using miniconda:
```
# git clone stori repository
cd stori
conda create -n stori-env python=3.10
conda activate stori-env
pip install -r requirements.txt
```

Play any game in interactive mode: (All modes used in Experiments are directly accessible)
![game mode](https://raw.githubusercontent.com/stori/stori/main/docs/mode_selection.png)
```
python play_game.py
```
