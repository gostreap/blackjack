# Blackjack

Blackjack Reinforcement Learning

## Setup

Let's start by installing the required modules. 

```
pip install -r requirements.txt
```

To use Deep Q-Learning, it is also necessary to install `pytorch`.

## Run

```
python src/temporal_difference.py
python src/sarsa.py
python src/q_learning.py
```

If `pytorch` is installed:

```
python src/dqn.py
```

The plot functions can be found in `plot.py` and `MC_sampling.py`.
