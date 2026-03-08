# Risk-Sensitive Derivative Hedging using Deep Learning + Reinforcement Learning

## Full Project Explanation (Beginner Friendly)

This document explains the entire project step-by-step, including:

* Finance concepts
* Mathematical models
* Reinforcement learning system design
* Deep learning architecture
* Data pipelines
* Code structure

The explanation assumes **no prior knowledge of quantitative finance**.

---

# 1. Project Goal

The project builds an AI system that learns how to **hedge an option**.

Traditional finance uses **Black–Scholes delta hedging**.

This project trains a **Reinforcement Learning (RL) agent** to learn a better hedging strategy.

Goal:

Minimize:

* hedging error
* transaction costs
* financial risk

---

# 2. Environment Setup

A Python virtual environment was created:

```
python -m venv venv
```

Purpose:

* isolate project dependencies
* avoid conflicts with other Python packages
* make the system reproducible

Activation (Windows):

```
venv\Scripts\activate
```

---

# 3. Dependencies Installed

Main libraries used:

| Library           | Purpose                            |
| ----------------- | ---------------------------------- |
| numpy             | numerical computation              |
| pandas            | dataset processing                 |
| scipy             | probability distributions          |
| torch             | deep learning framework            |
| gymnasium         | reinforcement learning environment |
| stable-baselines3 | RL algorithms                      |
| sb3-contrib       | recurrent RL algorithms            |
| yfinance          | financial data download            |
| matplotlib        | visualization                      |
| scikit-learn      | feature engineering                |

---

# 4. Reinforcement Learning Overview

Reinforcement Learning teaches an **agent** to make decisions through trial and error.

Structure:

```
Agent → Environment → Reward → Learning
```

Example:

Robot learning to walk by receiving rewards for balance.

In this project:

Agent = hedging model
Environment = financial market simulator

---

# 5. The Financial Problem

Banks sell **options** to clients.

Example call option:

```
Right to buy stock at $100 within 1 year
```

If the stock price rises above $100, the option buyer profits.

The bank selling the option may lose money.

Therefore the bank hedges risk by buying or selling shares.

---

# 6. What is Delta?

Delta measures how sensitive an option price is to stock price changes.

Formula:

```
Δ = ∂V / ∂S
```

Meaning:

If delta = 0.6

```
Stock increases $1 → Option increases $0.60
```

So the hedge rule becomes:

Hold **0.6 shares per option sold**.

---

# 7. Black–Scholes Model

The famous model for pricing options.

Call option price:

```
C = S·N(d1) − K·e^(−rT)·N(d2)
```

Where

S = stock price
K = strike price
T = time to maturity
r = interest rate
σ = volatility

Hidden variables:

```
d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
d2 = d1 − σ√T
```

Delta is:

```
Δ = N(d1)
```

This gives the **classical hedge ratio**.

---

# 8. Limitations of Black–Scholes

Black–Scholes assumes:

* continuous trading
* constant volatility
* zero transaction cost
* perfect market liquidity

Real markets violate these assumptions.

Therefore classical hedging is imperfect.

---

# 9. Idea of the Project

Instead of fixed formulas, train an **AI agent** that learns how to hedge.

Goal:

```
learn optimal hedge policy
```

using Reinforcement Learning.

---

# 10. RL Environment

The project implements a custom environment based on Gym.

Main functions:

```
reset()
step(action)
```

reset()

starts a new simulation episode.

step(action)

executes the agent's hedge decision and returns:

```
observation
reward
done
info
```

---

# 11. Episode Definition

An episode represents the lifetime of an option.

Example:

```
252 trading days (1 year).
```

Each day the agent decides how to hedge.

---

# 12. Synthetic Market Generation

Synthetic price data is generated using

**Geometric Brownian Motion (GBM)**.

Formula:

```
dS = μS dt + σS dW
```

Where

μ = drift (average growth)
σ = volatility
dW = random shock

Discrete simulation:

```
S(t+1) = S(t) * exp((μ − σ²/2)dt + σ√dt·Z)
```

Where

```
Z ~ Normal(0,1)
```

---

# 13. RL State Representation

The agent observes:

price
time remaining
option delta
option value
current hedge position

Example state:

```
[100.5, 0.96, 0.55, 5.2, 0.50]
```

---

# 14. RL Actions

Action = hedge ratio.

Example:

```
0.42
```

Meaning:

hold **0.42 shares per option**.

---

# 15. Reward Function

Reward encourages minimizing risk.

Example reward:

```
reward = −(hedging_error²) − transaction_cost
```

Where

```
hedging_error = portfolio_value − option_value
```

---

# 16. PPO Algorithm

The RL algorithm used is:

**Proximal Policy Optimization (PPO)**.

PPO improves policies gradually while preventing unstable updates.

Core objective:

```
L = min(r·A, clip(r,1−ε,1+ε)·A)
```

Where

r = probability ratio
A = advantage function

---

# 17. Neural Network Policy

The policy is a neural network built with PyTorch.

Architecture:

```
Input: state vector

state → dense layer → dense layer → output
```

Output:

hedge decision.

---

# 18. Synthetic Training

Script:

```
train.py
```

Pipeline:

```
GBM price simulation
→ RL environment
→ PPO training
→ policy update
```

Output model:

```
artifacts/ppo_hedging.zip
```

---

# 19. Historical Market Data

Real data downloaded using yfinance.

Example companies:

Vodafone
BP
HSBC

Saved to:

```
data/raw/lse_prices.csv
```

---

# 20. Feature Engineering

Script:

```
prepare_lse_data.py
```

Transforms raw prices into ML features.

Examples:

* log returns
* rolling volatility
* normalized price

Output:

```
data/processed/lse_features.csv
```

---

# 21. Recurrent RL Model

Historical training uses

**Recurrent PPO with LSTM**.

LSTM = Long Short-Term Memory neural network.

Advantage:

captures temporal patterns in financial data.

---

# 22. Historical Training

Script:

```
train_lse.py
```

Pipeline:

```
historical prices
→ feature dataset
→ RL environment
→ LSTM PPO training
```

Output model:

```
artifacts/recurrent_ppo_lse.zip
```

---

# 23. Visualization Dashboard

Frontend dashboard available.

Run:

```
python -m http.server 8000
```

Open:

```
http://localhost:8000/frontend/
```

Shows hedging performance and charts.

---

# 24. Full System Architecture

```
market data
→ feature engineering
→ RL environment
→ neural network policy
→ hedge decision
→ portfolio PnL
```

---

# 25. What the Model Learns

The AI learns:

* when to hedge aggressively
* when to hedge less
* how to reduce hedging risk
* how to adapt to volatility changes

---

# 26. Real-World Applications

This type of system is used in:

* quantitative hedge funds
* investment banks
* derivatives trading desks
* AI-driven risk management systems
