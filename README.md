# Dice: Dynamic Allocations for DNN Inference Tasks 
Dice is a token-based framework that introduces three dynamic allocation mechanisms for DNN inference workloads. 
Through theoretical analysis, we demonstrate that Dice guarantees diverse fairness and efficiency properties.
Additionally, we empirically analyze Dice by comparing it to state-of-the-art allocation mechanisms, showing that Dice outperforms the state of the art across various performance and efficiency metrics while providing provable fairness guarantees.

## Installation
Installation of Dice is too easy. At it is written in Python, you just need to create a virtual environment with python and install the dependencies:

```
git clone https://github.com/mhomidi/dice.git
cd dice
virtrualenv venv
source venv/bin/activate

# Do any change you want with sys_config_default.json

pip install -r requirement.txt
```

## Test
Evaluation scripts are written in `evals` directory. For running a test, you just need to run:
```zsh
python evals/run.py -p 1 -r 1 -d dd -q 0
```

You can get the help of the running with:
```zsh
python test.py [-p|--plot] <0|1> [-r|--should_run] <0|1> [-d|--deadline_type] <dd|wo_dd> [-q|--app_type_sub_id] <0..4> [-s|--sched] <scheduler>
```

* `scheduler`: specifies the type of scheduler. This can be one of these: `tant`,`ss`, `l_dice`, `s_dice`, `m_dice`.

* `should_run`: If you want to run the test it should be 1.

* `deadline_type`: It can be either `dd` when you have deadline or `wo_dd` when you do not have.

* `app_type_sub_id`: specifies the type of application you want to run.

* `sched`: specifies the type of scheduler you want to run. 

## Allocation Mechanims

### L-Dice
This is the lottery-based implementation of Dice.
In this mechanism, $m$ tokens are sequentially selected uniformly at random with replacement.
For every selected token, the associated user is then assigned to their most preferred node.
The allocated accelerator node is subsequently removed from all users' preference lists before moving on to the next winning token.

### S-Dice
In Dice's stride-based allocation mechanism, each user is assigned a *stride*, calculated by dividing a large number, $L$, by the number of tokens that the user holds, $s_i = L / t_i$ for all $i$.
Each user is also assigned a *pass*, initially set to zero, $p_i = 0$ for all $i$.
The mechanism selects a user with the minimum pass and assigns them to their most preferred node.
If multiple users share the same minimum pass, the one with the lowest "ID" is selected.
The pass of the chosen user is then advanced by their stride.
The allocated node is removed from all users' preference lists before selecting another user with the minimum pass.
This process continues until all $m$ accelerator nodes are allocated.

### M-Dice (Main Algorithm)
At each round $r$, Dice's market-based allocation mechanism determines the *competitive equilibrium* in a market where users utilize their tokens to purchase fractions of nodes, and the market establishes prices for each accelerator node.
A fractional allocation (A fractional allocation is an $n \times m$ matrix $x$, where $x_{i,j} \in [0, 1]$ is the fraction of node $j$ allocated to user $i$, and $\sum_i x_{i,j} = 1$ for all $j$)
$x^r$ is considered a competitive-equilibrium allocation if there exists a price vector $p = (p_1, \dots, p_m)$, defining the price of each accelerator node, such that $x^r_i$ represents the maximum allocation that each user $i$ can purchase with their tokens under prices $p$:

$x^r_i \in argmax_{x_i \in [0, 1]^m} ~~ u^r_i(x_i) ~~ \text{ s.t. } ~~ \sum_j x_{i,j}p_j \le t_i.$

A competitive-equilibrium allocation $x^r$ always exists for each round $r$, and it can be computed via the solution to the following convex program:

$Max. \sum_{i=1} t_i \log(u^r_i(x_i)),\quad s.t.\quad \sum_i x_{i,j} \le 1 , \quad \forall j \in 1, \dots, m.$

### Game-theoretic Properties
* L-Dice satisfies ex-ante Proportionality. *
* S-Dice satisfies Envy-freeness up-to-one and Proportionality up to one. *
* M-Dice is ex-ante Envy-freeness up-to-one and ex-ante Proportionality. It also satisfies per-round ex-ante Pareto Efficiency. *

\* Definitions and proofs are explained in the paper which is not publicly available until acceptance
