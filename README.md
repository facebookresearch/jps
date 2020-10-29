## Overview

Code for "Joint Policy Search for Collaborative Multi-agent Imperfect Information Games". Arxiv [link](https://arxiv.org/abs/2008.06495). The paper is published in NeurIPS 2020.

The project aims to find better equilibrium in multi-agent collaborative games with imperfect information by improving the policies of multiple agents simultaneously. This helps escape local equilibrium where unlaterial improvement of one player's policy is not helpful. To achieve that, we developed a novel value decomposition technique that decomposes the expected value changes into information sets where the policy differs, and search over candidate information sets via depth-first search. Each update can be proven not to degrade the performance in the tabular cases.

We open source the code to reproduce our results in simple games (see Def. 1-3 in the paper). The dataset and pre-trained Bridge model will be released later.

```
@inproceedings{tian2020jps,
    title={Joint Policy Search for Multi-agent Collaboration with Imperfect Information},
    author={Yuandong Tian and Qucheng Gong and Tina Jiang},
    booktitle={NeurIPS},
    year={2020}
}
```

## Requirements

Compiled with Linux and GCC 7.4
Pytorch 1.5+ (libtorch)

## Compilation

First initialize all submodules:

```
git submodule update --init --recursive
```

Then go to `simple_game`, and do the following to build 
```
mkdir build
cd build
cmake .. 
make
``` 
The executable `jps` is in `./build`.


## Examples 
To start, in the `build` directory, run the following to get CFR1k+JPS solution for Mini-Hanabi. Log [here](./simple_game/log/log2.txt): 
```
./jps --game comm2 --iter 100 --iter_cfr 1000
```
You might run with `--num_samples 1` to get the results for sampled-based version. E.g., run the following to get results of 100 trials:
```

for i in `seq 1 100`; do ./jps --game comm2 --iter 100 --iter_cfr 1000 --seed $i --num_samples 1; done > aa.txt
grep "CFRPure" aa.txt
```

Another example: Simple Bidding (N=16, d=3). Log [here](./simple_game/log/log1.txt).
```
./jps --game simplebidding --seed 1 --iter 100 --N_minibridge 16 --iter_cfr 1000 --max_depth 3
```

There are a few tabular imperfect information collaborative games implemented:
+ `comm`: Simple Communication Game (Def. 1 in the paper)
+ `simplebidding`: Simple Bidding (Def. 2 in the paper) 
+ `2suitedbridge`: 2-Suit Mini-Bridge (Def. 3 in the paper)
+ `comm2`: Mini-Hanabi introduced in [BAD paper](https://arxiv.org/abs/1811.01458).

Result
![Simple Game Result](./imgs/tabular.png)

Results of sample-based approach
![Sampled-based Result](./imgs/tabular_sampled.png)

## Visualize policy (for simple bidding)
In `build` folder, do:

```
./jps --game=simplebidding --N_minibridge=4 --seed=2 > aa.txt
python ../load_strategies.py --log aa.txt
```

The output is:
```
Optimal policies
score: 2.1875
        0       1        2        3
0  10 (0)  10 (1)  120 (2)  120 (2)
1  20 (0)  20 (2)   20 (2)  230 (4)
2  20 (2)  20 (2)   20 (2)  230 (4)
3  30 (0)  30 (4)   30 (4)   30 (4)
```
Note that "120" means P1 first bids 1, P2 then bids 2 and P1 bids 0 (Pass). The final contract is 2^{2-1} = 2, if card1 + card2 >= 2, then both of the players get reward 2 (shown in the parentheses), otherwise 0.

## Contribution
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
JPS is under CC-BY-NC 4.0 license, as found in the LICENSE file.
