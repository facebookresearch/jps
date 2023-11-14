# Overview

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

# Bridge

## Requirements
Compiled with Linux and GCC 12.1
Cuda 10.1
PyTorch 1.7.0
- Note that 1.7.1+ do not work due to some conflict with pybind11. The current submodule `third_party/pybind11` is set to be at commit `a1b71df`.
- You may try using later version of pybind11 and newer version of PyTorch (not tested yet).

## Pre-trained models

| Model                          | Description                               |
|--------------------------------|-------------------------------------------|
| agent-baseline-15-189.pth      | Baseline model1                           |
| agent-baseline-0704-26-199.pth | Baseline model2                           |
| agent-2day-5-250.pth           | JPS models (0.38 IMPs/b against WBridge5) |
| agent-2day-8-169.pth           | JPS models (0.44 IMPs/b against WBridge5) |
| agent-1610-0.63.pth            | JPS models (0.63 IMPs/b against WBridge5) |


## Conda environment installation and compilation
```
conda create --name bridge --file requirement.txt
conda activate bridge
git submodule update --init --recursive
cd third_party/pybind11
git checkout a1b71df

cd ../../
cmake ..
make

cp ./*.so ./python
cp ./rela/*.so ./python
cd python
```

## Bridge Dataset
Please download the dataset [here](https://dl.fbaipublicfiles.com/bridge/bridge_dataset.tar.gz). The dataset contains: 
- The training set (`dda.db`) contains 2.5M situations, 
- The evaluation set (`test.db`) contains 50K situations.
- The VS WBridge5 set (`vs_wb5.db`) contains 1000 situations.

Please untar it to `./bridge_data/` folder.

To check the record, you can install sqlite3 and run the following SQLs:
- Run `select count(*) from records;` to check the number of records
- Run `select * from records limit 1;` to check the first entry. 

Here is one example:
```
$ sqlite3
SQLite version 3.31.1 2020-01-27 19:55:54
Enter ".help" for usage hints.
Connected to a transient in-memory database.
Use ".open FILENAME" to reopen on a persistent database.
sqlite> .open dda.db
sqlite> select * from records limit 1;
0|{"pbn": "[Deal \"N:KT9743.AQT43.J.7 J85.9.Q6.KQJ9532 Q2.KJ765.T98.T64 A6.82.AK75432.A8\"]", "ddt": [0, 12, 0, 12, 0, 12, 0, 12, 10, 3, 10, 3, 9, 4, 9, 4, 0, 8, 0, 8]}
```

Note that each entry has a pre-computed double dummy table. DDS table is in the format of C (NESW), D (NESW), H (NESW), S (NESW), NT (NESW). For example, the 1D ddt table above means that
|   | C  | D  | H  | S | NT |
|---|----|----|----|---|----|
| N | 0  | 0  | 10 | 9 | 0  |
| E | 12 | 12 | 3  | 4 | 8  |
| S | 0  | 0  | 10 | 9 | 0  |
| W | 12 | 12 | 3  | 4 | 8  |

## Run evaluation
To run the evaluation, use the following command to make pre-trained models play with each other. 

```
python main2.py num_thread=200 game=bridge env_actor_gen.params.gen_type=basic seed=1 \
  trainer=selfplay actor_gen.params.batchsize=1024 epoch_len=10 eval_only=true \
  method=a2c agent.params.load_model=`pwd`/../bridge_models/agent-1610-0.63.pth \
  baseline=a2c  baseline.agent.params.load_model=`pwd`/../bridge_models/agent-baseline-15-189.pth \
  game.params.feature_version=old/single \
  game.params.train_dataset=`pwd`/../bridge_data/dda.db \
  game.params.test_dataset=`pwd`/../bridge_data/test.db
```

Here is the example output:
```
[2023-11-13 15:45:14,526][main2.py][INFO] - [-1] Time spent = 23.77 s
-1:eval_score_p0 [50000]: avg:   0.0318 (±   0.0012), min:  -0.8750[25576], max:   0.9167[14711]
-1:eval_score_p1 [50000]: avg:  -0.0318 (±   0.0012), min:  -0.9167[14711], max:   0.8750[25576]
-1:eval_score_p2 [50000]: avg:   0.0318 (±   0.0012), min:  -0.8750[25576], max:   0.9167[14711]
-1:eval_score_p3 [50000]: avg:  -0.0318 (±   0.0012), min:  -0.9167[14711], max:   0.8750[25576]
```

Here `0.0318` means that the JPS model `agent-1610-0.63.pth` is 0.0318 * 24 = 0.7632 IMPs/b better than the baseline `agent-baseline-15-189.pth`, since the reward score is normalzed to [-1,1] from [-24,24] in IMPs/b scale. Note that the bridge game has 4 players and the evaluation dumps the scores for each player over 50000 games of the evaluation set. 

For the three JPS models, we use the `old` feature while for baselines we use the `single` feature (which is a newer type of feature). Use `game.params.feature_version=old/single` to specify the features used for each model. 

Another example is to compete with baseline16:
```
python main2.py num_thread=200 game=bridge env_actor_gen.params.gen_type=basic seed=1 \
  trainer=selfplay actor_gen.params.batchsize=1024 epoch_len=10 eval_only=true \
  method=a2c agent.params.load_model=`pwd`/../bridge_models/agent-1610-0.63.pth \
  baseline=baseline16 \ 
  game.params.feature_version=old \
  game.params.train_dataset=`pwd`/../bridge_data/dda.db \
  game.params.test_dataset=`pwd`/../bridge_data/test.db
```

Here is the example output:
```
[2023-11-13 16:17:39,622][main2.py][INFO] - [-1] Time spent = 15.74 s
-1:eval_score_p0 [50000]: avg:   0.1276 (±   0.0011), min:  -0.7083[39419], max:   0.9167[13675]
-1:eval_score_p1 [50000]: avg:  -0.1276 (±   0.0011), min:  -0.9167[13675], max:   0.7083[39419]
-1:eval_score_p2 [50000]: avg:   0.1276 (±   0.0011), min:  -0.7083[39419], max:   0.9167[13675]
-1:eval_score_p3 [50000]: avg:  -0.1276 (±   0.0011), min:  -0.9167[13675], max:   0.7083[39419]
```

## Train model with self-play
To train the model, run the following:
```
python main2.py num_thread=200 game=bridge env_actor_gen.params.gen_type=basic seed=1 \
    trainer=selfplay actor_gen.params.batchsize=1024 method=a2c \
    game.params.train_dataset=`pwd`/../bridge_data/dda.db \
    game.params.test_dataset=`pwd`/../bridge_data/test.db
```

# Logs of our pre-trained model
Please check log of [3day model](./logs/jps_3days.log) and [14day model](./logs/jps_14days.log) competed against WBridge5. Here is an explanation of the log entry:

```
dealer is 0 [Vulnerability None]                                                      # The dealer is 0, no Vulnerability
[Deal "N:AK.J982.Q986.Q65 JT97.QT5.A5.AKJ7 Q82.AK743.32.T83 6543.6.KJT74.942"]        # All 4 hands
parScore: -140

Seat ♠   ♥   ♦   ♣   HCP   Actual Hand
0    2   4   4   3   12   ♠AK ♥J982 ♦Q986 ♣Q65                                        
1    4   3   2   4   15   ♠JT97 ♥QT5 ♦A5 ♣AKJ7                                        
2    3   5   2   3   9    ♠Q82 ♥AK743 ♦32 ♣T83
3    4   1   5   3   4    ♠6543 ♥6 ♦KJT74 ♣942

# At table 0, JPS at Seat 0 and Seat 2 (two AI doesn't know each other's hands), and WBridge5 at Seat 1 and Seat 3
# At table 1, JPS at Seat 1 and Seat 3, and WBridge5 at Seat 0 and Seat 2
# Bids in parentheses are from WBridge5 (the opponent)

Table 0, dealer: 0  1H (1N) P (P) P                                                   # Bidding. Seat 0 -> 1 -> 2 -> 3. JPS bids 1H, WBridge5 bids 1N (contract) 
Table 1, dealer: 0  (1D) P (1H) P (2H) X (3H) P (P) P                                 # Bidding. WBridge5 bids 1D, JPS bids P, Wbridge5 bids 1H, then 2H, then 3H (contract)

Table 0, Trick taken by declarer: 5, rawNSSeatScore: 100                              # Result After DDS. 1N down 2, declarer (Wbridge5) loses 100 points and JPS won 100 points.  
Table 1, Trick taken by declarer: 8, rawNSSeatScore: -50                              # 3H down 1, declarer (WBridge5) loses 50 points and JPS won 50 points. 
Final reward 0.166667                                                                 # Convert the two table scores into normalized IMPs (= IMP / 24) 
```

To compute the overall performance given these logs, please run the following script:
```
./jps$ python compute_score.py --log_file ./logs/jps_3days.log
mean = 0.442, std = 0.1993448629244666

./jps$ python compute_score.py --log_file ./logs/jps_14days.log
mean = 0.628, std = 0.1944979317253668
```

Note that the original log had one bug that miscalculated the declarer (the declarer should be the first player calling for the strain of the final contract, rather than the last player who finalizes it). This affects the `Final reward` entry so a simple `grep "Final reward" [log file]` didn't give you the right answer. Instead we provide you with [compute_score.py](./compute_score.py) to compute the final score correctly, with the help of DDS table of the 1k games stored [here](./logs/against_WBridge5.raw). 


# Simple Game

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
