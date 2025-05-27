# SAC Pytorch C++

This is an implementation of the [Soft Actor-Critic algorithm](https://doi.org/10.48550/arXiv.1801.01290) for the C++ API of Pytorch. It uses a simple `TestEnvironment` to test the algorithm. Below is a small visualization of the environment, the algorithm is tested in. 
<br>
<figure>
  <p align="center"><img src="img/test_mode_1.gif" width="50%" height="50%" hspace="0"></p>
 <figcaption style="text-align:center;">Fig. 1: The agent in testing mode.</figcaption>
</figure>
<br><br>

## Build
You first need to install PyTorch. For a clean installation from Anaconda, checkout this short [tutorial](https://gist.github.com/mhubii/1c1049fb5043b8be262259efac4b89d5), or this [tutorial](https://pytorch.org/cppdocs/installing.html), to only install the binaries.

Do
```shell
mkdir build
cd build
cmake ..
make
```

## Run
Run the executable with
```shell
cd build
./train_sac
```
To plot the results, run
```shell
cd ..
python plot.py --online_view --csv_file data/data.csv --epochs 1 10
```
It should produce something like shown below.
<br>
<figure>
  <p align="center"><img src="img/epoch_1.gif" width="50%" height="50%" hspace="0"><img src="img/epoch_10.gif" width="50%" height="50%" hspace="0"></p>
  <figcaption>Fig. 2: From left to right, the agent for successive epochs in training mode as it takes actions in the environment to reach the goal. </figcaption>
</figure>
<br><br>

The algorithm can also be used in test mode, once trained. Therefore, run
```shell
cd build
./test_sac
```
To plot the results, run
```shell
cd ..
python plot.py --online_view --csv_file data/data_test.csv --epochs 1
```
## Visualization
The results are saved to `data/data.csv` and can be visualized by running `python plot.py`. Run
```shell
python plot.py --help
```
for help.

## Note
You may also refer to the implementations based on the PPO (Proximal Policy Optimization) and TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithms, which are available in the following repositories:

**PPO** : https://github.com/mhubii/ppo_libtorch
**TD3** : https://github.com/hrshl212/TD3-libtorch