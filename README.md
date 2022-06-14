# Multi-Objective Evolutionary Neural Architecture Search Framework

## Requirements

- PyTorch, torchvision, numpy, scipy, ...
- [timm](https://github.com/rwightman/pytorch-image-models) (pip install git+https://github.com/rwightman/pytorch-image-models.git)
- [pymoo](https://github.com/anyoptimization/pymoo)
- [ofa](https://github.com/mit-han-lab/once-for-all) (pip install ofa)
- [cutmix](https://github.com/ildoonet/cutmix) (pip install git+https://github.com/ildoonet/cutmix)
- [torchprofile](https://github.com/zhijian-liu/torchprofile) (for FLOPs calculation)
- [pySOT](https://github.com/dme65/pySOT) (for RBF surrogate model) 
- [pydacefit](https://github.com/msu-coinlab/pydacefit) (for GP surrogate model) 

## Code Structure

- ``data_providers`` implements all the dataset, dataloader functions. 
- ``evaluation`` implements methods to inference an architecture or supernet on data.
- ``search``:
  - ``search_spaces`` implements different types of search spaces, e.g., ``ofa_search_space.py``. 
  - ``evaluators`` implements methods to assess the performance of candidate architecture, e.g., [accuracy](https://github.com/mikelzc1990/neural-architecture-transfer/blob/4eeaebbb968bc217f37f38e799e8c930b6dfedb0/search/evaluators/ofa_evaluator.py#L102), [#params](https://github.com/mikelzc1990/neural-architecture-transfer/blob/4eeaebbb968bc217f37f38e799e8c930b6dfedb0/search/evaluators/ofa_evaluator.py#L37), [#FLOPs](https://github.com/mikelzc1990/neural-architecture-transfer/blob/4eeaebbb968bc217f37f38e799e8c930b6dfedb0/search/evaluators/ofa_evaluator.py#L41), and [latency](https://github.com/mikelzc1990/neural-architecture-transfer/blob/4eeaebbb968bc217f37f38e799e8c930b6dfedb0/search/evaluators/ofa_evaluator.py#L46). 
  - ``algorithms`` implements a generic multi-objective evolutionary NAS framework (``evo_nas.py``) and NSGANetV2 (``msunas.py``).
  - ``surrogate_models`` implements all surrogate models to speed up the search. 
- ``train`` implements methods to train/fine-tune architectures and supernet

P.S. Almost all main function method scripts have a standalone example in ``__main__``, simply run it to see what's going on. 

### Food-101
- Download the dataset from http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz 
- Process the data using this script https://github.com/human-analysis/nsganetv2/blob/main/data_providers/datasets/prepare_food101.py
- Pretrained the supernet
```python
python warm_up_supernet.py --dataset food --data /path/to/food-101 --train-batch-size 128 --test-batch-size 200 --valid-size 5000 --phase 1 --save $Exp_dir 
python warm_up_supernet.py --dataset food --data /path/to/food-101 --train-batch-size 128 --test-batch-size 200 --valid-size 5000 --phase 2 --save $Exp_dir 
python warm_up_supernet.py --dataset food --data /path/to/food-101 --train-batch-size 128 --test-batch-size 200 --valid-size 5000 --phase 3 --save $Exp_dir 
```
- Kick-off MsuNAS
```python
python run_msunas.py --dataset food --data /path/to/food-101 --train-batch-size 128 --test-batch-size 200 --valid-size 5000 --save $Exp_dir 
```

### Flowers102

- Download the dataset from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
- Process the data using this script https://github.com/human-analysis/nsganetv2/blob/main/data_providers/datasets/prepare_flowers102.py
- Pretrained the supernet
```python
python warm_up_supernet.py --dataset flowers --data /path/to/flowers102 --train-batch-size 32 --test-batch-size 200 --valid-size 2000 --phase 1 --save $Exp_dir 
python warm_up_supernet.py --dataset flowers --data /path/to/flowers102 --train-batch-size 32 --test-batch-size 200 --valid-size 2000 --phase 2 --save $Exp_dir 
python warm_up_supernet.py --dataset flowers --data /path/to/flowers102 --train-batch-size 32 --test-batch-size 200 --valid-size 2000 --phase 3 --save $Exp_dir 
```
- Kick-off MsuNAS
```python
python run_msunas.py --dataset flowers --data /path/to/flowers102 --train-batch-size 32 --test-batch-size 200 --valid-size 2000 --save $Exp_dir 
```

### Three or more objectives

- For a quick demo on ImageNet, just run ``search/algorithms/msunas.py``, see [here](https://github.com/mikelzc1990/neural-architecture-transfer/blob/77750b8e6e5a6bf1e677701b8ceb791fe7abf437/search/algorithms/msunas.py#L338).

## NSGANetV2 Pipeline

Below I briefly walk you through the main steps of NSGANetV2 for a bi-objective search on CIFAR-10

### Step 1: Prepare Supernet
Assuming you have the ImageNet pretrained supernets from OFA, and you have put the weights (both ``ofa_mbv3_d234_e346_k357_w1.0`` and ``ofa_mbv3_d234_e346_k357_w1.2``) in ``pretrained/backbone/ofa_imagenet/``. 

We now need to first train the full capacity supernets, which serve as *teacher* to guide the subpart training during the supernet adaptation step later. Follow these steps, where (1) and (2) can be executed in parallel, while (3) requires both (1) and (2) to be finished. 

``$Exp_dir = /path/to/save/results``

```python
# for width multiplier 1.0 supernet
python3 warm_up_supernet.py --dataset cifar10 --data /path/to/data --phase 1 --save $Exp_dir 
```
- Takes ~150 seconds per epoch for 100 epochs. You should find the checkpoint and log files stored in ``$Exp_dir/cifar10/supernet/ofa_mbv3_d4_e6_k7_w1.0/``

```python
# for width multiplier 1.2 supernet
python3 warm_up_supernet.py --dataset cifar10 --data /path/to/data --phase 2 --save $Exp_dir 
```
- Takes ~210 seconds per epoch for 100 epochs. You should find the checkpoint and log files stored in ``$Exp_dir/cifar10/supernet/ofa_mbv3_d4_e6_k7_w1.2/``

```python
# warm-up supernet by uniform sampling
python3 warm_up_supernet.py --dataset cifar10 --data /path/to/data --phase 3 --save $Exp_dir 
```
- Takes ~8 mins per epoch for 150 epochs. You should find the checkpoint and log files stored in ``$Exp_dir/cifar10/supernet/``

### Step 2: Main Loop

The following line execute MsuNAS:
```python
# warm-up supernet by uniform sampling
# for cifar10, make sure the cifar10.1 folder is present in /path/to/data
python3 run_msunas.py --dataset cifar10 --data /path/to/data --save $Exp_dir 
```
- Takes ~25-28 mins for generation for searching (aux single-level problem + subset selection + re-evaluate archive) + 1 hour per generation for supernet adaptation. 
