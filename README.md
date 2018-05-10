# nnpu_tf
Code for reproducing experiments on MNIST and CIFAR-10 in paper "Positive-Unlabeled Learning with Non-Negative Risk Estimator". 

# Requirements
- Python 3.6.1
- tensorflow 1.4
- matplotlib
- numpy
- sklearn

# Usage

- MNIST experiment: 
```python train.py mnist 1```

- CIFAR-10 experiment: `python train.py cifar10 1`

When they are done, a figure will show up. And results are saved in the folder `./result`, then you can see the figures if you run `python show_result.py mnist` or `python show_result.py cifar10`
