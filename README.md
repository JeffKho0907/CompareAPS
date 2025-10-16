# ARC (Adaptive and Reliable Classification)

This package provides some statistical wrappers for machine learning classification tools in order to construct prediction sets for the label of a new test point with provably valid marginal coverage and approximate conditional coverage.

Accompanying paper (https://papers.nips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html):

    "Classification with Valid and Adaptive Coverage"
    Y. Romano, M. Sesia, E. Candès
    NeurIPS 2020 (spotlight).
    

## Contents

 - `arc/` Python package implementing our methods and some alternative benchmarks.
 - `third_party/` Third-party Python packages imported by our package.
 - `SYNTHETIC.py` Code for synthetic data experiment
 - `REAL.py` Code for real data experiment
 - `IMAGENET.py` Code for ImageNet experiment
  
## Third-party packages

This package builds upon the following non-standard Python packages provided in the "third-party" directory:

 - `nonconformist` https://github.com/donlnz/nonconformist
 - `cqr` https://github.com/yromano/cqr
 - `cqr-comparison` https://github.com/msesia/cqr-comparison
    
## Prerequisites

Prerequisites for the `arc` package:
 - numpy
 - scipy
 - sklearn
 - skgarden
 - torch
 - tqdm
 
Additional prerequisites for example notebooks:
 - pandas
 - matplotlib
 - seaborn

 


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.


This experiment uses the original code from the paper:

(https://papers.nips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html):

    "Classification with Valid and Adaptive Coverage"
    Y. Romano, M. Sesia, E. Candès
    NeurIPS 2020 (spotlight).

(https://doi.org/10.48550/arXiv.2009.14193):
    "Uncertainty Sets for Image Classifiers using Conformal Prediction"
    Anastasios N. Angelopoulos∗†, Stephen Bates∗
    , Jitendra Malik, & Michael I. Jordan
    ICLR 2021

and the code from the library torchcp(https://github.com/ml-stat-Sustech/TorchCP)


-This experiment compares empirical results of Adaptive Prediction Set, Regularized Adaptive Prediction Set and Sorted Adaptive Prediction set on synthetic tabular data, real tabular data(https://www.openml.org/search?type=data&status=active&id=42396) and ImageNet data. 

-Experiment on synthetic tabular data can be adjusted by users:
    -run: "python SYNTHETIC.py" to get results on K ∈ {10, 100, 1000}, p = 200
    -run: "python SYNTHETIC.py --k X --p Y" to get results on K = X and p = Y

-Experiment on real tabular data and ImageNet:
    -run: "python IMAGENET.py"
    -run: "python REAL.py"

