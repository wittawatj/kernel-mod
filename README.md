# kmod

[![Build Status](https://travis-ci.com/wittawatj/kmod.svg?token=yWUaYGwontVUwf9G8fLY&branch=master)](https://travis-ci.com/wittawatj/kmod)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/wittawatj/kmod/blob/master/LICENSE)

This repository contains a Python 3.6 implementation of the nonparametric
linear-time relative goodness-of-fit tests described in [our paper](https://arxiv.org/abs/1810.11630)

    Informative Features for Model Comparison
    Wittawat Jitkrittum, Heishiro Kanagawa, Patsorn Sangkloy, James Hays, Bernhard Schölkopf, Arthur Gretton
    NIPS 2018
    https://arxiv.org/abs/1810.11630

## How to install?

The package can be installed with the `pip` command.

    pip install git+https://github.com/wittawatj/kernel-mod.git

Once installed, you should be able to do `import kmod` without any error.


## Dependency

`autograd`, `matplotlib`, `numpy`, `scipy`, and the following two packages.

* The `kgof` package. This can be obtained from [its git
  repository](https://github.com/wittawatj/kernel-gof).

* The `freqopttest` (containing the UME two-sample test) package
  from  [its git repository](https://github.com/wittawatj/interpretable-test).

 In Python, make sure you can `import freqopttest` and `import kgof` without
   any error.


## Demo

To get started, check
[demo_kmod.ipynb](https://github.com/wittawatj/kernel-mod/blob/master/ipynb/demo_kmod.ipynb).
This is a Jupyter notebook which will guide you through from the beginning.
There are many Jupyter notebooks in `ipynb` folder demonstrating other
implemented tests. Be sure to check them if you would like to explore.


## Reproduce experimental results


### Experiments on test powers

All experiments which involve test powers can be found in
`kmod/ex/ex1_vary_n.py`, `kmod/ex/ex2_prob_params.py`, and
`kmod/ex/ex3_real_images.py`. Each file is runnable with a command line
argument. For example in
`ex1_vary_n.py`, we aim to check the test power of each testing algorithm
as a function of the sample size `n`. The script `ex1_vary_n.py` takes a
dataset name as its argument. See `run_ex1.sh` which is a standalone Bash
script on how to execute  `ex1_power_vs_n.py`.

We used [independent-jobs](https://github.com/wittawatj/independent-jobs)
package to parallelize our experiments over a
[Slurm](http://slurm.schedmd.com/) cluster (the package is not needed if you
just need to use our developed tests). For example, for
`ex1_vary_n.py`, a job is created for each combination of 

    (dataset, test algorithm, n, trial)

If you do not use Slurm, you can change the line 

    engine = SlurmComputationEngine(batch_parameters)

to 

    engine = SerialComputationEngine()

which will instruct the computation engine to just use a normal for-loop on a
single machine (will take a lot of time). Other computation engines that you
use might be supported. Running simulation will
create a lot of result files (one for each tuple above) saved as Pickle. Also, the `independent-jobs`
package requires a scratch folder to save temporary files for communication
among computing nodes. Path to the folder containing the saved results can be specified in 
`kmod/config.py` by changing the value of `expr_results_path`:

    # Full path to the directory to store experimental results.
    'expr_results_path': '/full/path/to/where/you/want/to/save/results/',

The scratch folder needed by the `independent-jobs` package can be specified in the same file
by changing the value of `scratch_path`

    # Full path to the directory to store temporary files when running experiments
    'scratch_path': '/full/path/to/a/temporary/folder/',

To plot the results, see the experiment's corresponding Jupyter notebook in the
`ipynb/` folder. For example, for `ex1_vary_n.py` see
`ipynb/ex1_results.ipynb` to plot the results.

### Experiments on images

* Preprocessing scripts for `celeba` and `cifar10` data can be found under
  `preprocessing/`.  See the readme files in the sub-folders under `proprocessing/`.

* The CNN feature extractor (used to define the kernel) in our Mnist experiment
  is trained with `kmod/mnist/classify.py`.
 
* Many GAN variants we used (i.e., in experiment 5 in the main text and in the
  appendix) were trained using the code from
  [https://github.com/janesjanes/GAN_training_code](https://github.com/janesjanes/GAN_training_code).

* Trained GAN models (Pytorch 0.4.1) used in this work can be found at
  [http://ftp.tuebingen.mpg.de/pub/is/wittawat/kmod_share/](http://ftp.tuebingen.mpg.de/pub/is/wittawat/kmod_share/).
  The readme files in the sub-folders under `preprocessing/` will tell you how
  to download the model files, for the purpose of reproducing the results.

## Coding guideline

* Use `autograd.numpy` instead of `numpy`. Part of the code relies on
  `autograd` to do automatic differentiation. Also use `np.dot(X, Y)` instead
  of `X.dot(Y)`. `autograd` cannot differentiate the latter. Also, do not use
  `x += ...`.  Use `x = x + ..` instead.


---------------

If you have questions or comments about anything related to this work, please
do not hesitate to contact [Wittawat Jitkrittum](http://wittawat.com) and
    [Heishiro Kanagawa](heishirok@gatsby.ucl.ac.uk)

