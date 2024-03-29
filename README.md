# Probabilistic Rainfall Estimation from Automotive Lidar
This repository contain code to reproduce results in the paper 'Probabilistic Rainfall Estimation from Automotive Lidar' published at IV 2022.

Paper link: [Probabilistic Rainfall Estimation from Automotive Lidar](https://arxiv.org/abs/2104.11467)

Shared public data (incl. pretrained models): [Google Drive directory](https://drive.google.com/drive/folders/1obveNAUWUdvUs5IQtXilB7VBWb7myk0p?usp=sharing)


### Dependency
Python3
  - version: 3.6
  - packages: 
    - matplotlib
    - numpy
    - pybind11
    - pyyaml
    - scikit-learn
    - tqdm

![header_image](https://user-images.githubusercontent.com/34254153/170935996-a26a1096-b279-43e3-afb4-a6c1f28c62ab.png)

# TL;DR
1. Initialize
<pre><code>$ git clone https://github.com/tier4/rainfall_modeling_open.git
$ cd rainfall_modeling_open
$ git submodule init
$ git submodule update
$ export PYTHONPATH=.
</code></pre>

2. Build library
<pre><code>$ cd ./boost_mst_lib
$ g++ -O3 -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` boost_mst_lib.cpp boost_mst.cpp -o boost_mst_lib`python3-config --extension-suffix`
$ cd ../
</code></pre>

3. Initialize experiment directory structure
<pre><code>python init_exp_dirs.py
</code></pre>

4. Download data and rainfall table for experiment --> experiment directory structure

[Google Drive directory](https://drive.google.com/drive/folders/1obveNAUWUdvUs5IQtXilB7VBWb7myk0p?usp=sharing)

5. Convert raw training data --> dataset

<pre><code>$ python conv_training_data_to_dataset.py dataset_primary_exp.yaml
</code></pre>

6. Convert dataset --> regression problem

<pre><code>$ python conv_dataset_to_reg_problem.py primary_experiments/datasets/dataset_training.gz primary_experiments/datasets/dataset_validation.gz dataset_X_exp.yaml --nprocs 4
</code></pre>

7. Train model

<pre><code>$ python train_model.py primary_experiment/regression_problems/reg_problem_train_reg_problem.gz primary_experiment/regression_problems/reg_problem_val_reg_problem.gz .
</code></pre>

# How to use

Note: In filenames, substitute `X` with `primary` or `secondary` depending desired experiment.

### Initialize repository

Note that the repository contains one submodule that also needs to be initialized.

<pre><code>$ git clone https://github.com/tier4/rainfall_modeling_open.git
$ cd rainfall_modeling_open

$ git submodule init
$ git submodule update

$ export PYTHONPATH=.
</code></pre>

### Build library

The C++ library `boost_mst_lib` need to be built before being used for feature vector generation.

<pre><code>$ cd ./boost_mst_lib

$ g++ -O3 -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` boost_mst_lib.cpp boost_mst.cpp -o boost_mst_lib`python3-config --extension-suffix`

$ cd ../
</code></pre>


### Step 1: Initialize experiment directory structure

Run `python init_exp_dirs.py` to generate empty directory stucture.

1. Download raw pointcloud data (rosbags pointclouds converted into Numpy arrays).

Example: All compressed `.gz` files in `rainfall_modeling_public_data/X_experiment/raw_data/crop_10/*.gz`.

2. Download rainfall measurement data (CSV file).

Example: `rainfall_table.csv` in `rainfall_modeling_public_data/X_experiment/rainfall_table.csv`.

Optional: Possible to download files for all stages of the process incl. trained models, in case one wish to try only a certain part.


### Step 2: Convert raw training data to dataset

Configure the dataset configuration YAML file `dataset_X_exp.yaml` to change sampling duration (e.g. 100 vs 150 consecutive examples, 10 samples = 1 sec) and crop size.

The default configuration represent the best 'all data' model with sampling duration 15 sec and crop box size 10 m.

Note: Comment out all 'moving' bags when using crop sizes > 10 (i.e. only use stationary data).

<pre><code>$ python conv_training_data_to_dataset.py dataset_X_exp.yaml
</code></pre>

Two dataset files `dataset_training.gz` and `dataset_validation.gz` will be generated in the directory `X_experiment/datasets/` by default.

Optional: Visualize pointclouds of a dataset frame-by-frame by running `python viz.py X_experiment/dataset/dataset_training.gz`


### Step 3: Convert dataset to regression problem

Generate the regression problem (i.e. sample feature vectors and sample target values represented as a data matrix and target vector from which to learn a regression model) by running the following command

<pre><code>$ python conv_dataset_to_reg_problem.py primary_experiment/datasets/dataset_training.gz primary_experiment/datasets/dataset_validation.gz dataset_primary_exp.yaml --nprocs N
</code></pre>

Two regression problem files `reg_prob_train_reg_problem.gz` and `reg_prob_val_reg_problem.gz` will be generated in the directory `primary_experiment_datasets/` by default. `N` denotes the number of processes for multiprocessing.

### Step 4: Train model

Train a model on a previously generated regression problem by modifying `train_model.py` with the desired gating tree depth and domain thresholds. The default configuration will train a tree depth 2 model with 'primary experiment' domain threshold values.

<pre><code>###########
#  Model
###########
...
tree_height = 2
domain_thresholds = [20., 10., 40.]
...

#################
#  Train model
#################
model.train_tree(dataset)
# model.train_tree_mp(dataset, 4)  <-- Exchange for multiprocessing w. 4 threads
</code></pre>

Run a pretrained model on a regression problem by commenting out the training code and uncomment the line which loads a model from file.

<pre><code>#################
#  Train model
#################
#model.train_tree(dataset)
...
model = read_compressed_pickle("tree_model.gz")
</code></pre>

Run the training program by specifing the paths for regression problems and model output directory.

<pre><code>python train_model.py primary_experiment/regression_problems/reg_problem_train_reg_problem.gz primary_experiment/regression_problems/reg_problem_val_reg_problem.gz .
</code></pre>

NOTE: Remember to set the value ranges when plotting (0 --> 70 for 'primary' and 0 --> 400 for 'secondary' experiment).

<pre><code>plt.xlim([0,70])
plt.ylim([0,70])
</code></pre>

The following files define the model training and inference part:
- rainfall_modeling/train_model.py
- rainfall_modeling/hierarchical_gmm.py
- rainfall_modeling/binary_tree.py

## File structure

```
rainfall_modeling_open
└───boost_mst_lib
│   │   boost_mst.cpp
│   │   boost_mst.hpp
│   │   boost_mst_lib.cpp
│   
└───feat_funcs
|   |   feat_comp.py
|   |   feat_intensity_statistics.py
|   |   feat_mean_statistics.py
|   |   feat_mst.py
|   |   feat_radial_statistics.py
|   |   mst_normalizer.py
|
|   variational_bayes_models          # Submodule with gate and expert models
|   LICENSE
│   README.md
|   binary_tree.py                    # Tree-related functions for hierarchical modeling
|   conv_dataset_to_reg_problem.py    # See 'Step 3'
|   conv_training_data_to_dataset.py  # see 'Step 2'
|   dataset_primary_exp.yaml          # Dataset and experiment configuration files 
|   dataset_secondary_exp.yaml
|   hierarchical_gmm.py               # Main model file
|   init_exp_dirs.py 
|   train_model.py  # Initialize a model for training and/or evaluation
|   utils.py        # Utility functions such as reading and storing files
|   viz.py          # Functions for visualizing pointclouds in datasets
```

## Feature vector specification

```
feature_vec (9, 1)
|
└───FeatMeanStdPnts
|   |   mean  [0]
|   |   std   [1]
|
└───FeatMeanStdIntensity
|   |   mean    [2]
|   |   median  [3]
|   |   std     [4]
|
└───FeatMeanStdRadial
|   |   mean  [5]
|   |   std   [6]
|
└───FeatMstLength
    |   mean  [7]
    |   std   [8]
```
