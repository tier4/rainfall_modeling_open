# rainfall_modeling_open
Code to reproduce results in the paper 'Probabilistic Rainfall Estimation from Automotive Lidar'


### Dependency
Python3
  - version: 3.6
  - packages: 
    - matplotlib
    - numpy
    - scikit-learn 
    - pybind11
    - pyyaml
    - tqdm


### TL;DR
1. Prepare `topics.txt`, `dataset.yaml`

2. Input following commands
<pre><code>
$ cd YOUR-PATH/OVP_ML_pipe

$ python2 convert_rosbags_to_training_data.py

$ python3 convert_training_data_to_dataset.py

$ cd ./boost_mst_lib

$ g++ -O3 -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` boost_mst_lib.cpp boost_mst.cpp -o boost_mst_lib`python3-config --extension-suffix`

$ cd ../

$ python3 gen_reg_problem.py PATH-TO-dataset_training.gz PATH-TO-dataset_validation.gz PATH-TO-OUTPUT-DIRECTORY

$ python3 vb_log_reg.py PATH-TO-TRAINING-REG-PROBLEM.gz PATH-TO-VALIDATION-REG-PROBLEM.gz PATH-TO-MODEL-OUTPUT-DIRECTORY

$ python3 insert_info_to_model.py PATH-TO-MODEL-FILE PATH-TO-MODEL-OUTPUT-DIRECTORY

</code></pre>

git clone --recurse-submodules git@github.com:rufuzzz0/rainfall_modeling_open.git


# How to use

Note: In filenames, `X` denotes `primary` or `secondary` depending desired experiment.

### Initialize repository

Note that the repository contains one submodule that also needs to be initialized.

<pre><code>
$ git clone git@github.com:rufuzzz0/rainfall_modeling_open.git

$ git submodule init

$ git submodule update

$ cd rainfall_modeling_open

$ export PYTHONPATH=.
</code></pre>

### Build library

The C++ library `boost_mst_lib` need to be built before being used for feature vector generation.

<pre><code>
$ cd ./boost_mst_lib

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

Run `python conv_training_data_to_dataset.py dataset_X_exp.yaml`

Two dataset files `dataset_training.gz` and `dataset_validation.gz` will be generated in the directory `X_experiment/datasets/` by default.

Optional: Visualize pointclouds of a dataset frame-by-frame by running `python viz.py X_experiment/dataset/dataset_training.gz`


### Step 3: Convert dataset to regression problem

Generate the regression problem (i.e. sample feature vectors and sample target values represented as a data matrix and target vector from which to learn a regression model) by running the following command

<pre><code>
$ `python conv_dataset_to_reg_problem.py X_experiments/datasets/dataset_training.gz X_experiments/datasets/dataset_validation.gz dataset_X_exp.yaml --nprocs N`
</code></pre>

Two regression problem files `reg_prob_train_reg_problem.gz` and `reg_prob_val_reg_problem.gz` will be generated in the directory `X_experiment_datasets/` by default. `N` denotes the number of processes for multiprocessing.

### Step 4: Train model

Train a model on a previously generated regression problem by modifying `train_model.py` with the desired gating tree depth and domain thresholds. The default configuration will train a tree depth 2 model with 'primary experiment' domain threshold values.

<pre><code>
###########
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
</code></pre>

Run a pretrained model on a regression problem by commenting out the training code and uncomment the line which loads a model from file.

<pre><code>
#################
#  Train model
#################
#model.train_tree(dataset)
...
model = read_compressed_pickle("tree_model.gz")
</code></pre>

Run the training program by specifing the paths for regression problems and model output directory.

<pre><code>
python train_model.py X_experiment/regression_problems/reg_problem_train_reg_problem.gz X_experiment/regression_problems/reg_problem_val_reg_problem.gz .
</code></pre>

NOTE: Remember to set the value ranges when plotting (0 --> 70 for 'primary' and 0 --> 400 for 'secondary' experiment).

<pre><code>
plt.xlim([0,70])
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
|   train_model.py     # Initialize a model for training and/or evaluation
|   utils.py       # Utility functions such as reading and storing files
|   viz.py         # Functions for visualizing pointclouds in datasets
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