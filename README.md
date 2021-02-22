# rainfall_modeling_open
Code to reproduce results in the paper 'Probabilistic Rainfall Estimation from Automotive Lidar'


### Dependency
Python3
  - version: 3.6 or higher
  - packages:
    - pybind11
    - pyyaml
    - matplotlib
    - numpy


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


### Initialize repository

<pre><code>
$ git clone --recurse-submodules git@github.com:rufuzzz0/rainfall_modeling_open.git

$ export PYTHONPATH=. python
</code></pre>

### Build library

The C++ library `boost_mst_lib` need to be built before being used for feature vector generation.

<pre><code>
$ cd ./boost_mst_lib

$ g++ -O3 -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` boost_mst_lib.cpp boost_mst.cpp -o boost_mst_lib`python3-config --extension-suffix`

$ cd ../
</code></pre>

### Step 1: Initialize experiment directory structure

Run program to generate empty directory stucture

Download raw data (rosbags pointclouds converted into Numpy arrays)

Optional: Possible to download files for all stages of the process


### Step 2: Convert raw training data to dataset

Run `python conv_training_data_to_dataset.py dataset_X_exp.yaml`

Two dataset files `dataset_training.gz` and `dataset_validation.gz` will be generated in the directory `X_experiment/datasets/` by default.

(Note: `X` denotes `primary` or `secondary` depending on which experiment is to be run)

Optional: Visualize pointclouds of a dataset by running `python viz.py X_experiment/dataset/dataset_training.gz`


### Step 3: Convert dataset to regression problem

Run `python conv_dataset_to_reg_problem.py X_experiments/datasets/dataset_training.gz X_experiments/datasets/dataset_validation.gz dataset_X_exp.yaml --nprocs N`

Two regression problem files `reg_prob_train_reg_problem.gz` and `reg_prob_val_reg_problem.gz` will be generated in the directory `X_experiment_datasets/` by default.

(Note: `X` denotes `primary` or `secondary` depending on which experiment is to be run)



Need to compile!
g++ -O3 -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` boost_mst_lib.cpp boost_mst.cpp -o boost_mst_lib`python3-config --extension-suffix`



Train model

rainfall_modeling/train_model.py
rainfall_modeling/hierarchical_gmm.py
rainfall_modeling/binary_tree.py



Visualization instruction
