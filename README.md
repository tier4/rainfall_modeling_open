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




Train model

rainfall_modeling/train_model.py
rainfall_modeling/hierarchical_gmm.py
rainfall_modeling/binary_tree.py



Visualization instruction
