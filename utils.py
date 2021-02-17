#!/usr/bin/env python3
import os
import yaml
import pickle
import gzip


def read_dataset_file(path_to_dataset_config_file):
    '''
    '''
    with open(path_to_dataset_config_file, "r") as file:
        try:
            dataset_yaml = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return dataset_yaml


def read_topics_file(path_to_topics_config_file):
    '''Returns a list of 'topic names'.
    '''
    topic_list = []

    with open(path_to_topics_config_file, "r") as file:
        for line in file:
            # Remove empty rows
            if line == '\n':
                continue
            # Skip comments
            if line[0] == '#':
                continue
            # Remove line break
            topic_list.append(line[:-1])

    return topic_list


def read_labels_file(path_to_labels_config_file):
    '''Returns a dictionary linking 'bag names' and 'label'.
    '''
    with open(path_to_labels_config_file, "r") as file:
        try:
            label_dic = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return label_dic


def write_compressed_pickle(obj, filename, write_dir, compresslevel=9):
    '''Converts an object into byte representation and writes a compressed file.

    Args:
        obj: Generic Python object.
        filename: Name of file without file ending.
        write_dir (str): Output path.
    '''
    filename_with_extension = filename + ".gz"
    path = os.path.join(write_dir, filename_with_extension)
    pkl_obj = pickle.dumps(obj, protocol=2)
    try:
        with gzip.open(path, "wb", compresslevel) as f:
            f.write(pkl_obj)
    except IOError as error:
        print(error)
        print(f"path: {path} in func write_compressed_pickle()")
        exit()


def read_compressed_pickle(path, compresslevel=9):
    '''Reads a compressed binary file and return the object.

    Args:
        path (str): Path to the file (incl. filename)
    '''
    try:
        with gzip.open(path, "rb", compresslevel) as f:
            pkl_obj = f.read()
            # For python 2 -> 3 compatibility
            obj = pickle.loads(pkl_obj, encoding="bytes")  
            return obj
    except IOError as error:
        print(error)
        print(f"path: {path} in func read_compressed_pickle()")
        exit()
