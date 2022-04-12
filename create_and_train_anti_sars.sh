#!/bin/bash

## Create the dataset
./clean_shorten_script.sh 100

sleep 2
# Separate the training in different files so that they can run on different procesors in batches of 100 each creating different dataset for each
# Entering the data directory
cd ./data/pre-training;

# The base directory for the dataset
data_dir="anti-sars"

# Set the split amount
split_amt=100
# count the number of lines in train.smi
lines=$(sed -n '$=' "$data_dir/train.smi")

# Get the number of directories which will be required if each of the train.smi contains 100 entries
num_dirs=$(expr $lines / $split_amt)

# The prefix for the training split file
split_prefix_name="anti-sars"

# Using the split functionality to split the data as eg anti-sars00, anti-sars01
split -d -l $split_amt "$data_dir/train.smi" anti-sars;

# Iterate over the range of directories
for ((i=0;i<= $num_dirs;i++)) do
    # The directory name
    dir_name="$data_dir-$i"
    # Create the directory with the given name
    rm -rf $dir_name;
    mkdir $dir_name;
    # Get the index of the directory with precision 2
    index=$(printf "%02d" $i)
    # The filename created by split which we have to mv to the currently created directory
    file_name="$split_prefix_name$index"
    # Moving the file to directory
    mv $file_name $dir_name   
    # Change the file name to train.smi by entering and exiting the directory 
    cd $dir_name
    mv $file_name "train.smi";
    # mv '../anti-sars/preprocessing_params.csv' .
    cd ../../../
    python submitPT.py $dir_name &
    cd ./data/pre-training;
done

# Deleting directories while in debug
# for ((i=0;i<= $num_dirs;i++)) do
#     # The directory name
#     dir_name="$data_dir-$i"
#     # Create the directory with the given name
#     rm -rf $dir_name;
# done

# Try and Reduce the max number of atoms, max_n_nodes in each record to some user defined value


# Convert the submitPT.py to accept input from command line and run the preprocess job on it for various datasets


# Combine the hdf files and change the combine_HDFs.py script to accept input form cmd