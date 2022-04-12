#!/bin/bash
rm -r data/pre-training/anti-sars
mkdir data/pre-training/anti-sars

python helper.py

# python tools/select_atom_types.py --smi ./data/pre-training/anti-sars/train.smi --dtb PERSONAL
# python tools/select_atom_types.py --smi ./data/pre-training/anti-sars/test.smi --dtb PERSONAL
# python tools/select_atom_types.py --smi ./data/pre-training/anti-sars/valid.smi --dtb PERSONAL

# mv ./data/pre-training/anti-sars/train_selected-atoms.smi ./data/pre-training/anti-sars/train.smi
# mv ./data/pre-training/anti-sars/test_selected-atoms.smi ./data/pre-training/anti-sars/test.smi
# mv ./data/pre-training/anti-sars/valid_selected-atoms.smi ./data/pre-training/anti-sars/valid.smi

python ./tools/remove_large_mols.py --smi ./data/pre-training/anti-sars/train.smi --perc $1
python ./tools/remove_large_mols.py --smi ./data/pre-training/anti-sars/test.smi --perc $1
python ./tools/remove_large_mols.py --smi ./data/pre-training/anti-sars/valid.smi --perc $1

mv ./data/pre-training/anti-sars/train_no-large.smi ./data/pre-training/anti-sars/train.smi
mv ./data/pre-training/anti-sars/test_no-large.smi ./data/pre-training/anti-sars/test.smi
mv ./data/pre-training/anti-sars/valid_no-large.smi ./data/pre-training/anti-sars/valid.smi
