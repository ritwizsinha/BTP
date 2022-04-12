import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data  = pd.read_csv('protease_only_id_smiles_IC50_deduped.csv')
data.drop('pIC50_std', 1, inplace=True)
data.drop('pIC50',1, inplace=True)
data.rename(columns={'smiles': 'SMILES'}, inplace=True)
train_data, test_data = train_test_split(data, test_size=0.1, random_state=1)
print(f'Split off {len(test_data)} mols for testing')

train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=2)
print(f'Split remaining mols into a {len(train_data)}/{len(val_data)} split for train and validation')

path="./data/pre-training/anti-sars/"
train_data.to_csv(path + 'train.smi', index=False)
val_data.to_csv(path + 'valid.smi', index=False)
test_data.to_csv(path + 'test.smi', index=False)