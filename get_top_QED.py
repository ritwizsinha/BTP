import csv
import numpy as np
from rdkit.Chem import MolFromSmiles, QED, AllChem
from rdkit.Chem.inchi import MolToInchiKey
from rdkit import DataStructs
import pickle
import sklearn
import argparse

"""
Computes the mean QED, mean DRD2 activity, fraction of active and unique mols, fraction unique, and average activity score for the molecules in the file.

To use script, run:
python tools/score_mols.py --smi path/to/file.smi
"""

# define the argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 add_help=False)

# define two potential arguments to use when drawing SMILES from a file
parser.add_argument("--smi",
                    type=str,
                    default="data/pre-training/chembl25_500k/test.smi",
                    help="SMILES file containing molecules to analyse.")
args = parser.parse_args()

def compute_score(filename):

    with open(filename) as f:
        reader = csv.reader(f, delimiter=" ")
        smiles = list(zip(*reader))[0]
    smiles = list(smiles)


    while '' in smiles:
        smiles.remove('')
    while 'SMILES' in smiles:
        smiles.remove('SMILES')
    while '[Xe]' in smiles:
        smiles.remove('[Xe]')

        
    n_mols = len(smiles)

    mols = [MolFromSmiles(smi) for smi in smiles]

    # QED
    qed = [QED.qed(mol) for mol in mols]
    qed = np.array(qed)
    qed = np.sort(qed)
    print(f'Top QED: {qed[-10:]}')

    # print(f"Mean score: {np.mean(score):.2f}", flush=True)


if __name__ == "__main__":
    compute_score(filename=args.smi)
    print("Done.", flush=True)
