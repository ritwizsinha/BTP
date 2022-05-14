from tensorflow.keras.models import load_model
import json
import argparse
import numpy as np
import csv
from molgym.mpnn.layers import custom_objects
from rdkit.Chem import MolFromSmiles
from molgym.utils.conversions import convert_rdkit_to_nx
from molgym.envs.rewards.mpnn import MPNNReward

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 add_help=False)

# define two potential arguments to use when drawing SMILES from a file
parser.add_argument("--smi",
                    type=str,
                    default="data/pre-training/chembl25_500k/test.smi",
                    help="SMILES file containing molecules to analyse.")
args = parser.parse_args()

def calculate_ic50_score(mols, ic50_model, atom_types, bond_types):
    scores=[]
    # calculate score for each mol
    for i in range(len(mols)):
        m = MolFromSmiles(mols[i])
        G = convert_rdkit_to_nx(m)
        reward = MPNNReward(ic50_model, atom_types=atom_types, bond_types=bond_types, maximize=False)
        scores.append(reward._call(G))
    return scores


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
    ic50_model = load_model('/home/btp/ritwiz-RL-GraphINVENT-updated/saved_models/mpnn_13_4_22', custom_objects=custom_objects)
    atom_types = None
    bond_types = None
    with open('/home/btp/ritwiz-RL-GraphINVENT-updated/atom_types.json') as fp:
        atom_types = json.load(fp)
    with open('/home/btp/ritwiz-RL-GraphINVENT-updated/bond_types.json') as fp:
        bond_types = json.load(fp)

    scores = calculate_ic50_score(smiles, ic50_model, atom_types, bond_types)
    print(f'Mean pIC50: {np.mean(scores):.2f}')

if __name__ == "__main__":
    compute_score(filename=args.smi)
    print("Done.", flush=True)