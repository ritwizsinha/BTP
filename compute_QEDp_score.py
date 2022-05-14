from tensorflow.keras.models import load_model
import json
import argparse
import numpy as np
import csv
from molgym.mpnn.layers import custom_objects
from rdkit.Chem import MolFromSmiles
from molgym.utils.conversions import convert_rdkit_to_nx, convert_nx_to_rdkit
from molgym.envs.rewards.mpnn import MPNNReward
from rdkit.Chem import QED
from molgym.envs.rewards import RewardFunction
import networkx as nx

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 add_help=False)

# define two potential arguments to use when drawing SMILES from a file
parser.add_argument("--smi",
                    type=str,
                    default="data/pre-training/chembl25_500k/test.smi",
                    help="SMILES file containing molecules to analyse.")
args = parser.parse_args()

class QEDReward(RewardFunction):
    """Quantitative measure of uncertainty"""

    def _call(self, graph: nx.Graph) -> float:
        mol = convert_nx_to_rdkit(graph)
        return QED.qed(mol)


class SAScore(RewardFunction):
    """Synthesis accessibility score

    Smaller values indicate greater "synthesizability" """

    def _call(self, graph: nx.Graph) -> float:
        mol = convert_nx_to_rdkit(graph)
        return calculateScore(mol)


class CycleLength(RewardFunction):
    """Reward based on the maximum number of cycles

    Taken from https://github.com/Bibyutatsu/FastJTNNpy3/blob/master/fast_bo/gen_latent.py"""

    def _call(self, graph: nx.Graph) -> float:
        cycle_list = nx.cycle_basis(graph)
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([len(j) for j in cycle_list])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6
        return float(cycle_length)

def calculate_QEDp_score(mols):
    scores = []
    for i in range(len(mols)):
        m = MolFromSmiles(mols[i])
        G = convert_rdkit_to_nx(m)
        score = QED.qed(m)
        saScore = SAScore()
        cLength = CycleLength()
        score -=saScore._call(G)
        score -=cLength._call(G)
        scores.append(score)
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

    scores = calculate_QEDp_score(smiles)
    print(f'Some values: {scores[:10]}')

if __name__ == "__main__":
    compute_score(filename=args.smi)
    print("Done.", flush=True)