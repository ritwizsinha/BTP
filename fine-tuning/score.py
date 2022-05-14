# load general packages and functions
import torch
from rdkit.Chem import MolFromSmiles, QED, Crippen, Descriptors, rdMolDescriptors, AllChem
import numpy as np
from rdkit import DataStructs
import sys
import os
from tensorflow.keras.models import load_model
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/btp/ritwiz-RL-GraphINVENT-updated')
from molgym.utils.conversions import convert_rdkit_to_nx, convert_nx_to_rdkit
# load program-specific functions
from parameters.constants import constants as C
from molgym.envs.rewards.mpnn import MPNNReward
from molgym.envs.rewards import RewardFunction
from molgym.mpnn.layers import custom_objects
import networkx as nx


def compute_score(graphs, termination_tensor, validity_tensor, uniqueness_tensor, smiles, drd2_model, ic50_model, atom_types, bond_types):

    if C.score_type == "reduce":
        # Reduce size
        n_nodes = graphs[2]
        n_graphs = len(n_nodes)
        max_nodes = C.max_n_nodes
        score = torch.ones(n_graphs, device="cpu") - torch.abs(n_nodes - 10.) / (max_nodes - 10 + 1)
    
    elif C.score_type == "augment":
        # Augment size
        n_nodes = graphs[2].float()
        n_graphs = len(n_nodes)
        max_nodes = C.max_n_nodes
        score = torch.ones(n_graphs, device="cpu") - torch.abs(n_nodes - 40.) / (max_nodes - 40)
    
    elif C.score_type == "qed":
        # QED
        score = [QED.qed(MolFromSmiles(smi)) for smi in smiles]
        norm = np.linalg.norm(score)
        score = score / norm
        score = torch.tensor(score, device="cpu")


    elif C.score_type == "activity":
        n_mols = len(smiles)

        mols = [MolFromSmiles(smi) for smi in smiles]

        # QED
        qed = [QED.qed(mol) for mol in mols]
        qed = torch.tensor(qed, device="cpu")
        qedMask = torch.where(qed > 0.5, torch.ones(n_mols, device="cpu", dtype=torch.uint8), torch.zeros(n_mols, device="cpu", dtype=torch.uint8))
        
        activity = compute_activity(mols, drd2_model)
        activityMask = torch.where(activity > 0.5, torch.ones(n_mols, device="cpu", dtype=torch.uint8), torch.zeros(n_mols, device="cpu", dtype=torch.uint8))


        score = qedMask*activityMask
    elif C.score_type == "ic50":
        score = calculate_ic50_score(smiles, ic50_model, atom_types, bond_types)
        score = torch.tensor(score, device="cpu")
    elif C.score_type == "QEDp":
        score = [QED.qed(MolFromSmiles(smi)) -  S for smi in smiles]
        score = torch.tensor(score, device="cpu")
    else:
        raise NotImplementedError("The score type chosen is not defined. Please choose among 'reduce', 'augment', 'qed' and 'activity'.")
    
    # remove non unique molecules from the score
    score = score * uniqueness_tensor

    # remove invalid molecules
    score = score * validity_tensor

    # remove non properly terminated molecules
    score = score * termination_tensor

    return score


def compute_activity(mols, drd2_model):

    n_mols = len(mols)

    activity = torch.zeros(n_mols, device="cpu")

    for idx, mol in enumerate(mols):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)   
        ecfp4 = np.zeros((2048,))                                       
        DataStructs.ConvertToNumpyArray(fp, ecfp4)
        activity[idx] = drd2_model.predict_proba([ecfp4])[0][1]
    
    return activity

        
def compute_activity_score(graphs, termination_tensor, validity_tensor, uniqueness_tensor, smiles):
    
    n_mols = len(smiles)

    mols = [MolFromSmiles(smi) for smi in smiles]

    # QED
    qed = [QED.qed(mol) for mol in mols]
    qed = torch.tensor(qed, device="cpu")
    qedMask = torch.where(qed > 0.5, torch.ones(n_mols, device="cpu", dtype=torch.uint8), torch.zeros(n_mols, device="cpu", dtype=torch.uint8))
    
    activity = compute_activity(mols)
    activityMask = torch.where(activity > 0.5, torch.ones(n_mols, device="cpu", dtype=torch.uint8), torch.zeros(n_mols, device="cpu", dtype=torch.uint8))


    score = qedMask*activityMask

    # remove non unique molecules from the score
    score = score * uniqueness_tensor

    # remove invalid molecules
    score = score * validity_tensor

    # remove non properly terminated molecules
    score = score * termination_tensor

    return score
    

def calculate_ic50_score(mols, ic50_model, atom_types, bond_types):
    scores=[]
    # calculate score for each mol
    for i in range(len(mols)):
        m = MolFromSmiles(mols[i])
        G = convert_rdkit_to_nx(m)
        reward = MPNNReward(ic50_model, atom_types=atom_types, bond_types=bond_types, maximize=False)
        scores.append(reward._call(G))
    return scores

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
        



# Load the SA score
env_path = os.path.dirname(os.path.dirname(sys.executable))
_sa_score_path = f"{env_path}/share/RDKit/Contrib/SA_Score/sascorer.py"
if not os.path.isfile(_sa_score_path):
    raise ValueError('SA_scorer file not found. You must edit the above lines to point to the right place. Sorry!')
sys.path.append(os.path.dirname(_sa_score_path))
from sascorer import calculateScore


class LogP(RewardFunction):
    """Water/octanol partition coefficient"""

    def _call(self, graph: nx.Graph) -> float:
        mol = convert_nx_to_rdkit(graph)
        return Crippen.MolLogP(mol)


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
