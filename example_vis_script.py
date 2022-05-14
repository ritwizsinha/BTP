import math
import random
import rdkit
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem.rdmolfiles import SmilesMolSupplier

smi_file = "output_anti-sars/13-4-22-ic50/job_0/generation/epoch150_0.smi"

# load molecules from file
mols = SmilesMolSupplier(smi_file, sanitize=True, nameColumn=-1)

n_samples = 100
mols_list = [mol for mol in mols]
mols_sampled = random.sample(mols_list, n_samples)  # sample 100 random molecules to visualize

mols_per_row = int(math.sqrt(n_samples))            # make a square grid

png_filename=smi_file[:-3] + "png"  # name of PNG file to create
labels=list(range(n_samples))       # label structures with a number

# draw the molecules (creates a PIL image)
img = MolsToGridImage(mols=mols_sampled,
                      molsPerRow=mols_per_row,
                      legends=[str(i) for i in labels])

img.save(png_filename)