import numpy as np
from Bio.PDB import PDBParser

def calculate_pocket_centroid(pdb_path: str, chain_id: str = "A") -> np.ndarray:
    parser = PDBParser(QUIET=True)
    # Load the minimized structure (T790M is a good representative)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]
    chain = model[chain_id]

    # The triad of residues wrapping the EGFR ATP cavity
    residues = [790, 745, 855]
    coords = []

    for res_id in residues:
        try:
            coord = chain[res_id]["CA"].coord
            coords.append(coord)
            print(f"Residue {res_id} CA: [{coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f}]")
        except KeyError:
            print(f"Warning: Could not find CA for residue {res_id}")

    # Calculate the arithmetic mean of the X, Y, and Z axes
    centroid = np.mean(coords, axis=0)
    return centroid

if __name__ == "__main__":
    # Point this to one of your minimized mutant structures
    pdb_file = "data/mutants/T790M.pdb" 
    
    print("Extracting coordinates...")
    center = calculate_pocket_centroid(pdb_file)
    
    print("\n--- NEW DOCKING CONFIG ---")
    print(f"center:")
    print(f"  x: {center[0]:.2f}")
    print(f"  y: {center[1]:.2f}")
    print(f"  z: {center[2]:.2f}")