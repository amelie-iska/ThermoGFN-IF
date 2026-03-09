from typing import List, Tuple
import numpy as np
import os
import glob
import tqdm


from data.all_atom_parse import load_structure_data, index_to_token, restype_3to1

def convert_aap_to_protein_sequences(structure_data_path: str, name: str = None) -> List[Tuple[str, str]]:
    if name is None:
        name = os.path.basename(structure_data_path).split(".")[0]
    structure_data = load_structure_data(structure_data_path)
    protein_residues = structure_data.residue_token[structure_data.is_center * structure_data.is_protein]
    chain_ids = structure_data.chain_id[structure_data.is_center * structure_data.is_protein]
    extracted_sequences = []

    for chain_id in np.sort(np.unique(chain_ids)):
        chain_id_int = int(chain_id)
        chain_name = f"{name}_{chain_id_int}"
        chain_residues = protein_residues[chain_ids == chain_id]
        extracted_sequence = "".join([restype_3to1.get(index_to_token[x], "X") for x in chain_residues])
        extracted_sequences.append((chain_name, extracted_sequence))
    
    return extracted_sequences

def parse_sequences_directory(directory: str, output_fasta_path: str):
    if os.path.isfile(output_fasta_path):
        print("Output path already exists, aborting")
        exit()

    structures = glob.glob(os.path.join(directory, "*.npz"))
    if len(structures) == 0:
        structures = glob.glob(os.path.join(directory, "*", "*.npz"))
    tmp = []
    for structure_path in tqdm.tqdm(structures):
        sequences = convert_aap_to_protein_sequences(structure_path)
        tmp.extend(sequences)
        if len(tmp) > 500:
            with open(output_fasta_path, "a") as f:
                for seq in tmp:
                    _ = f.write(f">{seq[0]}\n")
                    _ = f.write(f"{seq[1]}\n")
            tmp = []
    with open(output_fasta_path, "a") as f:
        for seq in tmp:
            _ = f.write(f">{seq[0]}\n")
            _ = f.write(f"{seq[1]}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_directory")
    parser.add_argument("output_fasta")
    args = parser.parse_args()
    parse_sequences_directory(args.npz_directory, args.output_fasta)