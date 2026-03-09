from typing import List
import prody
import torch
import numpy as np
import dataclasses
import os

from collections import OrderedDict
from Bio.Data import IUPACData
from scipy.spatial import cKDTree


restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "<UNK>",
}
restype_3to1 = {v: k for k, v in restype_1to3.items()}


# Tokenization goes from proteins to nucleic acids, to single atoms
residue_tokens = OrderedDict()
# For Kai
residue_tokens["<PAD>"] = len(residue_tokens)
residue_tokens["<MASK>"] = len(residue_tokens)


def add_flag_tokens(flag):
    for residue in prody.flagDefinition(flag):
        if residue not in residue_tokens:
            residue_tokens[residue] = len(residue_tokens)


add_flag_tokens("stdaa")

residue_tokens["<START>"] = len(residue_tokens)
residue_tokens["<END>"] = len(residue_tokens)
residue_tokens["<UNK>"] = len(residue_tokens)

# Extra tokens for ligands
add_flag_tokens("nonstdaa")
add_flag_tokens("nucleotide")
add_flag_tokens("nucleic")
add_flag_tokens("ion")

residue_tokens["<GLYCAN>"] = len(residue_tokens)


token_to_index = {token: i for i, token in enumerate(residue_tokens)}
index_to_token = {i: token for i, token in enumerate(residue_tokens)}
num_residue_tokens = len(residue_tokens)

num_protein_tokens = residue_tokens["XAA"] + 1

# Faster lookup for protein, nucleotide, or ion
protein_residues = set(
    list(prody.flagDefinition("stdaa")) + list(prody.flagDefinition("nonstdaa"))
)
nucleotide_residues = set(prody.flagDefinition("nucleotide"))
ion_residues = set(prody.flagDefinition("ion")).union({'FE', 'FE2', 'NI'})


# Element tokens
elements = list(IUPACData.atom_weights.keys())
_ = elements.pop(elements.index("H"))  # Remove hydrogen
elements.append("<ATOM_UNK>")
elements.append("<ATOM_PAD>")
elements = [e.upper() for e in elements]

element_to_index = {element: i for i, element in enumerate(elements)}
num_element_tokens = len(elements)


# AF3 crystallization aids
af3_crystallization_aids = [
    x.strip() for x in open("data/misc/af3_crystallization_aids.txt").readlines()
]
af3_crystallization_aids = set(af3_crystallization_aids)


# AF3 ligand exclusion list
af3_ligands_excluded = [
    x.strip() for x in open("data/misc/af3_ligands_excluded.txt").readlines()
]
af3_ligands_excluded = set(af3_ligands_excluded)

# Glycans
af3_glycans = [x.strip() for x in open("data/misc/af3_glycans.txt").readlines()]
af3_glycans = set(af3_glycans)


def get_token_index(token):
    return token_to_index.get(token, token_to_index["<UNK>"])


def get_element_index(element):
    return element_to_index.get(element.upper(), element_to_index["<ATOM_UNK>"])


@dataclasses.dataclass
class StructureData:
    residue_token: np.ndarray
    residue_index: np.ndarray
    residue_atom_index: np.ndarray
    occupancy: np.ndarray
    bfactor: np.ndarray
    chain_id: np.ndarray
    position: np.ndarray
    element_index: np.ndarray
    # atom_name: np.ndarray
    is_ion: np.ndarray
    is_protein: np.ndarray
    is_nucleotide: np.ndarray
    is_center: np.ndarray
    is_backbone: np.ndarray
    not_pad_mask: np.ndarray
    interact_non_protein_res: np.ndarray = np.nan
    interact_ion_res: np.ndarray = np.nan
    interact_nucleotide_res: np.ndarray = np.nan
    interact_molecule_res: np.ndarray = np.nan
    noisy_residue_token: np.ndarray = np.nan
    time_step: np.ndarray = np.nan

    def __len__(self):
        return len(self.residue_token)

    def num_residues(self):
        return len(np.unique(self.residue_index))


@dataclasses.dataclass
class BatchStructureData:
    residue_token: np.ndarray
    residue_index: np.ndarray
    residue_atom_index: np.ndarray
    occupancy: np.ndarray
    bfactor: np.ndarray
    batch_index: np.ndarray
    chain_id: np.ndarray
    position: np.ndarray
    element_index: np.ndarray
    # atom_name: np.ndarray
    is_ion: np.ndarray
    is_protein: np.ndarray
    is_nucleotide: np.ndarray
    is_center: np.ndarray
    is_backbone: np.ndarray
    not_pad_mask: np.ndarray
    interact_non_protein_res: np.ndarray
    interact_ion_res: np.ndarray
    interact_nucleotide_res: np.ndarray
    interact_molecule_res: np.ndarray
    noisy_residue_token: np.ndarray
    time_step: np.ndarray

    def __len__(self):
        return len(self.residue_token)


structure_data_fields = StructureData.__dataclass_fields__


def slice_structure_data(
    structure_data: StructureData,
    slice_array: np.array,
) -> StructureData:
    new_data = {}
    for key in structure_data_fields:
        key_obj = getattr(structure_data, key)
        if hasattr(key_obj, "shape") and len(key_obj.shape) > 0:
            if key_obj.shape[0] == slice_array.shape[0]:
                new_data[key] = getattr(structure_data, key)[slice_array]
        else:
            new_data[key] = getattr(structure_data, key)
        if key == "residue_index":
            new_data[key] -= np.min(structure_data.residue_index)
    return StructureData(**new_data)


def init_struct_data_dict():
    struct_data = {}
    struct_data["residue_token"] = []
    struct_data["residue_index"] = []
    struct_data["residue_atom_index"] = []
    struct_data["occupancy"] = []
    struct_data["bfactor"] = []
    struct_data["chain_id"] = []
    struct_data["position"] = []
    struct_data["element_index"] = []
    struct_data["is_ion"] = []
    struct_data["is_protein"] = []
    struct_data["is_nucleotide"] = []
    struct_data["is_center"] = []
    struct_data["is_backbone"] = []
    struct_data["not_pad_mask"] = []
    return struct_data


def extend_struct_data_dict(
    struct_data_dict, extension_data_dict
) -> bool:
    # Check if ligand or ion
    if len(extension_data_dict["position"]) == 0:
        return False
    if not (
        extension_data_dict["is_protein"][-1] or
        extension_data_dict["is_nucleotide"][-1] or
        extension_data_dict["is_ion"][-1]
    ):
        positions = np.array(extension_data_dict["position"])
        center_idx = np.argmin(
            np.linalg.norm(
                positions - np.mean(positions, axis=0, keepdims=True),
                axis=-1
            ),
            axis=0
        )
        extension_data_dict["is_center"] = [
            i == center_idx for i in range(len(extension_data_dict["position"]))
        ]
    if any(extension_data_dict["is_center"]):
        for key in struct_data_dict:
            struct_data_dict[key].extend(extension_data_dict[key])
        return True
    return False


def parse_structure(path_or_name: str):
    if ".cif" in path_or_name:
        structure = prody.parseMMCIF(path_or_name)
    elif ".pdb" in path_or_name:
        structure = prody.parsePDB(path_or_name)
    else:
        try:
            structure = prody.parseMMCIF(path_or_name)
        except:
            raise ValueError(
                "Could not parse file. Please provide a valid mmCIF or PDB file."
            )
    return structure


def parse_mmcif_to_structure_data(path_or_name,parser_chain_id = None) -> StructureData:
    """
    Parse a mmCIF file and return a StructureData object.
    """
    structure = parse_structure(path_or_name)
    atoms = structure.select("not water and not hydrogen")
    # atoms = atoms.select("occupancy > 0")

    # Store info on an atom basis
    # Each atom has a residue token, a residue index, and a chain ID
    # It also has a position in 3D space
    # Also an element and whether it is an ion atom or not
    # Additionally, a tag that specifies whether it is protein or not

    struct_data = init_struct_data_dict()
    temp_residue_data = init_struct_data_dict()

    prev_residue_index = ""
    prev_chain_id = ""
    internal_residue_index = 0
    internal_chain_index = -1
    residue_atom_count = 0

    for atom in atoms:
        residue = atom.getResname()
        if residue in af3_crystallization_aids:
            continue
        chainid = atom.getChid()
        if len(chainid) > 1:
            chainid = chainid[0]
        if parser_chain_id is not None and chainid not in  parser_chain_id:
            # print(f"Skipping chain {atom.getChid()} in {path_or_name}")
            continue
        resnum = str(atom.getChid()) + str(atom.getResnum()) + str(atom.getIcode())
        is_new_residue_number = prev_residue_index != resnum
        is_new_chain_id = prev_chain_id != atom.getChid()
        if is_new_residue_number or is_new_chain_id:
            prev_residue_index = resnum
            # Check if there is a center for the last residue
            # if there isn't, we skip that residue and don't increment
            # the internal_residue_index
            if extend_struct_data_dict(struct_data, temp_residue_data):
                internal_residue_index += 1
            temp_residue_data = init_struct_data_dict()
            residue_atom_count = 0
            if is_new_chain_id:
                prev_chain_id = atom.getChid()
                internal_chain_index += 1
        if residue in af3_glycans:
            residue = "<GLYCAN>"
        temp_residue_data["residue_token"].append(get_token_index(residue))
        temp_residue_data["residue_index"].append(internal_residue_index)
        temp_residue_data["residue_atom_index"].append(residue_atom_count)
        temp_residue_data["occupancy"].append(atom.getOccupancy())
        temp_residue_data["bfactor"].append(atom.getBeta())
        temp_residue_data["chain_id"].append(internal_chain_index)
        temp_residue_data["position"].append(atom.getCoords())
        temp_residue_data["element_index"].append(get_element_index(atom.getElement()))
        temp_residue_data["is_ion"].append(residue in ion_residues)
        temp_residue_data["is_protein"].append(residue in protein_residues)
        temp_residue_data["is_nucleotide"].append(residue in nucleotide_residues)
        temp_residue_data["is_backbone"].append(
            (temp_residue_data["is_protein"][-1])
            and (atom.getName() in ["N", "CA", "C", "O"])
        )
        temp_residue_data["not_pad_mask"].append(True)

        # Definition of a center atom is
        # 1) CA atom in proteins
        # 2) C1' atom in nucleotides
        # 3) The first atom for everything else
        if temp_residue_data["is_protein"][-1]:
            temp_residue_data["is_center"].append(atom.getName() == "CA")
        elif temp_residue_data["is_nucleotide"][-1]:
            temp_residue_data["is_center"].append(atom.getName() == "C1'")
        elif temp_residue_data["is_ion"][-1]:
            temp_residue_data["is_center"].append(residue_atom_count == 0)
        else:
            # Ligand is_center should be based on the 
            # average location of ligand atoms
            temp_residue_data["is_center"].append(residue_atom_count == 0)

        residue_atom_count += 1

    # Merge last residue
    extend_struct_data_dict(struct_data, temp_residue_data)

    # Make numpy arrays
    for key in struct_data:
        struct_data[key] = np.array(struct_data[key])
    # Set first residue_index to 0
    struct_data["residue_index"] -= np.min(struct_data["residue_index"])

    struct_data = StructureData(**struct_data)
    return struct_data


def mask_structure_data(struct_data: StructureData, mask_array: np.ndarray):
    """
    Mask the structure data with a boolean array.
    """
    # Masking the data consists of putting in <MASK> tokens for the protein residues
    # And removing the non-backbone atoms for the masked residues

    # The mask array is the same length as the numbe of residues
    # We need to calculate the atom_mask_array
    # This is a boolean array of the same length as the number of atoms

    masked_residues = np.nonzero(mask_array)[0]
    atom_mask_array = np.isin(struct_data.residue_index, masked_residues)
    atom_mask_array = np.logical_and(
        atom_mask_array, np.logical_not(struct_data.is_backbone)
    )
    atom_keep_array = np.logical_not(atom_mask_array)

    struct_data.residue_token[atom_mask_array] = token_to_index["<MASK>"]

    masked_data = {}
    for key in structure_data_fields:
        masked_data[key] = struct_data.__dict__[key][atom_keep_array]
    return StructureData(**masked_data)


def get_closest_n_residues(
    struct_data: StructureData, position: np.ndarray, n: int
) -> StructureData:
    """
    Get the n closest residues to a position.
    """
    # Going to pick residues by center atom
    center_atoms = struct_data.position[struct_data.is_center]
    kdtree = cKDTree(center_atoms)
    _, closest_residue_indices = kdtree.query(position, n)
    closest_residue_indices = np.sort(closest_residue_indices)
    # Now remove residues not selected
    keep_array = np.isin(struct_data.residue_index, closest_residue_indices)

    return slice_structure_data(struct_data, keep_array)


def get_closest_n_atoms(
    struct_data: StructureData,
    position: np.ndarray,
    n: int,
    remove_incomplete_residues: bool = True,
) -> StructureData:
    """
    Get the n closest atoms to a position.
    Throw out atoms that are partially picked from a residue
    """
    kdtree = cKDTree(struct_data.position)
    _, closest_atom_indices = kdtree.query(position, n)
    closest_atom_indices = np.sort(closest_atom_indices)
    closest_atom_mask = np.zeros(len(struct_data.position), dtype=bool)
    closest_atom_mask[closest_atom_indices] = True
    if remove_incomplete_residues:
        residues_to_throw = np.unique(struct_data.residue_index[~closest_atom_mask])
        atoms_to_throw = np.isin(struct_data.residue_index, residues_to_throw)
        closest_atom_mask[atoms_to_throw] = False

    return slice_structure_data(struct_data, closest_atom_mask)


pad_constants = {
    "residue_token": token_to_index["<PAD>"],
    "residue_index": -1,
    "residue_atom_index": -1,
    "occupancy": 0.0,
    "bfactor": 0.0,
    "chain_id": -1,
    "position": 0.0,
    "element_index": element_to_index["<ATOM_PAD>"],
    "atom_name": "",
    "is_ion": False,
    "is_protein": False,
    "is_nucleotide": False,
    "is_center": False,
    "is_backbone": False,
    "not_pad_mask": False,
    'interact_non_protein_res': False,
    'interact_ion_res': False,
    'interact_nucleotide_res': False,
    'interact_molecule_res': False,
    "noisy_residue_token": token_to_index["<PAD>"],
    "time_step": -1,
}


def pad_structure_data(struct_data: StructureData, pad_length: int):
    """
    Pad a structure data to a certain length.
    """
    pad_data = {}
    for key in structure_data_fields:
        if key != "position" and key != "time_step":
            pad_data[key] = np.pad(
                struct_data.__dict__[key],
                (0, pad_length - len(struct_data)),
                mode="constant",
                constant_values=pad_constants[key],
            )
        elif key == "time_step":
            pad_data[key] = struct_data.__dict__[key]
        else:
            pad_values = np.zeros((pad_length - len(struct_data), 3))
            pad_data[key] = np.concatenate(
                [struct_data.__dict__[key], pad_values], axis=0
            )

    return StructureData(**pad_data)


def from_numpy(x_array: np.array) -> torch.Tensor:
    if x_array.dtype == np.float64:
        x_array = x_array.astype(np.float32)
    return torch.from_numpy(x_array)


def batch_structure_data_list(
    struct_data_list: List[StructureData],
    pad_length: int = None,
    to_torch: bool = False,
) -> BatchStructureData:
    """
    Batch a list of structure data.
    """
    try:
        if None in struct_data_list:
            struct_data_list.remove(None)
        # struct_data_list = [struct_data for struct_data in struct_data_list if struct_data.is_center.sum() <= 512 and np.unique(struct_data.residue_index).shape[0] == struct_data.is_center.sum()]

        stack_fn = np.stack if not to_torch else torch.stack
        full_fn = np.full if not to_torch else torch.full
        convert_fn = (lambda x: x) if not to_torch else from_numpy

        if pad_length is None:
            pad_length = max(len(struct_data) for struct_data in struct_data_list)

        batch_data = {}
        new_struct_data_list = []
        for struct_data in struct_data_list:
            if len(struct_data) < pad_length:
                new_struct_data_list.append(pad_structure_data(struct_data, pad_length))
            else:
                new_struct_data_list.append(struct_data)

        for key in structure_data_fields:
            if key not in BatchStructureData.__dataclass_fields__ or key in [
                "time_step"
            ]:
                continue
            batch_data[key] = stack_fn(
                [
                    convert_fn(struct_data.__dict__[key])
                    for struct_data in new_struct_data_list
                ]
            )
        batch_data["batch_index"] = stack_fn(
            [full_fn((pad_length,), i) for i in range(len(new_struct_data_list))]
        )
        batch_data["time_step"] = stack_fn(
            [convert_fn(struct_data.time_step) for struct_data in new_struct_data_list]
        )
        return BatchStructureData(**batch_data)
    
    except Exception as e:
        print(e)
        return None


def dump_structure_data(struct_data: StructureData, path: str):
    """
    Dump a structure data to a file.
    """
    np.savez_compressed(path, **struct_data.__dict__)


def load_structure_data(path: str) -> StructureData:
    """
    Load a structure data from a file.
    """
    struct_data = np.load(path)
    return StructureData(**struct_data)


def parse_or_load_mmcif(path_or_name,parser_chain_id = None) -> StructureData:
    if os.path.splitext(path_or_name)[1] == ".npz":
        return load_structure_data(path_or_name)
    else:
        return parse_mmcif_to_structure_data(path_or_name,parser_chain_id)


def get_example_batch():
    struct_data1 = parse_mmcif_to_structure_data("5a00")
    struct_data2 = parse_mmcif_to_structure_data("7npm")
    struct_data3 = parse_mmcif_to_structure_data("dataset/test_nucleotide/1bc7.pdb")
    struct_data_list = [struct_data1, struct_data2,struct_data3]
    batch_data = batch_structure_data_list(struct_data_list)
    return batch_data



def propagate_mask_vectorized(mask, index):
    # Ensure inputs are tensors
    mask = torch.as_tensor(mask, dtype=torch.bool)
    index = torch.as_tensor(index)
    
    # Get unique indices and their inverse mapping
    unique_indices, inverse_indices = torch.unique(index, return_inverse=True)
    
    # Create a tensor to hold the "any True" status for each unique index
    any_true = torch.zeros_like(unique_indices, dtype=torch.bool)
    
    # Use scatter_reduce to check if any mask is True for each unique index
    any_true.scatter_reduce_(0, inverse_indices, mask, reduce='amax')
    
    # Expand the any_true tensor to match the original shape
    return any_true[inverse_indices]


def interact_residue(data):
    protein_pos = np.expand_dims(data.position[data.is_protein],axis=1)
    non_protein_pos = np.expand_dims(data.position[~data.is_protein],axis=0)
    ion_pos = np.expand_dims(data.position[data.is_ion],axis=0)
    nucleotide_pos = np.expand_dims(data.position[data.is_nucleotide],axis=0)
    molecule_pos = np.expand_dims(data.position[~data.is_protein&~data.is_ion&~data.is_nucleotide],axis=0)

    dist_non_protein = np.sqrt((np.sum((protein_pos-non_protein_pos)**2,axis=-1)))
    interact_non_protein = np.any(dist_non_protein<5,axis=1)
    dist_ion = np.sqrt((np.sum((protein_pos-ion_pos)**2,axis=-1)))
    interact_ion = np.any(dist_ion<5,axis=1)
    dist_nucleotide = np.sqrt((np.sum((protein_pos-nucleotide_pos)**2,axis=-1)))
    interact_nucleotide = np.any(dist_nucleotide<5,axis=1)
    dist_molecule = np.sqrt((np.sum((protein_pos-molecule_pos)**2,axis=-1)))
    interact_molecule = np.any(dist_molecule<5,axis=1)

    interact_non_protein_res = propagate_mask_vectorized(interact_non_protein,data.residue_index[data.is_protein])
    interact_ion_res = propagate_mask_vectorized(interact_ion,data.residue_index[data.is_protein])
    interact_nucleotide_res = propagate_mask_vectorized(interact_nucleotide,data.residue_index[data.is_protein])
    interact_molecule_res = propagate_mask_vectorized(interact_molecule,data.residue_index[data.is_protein])


    data.interact_non_protein_res = interact_non_protein_res.numpy()
    data.interact_ion_res = interact_ion_res.numpy()
    data.interact_nucleotide_res = interact_nucleotide_res.numpy()
    data.interact_molecule_res = interact_molecule_res.numpy()

    return data

def pdb2data(pdb_file,device = 'cpu'):
    data = parse_mmcif_to_structure_data(pdb_file)
    data = interact_residue(data)
    data = {k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else torch.Tensor([v]) for k, v in data.__dict__.items()}
    data['batch_index'] = torch.zeros_like(data['residue_index'])
    data = {
        k: v.float() if isinstance(v, torch.Tensor) and v.dtype == torch.float64 else v for k, v in data.items()
    }
    return data



def _switch_pdb_order_add_ligand(new_pdb_path, original_pdb_path):
    '''
    new_pdb_path is the sidechain packing pdb
    original_pdb_path is the original pdb
    '''
    structure = prody.parsePDB(new_pdb_path)
    protein_atom = structure.select("protein")

    structure2 = prody.parsePDB(original_pdb_path)
    ligand_atom = structure2.select("not protein and not water and not hydrogen")

    merged =  protein_atom.toAtomGroup() + ligand_atom.toAtomGroup()

    prody.proteins.pdbfile.writePDB(new_pdb_path,merged)

def switch_pdb_order_add_ligand(new_pdb_path, original_pdb_path, confidence=None):
    new_lines = []
    temp_line = None
    ligand_lines = []

    # Read the original PDB to extract ligand information
    with open(original_pdb_path, 'r') as f:
        for line in f:
            if line.startswith('HETATM'):
                ligand_lines.append(line)

    # Process the new PDB file
    current_residue = None
    current_chain = None
    confidence_index = -1
    with open(new_pdb_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('ATOM'):
                residue_number = int(line[22:26])
                chain_id = line[21]
                if residue_number != current_residue or chain_id != current_chain:
                    current_residue = residue_number
                    current_chain = chain_id
                    if confidence is not None:
                        confidence_index = (confidence_index + 1) % len(confidence)

                atom_name = line[12:16].strip()
                if atom_name == "CB":
                    atom_index = int(line[6:11])
                    temp_line = line[:6] + f"{atom_index + 1:5}" + line[11:]
                    if confidence is not None:
                        temp_line = temp_line[:60] + f"{confidence[confidence_index]:6.2f}" + temp_line[66:]
                elif atom_name == "O":
                    atom_index = int(line[6:11])
                    new_line = line[:6] + f"{atom_index - 1:5}" + line[11:]
                    if confidence is not None:
                        new_line = new_line[:60] + f"{confidence[confidence_index]:6.2f}" + new_line[66:]
                    new_lines.append(new_line)
                    if temp_line is not None:
                        new_lines.append(temp_line)
                    temp_line = None
                else:
                    new_line = line
                    if confidence is not None:
                        new_line = new_line[:60] + f"{confidence[confidence_index]:6.2f}" + new_line[66:]
                    new_lines.append(new_line)
            elif line.startswith('TER'):
                new_lines.append(line)
            elif line.startswith('END'):
                new_lines.append(line)
                new_lines.extend(ligand_lines)
                break
            else:
                new_lines.append(line)

    # Write the new PDB file
    with open(new_pdb_path, 'w') as f:
        f.writelines(new_lines)


def make_merged_pdb(original_pdb_path: str, sc_packing_pdb_path: str):
    og_pdb = prody.parsePDB(original_pdb_path)
    sc_pdb = prody.parsePDB(sc_packing_pdb_path)

    iter_og_pdb = og_pdb.iterResidues()
    iter_sc_pdb = sc_pdb.iterResidues()
    internal_residue_num = 1

    new_residues = []

    while True:
        try:
            og_residue = next(iter_og_pdb)
        except StopIteration:
            break
        is_protein = og_residue.select("not protein") is None
        if is_protein:
            updated_residue = next(iter_sc_pdb)
        else:
            updated_residue = og_residue
        updated_atom_group = updated_residue.toAtomGroup()
        updated_atom_group.setCoords(updated_residue.getCoords())
        updated_atom_group.setResnums(
            [internal_residue_num] * len(updated_atom_group.getCoords())
        )
        new_residues.append(updated_atom_group)
        internal_residue_num += 1
    
    new_atom_group = sum(
        [residue for residue in new_residues[1:]],
        start=new_residues[0]
    )
    new_atom_group.setTitle(og_pdb.getTitle())
    prody.proteins.writePDB(sc_packing_pdb_path, new_atom_group)


if __name__ == "__main__":
    np.random.seed(0)
    struct_data = parse_mmcif_to_structure_data("out/3eka__1__2.A_3.A__1.B.cif")
    print(struct_data)
    print(f"Num tokens is: {np.max(struct_data.residue_token) + 1}")

    # Let's test masking
    mask_array = np.random.rand(np.max(struct_data.residue_token) + 1) > 0.2
    print(np.sum(mask_array))
    masked_struct_data = mask_structure_data(struct_data, mask_array)
    print(masked_struct_data)
