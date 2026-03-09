import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openfold.data.data_transforms as data_transforms
import torch
import torch.nn.functional as F
from einops import rearrange
from openfold.utils.feats import atom14_to_atom37
from openfold.utils.rigid_utils import Rigid, Rotation
from torchtyping import TensorType

from fampnn.data import protein
from fampnn.data import residue_constants as rc

FEATURES_LONG = ("residue_index", "chain_index", "aatype")

def load_feats_from_pdb(pdb, chain_ids_override: str = None, max_conformers: int = 1):
    """
    Load model input features from a PDB file or mmcif file.
    - chain_residx_gap: Gap to add between residue indices in different chains.
    - max_conformers: Handle disordered atoms, max number of altlocs to store. If > 1, returns coords with shape [seqlen, num_atoms, max_conformers, 3]
    """
    feats = {}
    protein_obj, chain_id_mapping = protein.read_pdb(pdb, chain_ids_override=chain_ids_override, max_conformers=max_conformers)
    for k, v in vars(protein_obj).items():
        if isinstance(v, list) and all(isinstance(i, np.ndarray) for i in v):
            v = np.array(v)  # Convert list of NumPy arrays into a single NumPy array
        feats[k] = torch.tensor(v)

    feats["all_atom_positions"] = feats.pop("atom_positions")
    feats["all_atom_mask"] = feats.pop("atom_mask")
    feats["aatype"] = feats["aatype"].long()

    # Add one-hot encoding of amino acid types
    feats["target_feat"] = F.one_hot(feats["aatype"], num_classes=len(rc.restypes_with_x)).float()

    # Add AF2 features, uncomment if needed
    feats = data_transforms.make_seq_mask(feats)

    # Handle the distinction between missing atoms and ghost atoms in the atom mask
    ghost_atom_mask = 1 - torch.tensor(rc.restype_atom37_mask)[feats["aatype"]]  # 1 for atoms that are not in the residue type; ghost atoms
    if max_conformers > 1:
        ghost_atom_mask = rearrange(ghost_atom_mask, "n a -> n 1 a").expand(-1, max_conformers, -1)  # [n, c, a]

    missing_atom_mask = (1 - feats["all_atom_mask"]) * (1 - ghost_atom_mask)  # 1 for atoms that are missing in the PDB file; missing if not in atom_mask but not a ghost atom

    feats["ghost_atom_mask"] = ghost_atom_mask  # [n, a] or [n, c, a]
    feats["missing_atom_mask"] = missing_atom_mask  # [n, a] or [n, c, a]
    feats["interface_residue_mask"] = get_interface_residue_mask(feats['all_atom_positions'], feats['chain_index'])

    # Mapping from chain letter to chain index
    feats["chain_id_mapping"] = chain_id_mapping

    return feats

def get_interface_residue_mask(x, chain_index):
    # Extract C-alpha atoms' positions
    x_ca = x[:, 1, :]

    # Calculate pairwise Euclidean distances between C-alpha atoms
    d_ca = x_ca[None, :, :] - x_ca[:,  None, :]
    d_ca = torch.sqrt(torch.sum(d_ca ** 2, dim=2))

    # Create a mask for residues within the same chain
    same_chain_mask = torch.eq(chain_index[:, None], chain_index[None, :])
    d_ca[same_chain_mask] = np.inf  # Set distances within the same chain to infinity

    # Apply cutoff to get interface residues
    within_cutoff = d_ca < rc.interface_cutoff
    interface_residue_mask = torch.any(within_cutoff, dim=1).to(dtype=torch.bool)
    return interface_residue_mask

def check_valid_interface(x, atom_mask, chain_index):
    num_residues, num_atoms_per_residue, _ = x.shape
    x_flat = x.reshape(-1, 3)
    atom_mask_flat = atom_mask.reshape(-1)

    # Create residue index mapping
    residue_index = torch.arange(num_residues, device=x.device).repeat_interleave(num_atoms_per_residue)

    # Mask to filter valid atoms
    valid_mask = atom_mask_flat.bool()
    x_valid = x_flat[valid_mask]
    residue_index_valid = residue_index[valid_mask]
    chain_index_valid = chain_index[residue_index_valid]

    # Calculate pairwise distances only for valid atoms
    d_valid = torch.cdist(x_valid, x_valid, p=2)

    # Mask out same-chain residues
    same_chain_mask = chain_index_valid[:, None] == chain_index_valid[None, :]
    d_valid[same_chain_mask] = float('inf')

    # Check if any inter-chain distance is below the threshold
    return torch.any(d_valid < 5.01)


def pad_to_max_len(batch: Dict[str, TensorType["b n ..."]], max_len: int):
    """
    Inverse of trim_to_max_len; pads a batch to a fixed length.
    """
    padded_example = {}
    for k, v in batch.items():
        # features which aren't padded
        if k in ['pdb_key', 'cond_labels_in', 'chain_ids']:
            padded_example[k] = v
        else:
            B, N, *extra_shape = v.shape
            padding = torch.zeros((B, max_len - N, *extra_shape), device=v.device, dtype=v.dtype)
            padded_example[k] = torch.cat([v, padding], dim=1)
    return padded_example


def atom14_aatype_to_atom37(atom14_pos: TensorType["b n 14 3", float],
                            aatype: TensorType["b n", int]
                            ) -> TensorType["b n 37 3", float]:
    feats = {}
    feats["aatype"] = aatype
    feats = data_transforms.make_atom14_masks(feats)
    return atom14_to_atom37(atom14_pos, feats)


def torch_kabsch(a: TensorType["b n x"],
                 b: TensorType["b n x"]
                 ) -> TensorType["b x x"]:
    """
    get alignment matrix for two sets of coordinates using PyTorch

    adapted from: https://github.com/sokrypton/ColabDesign/blob/ed4b01354928b60cd1347f570e9b248f78f11c6d/colabdesign/shared/protein.py#L128
    """
    with torch.autocast(device_type=a.device.type, enabled=False):
        ab = a.transpose(-1, -2) @ b
        u, s, vh = torch.linalg.svd(ab, full_matrices=False)
        flip = torch.det(u @ vh) < 0
        u_ = torch.where(flip, -u[..., -1].T, u[..., -1].T).T
    u = torch.cat([u[..., :-1], u_[..., None]], dim=-1)
    return u @ vh


def torch_rmsd_weighted(a: TensorType["b n x", float],
                        b: TensorType["b n x", float],
                        weights: Optional[TensorType["b n", float]],
                        return_aligned: bool = False
                        ) -> TensorType["b", float]:
    """
    Compute weighted RMSD of coordinates after weighted alignment. Batched.

    For masked RMSD, set weights to 0 for masked atoms.

    Aligns a to b using Kabsch algorithm, then computes RMSD.
    If return_aligned is True, returns the aligned structures as well.

    Adapted from: https://github.com/sokrypton/ColabDesign/blob/main/colabdesign/af/loss.py#L445
    """
    if weights is None:
        weights = torch.ones(a.shape[:-1], device=a.device, dtype=a.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True)  # normalize weights

    # Align
    W = weights[..., None]
    a_mu = (a * W).sum(dim=-2, keepdim=True)
    b_mu = (b * W).sum(dim=-2, keepdim=True)

    R = torch_kabsch((a - a_mu) * W, b - b_mu)
    aligned_a = (a - a_mu) @ R + b_mu

    weighted_msd = (W * ((aligned_a - b) ** 2)).sum(dim=(-1, -2))
    weighted_rmsd = torch.sqrt(weighted_msd + 1e-8)

    if return_aligned:
        return weighted_rmsd, (aligned_a, b)
    return weighted_rmsd


def uniform_rand_rotation(batch_size):
    # Creates a shape (batch_size, 3, 3) rotation matrix uniformly at random in SO(3)
    # Uses quaternionic multiplication to generate independent rotation matrices for each batch
    q = torch.randn(batch_size, 4)
    q /= torch.norm(q, dim=1, keepdim=True)
    rotation = torch.zeros(batch_size,3,3).to(q)
    a, b, c, d = q[:,0], q[:,1], q[:,2], q[:,3]
    rotation[:,0,:] = torch.stack([2*a**2 -1 + 2*b**2,   2*b*c - 2*a*d,        2*b*d + 2*a*c]).T
    rotation[:,1,:] = torch.stack([2*b*c + 2*a*d,        2*a**2 -1 + 2*c**2,   2*c*d - 2*a*b]).T
    rotation[:,2,:] = torch.stack([2*b*d - 2*a*c,        2*c*d + 2*a*b,        2*a**2 -1 + 2*d**2]).T
    return rotation


def center_random_augmentation(coords_in: TensorType["n a 3", float],
                               seq_mask: TensorType["n", float],
                               atom_mask: TensorType["n a", float],
                               missing_atom_mask: TensorType["n a", float],
                               translation_scale=1.0,
                               return_transforms=False
                               ):
    """
    Batched or unbatched.
    Mean center on CA atoms, then apply random rotation and translation.
    Ensures that missing/ghost/padding atoms are set back to 0.

    Inputs:
        - seq_mask: 0 if residue is padding
        - atom_mask: 1 if not ghost and not missing atom, 0 otherwise
        - missing_atom_mask: 1 if atom is missing, 0 if present
    """
    input_dim = coords_in.dim()
    if input_dim == 3:
        # unbatched; add batch dimension
        coords_in = coords_in.unsqueeze(0)
        atom_mask = atom_mask.unsqueeze(0)
        missing_atom_mask = missing_atom_mask.unsqueeze(0)
        seq_mask = seq_mask.unsqueeze(0)

    X = coords_in[:, :, 1:2]  # [b n 1 3]

    # Center coords
    M = (1 - missing_atom_mask[:, :, 1:2]) * seq_mask[:, :, None]  # [b n 1]
    M_sum = M.sum(dim=1, keepdim=True)[..., None]  # [b 1 1 1]
    coords_mean = (X * M[..., None]).sum(dim=1, keepdim=True) / M_sum  # [b 1 1 3]
    coords_in = coords_in - coords_mean

    # Apply random rotation
    random_rot = uniform_rand_rotation(coords_in.shape[0]).to(coords_in.device)
    coords_in = torch.einsum("b n a i, b i j -> b n a j", coords_in, random_rot)

    # Apply random translation
    random_trans = torch.randn_like(coords_mean) * translation_scale
    coords_in = coords_in + random_trans

    # Zero out padding + missing / ghost atoms
    coords_in = coords_in * rearrange(seq_mask, "b n -> b n 1 1")
    coords_in = coords_in * atom_mask[..., None]

    transforms = (coords_mean, random_rot, random_trans)
    if input_dim == 3:
        # unbatched; remove batch dimension
        coords_in = coords_in.squeeze(0)
        transforms = tuple(t.squeeze(0) for t in transforms)

    if return_transforms:
        return coords_in, transforms

    return coords_in


def apply_random_augmentation(coords_in: TensorType["b n a 3", float],
                              transforms: Tuple[TensorType["b 1 1 3", float], TensorType["b 3 3", float], TensorType["b 1 1 3", float]],
                              seq_mask: TensorType["b n", float],
                              atom_mask: TensorType["b n a", float]) -> TensorType["b n a 3", float]:
    """
    Batched or unbatched.

    Given the output transforms of center_random_augmentation, applies the same transformation to a set of coordinates.
    Ensures that missing/ghost/padding atoms are set back to 0.
    """
    input_dim = coords_in.dim()
    if input_dim == 3:
        # unbatched; add batch dimension
        coords_in = coords_in.unsqueeze(0)
        transforms = tuple(t.unsqueeze(0) for t in transforms)

    coords_mean, random_rot, random_trans = transforms

    # Apply transforms
    coords_in = coords_in - coords_mean
    coords_in = torch.einsum("b n a i, b i j -> b n a j", coords_in, random_rot)
    coords_in = coords_in + random_trans

    # Zero out padding + missing / ghost atoms
    coords_in = coords_in * rearrange(seq_mask, "b n -> b n 1 1")
    coords_in = coords_in * atom_mask[..., None]

    if input_dim == 3:
        # unbatched; remove batch dimension
        coords_in = coords_in.squeeze(0)

    return coords_in


def cat_bb_scn(x_bb: TensorType["... a1 3", float],
               x_scn: TensorType["... a2 3", float]) -> TensorType["... a 3", float]:
    """
    Concatenate the bb and scn atoms to their corresponding indices.
    """
    A = x_bb.shape[-2] + x_scn.shape[-2]
    x = torch.zeros(x_bb.shape[:-2] + (A, 3), device=x_bb.device, dtype=x_bb.dtype)
    x[..., rc.bb_idxs, :] = x_bb
    x[..., rc.non_bb_idxs, :] = x_scn
    return x

def atom14_aatype_to_atom37(atom14_pos: TensorType["b n 14 3", float],
                            aatype: TensorType["b n", int]) -> TensorType["b n 37 3", float]:
    feats = {}
    feats["aatype"] = aatype
    feats = data_transforms.make_atom14_masks(feats)
    return atom14_to_atom37(atom14_pos, feats)

def get_rc_tensor(rc_np, aatype):
    return torch.tensor(rc_np, device=aatype.device)[aatype]

def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]

def atom37_to_atom14(aatype: TensorType["... n", int],
                     all_atom_pos: TensorType["... n 37 3", float],
                     atom37_mask: Optional[TensorType["... n 37", float]] = None):
    """Convert Atom37 positions to Atom14 positions."""
    if atom37_mask is None:
        atom37_mask = get_rc_tensor(rc.STANDARD_ATOM_MASK_WITH_X, aatype)

    residx_atom14_to_atom37 = get_rc_tensor(
        rc.RESTYPE_ATOM14_TO_ATOM37, aatype
    )

    no_batch_dims = len(aatype.shape) - 1

    atom14_mask = batched_gather(
        atom37_mask,
        residx_atom14_to_atom37,
        dim=no_batch_dims + 1,
        no_batch_dims=no_batch_dims + 1,
    ).to(all_atom_pos.dtype)

    # create a mask for known groundtruth positions
    atom14_mask *= get_rc_tensor(rc.RESTYPE_ATOM14_MASK_WITH_X, aatype)

    # gather the groundtruth positions
    atom14_positions = batched_gather(
        all_atom_pos,
        residx_atom14_to_atom37,
        dim=no_batch_dims + 1,
        no_batch_dims=no_batch_dims + 1,
    )

    return atom14_positions, atom14_mask

### GVP UTILS

def nan_to_num(ts, val=0.0):
    """
    Replaces nans in tensor with a fixed value.
    """
    val = torch.tensor(val, dtype=ts.dtype, device=ts.device)
    return torch.where(~torch.isfinite(ts), val, ts)


def rbf(values, v_min, v_max, n_bins=16):
    """
    Returns RBF encodings in a new dimension at the end.
    """
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device)
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    v_expand = torch.unsqueeze(values, -1)
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-z ** 2)


def norm(tensor, dim, eps=1e-8, keepdim=False):
    """
    Returns L2 norm along a dimension.
    """
    return torch.sqrt(
            torch.sum(torch.square(tensor), dim=dim, keepdim=keepdim) + eps)


def normalize(tensor, dim=-1):
    """
    Normalizes a tensor along a dimension after removing nans.
    """
    return nan_to_num(
        torch.div(tensor, norm(tensor, dim=dim, keepdim=True))
    )

def orientations(X):
    forward = normalize(X[:, 1:] - X[:, :-1])
    backward = normalize(X[:, :-1] - X[:, 1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

def sidechains(X):
    n, origin, c = X[:, :, 0], X[:, :, 1], X[:, :, 2]
    c, n = normalize(c - origin), normalize(n - origin)
    bisector = normalize(c + n)
    perp = normalize(torch.cross(c, n, dim=-1))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec

def dihedrals(X, eps=1e-7):
    X = torch.flatten(X[:, :, :3], 1, 2)
    bsz = X.shape[0]
    dX = X[:, 1:] - X[:, :-1]
    U = normalize(dX, dim=-1)
    u_2 = U[:, :-2]
    u_1 = U[:, 1:-1]
    u_0 = U[:, 2:]

    # Backbone normals
    n_2 = normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
    n_1 = normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [bsz, -1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], -1)
    return D_features

def positional_embeddings(edge_index,
                           num_embeddings=None,
                           num_positional_embeddings=16,
                           period_range=[2, 1000]):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    num_embeddings = num_embeddings or num_positional_embeddings
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32,
            device=edge_index.device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E

def dist(X, E_idx, padding_mask):
    """ Pairwise euclidean distances """
    residue_mask = ~padding_mask
    residue_mask_2D = torch.unsqueeze(residue_mask,1) * torch.unsqueeze(residue_mask,2)
    dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
    D = norm(dX, dim=-1)

    # sorting preference: first those with coords,then the
    # residues that came from padding are last
    D_adjust = nan_to_num(D) + (~residue_mask_2D) * (1e10)
    D_neighbors = torch.gather(D_adjust, 2, E_idx)

    residue_mask_neighbors = (D_neighbors < 5e9)
    return D_neighbors, residue_mask_neighbors

def rotate(v, R):
    """
    Rotates a vector by a rotation matrix.

    Args:
        v: 3D vector, tensor of shape (length x batch_size x channels x 3)
        R: rotation matrix, tensor of shape (length x batch_size x 3 x 3)

    Returns:
        Rotated version of v by rotation matrix R.
    """
    R = R.unsqueeze(-3)
    v = v.unsqueeze(-1)
    return torch.sum(v * R, dim=-2)


def get_rotation_frames(coords):
    """
    Returns a local rotation frame defined by N, CA, C positions.

    Args:
        coords: coordinates, tensor of shape (batch_size x length x 3 x 3)
        where the third dimension is in order of N, CA, C

    Returns:
        Local relative rotation frames in shape (batch_size x length x 3 x 3)
    """
    v1 = coords[:, :, 2] - coords[:, :, 1]
    v2 = coords[:, :, 0] - coords[:, :, 1]
    e1 = normalize(v1, dim=-1)
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
    e2 = normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-2)
    return R


#### PROTEIN-MPNN UTILS

# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.reshape((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


def backbone_coords_to_frames(x_bb: TensorType["... 4 3", float],
                              atom_mask: TensorType["... 4", float],
                              eps=1e-8):
    """
    Convert backbone coordinates to local frames (rotation + translation) for each residue.
    """
    base_atom_names = ["C", "CA", "N"]
    rigid_group_base_atom37_idx = torch.tensor([rc.bb_atom_order[atom] for atom in base_atom_names])
    base_atom_pos = x_bb[..., rigid_group_base_atom37_idx, :]
    gt_frames = Rigid.from_3_points(
            p_neg_x_axis=base_atom_pos[..., 0, :],
            origin=base_atom_pos[..., 1, :],
            p_xy_plane=base_atom_pos[..., 2, :],
            eps=eps,
    )
    gt_exists = torch.min(atom_mask[..., rigid_group_base_atom37_idx], dim=-1)[0]

    rots = torch.eye(3, dtype=x_bb.dtype, device=x_bb.device)
    rots = torch.tile(rots, (*x_bb.shape[:-2], 1, 1))
    rots[..., 0, 0] = -1
    rots[..., 2, 2] = -1

    rots = Rotation(rot_mats=rots)
    gt_frames = gt_frames.compose(Rigid(rots, None))

    gt_frames_tensor = gt_frames.to_tensor_4x4()

    return gt_frames_tensor, gt_exists


def transform_sidechain_frame(x_scn: TensorType["b n 33 3", float],
                              x_bb: TensorType["b n 4 3", float],
                              atom_mask_scn: TensorType["b n 33", float],
                              atom_mask_bb: TensorType["b n 4", float],
                              to_local: bool) -> Tuple[
                                  TensorType["b n 33 3", float],
                                  TensorType["b n", float]
                              ]:
    """
    Transform sidechain coordinates based on the backbone frame.
    If to_local, transform from global to local frame. Otherwise, transform from local to global frame.
    """
    bb_frames, bb_frames_exists = backbone_coords_to_frames(x_bb, atom_mask_bb)
    T = Rigid.from_tensor_4x4(bb_frames[..., None, :, :])

    if to_local:
        # Transform from global to local frame, ghost atom value is at 0
        x_scn = T.invert_apply(x_scn)
        ghost_atom_value = 0
    else:
        # Transform from local to global frame, ghost atom value is at CA
        x_scn = T.apply(x_scn)
        ca_idx = rc.bb_atom_order["CA"]
        ghost_atom_value = x_bb[..., ca_idx:ca_idx + 1, :]

    x_scn = torch.where(atom_mask_scn[..., None].bool(), x_scn, ghost_atom_value)  # "zero out" ghost atoms and missing atoms
    x_scn = torch.where(bb_frames_exists[..., None, None].bool(), x_scn, ghost_atom_value)  # "zero out" sidechain atoms where backbone frame does not exist

    return x_scn, bb_frames_exists


def process_single_pdb(data):
    example = {}

    # Use raw coordinates
    x = data["all_atom_positions"]  # [n, a, 3]
    atom_mask = data["all_atom_mask"]  # [n, a]
    seq_mask = data["seq_mask"]  # [n]
    x = x * atom_mask[..., None]  # we first ensure missing & ghost atoms are zeroed out

    # per-channel mask for x, used for loss.
    # We only mask out missing atoms from PDB files, not ghost atoms.
    x_mask = rearrange(1 - data["missing_atom_mask"], "n a -> n a 1").expand_as(x)

    # Construct example
    example["x"] = x * atom_mask[..., None]
    example["seq_mask"] = seq_mask
    example["x_mask"] = x_mask
    example["residue_index"] = data["residue_index"]
    example["chain_index"] = data["chain_index"]
    example["aatype"] = data["aatype"]  # not one-hot encoded
    example["ghost_atom_mask"] = data["ghost_atom_mask"]
    example["missing_atom_mask"] = data["missing_atom_mask"]
    example["atom_mask"] = atom_mask
    example["seq_unk_mask"] = (data["aatype"] == rc.restype_order_with_x["X"])
    example['interface_residue_mask'] = data['interface_residue_mask']
    example['chain_ids'] = data['chain_ids']

    # Convert data types
    example_out = {}

    for k, v in example.items():
        if k in FEATURES_LONG:
            example_out[k] = v.long()
        else:
            example_out[k] = v.float()

    return example_out