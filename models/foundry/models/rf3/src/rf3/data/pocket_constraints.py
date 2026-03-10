from __future__ import annotations

import numpy as np
import torch
from atomworks.enums import ChainType
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import get_af3_token_center_masks
from beartype.typing import Any
from biotite.structure import AtomArray

from rf3.data.ground_truth_template import DEFAULT_DISTOGRAM_BINS


def _as_tensor_bool(value: Any, shape: tuple[int, int]) -> torch.Tensor:
    if value is None:
        return torch.zeros(shape, dtype=torch.bool)
    return torch.as_tensor(value, dtype=torch.bool).clone()


def _as_tensor_float(value: Any, shape: tuple[int, ...]) -> torch.Tensor:
    if value is None:
        return torch.zeros(shape, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32).clone()


def _normalize_pocket_constraints(constraints: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not constraints:
        return normalized
    if not isinstance(constraints, list):
        raise TypeError("constraints must be a list")
    for item in constraints:
        if not isinstance(item, dict):
            raise TypeError("constraint entries must be dicts")
        pocket = item.get("pocket")
        if not pocket:
            continue
        if not isinstance(pocket, dict):
            raise TypeError("pocket constraint must be a dict")
        normalized.append(pocket)
    return normalized


def _fallback_chain_id(
    *,
    target_chain_id: str,
    token_chain_ids: np.ndarray,
    token_chain_types: np.ndarray,
    want_non_polymer: bool,
) -> str | None:
    if target_chain_id in set(token_chain_ids.tolist()):
        return target_chain_id

    allowed_types = (
        ChainType.get_non_polymers() if want_non_polymer else ChainType.get_polymers()
    )
    mask = np.isin(token_chain_types, allowed_types)
    unique = np.unique(token_chain_ids[mask])
    if len(unique) == 1:
        return str(unique[0])
    return None


class AddPocketConstraintDistogram(Transform):
    """Convert inference-time pocket constraints into RF3 template features."""

    def __init__(self, distogram_bins: torch.Tensor = DEFAULT_DISTOGRAM_BINS):
        self.distogram_bins = torch.as_tensor(distogram_bins, dtype=torch.float32)

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        constraints = _normalize_pocket_constraints(data.get("constraints"))
        if not constraints:
            return data

        atom_array = data.get("atom_array")
        if not isinstance(atom_array, AtomArray):
            raise TypeError("atom_array is required for pocket constraints")

        center_mask = get_af3_token_center_masks(atom_array)
        token_chain_ids = np.asarray(atom_array.chain_id[center_mask]).astype(str)
        token_res_ids = np.asarray(atom_array.res_id[center_mask]).astype(int)
        token_chain_types = np.asarray(atom_array.chain_type[center_mask])
        n_token = len(token_chain_ids)
        n_bins = int(self.distogram_bins.shape[0]) + 1

        new_has = torch.zeros((n_token, n_token), dtype=torch.bool)
        new_dist = torch.zeros((n_token, n_token, n_bins), dtype=torch.float32)
        new_noise = torch.zeros((n_token,), dtype=torch.float32)

        for pocket in constraints:
            binder_chain_id = str(pocket.get("binder") or "").strip()
            if not binder_chain_id:
                raise ValueError("pocket constraint is missing binder chain id")

            resolved_binder_chain_id = _fallback_chain_id(
                target_chain_id=binder_chain_id,
                token_chain_ids=token_chain_ids,
                token_chain_types=token_chain_types,
                want_non_polymer=True,
            )
            if resolved_binder_chain_id is None:
                raise ValueError(
                    f"Could not resolve binder chain '{binder_chain_id}' from token chain ids {sorted(set(token_chain_ids.tolist()))}"
                )

            binder_mask = token_chain_ids == resolved_binder_chain_id
            binder_indices = np.where(binder_mask)[0]
            if binder_indices.size == 0:
                raise ValueError(f"No binder tokens found for chain '{resolved_binder_chain_id}'")

            contacts = pocket.get("contacts") or []
            if not isinstance(contacts, list) or not contacts:
                raise ValueError("pocket constraint contacts must be a non-empty list")

            max_distance = float(pocket.get("max_distance", 8.0))
            if max_distance <= 0:
                raise ValueError("pocket constraint max_distance must be > 0")

            force = bool(pocket.get("force", False))
            scale = 1.0 if force else 0.5
            threshold_feature = torch.zeros((n_bins,), dtype=torch.float32)
            upper_bin = int(torch.bucketize(torch.tensor(max_distance), self.distogram_bins).item())
            threshold_feature[: upper_bin + 1] = scale

            for contact in contacts:
                if not isinstance(contact, (list, tuple)) or len(contact) != 2:
                    raise ValueError(
                        f"pocket contact must be [chain_id, residue_id], got {contact!r}"
                    )
                contact_chain_id = str(contact[0]).strip()
                contact_res_id = int(contact[1])

                resolved_contact_chain_id = _fallback_chain_id(
                    target_chain_id=contact_chain_id,
                    token_chain_ids=token_chain_ids,
                    token_chain_types=token_chain_types,
                    want_non_polymer=False,
                )
                if resolved_contact_chain_id is None:
                    raise ValueError(
                        f"Could not resolve contact chain '{contact_chain_id}' from token chain ids {sorted(set(token_chain_ids.tolist()))}"
                    )

                residue_mask = (token_chain_ids == resolved_contact_chain_id) & (
                    token_res_ids == contact_res_id
                )
                residue_indices = np.where(residue_mask)[0]
                if residue_indices.size == 0:
                    raise ValueError(
                        f"No residue tokens found for chain '{resolved_contact_chain_id}' residue '{contact_res_id}'"
                    )

                new_noise[binder_indices] = torch.maximum(
                    new_noise[binder_indices],
                    torch.full((binder_indices.size,), max_distance, dtype=torch.float32),
                )
                new_noise[residue_indices] = torch.maximum(
                    new_noise[residue_indices],
                    torch.full((residue_indices.size,), max_distance, dtype=torch.float32),
                )

                for binder_idx in binder_indices.tolist():
                    for residue_idx in residue_indices.tolist():
                        new_has[binder_idx, residue_idx] = True
                        new_has[residue_idx, binder_idx] = True
                        new_dist[binder_idx, residue_idx] = torch.maximum(
                            new_dist[binder_idx, residue_idx], threshold_feature
                        )
                        new_dist[residue_idx, binder_idx] = torch.maximum(
                            new_dist[residue_idx, binder_idx], threshold_feature
                        )

        feats = data.setdefault("feats", {})
        has_shape = (n_token, n_token)
        dist_shape = (n_token, n_token, n_bins)
        old_has = _as_tensor_bool(feats.get("has_distogram_condition"), has_shape)
        old_dist = _as_tensor_float(feats.get("distogram_condition"), dist_shape)
        old_noise = _as_tensor_float(feats.get("distogram_condition_noise_scale"), (n_token,))

        feats["has_distogram_condition"] = torch.logical_or(old_has, new_has)
        feats["distogram_condition"] = torch.maximum(old_dist, new_dist)
        feats["distogram_condition_noise_scale"] = torch.maximum(old_noise, new_noise)
        return data
