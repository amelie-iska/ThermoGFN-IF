import torch
import torch.nn as nn


from model.zoidberg.atom_encoder import AtomEncoder
from model.zoidberg.local_atom_attention import LocalAtomAttention
from model.zoidberg.atom_to_node_embedder import AtomToNodeEmbedder
from model.zoidberg.node_to_atom_embedder import NodeToAtomEmbedder
from model.zoidberg.transition_block import TransitionBlock
from model.zoidberg.utils import (
    gather_residue_average_from_atoms,
    transform_residue_index,
    StartEndPad,
)


class Zoidberg(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        encoder_hidden_dim: int,
        num_blocks: int,
        num_heads: int,
        k: int,
        num_tfmr_layers: int,
        num_tfmr_heads: int,
        max_distance_ang: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.k = k
        self.num_tfmr_layers = num_tfmr_layers
        self.num_tfmr_heads = num_tfmr_heads
        self.max_distance_ang = max_distance_ang
        self.layers = nn.ModuleDict()
        self.layers["atom_encoder"] = AtomEncoder(hidden_dim, encoder_hidden_dim)
        self.layers["start_end_pad"] = StartEndPad(hidden_dim)

        for i in range(num_blocks):
            self.layers[f"local_atom_attention_{i}"] = LocalAtomAttention(
                hidden_dim, num_heads, k, max_distance_ang=max_distance_ang
            )
            self.layers[f"atom_to_node_embedder_{i}"] = AtomToNodeEmbedder(hidden_dim)
            self.layers[f"tfmr_block_{i}"] = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_tfmr_heads,
                    dim_feedforward=num_tfmr_heads * 64,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                ),
                num_layers=num_tfmr_layers,
            )
            if i < num_blocks - 1:
                self.layers[f"node_to_atom_embedder_{i}"] = NodeToAtomEmbedder(
                    hidden_dim
                )
                self.layers[f"transition_block_{i}"] = TransitionBlock(
                    hidden_dim, input_dim=hidden_dim
                )

    def forward(self, batch_dict: dict, timestep: torch.Tensor):
        atom_x = self.layers["atom_encoder"](batch_dict, timestep)
        unique_residue_index = transform_residue_index(
            batch_dict["residue_index"], batch_dict["not_pad_mask"]
        )

        # Calculate
        for i in range(self.num_blocks):
            atom_x = self.layers[f"local_atom_attention_{i}"](
                atom_x,
                batch_dict["batch_index"],
                batch_dict["position"],
                batch_dict["not_pad_mask"],
            )
            residue_x, residue_mask = self.layers[f"atom_to_node_embedder_{i}"](
                atom_x,
                batch_dict["is_center"],
                unique_residue_index,
                batch_dict["not_pad_mask"],
            )
            residue_x = self.layers[f"tfmr_block_{i}"](
                residue_x, src_key_padding_mask=~residue_mask
            )  # Double check masking
            if i < self.num_blocks - 1:
                # TODO: Is it better to do residue -> all atoms in residue update
                # TODO: Or residue -> center atom update
                atom_x = self.layers[f"node_to_atom_embedder_{i}"](
                    atom_x,
                    residue_x,
                    unique_residue_index,
                    batch_dict["not_pad_mask"],
                )
                atom_x = self.layers[f"transition_block_{i}"](atom_x)

        # Lastly, just grab information for the protein residues
        # Need to pad start and end tokens (learnable)
        protein_mask, _ = gather_residue_average_from_atoms(
            batch_dict["is_protein"].float().unsqueeze(-1),
            unique_residue_index,
            batch_dict["not_pad_mask"],
        )
        protein_mask = protein_mask.squeeze(-1) > 0.99
        protein_mask = residue_mask * protein_mask
        x = self.layers["start_end_pad"](residue_x, protein_mask)
        return x


if __name__ == "__main__":
    import torch
    from data.all_atom_parse import get_example_batch

    batch_dict = get_example_batch().__dict__
    batch_dict = {
        k: torch.from_numpy(v) for k, v in batch_dict.items() if k != "atom_name"
    }
    batch_dict = {
        k: v.float() if v.dtype == torch.float64 else v for k, v in batch_dict.items()
    }
    timestep = torch.rand_like(batch_dict["batch_index"].float())

    model = Zoidberg(
        hidden_dim=32,
        encoder_hidden_dim=64,
        num_blocks=3,
        num_heads=4,
        k=4,
        num_tfmr_layers=2,
        num_tfmr_heads=4,
    )
    out = model(batch_dict, timestep)
    print(out.shape)
