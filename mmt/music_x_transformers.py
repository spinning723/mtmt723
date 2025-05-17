import argparse
import logging
import pathlib
import pprint
import sys

import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn
from x_transformers.autoregressive_wrapper import (
    ENTMAX_ALPHA,
    entmax,
    exists,
    top_a,
    top_k,
    top_p,
)
from x_transformers.x_transformers import (
    AbsolutePositionalEmbedding,
    AttentionLayers,
    Decoder,
    TokenEmbedding,
    always,
    default,
    exists,
)

import representation
import utils


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=("sod", "lmd"),
        required=True,
        help="dataset key",
    )
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    # Add weight decay as a command-line argument
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="L2 regularization (weight decay) for optimizer",
    )
    return parser.parse_args(args=args, namespace=namespace)


class MusicTransformerWrapper(nn.Module):
    def __init__(
            self,
            *,
            encoding,
            max_seq_len,
            attn_layers,
            emb_dim=None,
            max_beat=None,
            max_mem_len=0.0,
            shift_mem_down=0,
            emb_dropout=0.0,
            num_memory_tokens=None,
            tie_embedding=False,
            use_abs_pos_emb=True,
            l2norm_embed=False,
    ):
        super().__init__()
        assert isinstance(
            attn_layers, AttentionLayers
        ), "attention layers must be one of Encoder or Decoder"

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        self.encoding = encoding
        self.field_names = encoding["dimensions"]
        self.type_idx = self.field_names.index("type")
        self.beat_idx = self.field_names.index("beat")
        self.position_idx = self.field_names.index("position")
        self.pitch_idx = self.field_names.index("pitch")
        self.duration_idx = self.field_names.index("duration")
        self.instrument_idx = self.field_names.index("instrument")

        n_tokens = encoding["n_tokens"]
        if max_beat is not None:
            beat_dim = self.beat_idx
            n_tokens[beat_dim] = max_beat + 1

        self.l2norm_embed = l2norm_embed
        self.token_emb = nn.ModuleList(
            [
                TokenEmbedding(emb_dim, n, l2norm_embed=l2norm_embed)
                for n in n_tokens
            ]
        )
        self.pos_emb = (
            AbsolutePositionalEmbedding(
                emb_dim, max_seq_len, l2norm_embed=l2norm_embed
            )
            if (use_abs_pos_emb and not attn_layers.has_pos_emb)
            else always(0)
        )

        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = (
            nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        )

        # --- Cross attention blocks ---
        self.cross_attn_pitch = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.cross_attn_duration = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)

        # --- attn_layers is now after cross attention ---
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        self.feedforward = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(emb_dropout)
        )
        self.to_one = nn.Linear(dim, 1, bias=False)

        self.init_()

        self.to_logits = (
            nn.ModuleList([nn.Linear(dim, n) for n in n_tokens])
            if not tie_embedding
            else [lambda t: t @ emb.weight.t() for emb in self.token_emb]
        )

        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(
                torch.randn(num_memory_tokens, dim)
            )

    def init_(self):
        if self.l2norm_embed:
            for emb in self.token_emb:
                nn.init.normal_(emb.emb.weight, std=1e-5)
            nn.init.normal_(self.pos_emb.emb.weight, std=1e-5)
            return
        for emb in self.token_emb:
            nn.init.kaiming_normal_(emb.emb.weight)

    def forward(
            self,
            x,  # shape : (b, n , d)
            return_embeddings=False,
            mask=None,
            return_mems=False,
            return_attn=False,
            mems=None,
            **kwargs,
    ):
        b, seq_len, d = x.shape
        num_mem = self.num_memory_tokens

        # --- Use only type, beat, position, instrument as input ---
        input_idxs = [self.type_idx, self.beat_idx, self.position_idx, self.instrument_idx]
        x_in = sum(
            self.token_emb[i](x[..., i]) for i in input_idxs
        ) + self.pos_emb(x)
        x_in = self.emb_dropout(x_in)
        x_in = self.project_emb(x_in)

        if num_mem > 0:
            mem = repeat(self.memory_tokens, "n d -> b n d", b=b)
            x_in = torch.cat((mem, x_in), dim=1)
            if exists(mask):
                mask = F.pad(mask, (num_mem, 0), value=True)

        # --- Cross attention: pitch as query ---
        pitch_emb = self.token_emb[self.pitch_idx](x[..., self.pitch_idx])  # (b, seq_len, emb_dim)
        pitch_emb_proj = self.project_emb(pitch_emb)
        x2, _ = self.cross_attn_pitch(
            query=pitch_emb_proj,
            key=x_in,
            value=x_in,
            need_weights=False,
            attn_mask=None
        )

        # --- Cross attention: duration as query ---
        duration_emb = self.token_emb[self.duration_idx](x[..., self.duration_idx])  # (b, seq_len, emb_dim)
        duration_emb_proj = self.project_emb(duration_emb)
        x3, _ = self.cross_attn_duration(
            query=duration_emb_proj,
            key=x2,
            value=x2,
            need_weights=False,
            attn_mask=None
        )

        # --- Now pass through attn_layers ---
        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = (
                mems[: self.shift_mem_down],
                mems[self.shift_mem_down:],
            )
            mems = [*mems_r, *mems_l]

        x4, intermediates = self.attn_layers(
            x3, mask=mask, mems=mems, return_hiddens=True, **kwargs
        )
        x4 = self.norm(x4)

        out_one = self.to_one(self.feedforward(x4))  # (B, S+num_mem, 1)
        out_one = out_one[:, -1, :]

        mem, x_remain = x4[:, :num_mem], x4[:, num_mem:]

        out = (
            [to_logit(x_remain) for to_logit in self.to_logits]
            if not return_embeddings
            else x_remain
        )

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = (
                list(
                    map(
                        lambda pair: torch.cat(pair, dim=-2),
                        zip(mems, hiddens),
                    )
                )
                if exists(mems)
                else hiddens
            )
            new_mems = list(
                map(
                    lambda t: t[..., -self.max_mem_len:, :].detach(), new_mems
                )
            )
            return out, new_mems

        if return_attn:
            attn_maps = list(
                map(
                    lambda t: t.post_softmax_attn,
                    intermediates.attn_intermediates,
                )
            )
            return out, attn_maps

        return out_one


class MusicAutoregressiveWrapper(nn.Module):
    def __init__(self, net, encoding, ignore_index=-100, pad_value=0):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

        self.encoding = encoding
        self.field_names = encoding["dimensions"]
        self.type_idx = self.field_names.index("type")
        self.beat_idx = self.field_names.index("beat")
        self.position_idx = self.field_names.index("position")
        self.pitch_idx = self.field_names.index("pitch")
        self.duration_idx = self.field_names.index("duration")
        self.instrument_idx = self.field_names.index("instrument")

        # Get the type codes
        self.sos_type_code = encoding["type_code_map"]["start-of-song"]
        self.eos_type_code = encoding["type_code_map"]["end-of-song"]
        self.son_type_code = encoding["type_code_map"]["start-of-notes"]
        self.instrument_type_code = encoding["type_code_map"]["instrument"]
        self.note_type_code = encoding["type_code_map"]["note"]

        # Get the dimension indices
        self.dimensions = {
            key: encoding["dimensions"].index(key)
            for key in (
                "type",
                "beat",
                "position",
                "pitch",
                "duration",
                "instrument",
            )
        }
        assert self.dimensions["type"] == 0

    @torch.no_grad()
    def generate(
        self,
        start_tokens,  # shape : (b, n, d)
        seq_len,
        eos_token=None,
        temperature=1.0,  # int or list of int
        filter_logits_fn="top_k",  # str or list of str
        filter_thres=0.9,  # int or list of int
        min_p_pow=2.0,
        min_p_ratio=0.02,
        monotonicity_dim=None,
        return_attn=False,
        **kwargs,
    ):
        # No change: keep same input/output format as before
        _, t, dim = start_tokens.shape

        if isinstance(temperature, (float, int)):
            temperature = [temperature] * dim
        elif len(temperature) == 1:
            temperature = temperature * dim
        else:
            assert (
                len(temperature) == dim
            ), f"`temperature` must be of length {dim}"

        if isinstance(filter_logits_fn, str):
            filter_logits_fn = [filter_logits_fn] * dim
        elif len(filter_logits_fn) == 1:
            filter_logits_fn = filter_logits_fn * dim
        else:
            assert (
                len(filter_logits_fn) == dim
            ), f"`filter_logits_fn` must be of length {dim}"

        if isinstance(filter_thres, (float, int)):
            filter_thres = [filter_thres] * dim
        elif len(filter_thres) == 1:
            filter_thres = filter_thres * dim
        else:
            assert (
                len(filter_thres) == dim
            ), f"`filter_thres` must be of length {dim}"

        if isinstance(monotonicity_dim, str):
            monotonicity_dim = [self.dimensions[monotonicity_dim]]
        else:
            monotonicity_dim = [self.dimensions[d] for d in monotonicity_dim]

        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 2:
            start_tokens = start_tokens[None, :, :]

        self.net.eval()
        out = start_tokens
        mask = kwargs.pop("mask", None)

        if mask is None:
            mask = torch.ones(
                (out.shape[0], out.shape[1]),
                dtype=torch.bool,
                device=out.device,
            )

        if monotonicity_dim is not None:
            current_values = {
                d: torch.max(start_tokens[:, :, d], 1)[0]
                for d in monotonicity_dim
            }
        else:
            current_values = None

        instrument_dim = self.dimensions["instrument"]
        type_dim = self.dimensions["type"]
        for _ in range(seq_len):
            x = out[:, -self.max_seq_len :]
            mask = mask[:, -self.max_seq_len :]

            # No change needed here, input/output format is unchanged for .net
            if return_attn:
                logits, attn = self.net(
                    x, mask=mask, return_attn=True, **kwargs
                )
                logits = [l[:, -1, :] for l in logits]
            else:
                logits = [
                    l[:, -1, :] for l in self.net(x, mask=mask, **kwargs)
                ]

            # Enforce monotonicity
            if monotonicity_dim is not None and 0 in monotonicity_dim:
                for i, v in enumerate(current_values[0]):
                    logits[0][i, :v] = -float("inf")

            # Filter out sos token
            logits[0][type_dim, 0] = -float("inf")

            # Sample from the logits
            sample_type = sample(
                logits[0],
                filter_logits_fn[0],
                filter_thres[0],
                temperature[0],
                min_p_pow,
                min_p_ratio,
            )

            # Update current values
            if monotonicity_dim is not None and 0 in monotonicity_dim:
                current_values[0] = torch.maximum(
                    current_values[0], sample_type.reshape(-1)
                )

            # Iterate after each sample
            samples = [[s_type] for s_type in sample_type]
            for idx, s_type in enumerate(sample_type):
                # A start-of-song, end-of-song or start-of-notes code
                if s_type in (
                    self.sos_type_code,
                    self.eos_type_code,
                    self.son_type_code,
                ):
                    samples[idx] += [torch.zeros_like(s_type)] * (
                        len(logits) - 1
                    )
                # An instrument code
                elif s_type == self.instrument_type_code:
                    samples[idx] += [torch.zeros_like(s_type)] * (
                        len(logits) - 2
                    )
                    logits[instrument_dim][:, 0] = -float("inf")  # avoid none
                    sampled = sample(
                        logits[instrument_dim][idx : idx + 1],
                        filter_logits_fn[instrument_dim],
                        filter_thres[instrument_dim],
                        temperature[instrument_dim],
                        min_p_pow,
                        min_p_ratio,
                    )[0]
                    samples[idx].append(sampled)
                # A note code
                elif s_type == self.note_type_code:
                    for d in range(1, dim):
                        # Enforce monotonicity
                        if (
                            monotonicity_dim is not None
                            and d in monotonicity_dim
                        ):
                            logits[d][idx, : current_values[d][idx]] = -float(
                                "inf"
                            )

                        # Sample from the logits
                        logits[d][:, 0] = -float("inf")  # avoid none
                        sampled = sample(
                            logits[d][idx : idx + 1],
                            filter_logits_fn[d],
                            filter_thres[d],
                            temperature[d],
                            min_p_pow,
                            min_p_ratio,
                        )[0]
                        samples[idx].append(sampled)

                        # Update current values
                        if (
                            monotonicity_dim is not None
                            and d in monotonicity_dim
                        ):
                            current_values[d][idx] = torch.max(
                                current_values[d][idx], sampled
                            )[0]
                else:
                    raise ValueError(f"Unknown event type code: {s_type}")

            stacked = torch.stack(
                [torch.cat(s).expand(1, -1) for s in samples], 0
            )
            out = torch.cat((out, stacked), dim=1)
            mask = F.pad(mask, (0, 1), value=True)

            if exists(eos_token):
                is_eos_tokens = out[..., 0] == eos_token

                # Mask out everything after the eos tokens
                if is_eos_tokens.any(dim=1).all():
                    for i, is_eos_token in enumerate(is_eos_tokens):
                        idx = torch.argmax(is_eos_token.byte())
                        out[i, idx + 1 :] = self.pad_value
                    break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)

        if return_attn:
            return out, attn

        return out

    def forward(self, x, return_list=False, **kwargs):
        xi = x[:, :-1]
        xo = x[:, 1:]

        # help auto-solve a frequent area of confusion around input masks in auto-regressive
        # if user supplies a mask that is only off by one from the source sequence, resolve it for them
        mask = kwargs.get("mask", None)
        if mask is not None and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
            kwargs["mask"] = mask

        out = self.net(xi, **kwargs)

        losses = []
        target_one = torch.ones_like(out)

        criterion = nn.MSELoss()
        loss = criterion(out, target_one)

        if return_list:
            return loss, losses
        return loss


class MusicXTransformer(nn.Module):
    def __init__(self, *, dim, encoding, **kwargs):
        super().__init__()
        assert "dim" not in kwargs, "dimension must be set with `dim` keyword"
        transformer_kwargs = {
            "max_seq_len": kwargs.pop("max_seq_len"),
            "max_beat": kwargs.pop("max_beat"),
            "emb_dropout": kwargs.pop("emb_dropout", 0),
            "use_abs_pos_emb": kwargs.pop("use_abs_pos_emb", True),
        }
        self.decoder = MusicTransformerWrapper(
            encoding=encoding,
            attn_layers=Decoder(dim=dim, **kwargs),
            **transformer_kwargs,
        )
        self.decoder = MusicAutoregressiveWrapper(
            self.decoder, encoding=encoding
        )

    @torch.no_grad()
    def generate(self, seq_in, seq_len, **kwargs):
        return self.decoder.generate(seq_in, seq_len, **kwargs)

    def forward(self, seq, mask=None, **kwargs):
        return self.decoder(seq, mask=mask, **kwargs)


def main():
    """Main function."""
    # Parse the command-line arguments
    args = parse_args()

    # Set default arguments
    if args.dataset is not None:
        if args.in_dir is None:
            args.in_dir = pathlib.Path(f"data/{args.dataset}/processed/notes")

    # Set up the logger
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Load the encoding
    encoding = representation.load_encoding(args.in_dir / "encoding.json")

    # Create the model
    model = MusicXTransformer(
        dim=64,
        encoding=encoding,
        depth=3,
        heads=4,
        max_seq_len=256,
        max_beat=64,
        rel_pos_bias=True,  # relative positional bias
        rotary_pos_emb=True,  # rotary positional encoding
        emb_dropout=0.1,
        attn_dropout=0.1,
        ff_dropout=0.1,
    )

    # Summarize the model
    n_parameters = sum(p.numel() for p in model.parameters())
    n_trainables = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Number of parameters: {n_parameters}")
    print(f"Number of trainable parameters: {n_trainables}")

    # Create test data
    seq = torch.randint(0, 4, (1, 1024, 6))
    mask = torch.ones((1, 1024)).bool()

    # Pass test data through the model
    # --------- REGULARIZATION IN OPTIMIZER -----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=args.weight_decay)
    # --------------------------------------------------
    loss = model(seq, mask=mask)
    loss.backward()


if __name__ == "__main__":
    main()