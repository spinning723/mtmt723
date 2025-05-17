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

        n_tokens = encoding["n_tokens"]
        if max_beat is not None:
            beat_dim = encoding["dimensions"].index("beat")
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

        # indices for fields
        self.type_idx = encoding["dimensions"].index("type")
        self.beat_idx = encoding["dimensions"].index("beat")
        self.position_idx = encoding["dimensions"].index("position")
        self.pitch_idx = encoding["dimensions"].index("pitch")
        self.duration_idx = encoding["dimensions"].index("duration")
        self.instrument_idx = encoding["dimensions"].index("instrument")

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
            cross_query=None,  # <-- pass pitch/duration embedding here
            **kwargs,
    ):
        b, _, d = x.shape
        num_mem = self.num_memory_tokens

        # Only embed type, beat, position, instrument (indices 0,1,2,5)
        input_fields = [self.type_idx, self.beat_idx, self.position_idx, self.instrument_idx]
        x_in = sum(
            self.token_emb[i](x[..., i]) for i in input_fields
        ) + self.pos_emb(x)
        x_in = self.emb_dropout(x_in)
        x_in = self.project_emb(x_in)

        # Prepare cross-attention query: embed pitch and duration (indices 3,4)
        if cross_query is None:
            pitch_emb = self.token_emb[self.pitch_idx](x[..., self.pitch_idx])
            duration_emb = self.token_emb[self.duration_idx](x[..., self.duration_idx])
            cross_query = pitch_emb + duration_emb  # shape: (b, n, emb_dim or dim)
            cross_query = self.project_emb(cross_query)

        if num_mem > 0:
            mem = repeat(self.memory_tokens, "n d -> b n d", b=b)
            x_in = torch.cat((mem, x_in), dim=1)
            cross_query = torch.cat((mem, cross_query), dim=1)
            if exists(mask):
                mask = F.pad(mask, (num_mem, 0), value=True)

        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = (
                mems[: self.shift_mem_down],
                mems[self.shift_mem_down:],
            )
            mems = [*mems_r, *mems_l]

        # Pass both input and cross_query to attn_layers
        if hasattr(self.attn_layers, "forward_with_cross"):
            x_out, intermediates = self.attn_layers.forward_with_cross(
                x_in, cross_query=cross_query, mask=mask, mems=mems, return_hiddens=True, **kwargs
            )
        else:
            # fallback if attn_layers does not support cross attention
            x_out, intermediates = self.attn_layers(
                x_in, mask=mask, mems=mems, return_hiddens=True, **kwargs
            )

        x_out = self.norm(x_out)

        out_one = self.to_one(self.feedforward(x_out))
        out_one = out_one[:, -1, :]

        mem, x_last = x_out[:, :num_mem], x_out[:, num_mem:]

        out = (
            [to_logit(x_last) for to_logit in self.to_logits]
            if not return_embeddings
            else x_last
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
        temperature=1.0,
        filter_logits_fn="top_k",
        filter_thres=0.9,
        min_p_pow=2.0,
        min_p_ratio=0.02,
        monotonicity_dim=None,
        return_attn=False,
        **kwargs,
    ):
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

            # Prepare cross-attention query (pitch+duration embedding)
            # x: (B, S, D)
            cross_query = None  # Will be handled inside MusicTransformerWrapper

            if return_attn:
                logits, attn = self.net(
                    x, mask=mask, return_attn=True, cross_query=cross_query, **kwargs
                )
                logits = [l[:, -1, :] for l in logits]
            else:
                logits = [
                    l[:, -1, :] for l in self.net(x, mask=mask, cross_query=cross_query, **kwargs)
                ]

            # Enforce monotonicity
            if monotonicity_dim is not None and 0 in monotonicity_dim:
                for i, v in enumerate(current_values[0]):
                    logits[0][i, :v] = -float("inf")

            # Filter out sos token
            logits[0][type_dim, 0] = -float("inf")

            sample_type = sample(
                logits[0],
                filter_logits_fn[0],
                filter_thres[0],
                temperature[0],
                min_p_pow,
                min_p_ratio,
            )

            if monotonicity_dim is not None and 0 in monotonicity_dim:
                current_values[0] = torch.maximum(
                    current_values[0], sample_type.reshape(-1)
                )

            samples = [[s_type] for s_type in sample_type]
            for idx, s_type in enumerate(sample_type):
                if s_type in (
                    self.sos_type_code,
                    self.eos_type_code,
                    self.son_type_code,
                ):
                    samples[idx] += [torch.zeros_like(s_type)] * (
                        len(logits) - 1
                    )
                elif s_type == self.instrument_type_code:
                    samples[idx] += [torch.zeros_like(s_type)] * (
                        len(logits) - 2
                    )
                    logits[instrument_dim][:, 0] = -float("inf")
                    sampled = sample(
                        logits[instrument_dim][idx : idx + 1],
                        filter_logits_fn[instrument_dim],
                        filter_thres[instrument_dim],
                        temperature[instrument_dim],
                        min_p_pow,
                        min_p_ratio,
                    )[0]
                    samples[idx].append(sampled)
                elif s_type == self.note_type_code:
                    for d in range(1, dim):
                        if (
                            monotonicity_dim is not None
                            and d in monotonicity_dim
                        ):
                            logits[d][idx, : current_values[d][idx]] = -float(
                                "inf"
                            )
                        logits[d][:, 0] = -float("inf")
                        sampled = sample(
                            logits[d][idx : idx + 1],
                            filter_logits_fn[d],
                            filter_thres[d],
                            temperature[d],
                            min_p_pow,
                            min_p_ratio,
                        )[0]
                        samples[idx].append(sampled)
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

        mask = kwargs.get("mask", None)
        if mask is not None and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
            kwargs["mask"] = mask

        # Prepare cross attention query (pitch+duration)
        cross_query = None  # handled in wrapper

        out = self.net(xi, cross_query=cross_query, **kwargs)

        losses = []
        target_one = torch.ones_like(out)

        criterion = nn.MSELoss()
        loss = criterion(out, target_one)

        if return_list:
            return loss, losses
        return loss