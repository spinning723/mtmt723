from x_transformers import Decoder, CrossAttender
import torch
import torch.nn as nn

class TransformerCrossWrapper(nn.Module):
    def __init__(
        self,
        *,
        vocab_sizes,        # List[int], length = 6
        field_embed_dims,   # List[int], length = 6
        max_seq_len: int,
        dim: int = None,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        use_abs_pos_emb: bool = True,
        **kwargs
    ):
        super().__init__()
        assert len(vocab_sizes) == 6 and len(field_embed_dims) == 6

        # Embeddings for first five fields
        self.input_embeds = nn.ModuleList([
            nn.Embedding(vocab_sizes[i], field_embed_dims[i])
            for i in range(5)
        ])
        self.input_embed_dim = sum(field_embed_dims[:5])

        # Embedding for the query field (6th field)
        self.query_embed = nn.Embedding(vocab_sizes[5], field_embed_dims[5])

        # Model dimension
        if dim is None:
            dim = self.input_embed_dim
        assert dim == self.input_embed_dim, "Sum of input field dims must equal model dim"
        self.decoder = Decoder(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            use_abs_pos_emb=use_abs_pos_emb,
            **kwargs
        )
        self.pos_emb = nn.Embedding(max_seq_len, dim) if use_abs_pos_emb else None

        # Project decoder output to query embedding dimension for cross attention
        self.project_context = nn.Linear(dim, field_embed_dims[5])

        # Cross attention: query_dim is the embedding dim of the 6th field
        self.cross_attn = CrossAttender(
            dim=field_embed_dims[5],     # query dim
            context_dim=field_embed_dims[5],  # context dim, now matches projected decoder output
            heads=heads,
            dim_head=dim_head,
            depth=1,                     # depth is required by CrossAttender
        )

        # Output head (customize as needed)
        self.to_logits = nn.Linear(field_embed_dims[5], vocab_sizes[5])

    def forward(self, x):
        """
        x: (batch, seq_len, 6)
        First 5 fields: input tokens for decoder
        Last field: query tokens for cross attention
        """
        b, n, f = x.shape
        assert f == 6

        # Embed and concatenate the first 5 fields
        input_emb = [
            emb_layer(x[:,:,i]) for i, emb_layer in enumerate(self.input_embeds)
        ]
        tokens = torch.cat(input_emb, dim=-1)  # (b, n, model_dim)
        if self.pos_emb is not None:
            positions = torch.arange(n, device=x.device).unsqueeze(0).expand(b, n)
            tokens = tokens + self.pos_emb(positions)
        dec_out = self.decoder(tokens)  # (b, n, model_dim)

        # Project decoder output to match the query embedding dimension
        context = self.project_context(dec_out)  # (b, n, query_dim)

        # Embed the last field (query)
        query = self.query_embed(x[:,:,5])  # (b, n, query_dim)

        # Cross attention: query attends to decoder output (context)
        out = self.cross_attn(query, context=context)  # (b, n, query_dim)

        # Output logits if needed
        logits = self.to_logits(out)  # (b, n, vocab_size of query field)
        return logits

if __name__ == "__main__":
    vocab_sizes = [100, 50, 200, 10, 500, 30]
    field_embed_dims = [16, 8, 32, 4, 40, 24]  # last field can be any dim you want
    model = TransformerCrossWrapper(
        vocab_sizes=vocab_sizes,
        field_embed_dims=field_embed_dims,
        max_seq_len=128,
        dim=sum(field_embed_dims[:5]),
    )
    # For demonstration, ensure each field value is within its vocab range
    batch, seq_len = 4, 128
    x = torch.stack([
        torch.randint(0, vocab_sizes[i], (batch, seq_len))
        for i in range(6)
    ], dim=-1)  # shape (batch, seq_len, 6)
    logits = model(x)
    print(logits.shape)  # (4, 128, vocab_sizes[5])