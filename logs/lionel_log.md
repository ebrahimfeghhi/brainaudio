### Lionel's work log
- 2025-10-06: Trying large embedder, with the original time-masked transformer, hparams held constant
    1. config: `tm_transformer_b2t_24+25_WideEmb`
        - Wider embedder with `embed_mlp_ratio = 2` $\;\Longrightarrow\;$ killed: loss high, PER rate for Frank really high and not ideal for Sergey
    2. config: `tm_transformer_b2t_24+25_sWideEmb`
        - Wider embedder with `embed_mlp_ratio = 1.5` $\;\Longrightarrow\;$ finished training: Bad results. The idea is that the wider embedder might just overfit the data.