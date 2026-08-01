# -*- coding: utf-8 -*-
"""Construção dos modelos Transformer usados na comparação entre backends.

Os três backends partem da mesma configuração e dos mesmos pesos aleatórios
(seed fixa): pesos pré-treinados não são necessários para medir compilação e
latência, e evitá-los remove o download do caminho cronometrado. As entradas
são `(batch, seq_len)` de IDs inteiros, como nas tabelas do artigo.
"""

from __future__ import annotations

from typing import Any


BERT_SHAPES = ((1, 64), (1, 128), (8, 128))
GPT2_SHAPES = ((1, 16), (1, 128), (8, 128), (8, 256))
MODEL_SHAPES = {"bert": BERT_SHAPES, "gpt2": GPT2_SHAPES}


def hf_config(model: str, seed: int = 0):
    """Configuração canônica (base) de cada modelo, sem pesos pré-treinados."""
    import transformers

    if model == "bert":
        return transformers.BertConfig()  # bert-base-uncased
    if model == "gpt2":
        config = transformers.GPT2Config()  # gpt2 (124M)
        config.use_cache = False
        return config
    raise ValueError(f"modelo desconhecido: {model}")


def make_persistent_buffers(module):
    """Torna persistentes os buffers que ficam fora do `state_dict`.

    O importador de `ExportedProgram` do TVM procura todo buffer declarado no
    grafo dentro do `state_dict`; buffers não-persistentes (por exemplo
    `bert.embeddings.token_type_ids`) quebram a importação com `KeyError`. A
    troca não altera valores nem semântica, apenas a serialização.
    """
    for submodule in module.modules():
        state = submodule.state_dict()
        for name, buffer in list(submodule.named_buffers(recurse=False)):
            if name not in state:
                submodule.register_buffer(name, buffer, persistent=True)
    return module


def build_torch_model(model: str, seed: int = 0, seq_len: int | None = None):
    """Modelo PyTorch em modo inference, com pesos determinísticos."""
    import torch
    import transformers

    torch.manual_seed(seed)
    config = hf_config(model, seed)
    if model == "bert":
        network = transformers.BertModel(config, add_pooling_layer=False)
    else:
        network = transformers.GPT2Model(config)

    class LastHiddenState(torch.nn.Module):
        """Envelope com uma única saída tensorial, comparável entre backends."""

        def __init__(self, inner, model_name: str, fixed_seq_len: int | None):
            super().__init__()
            self.inner = inner
            self.model_name = model_name
            if model_name == "gpt2":
                if fixed_seq_len is None:
                    raise ValueError("seq_len is required for the fixed GPT-2 mask")
                mask = torch.full(
                    (1, 1, fixed_seq_len, fixed_seq_len),
                    torch.finfo(torch.float32).min,
                )
                self.register_buffer(
                    "fixed_causal_mask", torch.triu(mask, diagonal=1), persistent=True
                )

        def forward(self, input_ids):
            if self.model_name == "gpt2":
                return self.inner(
                    input_ids=input_ids, attention_mask=self.fixed_causal_mask
                ).last_hidden_state
            return self.inner(input_ids=input_ids).last_hidden_state

    # BERT's token_type_ids must be persistent for TVM's importer. GPT-2 has
    # non-persistent boolean causal-mask buffers; forcing those into the state
    # dict makes TVM attempt to bind an unsupported boolean DLPack parameter.
    prepared = make_persistent_buffers(network) if model == "bert" else network
    return LastHiddenState(prepared, model, seq_len).eval()


def torch_inputs(model: str, batch: int, seq_len: int, seed: int = 0):
    import torch

    generator = torch.Generator().manual_seed(seed)
    vocab = hf_config(model).vocab_size
    return torch.randint(0, vocab, (batch, seq_len), generator=generator)


def build_flax_model(model: str, batch: int, seq_len: int, seed: int = 0):
    """Modelo Flax equivalente + parâmetros inicializados, para o caminho XLA."""
    import jax
    import numpy as np
    import transformers

    config = hf_config(model, seed)
    if model == "bert":
        flax_model = transformers.FlaxBertModel(
            config, seed=seed, _do_init=True, add_pooling_layer=False
        )
    else:
        flax_model = transformers.FlaxGPT2Model(config, seed=seed, _do_init=True)

    rng = np.random.default_rng(seed)
    input_ids = jax.numpy.asarray(
        rng.integers(0, config.vocab_size, size=(batch, seq_len), dtype=np.int32)
    )

    # O módulo Flax exige explicitamente as máscaras/posições que a API de alto
    # nível preenche sozinha; passá-las mantém o grafo igual ao do caminho Torch
    # (atenção plena, sem padding) e deixa a chamada compilável por jax.jit.
    ones = jax.numpy.ones((batch, seq_len), dtype="i4")
    positions = jax.numpy.broadcast_to(
        jax.numpy.arange(seq_len, dtype="i4"), (batch, seq_len)
    )
    if model == "bert":

        def forward(params, ids):
            return flax_model.module.apply(
                {"params": params},
                input_ids=ids,
                attention_mask=ones,
                token_type_ids=jax.numpy.zeros_like(ids),
                position_ids=positions,
                deterministic=True,
                return_dict=True,
            ).last_hidden_state

    else:

        def forward(params, ids):
            return flax_model.module.apply(
                {"params": params},
                input_ids=ids,
                attention_mask=ones,
                position_ids=positions,
                deterministic=True,
                return_dict=True,
            ).last_hidden_state

    return forward, flax_model.params, input_ids


def describe(model: str) -> dict[str, Any]:
    config = hf_config(model)
    return {
        "model": model,
        "hidden_size": getattr(config, "hidden_size", getattr(config, "n_embd", None)),
        "num_layers": getattr(
            config, "num_hidden_layers", getattr(config, "n_layer", None)
        ),
        "vocab_size": config.vocab_size,
        "weights": "random_init",
    }
