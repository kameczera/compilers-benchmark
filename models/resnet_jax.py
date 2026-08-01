
# -*- coding: utf-8 -*-
from __future__ import annotations

import jax
import jax.numpy as jnp

def load_resnet18_from_flaxmodels(input_shape=(1, 224, 224, 3), dtype=jnp.float32):
    """Instancia ResNet18 do pacote flaxmodels. Retorna (apply_fn, variables, dummy_input)."""
    x = jnp.ones(input_shape, dtype)
    rng = jax.random.PRNGKey(0)

    last_err = None
    try:
        from flaxmodels.resnet import ResNet18  # type: ignore
        model = ResNet18(
            output="logits",
            pretrained=None,
            normalize=False,
            num_classes=1000,
            dtype="float32",
        )
        variables = model.init(rng, x, train=False)
        def apply_fn(vs, inp):
            return model.apply(vs, inp, train=False)
        return apply_fn, variables, x
    except Exception as e1:
        last_err = e1

    try:
        import flaxmodels as fm  # type: ignore
        model = fm.ResNet18(
            output="logits",
            pretrained=None,
            normalize=False,
            num_classes=1000,
            dtype="float32",
        )
        variables = model.init(rng, x, train=False)
        def apply_fn(vs, inp):
            return model.apply(vs, inp, train=False)
        return apply_fn, variables, x
    except Exception as e2:
        last_err = e2

    raise RuntimeError(
        "Não foi possível instanciar ResNet18 do flaxmodels. Verifique instalação/API.\n"
        f"Erro: {type(last_err).__name__}: {last_err}"
    )


def load_resnet50_from_flaxmodels(input_shape=(1, 224, 224, 3), dtype=jnp.float32):
    """Instancia ResNet50 do pacote flaxmodels. Retorna (apply_fn, variables, dummy_input)."""
    x = jnp.ones(input_shape, dtype)
    rng = jax.random.PRNGKey(0)

    last_err = None
    try:
        from flaxmodels.resnet import ResNet50  # type: ignore
        model = ResNet50(
            output="logits",
            pretrained=None,
            normalize=False,
            num_classes=1000,
            dtype="float32",
        )
        variables = model.init(rng, x, train=False)
        def apply_fn(vs, inp):
            return model.apply(vs, inp, train=False)
        return apply_fn, variables, x
    except Exception as e1:
        last_err = e1

    try:
        import flaxmodels as fm  # type: ignore
        model = fm.ResNet50(
            output="logits",
            pretrained=None,
            normalize=False,
            num_classes=1000,
            dtype="float32",
        )
        variables = model.init(rng, x, train=False)
        def apply_fn(vs, inp):
            return model.apply(vs, inp, train=False)
        return apply_fn, variables, x
    except Exception as e2:
        last_err = e2

    raise RuntimeError(
        "Não foi possível instanciar ResNet50 do flaxmodels. Verifique instalação/API.\n"
        f"Erro: {type(last_err).__name__}: {last_err}"
    )
