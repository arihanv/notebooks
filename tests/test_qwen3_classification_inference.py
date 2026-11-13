# Unsloth Notebooks - Notebooks for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

TEMPLATE = Path(__file__).resolve().parents[1] / "original_template/Qwen3_(1.7B)-Classification.ipynb"


@pytest.mark.parametrize("num_labels, expected", [(2, [1, 0]), (77, [76, 75, 74])])
def test_classify_returns_available_predictions(num_labels, expected):
    cells = json.loads(TEMPLATE.read_text(encoding="utf-8"))["cells"]
    sources = ("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")
    source = next(source for source in sources if "def classify(" in source)
    classify = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "classify"
    )

    logits = torch.arange(num_labels, dtype=torch.float32).unsqueeze(0)
    model = Mock(device="cpu", return_value=SimpleNamespace(logits=logits))
    tokenizer = Mock()
    tokenizer.return_value.to.return_value = {}
    namespace = {
        "torch": torch, "model": model, "tokenizer": tokenizer,
        "max_seq_length": 128, "id2label": {i: str(i) for i in range(num_labels)},
    }
    exec(compile(ast.Module(body=[classify], type_ignores=[]), str(TEMPLATE), "exec"), namespace)

    (predictions,) = namespace["classify"](["Where is my card?"])
    assert [label for label, _ in predictions] == [str(i) for i in expected]
