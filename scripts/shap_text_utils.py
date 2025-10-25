"""Utility helpers for working with SHAP text explanations."""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np


def _ensure_list_of_tokens(tokens_data: Any) -> List[List[str]]:
    """Normalise SHAP token metadata to a list of token sequences.

    SHAP's text explainers may return ``data`` either as a flat sequence of tokens
    (when explaining a single example) or as a sequence of token sequences (when
    multiple examples are processed). This helper converts both representations
    into a consistent ``List[List[str]]`` form so downstream consumers can index
    tokens per example safely.
    """

    if isinstance(tokens_data, np.ndarray):
        # ``tolist`` preserves nested structure for object arrays, which is what
        # SHAP uses when returning per-example token sequences.
        tokens_data = tokens_data.tolist()

    if isinstance(tokens_data, (list, tuple)):
        if not tokens_data:
            return []
        first_item = tokens_data[0]
        if isinstance(first_item, (str, bytes)):
            # We are dealing with a single example: ``tokens_data`` is already the
            # token sequence for that example.
            return [list(tokens_data)]
        return [list(item) for item in tokens_data]

    # Fall back to representing the payload as a single-token example. This is a
    # defensive branch and should rarely trigger, but it keeps the caller from
    # crashing in edge cases where SHAP returns an unexpected type.
    return [[str(tokens_data)]]


def normalise_text_explanations(
    values: Sequence[Any] | np.ndarray,
    tokens_data: Any,
) -> Tuple[np.ndarray, List[List[str]]]:
    """Convert SHAP text explanation outputs into a predictable structure.

    SHAP's ``Text`` masker collapses the leading dimension of the returned arrays
    when only a single example is explained. Depending on the number of output
    labels, ``values`` can therefore have rank 1 (single example, single label),
    rank 2 (either ``examples × tokens`` or ``labels × tokens``), or rank 3
    (``examples × labels × tokens``). This function restores the missing example
    dimension when necessary and pairs the resulting array with token sequences
    per example.

    Args:
        values: The ``values`` attribute from a :class:`shap.Explanation`.
        tokens_data: The ``data`` attribute from the same explanation.

    Returns:
        A tuple ``(values_array, tokens_per_example)`` where ``values_array`` is a
        numpy array with shape ``(num_examples, ...)`` and ``tokens_per_example``
        is a list containing the token sequence for each explained example.

    Raises:
        RuntimeError: If the shapes cannot be aligned into a supported format.
    """

    tokens_per_example = _ensure_list_of_tokens(tokens_data)

    values_array = np.asarray(values)
    if values_array.ndim == 0:
        values_array = values_array.reshape(1, 1)
    elif values_array.ndim == 1:
        values_array = values_array.reshape(1, -1)
    elif values_array.ndim == 2:
        if len(tokens_per_example) == values_array.shape[0]:
            # Shape already corresponds to ``examples × tokens``.
            pass
        elif len(tokens_per_example) == 1 and values_array.shape[-1] == len(
            tokens_per_example[0]
        ):
            # Single example but multiple output labels; restore the example
            # dimension so callers can index ``values_array[example_idx]``.
            values_array = values_array.reshape(1, *values_array.shape)
        else:
            raise RuntimeError(
                "Cannot align SHAP values with tokens: "
                f"values shape {values_array.shape}, "
                f"tokens length {len(tokens_per_example)}."
            )
    elif values_array.ndim == 3:
        # ``examples × labels × tokens`` – nothing to do.
        pass
    else:
        raise RuntimeError(
            "Unexpected SHAP values rank "
            f"{values_array.ndim}; expected rank between 0 and 3."
        )

    if len(tokens_per_example) != values_array.shape[0]:
        raise RuntimeError(
            "Mismatch between number of examples in SHAP values "
            f"({values_array.shape[0]}) and tokens ({len(tokens_per_example)})."
        )

    return values_array, tokens_per_example

