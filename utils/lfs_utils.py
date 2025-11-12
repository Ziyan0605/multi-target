import os
import warnings
from typing import Optional


_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def _read_file_prefix(path: str, num_bytes: int = 128) -> bytes:
    """Safely read the first ``num_bytes`` from ``path``.

    The helper opens the file in binary mode so it works for both text and
    binary artefacts.
    """

    with open(path, "rb") as f:
        return f.read(num_bytes)


def is_lfs_pointer(path: str) -> bool:
    """Return ``True`` if ``path`` exists and looks like a Git-LFS pointer."""

    if not os.path.exists(path):
        return False

    try:
        prefix = _read_file_prefix(path, len(_LFS_HEADER))
    except OSError:
        return False

    return prefix.startswith(_LFS_HEADER)


def ensure_not_lfs_pointer(
    path: str,
    hint: Optional[str] = None,
    *,
    strict: bool = True,
) -> bool:
    """Validate that ``path`` is not a Git-LFS pointer.

    Parameters
    ----------
    path:
        File path to check.
    hint:
        Optional instruction string appended to the raised error to guide the
        user on how to fetch the missing artefact.
    strict:
        If ``True`` (default), raise a ``ValueError`` when a Git-LFS pointer is
        detected. When ``False``, emit a warning and return ``False`` so callers
        may decide how to recover.

    Returns
    -------
    bool
        ``True`` when the file exists and is not a pointer. ``False`` is
        returned when the file is a pointer and ``strict`` is ``False``.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file '{path}' does not exist.")

    if is_lfs_pointer(path):
        message = [
            f"The file '{path}' is a Git-LFS pointer and needs to be downloaded via git-lfs.",
        ]
        if hint:
            message.append(hint)
        text = " ".join(message)
        if strict:
            raise ValueError(text)
        warnings.warn(text)
        return False

    return True
