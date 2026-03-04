from __future__ import annotations

"""cer_core.bilanciamento.fingerprint
================================

Utility di hashing deterministico (SHA-256) per supportare:

- *content fingerprinting* di oggetti JSON (es. `scenario.json`) in modo riproducibile;
- hashing di file e testo;
- fingerprint ricorsivo di una directory basato esclusivamente su path relativo e contenuto.

Scelte di progetto
------------------
- L'algoritmo utilizzato è **SHA-256** (hex digest).
- Le funzioni sono *pure* (nessun side-effect), eccetto la lettura da filesystem
  nelle funzioni `sha256_file()` e `sha256_dir_files()`.
- Per JSON: viene usata una serializzazione canonica con:
  - `sort_keys=True` per stabilità rispetto all'ordine delle chiavi;
  - `separators=(',', ':')` per rimuovere whitespace non significativo;
  - `ensure_ascii=False` per preservare UTF-8 senza escape superflui.

Limitazioni note
----------------
- Il fingerprint JSON dipende dalla rappresentazione Python dell'oggetto:
  - liste: l'ordine è significativo;
  - float: eventuali NaN/Infinity non sono rappresentabili in JSON standard;
  - oggetti non serializzabili da `json.dumps` generano eccezione.

Questo modulo è usato dal bilanciamento energetico per il fingerprint di scenario
(escludendo i timestamp di update/creazione), così da ottenere un identificatore
stabile del contenuto logico.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Set


def sha256_bytes(data: bytes) -> str:
    """Calcola l'hash SHA-256 di un blob binario.

    Parameters
    ----------
    data : bytes
        Payload binario.

    Returns
    -------
    str
        Hex digest SHA-256 (64 caratteri).
    """
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_text(text: str, *, encoding: str = "utf-8") -> str:
    """Calcola l'hash SHA-256 di una stringa.

    Parameters
    ----------
    text : str
        Testo in input.
    encoding : str, default "utf-8"
        Encoding usato per convertire il testo in bytes.

    Returns
    -------
    str
        Hex digest SHA-256.
    """
    return sha256_bytes(text.encode(encoding))


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Calcola l'hash SHA-256 del contenuto di un file.

    Parameters
    ----------
    path : pathlib.Path
        Path del file.
    chunk_size : int, default 1 MiB
        Dimensione chunk per la lettura streaming.

    Returns
    -------
    str
        Hex digest SHA-256.

    Raises
    ------
    OSError
        Se il file non è leggibile.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_json_canonical(obj: Any) -> str:
    """Calcola un fingerprint SHA-256 di un oggetto serializzato in JSON canonico.

    La canonicalizzazione è ottenuta con:
    - `sort_keys=True` per stabilità rispetto all'ordine delle chiavi;
    - `separators=(',', ':')` per eliminare whitespace non significativo;
    - `ensure_ascii=False` per preservare UTF-8.

    Parameters
    ----------
    obj : Any
        Oggetto JSON-serializzabile (dict/list/scalari).

    Returns
    -------
    str
        Hex digest SHA-256 del JSON canonico.

    Raises
    ------
    TypeError
        Se `obj` non è serializzabile con `json.dumps`.
    ValueError
        Se `json.dumps` fallisce per contenuti non rappresentabili.
    """
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(s)


def sha256_dir_files(
    root: Path,
    *,
    exclude_names: Optional[Set[str]] = None,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Fingerprint ricorsivo di una directory basato su path relativo e contenuto.

    L'hash risultante dipende unicamente da:
    - path relativo (normalizzato con separatore `/`)
    - contenuto binario dei file

    Non dipende da mtime/ctime, permessi, owner.

    Parameters
    ----------
    root : pathlib.Path
        Directory radice.
    exclude_names : Optional[Set[str]], default None
        Insieme di nomi file da escludere (match su `Path.name`).
    chunk_size : int, default 1 MiB
        Dimensione chunk per la lettura streaming dei file.

    Returns
    -------
    str
        Hex digest SHA-256 del contenuto complessivo.

    Raises
    ------
    OSError
        Se un file non è leggibile.
    """
    exclude = exclude_names or set()
    files = [p for p in root.rglob("*") if p.is_file() and p.name not in exclude]
    files.sort(key=lambda p: str(p.relative_to(root)).replace("\\", "/"))

    h = hashlib.sha256()
    for p in files:
        rel = str(p.relative_to(root)).replace("\\", "/")
        h.update(rel.encode("utf-8"))
        h.update(b"\0")

        with p.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)

        h.update(b"\0")

    return h.hexdigest()
