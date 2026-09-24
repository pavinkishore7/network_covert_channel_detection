"""Persisted trust-on-first-use (TOFU) key store for ReauthClient.

Same convention as pqc_auth/dilithium.py's ``OqsDilithiumSigner`` key_path:
a plaintext file on disk, read on startup, written when new state needs
recording. This file is PLAINTEXT and UNAUTHENTICATED AT REST -- anyone
with filesystem access to it can read or edit it, and nothing in this
module detects that. Acceptable only for this project's demo/review
purposes, matching every other plaintext-on-disk mechanism already in
pqc_auth/ (dilithium.py's key_path, audit_log.py's log file) -- this is
not a new convention invented here.

What TOFU is, and is NOT (the same trust model SSH uses for host keys):
  TOFU turns "do I trust this key" into "is this the same key I saw last
  time for this identity". That defends against a server identity being
  silently swapped out AFTER a legitimate first connection -- a later
  active MITM, or a different/compromised server standing in under the
  same ``server_id``. It does NOT defend against an attacker already
  sitting in the path on the VERY FIRST connection ever made to a given
  ``server_id``: that first trust decision is, and remains, entirely
  unverified by anything in this code -- a legitimate first contact and an
  attacker-controlled first contact are indistinguishable here, by
  construction. This is TOFU's own well-known limitation (SSH has the
  identical gap the first time you connect to a new host), not a shortcut
  taken in this implementation. See pqc_auth/README.md for the fuller
  writeup, including how this interacts with public-key pinning.

Signed rotation (pqc_auth/key_rotation.py) is the one automatic key-change
path: a new key is accepted only with a statement signed by the CURRENTLY
pinned key, carrying an epoch above the one stored here. Each entry
therefore records the key AND its epoch (0 for a key learned on first
contact or set by ``force_retrust()`` without one). ``accept_rotation()``
is the only method that moves a pin because of a statement, and it is a
compare-and-set: it refuses unless the stored key is still the one the
statement rotated away from and the new epoch is higher.

Explicitly OUT of scope: certificate authorities, revocation, and recovery
from a compromised old key. A key change WITHOUT a valid rotation
statement still looks identical to an attack and still needs a deliberate
``force_retrust()`` call.
"""

from __future__ import annotations

import json
from pathlib import Path


class TrustStoreError(Exception):
    """Raised on a trust-store operation that would violate TOFU semantics
    (e.g. calling ``trust_first_contact`` for a ``server_id`` that already
    has a trusted key -- that is a key CHANGE, a distinct, explicit path,
    not first contact)."""


class TrustStore:
    """Wraps one plaintext JSON file mapping
    ``server_id -> {"public_key": hex, "epoch": int}``. An entry written
    before rotation existed (a bare hex string) is read as epoch 0.

    A ``server_id`` is a caller-supplied string identifying a logical
    server identity -- deliberately NOT ``host:port``, since a server can
    restart on a different port without its identity actually changing.
    Callers (e.g. ``pqc_auth/demo.py``, ``pqc_auth/live_loop.py``) decide
    what string makes sense for their deployment (e.g.
    ``"pqc_auth-demo-server"``).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, dict]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return {}
        data = json.loads(self.path.read_text())
        return {
            server_id: {"public_key": entry, "epoch": 0} if isinstance(entry, str) else entry
            for server_id, entry in data.items()
        }

    def _save(self, data: dict[str, dict]) -> None:
        self.path.write_text(json.dumps(data, sort_keys=True, indent=2))

    def get_trusted_key(self, server_id: str) -> bytes | None:
        """The currently-trusted key for ``server_id``, or ``None`` if this
        identity has never been seen before."""
        entry = self._load().get(server_id)
        return bytes.fromhex(entry["public_key"]) if entry is not None else None

    def get_epoch(self, server_id: str) -> int | None:
        """The rotation epoch of the trusted key, or ``None`` if unknown."""
        entry = self._load().get(server_id)
        return int(entry["epoch"]) if entry is not None else None

    def trust_first_contact(self, server_id: str, public_key: bytes, epoch: int = 0) -> None:
        """Record ``public_key`` as trusted for a ``server_id`` that has
        NEVER been seen before. This is the core TOFU write path -- it is
        deliberately a distinct method from ``force_retrust()``, not the
        same code path with a default, and it refuses (raises) rather than
        silently overwriting if ``server_id`` already has a trusted key:
        that situation is a key CHANGE, which must go through
        ``force_retrust()`` or ``accept_rotation()`` instead so it's always
        an explicit action, never an accidental side effect of calling this
        method twice.

        ``epoch`` is whatever the first response claimed for its key (see
        ReauthClient) and is exactly as unverified as the key itself.
        """
        data = self._load()
        if server_id in data:
            raise TrustStoreError(
                f"server_id {server_id!r} already has a trusted key -- "
                "trust_first_contact() refuses to overwrite it; call "
                "force_retrust() if you deliberately intend to accept a "
                "new key for this already-known identity"
            )
        data[server_id] = {"public_key": public_key.hex(), "epoch": int(epoch)}
        self._save(data)

    def force_retrust(self, server_id: str, public_key: bytes, epoch: int = 0) -> None:
        """Deliberately accept ``public_key`` as the new trusted key for
        ``server_id``, whether or not one was already recorded. Simulates
        an operator consciously accepting a key change that came without a
        valid rotation statement. Nothing in this module or in ReauthClient
        calls this automatically -- a caller must invoke it on purpose.
        ``epoch`` is the operator's statement of which rotation epoch the
        key belongs to; the default 0 accepts any later signed rotation.
        """
        data = self._load()
        data[server_id] = {"public_key": public_key.hex(), "epoch": int(epoch)}
        self._save(data)

    def accept_rotation(self, server_id: str, old_key: bytes, new_key: bytes, new_epoch: int) -> None:
        """Move the pin for ``server_id`` from ``old_key`` to ``new_key``
        after a verified rotation chain (pqc_auth/key_rotation.py). The
        caller has already verified the signatures; this re-checks, against
        what is on disk right now, that the pin is still ``old_key`` and
        that ``new_epoch`` is higher than the stored epoch, and raises
        otherwise -- so a stale or concurrent caller can never roll the
        pin or its epoch backwards."""
        data = self._load()
        entry = data.get(server_id)
        if entry is None or bytes.fromhex(entry["public_key"]) != old_key:
            raise TrustStoreError(f"server_id {server_id!r} is no longer pinned to the key this rotation replaces")
        if new_epoch <= int(entry["epoch"]):
            raise TrustStoreError(f"rotation epoch {new_epoch} is not above stored epoch {entry['epoch']}")
        data[server_id] = {"public_key": new_key.hex(), "epoch": int(new_epoch)}
        self._save(data)
