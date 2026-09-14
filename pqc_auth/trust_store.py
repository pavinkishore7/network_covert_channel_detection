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

Explicitly OUT of scope: certificate authorities, any signed-key
distribution scheme, key revocation, and rotation-with-continuity -- a
legitimate, planned key rotation by the server looks IDENTICAL to an
attack under this scheme. Accepting a new key for an already-known
``server_id`` requires an explicit, deliberate call to ``force_retrust()``;
there is no automatic "this looks like a safe rotation" path, and this
module does not attempt to build one.
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
    """Wraps one plaintext JSON file mapping ``server_id -> hex public key``.

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

    def _load(self) -> dict[str, str]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return {}
        return json.loads(self.path.read_text())

    def _save(self, data: dict[str, str]) -> None:
        self.path.write_text(json.dumps(data, sort_keys=True, indent=2))

    def get_trusted_key(self, server_id: str) -> bytes | None:
        """The currently-trusted key for ``server_id``, or ``None`` if this
        identity has never been seen before."""
        hex_key = self._load().get(server_id)
        return bytes.fromhex(hex_key) if hex_key is not None else None

    def trust_first_contact(self, server_id: str, public_key: bytes) -> None:
        """Record ``public_key`` as trusted for a ``server_id`` that has
        NEVER been seen before. This is the core TOFU write path -- it is
        deliberately a distinct method from ``force_retrust()``, not the
        same code path with a default, and it refuses (raises) rather than
        silently overwriting if ``server_id`` already has a trusted key:
        that situation is a key CHANGE, which must go through
        ``force_retrust()`` instead so it's always an explicit, deliberate
        action, never an accidental side effect of calling this method
        twice.
        """
        data = self._load()
        if server_id in data:
            raise TrustStoreError(
                f"server_id {server_id!r} already has a trusted key -- "
                "trust_first_contact() refuses to overwrite it; call "
                "force_retrust() if you deliberately intend to accept a "
                "new key for this already-known identity"
            )
        data[server_id] = public_key.hex()
        self._save(data)

    def force_retrust(self, server_id: str, public_key: bytes) -> None:
        """Deliberately accept ``public_key`` as the new trusted key for
        ``server_id``, whether or not one was already recorded. Simulates
        an operator consciously accepting a known key rotation. Nothing in
        this module or in ReauthClient calls this automatically -- a
        caller must invoke it on purpose, exactly once per rotation it
        chooses to accept.
        """
        data = self._load()
        data[server_id] = public_key.hex()
        self._save(data)
