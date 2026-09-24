"""Import liboqs before anything can import TensorFlow.

Importing tensorflow before oqs and then using oqs aborts the process with
a native allocator conflict (free(): invalid pointer); importing oqs first
avoids it (see pqc_auth/README.md, "liboqs and TensorFlow must be imported
in a specific order"). tests/test_cnn_scan_detector.py imports TensorFlow
at collection time, before any liboqs test runs, so the whole suite needs
oqs loaded first. No effect when liboqs isn't installed.
"""

try:
    import oqs  # noqa: F401  # type: ignore[import-not-found]
except (ImportError, RuntimeError, SystemExit):
    pass
