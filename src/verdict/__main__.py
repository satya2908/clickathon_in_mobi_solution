"""Makes ``python -m verdict`` equivalent to the ``verdict`` console script.

The ingest endpoint launches the CLI as ``sys.executable -m verdict``, rather than by name, so it
runs the interpreter it is already running under. Resolving ``verdict`` from PATH instead would
find whichever one happens to be first there, which in a virtualenv-inside-a-container is not
reliably the same installation.
"""

from .cli import main

if __name__ == "__main__":
    main()
