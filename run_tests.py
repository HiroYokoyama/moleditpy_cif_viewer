#!/usr/bin/env python
"""
Helper script to run tests for CIF Viewer plugin.
Automatically configures environment variables (e.g. PYTEST_QT_API) and runs pytest.
If PyQt6 is not installed, it runs only the unit tests (skipping GUI/integration tests).
"""

import os
import sys


def main():
    # Configure Qt backend for pytest-qt
    os.environ["PYTEST_QT_API"] = "pyqt6"

    # Try to import pytest
    try:
        import pytest
    except ImportError:
        print(
            "Error: 'pytest' is required to run the tests. Please install it with 'pip install pytest'."
        )
        sys.exit(1)

    # Check if PyQt6 is installed
    has_pyqt6 = False
    try:
        import PyQt6

        has_pyqt6 = True
    except ImportError:
        pass

    # Build pytest arguments
    args = []

    if not has_pyqt6:
        print("=" * 80)
        print("WARNING: PyQt6 is not installed in the current Python environment.")
        print("Skipping GUI and integration tests (test_plugin_integration.py).")
        print("Running pure unit tests only (test_parser.py, test_pymatgen_parser.py).")
        print("=" * 80)
        args.extend(["tests/test_parser.py", "tests/test_pymatgen_parser.py"])
    else:
        print("PyQt6 detected. Running the full test suite...")
        # Add any command line arguments passed to this script
        if len(sys.argv) > 1:
            args.extend(sys.argv[1:])
        else:
            args.append("tests/")

    # Run pytest programmatically
    result = pytest.main(args)
    sys.exit(result)


if __name__ == "__main__":
    main()
