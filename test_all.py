#!/usr/bin/env python
"""
test_all.py - Test runner script for CIF Viewer plugin.
Automatically configures the environment and runs the test suite.
If PyQt6 is not available or if the user passes --unit-only, it will skip Qt integration tests.
"""

import os
import sys


def main():
    # 1. Configure Qt backend for pytest-qt
    os.environ["PYTEST_QT_API"] = "pyqt6"

    # 2. Set QPA platform to offscreen by default for headless operation if not already configured
    if "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    # 3. Import pytest
    try:
        import pytest
    except ImportError:
        print(
            "Error: 'pytest' is required to run tests. Please install it with: pip install pytest"
        )
        sys.exit(1)

    # 4. Check for command-line overrides to run unit tests only
    unit_only = False
    pytest_args = []

    for arg in sys.argv[1:]:
        if arg in ("--unit-only", "--skip-qt", "-u"):
            unit_only = True
        else:
            pytest_args.append(arg)

    # Enable coverage reporting by default unless skipped
    if not any("--cov" in arg for arg in pytest_args):
        pytest_args.extend(["--cov=cif_viewer", "--cov-report=term-missing"])

    # 5. Check if PyQt6 can be imported and initialized
    has_qt = False
    if not unit_only:
        try:
            from PyQt6.QtWidgets import QApplication

            # Attempt to retrieve or create the QApplication instance
            app = QApplication.instance()
            if app is None:
                app = QApplication([])
            has_qt = True
        except Exception as e:
            print(f"Notice: PyQt6/Qt cannot be initialized ({e}).")
            print("Falling back to unit tests only.")

    # 6. Define target files/directories
    unit_test_files = ["tests/test_parser.py", "tests/test_pymatgen_parser.py"]

    # 7. Build pytest arguments list
    args = []
    if unit_only or not has_qt:
        print("=" * 80)
        if unit_only:
            print("Running unit tests only (explicitly requested).")
        else:
            print("PyQt6/Qt GUI environment not available or failed to initialize.")
            print("Skipping GUI and integration tests (test_plugin_integration.py).")
        print("Running unit tests:")
        for f in unit_test_files:
            print(f"  - {f}")
        print("=" * 80)

        args.extend(unit_test_files)
        args.extend(pytest_args)
    else:
        print("=" * 80)
        print("Running the full test suite (including Qt/GUI integration tests)...")
        print("=" * 80)
        if pytest_args:
            args.extend(pytest_args)
        else:
            args.append("tests/")

    # 8. Run pytest
    result = pytest.main(args)
    sys.exit(result)


if __name__ == "__main__":
    main()
