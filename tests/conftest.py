import os
import pytest

@pytest.fixture(autouse=True)
def clean_settings_json():
    """
    Autouse fixture to ensure cif_viewer/settings.json is removed
    before and after every test, preventing it from polluting the workspace.
    """
    settings_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "cif_viewer",
        "settings.json"
    )
    
    # Clean up before test
    if os.path.exists(settings_path):
        try:
            os.remove(settings_path)
        except Exception:
            pass

    yield

    # Clean up after test
    if os.path.exists(settings_path):
        try:
            os.remove(settings_path)
        except Exception:
            pass
