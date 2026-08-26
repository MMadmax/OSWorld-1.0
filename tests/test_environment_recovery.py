import pytest

from lib_run_single import RecoverableEnvironmentError, _require_screenshot


def test_require_screenshot_accepts_bytes():
    assert _require_screenshot({"screenshot": b"image"}, "test") == b"image"


@pytest.mark.parametrize("obs", [None, {}, {"screenshot": None}, {"screenshot": b""}])
def test_require_screenshot_rejects_missing_desktop_image(obs):
    with pytest.raises(RecoverableEnvironmentError, match="restart the environment"):
        _require_screenshot(obs, "step 2")
