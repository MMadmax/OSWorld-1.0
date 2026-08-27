from unittest.mock import Mock, patch

import pytest

from desktop_env.desktop_env import DesktopEnv


def _bare_env(provider_name="docker"):
    env = DesktopEnv.__new__(DesktopEnv)
    env.provider_name = provider_name
    env.setup_controller = Mock()
    env.controller = Mock()
    env.controller.get_screenshot.return_value = b"png"
    return env


def test_docker_xcursor_patch_uploads_helper_and_checks_screenshot():
    env = _bare_env()

    with patch.dict(
        "os.environ",
        {
            "OSWORLD_DOCKER_XCURSOR_PATCH": "1",
            "OSWORLD_DOCKER_XCURSOR_RELOAD_DELAY": "0",
        },
    ):
        env._inject_docker_xcursor_patch()

    files = env.setup_controller._upload_file_setup.call_args.args[0]
    assert len(files) == 1
    assert files[0]["local_path"].endswith("desktop_env/server/pyxcursor.py")
    assert files[0]["path"] == "/home/user/server/pyxcursor.py"
    env.controller.get_screenshot.assert_called_once_with()


def test_docker_xcursor_patch_can_be_disabled():
    env = _bare_env()

    with patch.dict("os.environ", {"OSWORLD_DOCKER_XCURSOR_PATCH": "0"}):
        env._inject_docker_xcursor_patch()

    env.setup_controller._upload_file_setup.assert_not_called()
    env.controller.get_screenshot.assert_not_called()


def test_docker_xcursor_patch_is_not_applied_to_other_providers():
    env = _bare_env(provider_name="vmware")

    env._inject_docker_xcursor_patch()

    env.setup_controller._upload_file_setup.assert_not_called()
    env.controller.get_screenshot.assert_not_called()


def test_docker_xcursor_patch_requires_server_recovery():
    env = _bare_env()
    env.controller.get_screenshot.return_value = None

    with patch.dict(
        "os.environ",
        {"OSWORLD_DOCKER_XCURSOR_RELOAD_DELAY": "0"},
    ):
        with pytest.raises(RuntimeError, match="did not recover"):
            env._inject_docker_xcursor_patch()
