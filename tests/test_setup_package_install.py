import pytest

from desktop_env.controllers.setup import _APT_LOCK_FILES, _build_install_packages_command


def test_install_packages_command_covers_packagekit_and_all_apt_locks():
    command = _build_install_packages_command(["jq", "sysstat"], "password")

    assert "systemctl stop packagekit.service" in command
    assert "DPkg::Lock::Timeout=120" in command
    assert "Acquire::Retries=3" in command
    assert "jq sysstat" in command
    for lock_file in _APT_LOCK_FILES:
        assert lock_file in command


def test_install_packages_command_rejects_shell_injection():
    with pytest.raises(ValueError, match="Invalid apt package"):
        _build_install_packages_command(["jq; reboot"], "password")
