import errno
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from para_joint_solver.visualization import inspector as module


@pytest.fixture
def startup(monkeypatch):
    inspector = Mock()
    monkeypatch.setattr(module, "Inspector", Mock(return_value=inspector))
    browser = Mock()
    monkeypatch.setattr(module.webbrowser, "open", browser)
    solver = SimpleNamespace(mechanism=SimpleNamespace(id="test"))
    return solver, inspector, browser


@pytest.mark.parametrize("port,occupied", [(8765, 0), (8765, 2), (9000, 1), (0, 0), (65534, 1)])
def test_startup_skips_busy_ports_and_uses_bound_url(monkeypatch, capsys, startup, port, occupied):
    solver, inspector, browser = startup
    server = MagicMock()
    server.__enter__.return_value = server
    server.server_port = port + occupied if port else 45678
    factory = Mock(side_effect=[OSError(errno.EADDRINUSE, "Address already in use")]
                   * occupied + [server])
    monkeypatch.setattr(module, "HTTPServer", factory)

    module.serve(solver, port=port)

    assert [call.args[0] for call in factory.call_args_list] == [
        ("127.0.0.1", port + i) for i in range(occupied + 1)
    ]
    url = f"http://127.0.0.1:{server.server_port}/"
    browser.assert_called_once_with(url)
    assert url in capsys.readouterr().out
    server.serve_forever.assert_called_once_with()
    server.__exit__.assert_called_once()
    inspector.renderer.close.assert_called_once_with()


@pytest.mark.parametrize("port,error", [
    (8765, errno.EACCES), (8765, errno.EADDRNOTAVAIL),
    (65535, errno.EADDRINUSE), (0, errno.EADDRINUSE),
])
def test_startup_preserves_other_errors_and_closes_renderer(monkeypatch, startup, port, error):
    solver, inspector, browser = startup
    failure = OSError(error, "Cannot bind")
    factory = Mock(side_effect=failure)
    monkeypatch.setattr(module, "HTTPServer", factory)

    with pytest.raises(OSError) as caught:
        module.serve(solver, port=port)

    assert caught.value is failure
    factory.assert_called_once()
    browser.assert_not_called()
    inspector.renderer.close.assert_called_once_with()


@pytest.mark.parametrize("port", [-1, 65536])
def test_invalid_port_is_rejected_before_renderer_creation(startup, port):
    solver, _, browser = startup
    with pytest.raises(ValueError, match="Port must be between 0 and 65535"):
        module.serve(solver, port=port)
    module.Inspector.assert_not_called()
    browser.assert_not_called()
