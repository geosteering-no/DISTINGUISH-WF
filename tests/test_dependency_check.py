from types import SimpleNamespace

from wf_demo.dependency_check import (
    TORCH_INSTALL,
    UDAR_PROXY_INSTALL,
    check_dependencies,
)


def test_missing_smooth_proxy_reports_one_actionable_root_cause():
    imported = []

    def import_module(name):
        imported.append(name)
        if name == "udar_proxi.mymodel":
            return SimpleNamespace()
        return SimpleNamespace()

    problems = check_dependencies(import_module)

    assert len(problems) == 1
    assert "SmoothEMConvModel" in problems[0]
    assert UDAR_PROXY_INSTALL in problems[0]
    assert "GeoSim.sim" not in imported
    assert "pathoptim.pathOPTIM" not in imported


def test_compatible_proxy_allows_downstream_import_checks():
    imported = []

    def import_module(name):
        imported.append(name)
        if name == "udar_proxi.mymodel":
            return SimpleNamespace(SmoothEMConvModel=object)
        return SimpleNamespace()

    assert check_dependencies(import_module) == []
    assert "GeoSim.sim" in imported
    assert "pathoptim.pathOPTIM" in imported


def test_broken_torchvision_reports_matched_wheel_command_once():
    imported = []

    def import_module(name):
        imported.append(name)
        if name == "udar_proxi.mymodel":
            return SimpleNamespace(SmoothEMConvModel=object)
        if name == "torchvision":
            raise RuntimeError("operator torchvision::nms does not exist")
        return SimpleNamespace()

    problems = check_dependencies(import_module)

    assert len(problems) == 1
    assert "torchvision::nms" in problems[0]
    assert TORCH_INSTALL in problems[0]
    assert "GeoSim.sim" not in imported
    assert "pathoptim.pathOPTIM" not in imported
