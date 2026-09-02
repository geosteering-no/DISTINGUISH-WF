import importlib


UDAR_PROXY_INSTALL = (
    "python -m pip install --upgrade --force-reinstall "
    "'git+https://github.com/KriFos1/UTA-proxy-final.git"
    "@2fdb9a356ad2cf495c04eb402f272724cd9f00dd'"
)
TORCH_INSTALL = (
    "python -m pip install --upgrade --force-reinstall "
    "'torch==2.13.0' 'torchvision==0.28.0'"
)


def check_dependencies(import_module=importlib.import_module):
    checks = [
        ("geostat.decomp", "Geostatistics prior sampling (geostat)"),
        ("misc.system_tools.environ_var", "PET system tools (misc)"),
        ("pipt.loop.assimilation", "PET data assimilation (pipt)"),
    ]
    problems = []
    for module, label in checks:
        try:
            import_module(module)
        except Exception as error:
            problems.append(
                f"{label} (`{module}`): {type(error).__name__}: {error}"
            )

    try:
        udar_model = import_module("udar_proxi.mymodel")
        if not hasattr(udar_model, "SmoothEMConvModel"):
            raise ImportError("installed revision does not provide SmoothEMConvModel")
    except Exception as error:
        problems.append(
            "UDAR smooth proxy (`udar_proxi.mymodel.SmoothEMConvModel`): "
            f"{type(error).__name__}: {error}. Reinstall pinned dependency with "
            f"`{UDAR_PROXY_INSTALL}`"
        )
        return problems

    try:
        import_module("torchvision")
    except Exception as error:
        problems.append(
            "Torch/Torchvision binary compatibility (`torchvision`): "
            f"{type(error).__name__}: {error}. Reinstall matched wheels with "
            f"`{TORCH_INSTALL}`, then restart Streamlit."
        )
        return problems

    for module, label in (
        ("GeoSim.sim", "GeoSim forward simulator"),
        ("pathoptim.pathOPTIM", "Decision support (pathoptim)"),
    ):
        try:
            import_module(module)
        except Exception as error:
            problems.append(
                f"{label} (`{module}`): {type(error).__name__}: {error}"
            )
    return problems
