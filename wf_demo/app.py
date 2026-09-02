import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")  # pages import matplotlib via pathoptim

import streamlit as st

from wf_demo.dependency_check import check_dependencies


st.set_page_config(
    page_title="DISTINGUISH",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    [data-testid="stMainBlockContainer"] {
        width: min(80vw, 100%);
        max-width: none;
    }
    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"] {
            width: 100%;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def heal_cv2_path_leak():
    # A failed OpenCV import leaks its package dir into sys.path (the restore
    # in cv2/__init__ never runs on failure). That entry shadows PET's
    # top-level `misc` package with `cv2/misc` and breaks `import pipt`.
    leaked = [
        p for p in sys.path
        if os.path.basename(os.path.normpath(p)) == "cv2"
        and os.path.isfile(os.path.join(p, "misc", "__init__.py"))
    ]
    for p in leaked:
        sys.path.remove(p)


heal_cv2_path_leak()
_startup_problems = check_dependencies()
if _startup_problems:
    st.error(
        "Application dependencies failed to import — the app cannot start. "
        "Fix the following and reload:\n\n"
        + "\n\n".join(f"- {p}" for p in _startup_problems)
    )
    st.stop()


pages = [
    st.Page("earth_model_page.py", title="earth-model", url_path="earth-model", default=True),
    st.Page("geosteering.py", title="geosteering", url_path="geosteering"),
]

st.navigation(pages).run()
