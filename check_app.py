"""Check of the app, not part of the demo: runs app.py without a browser
(streamlit.testing), changes one control after the other and measures how long
the page takes to answer. Writes results/app_timing.txt.

    python check_app.py
"""
import torch  # noqa: F401  keep first: on Windows torch must load before scikit-learn

import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

HERE = Path(__file__).parent
app = AppTest.from_file(str(HERE / "app.py"), default_timeout=300)
lines = []


def measure(label, change=None):
    if change:
        change()
    start = time.perf_counter()
    app.run()
    seconds = time.perf_counter() - start
    errors = [e.value for e in app.exception]
    lines.append(f"{seconds:6.1f} s  {label}  errors: {errors or 'none'}")
    print(lines[-1], flush=True)


# app.sidebar exists only after the first run, so it is looked up inside the lambdas.
def button(label):
    return next(b for b in app.button if b.label == label)


measure("first load: qsar-biodeg, 0 added columns, 50 rows (TabPFN v2 always runs)")
measure("added columns 0 -> 10", lambda: app.sidebar.slider[0].set_value(10))
measure("added columns 10 -> 30", lambda: app.sidebar.slider[0].set_value(30))
measure("training rows 50 -> 200", lambda: app.sidebar.slider[1].set_value(200))
measure("table -> sylvine (30 added, 200 rows)", lambda: app.sidebar.selectbox[0].set_value("sylvine"))
measure("walkthrough: draw a new table", lambda: button("Draw a new table").click())
measure("walkthrough: 30 columns", lambda: app.select_slider(key="w_d").set_value(30))
measure("walkthrough: 20 unrelated", lambda: app.select_slider(key="w_unrelated").set_value(20))
measure("walkthrough: 3 hidden layers, sin in layer 3", lambda: (app.select_slider(key="w_depth").set_value(3),
                                                                app.select_slider(key="w_act3").set_value("sin")))
measure("walkthrough: prior A", lambda: app.radio[0].set_value("all"))
assert app.select_slider(key="w_unrelated_a").value == 0 and app.select_slider(key="w_unrelated_a").disabled
measure("walkthrough: prior A, draw a new table", lambda: button("Draw a new table").click())
assert app.select_slider(key="w_d").value == app.session_state["walk"]["d"]
measure("walkthrough: next step", lambda: button("Next ▶").click())
measure("walkthrough: draw new training rows", lambda: button("Draw new training rows").click())
measure("walkthrough: move the star", lambda: next(s for s in app.slider if s.label.startswith("x1")).set_value(1.5))
(HERE / "results" / "app_timing.txt").write_text("\n".join(lines) + "\n")
