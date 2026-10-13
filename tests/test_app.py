from streamlit.testing.v1 import AppTest


def test_app_runs_with_demo_data():
    at = AppTest.from_file("../streamlit_app.py", default_timeout=120).run()
    assert not at.exception
    assert len(at.metric) == 4


def test_classifier_button():
    at = AppTest.from_file("../streamlit_app.py", default_timeout=240).run()
    at.button(key="run_clf").click().run()
    assert not at.exception
    assert "clf" in at.session_state
