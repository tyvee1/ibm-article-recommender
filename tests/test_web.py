"""The web app's safety rules, each with a control that shows the test can fail."""

import shutil

import pytest

from ibm_recs.web import MAX_QUERY, create_app, create_server, settings

XSS = "<script>alert(1)</script>"


@pytest.fixture(scope="module")
def client(model):
    app = create_app(model=model, config=settings({}))
    return app.test_client()


def assert_escaped(client):
    """The search page shows a script tag as text, never as markup."""
    page = client.get("/search", query_string={"q": XSS}).get_data(as_text=True)
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert XSS not in page


def test_user_input_is_escaped(client):
    assert_escaped(client)


def test_escaping_control_fires_when_autoescape_is_off(model):
    app = create_app(model=model, config=settings({}))
    app.jinja_env.autoescape = False
    with pytest.raises(AssertionError):
        assert_escaped(app.test_client())


def test_health_reports_the_model(client):
    body = client.get("/health").get_json()
    assert body["status"] == "ok" and body["users"] == 5149 and body["articles"] == 714


def test_pages_answer_for_known_and_new_users(client):
    known = client.get("/user?id=1")
    new = client.get("/user?id=0")
    assert known.status_code == 200 and "Already read" in known.get_data(as_text=True)
    assert new.status_code == 200 and "most-viewed" in new.get_data(as_text=True)


@pytest.mark.parametrize("bad", ["", "abc", "-1", "1.5", "1 OR 1=1", "1234567890"])
def test_ids_that_are_not_ids_are_refused(client, bad):
    assert client.get("/user", query_string={"id": bad}).status_code == 400


def test_unknown_article_is_404_without_internals(client):
    response = client.get("/article?id=999999")
    page = response.get_data(as_text=True)
    assert response.status_code == 404 and "Traceback" not in page and "Request id" in page


def test_search_length_is_capped_at_the_limit(client):
    assert client.get("/search", query_string={"q": "a" * MAX_QUERY}).status_code == 200  # near miss
    assert client.get("/search", query_string={"q": "a" * (MAX_QUERY + 1)}).status_code == 400


def test_only_get_is_accepted(client):
    assert client.post("/search", data={"q": "x"}).status_code == 405


def test_debug_is_off_unless_asked_for(model):
    assert create_app(model=model, config=settings({})).debug is False
    assert create_app(model=model, config=settings({"IBM_RECS_DEBUG": "1"})).debug is True  # control


def test_listens_on_this_machine_only_by_default(model):
    app = create_app(model=model, config=settings({}))
    server = create_server(app, dict(settings({}), port=0))
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()
    wide = create_server(app, dict(settings({"IBM_RECS_HOST": "0.0.0.0"}), port=0))  # control
    try:
        assert wide.server_address[0] == "0.0.0.0"
    finally:
        wide.server_close()


def test_startup_refuses_a_missing_or_changed_data_file(data_path, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        create_app(config=settings({"IBM_RECS_DATA": str(tmp_path / "nope.csv")}))
    copy = tmp_path / "data.csv"
    shutil.copyfile(data_path, copy)
    with open(copy, "a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="sha256"):
        create_app(config=settings({"IBM_RECS_DATA": str(copy)}))


def test_access_log_does_not_record_what_users_type(model, caplog):
    """The app's log names the path, never the query text; Werkzeug's own
    access log (which records the whole query string) is switched off."""
    import logging
    import threading
    import urllib.request

    from ibm_recs.web import QuietRequestHandler

    app = create_app(model=model, config=settings({}))
    server = create_server(app, dict(settings({}), port=0))
    assert server.RequestHandlerClass is QuietRequestHandler
    thread = threading.Thread(target=server.handle_request)
    thread.start()
    with caplog.at_level(logging.INFO):
        port = server.server_address[1]
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        opener.open(f"http://127.0.0.1:{port}/search?q=secret-words", timeout=10).read()
        thread.join(10)
    server.server_close()
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "/search" in logged and "secret-words" not in logged
