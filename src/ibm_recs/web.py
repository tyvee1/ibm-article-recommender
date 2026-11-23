"""A small local web app on top of ArticleRecommender.

Run it from the repository folder:

    python -m ibm_recs.web

then open http://127.0.0.1:8050. Pages: recommendations for a user id,
similar articles for an article id, and a title search.

Configuration (environment variables, all optional):

    IBM_RECS_DATA         interaction CSV (default data/user-item-interactions.csv)
    IBM_RECS_DATA_SHA256  required hash of that file (default: the course's file)
    IBM_RECS_HOST         address to listen on (default 127.0.0.1, this machine only)
    IBM_RECS_PORT         port (default 8050)
    IBM_RECS_DEBUG        "1" turns on Flask debug mode; never set it beyond your
                          own machine, because the debugger runs typed-in code

The app reads the CSV once at startup, after checking its hash, and keeps the
fitted model in memory. It only answers GET requests and never writes files.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import sys
import time
import uuid

from flask import Flask, abort, g, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
from werkzeug.serving import WSGIRequestHandler, make_server

from . import __version__
from .recommender import ArticleRecommender, file_sha256, load_interactions

COURSE_DATA_SHA256 = "07df24977199e629339723b644a93b8a6452d3a5c70868f7b0a143b14ab59bc6"
MAX_QUERY = 100        # characters in a title search
MAX_ID_DIGITS = 9      # an id longer than this is not an id
RESULTS = 10           # recommendations per list
ID_PATTERN = re.compile(rf"^[0-9]{{1,{MAX_ID_DIGITS}}}$")

log = logging.getLogger("ibm_recs.web")


def settings(environ=None):
    """Read the configuration from the environment, with safe defaults."""
    env = os.environ if environ is None else environ
    return {
        "data": env.get("IBM_RECS_DATA", os.path.join("data", "user-item-interactions.csv")),
        "data_sha256": env.get("IBM_RECS_DATA_SHA256", COURSE_DATA_SHA256),
        "host": env.get("IBM_RECS_HOST", "127.0.0.1"),
        "port": int(env.get("IBM_RECS_PORT", "8050")),
        "debug": env.get("IBM_RECS_DEBUG", "") == "1",
    }


def parse_id(raw):
    """Return the id in a query value as an int, or None when it is not one.

    Only 1 to MAX_ID_DIGITS plain digits are accepted: no sign, spaces,
    decimals or anything longer, so the value is checked before any lookup.
    """
    if raw is None or not ID_PATTERN.match(raw):
        return None
    return int(raw)


def create_app(model=None, config=None):
    """Build the Flask app.

    INPUT:
    model - a fitted ArticleRecommender (tests pass one in); when None, the
            CSV named in config is hash-checked, read and fitted here
    config - settings() output; defaults to the environment

    OUTPUT:
    app - the Flask application. Raises ValueError at startup when the data
          file is missing or its hash differs, so a bad file never serves.
    """
    config = settings() if config is None else config
    data_sha = None
    if model is None:
        if not os.path.isfile(config["data"]):
            raise ValueError(f"data file not found: {config['data']}")
        data_sha = file_sha256(config["data"])
        model = ArticleRecommender().fit(load_interactions(config["data"], expected_sha256=config["data_sha256"]))
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=16 * 1024, JSON_SORT_KEYS=False)
    app.debug = bool(config["debug"])
    app.extensions["recommender"] = model

    @app.before_request
    def start():
        g.request_id = uuid.uuid4().hex[:12]
        g.started = time.perf_counter()
        if request.method not in ("GET", "HEAD"):
            abort(405)

    @app.after_request
    def finish(response):
        response.headers["X-Request-Id"] = g.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self' 'unsafe-inline'"
        response.headers["Referrer-Policy"] = "no-referrer"
        # Log the route and timing, never the raw query text.
        log.info("%s %s %s %.1fms", g.request_id, request.path, response.status_code, 1000 * (time.perf_counter() - g.started))
        return response

    @app.errorhandler(HTTPException)
    def http_error(error):
        return render_template("error.html", code=error.code, message=error.description, request_id=g.get("request_id", "-")), error.code

    @app.errorhandler(Exception)
    def server_error(error):
        log.exception("request %s failed", g.get("request_id", "-"))
        return render_template("error.html", code=500, message="Something went wrong on our side.", request_id=g.get("request_id", "-")), 500

    @app.get("/")
    def home():
        return render_template("home.html", top=model.top_articles(RESULTS))

    @app.get("/user")
    def user():
        user_id = parse_id(request.args.get("id"))
        if user_id is None:
            abort(400, description="A user id is a whole number, for example 1.")
        known = model.has_user(user_id)
        return render_template(
            "user.html",
            user_id=user_id,
            known=known,
            read=[(a, model.title(a)) for a in model.user_articles(user_id)] if known else [],
            recs=model.recommend_for_user(user_id, RESULTS),
        )

    @app.get("/article")
    def article():
        article_id = parse_id(request.args.get("id"))
        if article_id is None:
            abort(400, description="An article id is a whole number, for example 1429.")
        if not model.has_article(article_id):
            abort(404, description=f"No article {article_id} in this data.")
        return render_template(
            "article.html",
            article_id=article_id,
            title=model.title(article_id),
            by_title=model.similar_by_title(article_id, RESULTS),
            by_readers=model.similar_by_readers(article_id, RESULTS),
        )

    @app.get("/search")
    def search():
        query = request.args.get("q", "")
        if len(query) > MAX_QUERY:
            abort(400, description=f"Searches are limited to {MAX_QUERY} characters.")
        needle = query.strip().lower()
        hits = [(a, model.title(a)) for a in model.view_ranking if needle and needle in model.title(a).lower()][:20]
        return render_template("search.html", query=query, hits=hits)

    @app.get("/health")
    def health():
        return jsonify(
            status="ok",
            version=__version__,
            data_sha256=data_sha,
            users=len(model.user_ids),
            articles=len(model.article_ids),
        )

    return app


class QuietRequestHandler(WSGIRequestHandler):
    """Werkzeug's handler without its access log, which records the full
    query string (what the user typed). The app logs path, status and time."""

    def log_request(self, *args, **kwargs):
        pass


def create_server(app, config):
    """A WSGI server bound to config's host and port (not yet serving)."""
    return make_server(config["host"], config["port"], app, threaded=False, request_handler=QuietRequestHandler)


def main():
    """Start the app; Ctrl-C stops it with exit code 130 and no traceback."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # Ctrl-C must stop the server even when the shell started it with the
    # interrupt signal ignored (as background jobs are). Werkzeug's
    # serve_forever swallows KeyboardInterrupt, so the handler also records
    # that it fired, and the exit code says the run was interrupted.
    interrupted = []

    def on_interrupt(signum, frame):
        interrupted.append(signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, on_interrupt)
    config = settings()
    try:
        app = create_app(config=config)
        server = create_server(app, config)
    except (ValueError, OSError) as error:
        print(f"Could not start: {error}", file=sys.stderr)
        return 2
    print(f"Serving on http://{config['host']}:{config['port']}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    if interrupted:
        print("\nStopped.", flush=True)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
