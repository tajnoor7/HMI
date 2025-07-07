#!/usr/bin/env python
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import atexit

import multiprocessing
import warnings

# Ingestion pipeline
from scripts.ingest_emails import (
    create_database,
    load_and_buffer,
    compute_threads,
    commit_buffer
)

# Search API
from scripts.hybrid_search import search_emails, cleanup


# 🔧 Disable parallelism warnings from HuggingFace tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 🔧 Ensure safe process spawning (fixes semaphore leak warnings on some systems)
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # Already set in current process
    pass

# 🔧 Suppress resource tracker warnings for leaked semaphores at shutdown
warnings.filterwarnings(
    "ignore",
    message="resource_tracker: There appear to be .* leaked semaphore objects"
)

# Register cleanup to run when Flask app shuts down
atexit.register(cleanup)

# Compute the absolute path to db/emails.db
BASE_DIR = os.path.dirname(__file__)
DB_PATH  = os.path.join(BASE_DIR, 'db', 'emails.db')

app = Flask(__name__)
CORS(app)

def startup():
    """
    Run ingestion/threading if the DB doesn't yet exist.
    """
    if not os.path.exists(DB_PATH):
        app.logger.info("🆕 No database found — running ingestion pipeline…")
        create_database()
        buffer = load_and_buffer()
        threaded = compute_threads(buffer)
        commit_buffer(threaded)
    else:
        app.logger.info("✅ Database already exists — skipping ingestion.")

@app.route('/search')
def search():
    """
    Hybrid search endpoint:
      GET /search?q=<query>
    Returns JSON { query, lexical: [...], semantic: [...] }
    """
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify({'error': 'Missing required "q" parameter.'}), 400

    try:
        print("working from here")
        results = search_emails(q)
        return jsonify({
            'query': q,
            # 'lexical': results['lexical'],
            'semantic': results['semantic']
        })
    except Exception as e:
        app.logger.error("Search error", exc_info=e)
        return jsonify({'error': str(e)}), 500

def main():
    try:
        # 1) Ingest/thread upfront if needed
        startup()

        # 2) Launch Flask server
        #    In production, run via gunicorn or uwsgi instead
        app.run(host='0.0.0.0', port=5050, debug=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
        cleanup()
    except Exception as e:
        print(f"❌ Error: {e}")
        cleanup()
        raise

if __name__ == '__main__':
    main()
