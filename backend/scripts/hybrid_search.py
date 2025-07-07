import os
import sqlite3
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import pipeline
import torch
from scripts.info_extraction import enrich_with_info_extraction
from utils.entity_extraction import extract_actors, extract_events, extract_relationships

# Force single-process mode to prevent semaphore leaks
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

DB_PATH     = 'db/emails.db'
EMBED_MODEL = 'all-MiniLM-L6-v2'
LEX_LIMIT   = 10
SEM_LIMIT   = 10

# preload model
_embedder = SentenceTransformer(EMBED_MODEL)

_summarizer = None

# Classification labels
_LABELS = [
    "legal", "financial", "project discussion",
    "human resources", "operations", "general"
]

_classifier = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        device = 0 if torch.cuda.is_available() else -1
        _summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=device
        )
    return _summarizer

def get_classifier():
    global _classifier
    if _classifier is None:
        device = 0 if torch.cuda.is_available() else -1
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device
        )
    return _classifier

# in‐memory caches
_ALL = {
    'ids': None,
    'subjects': None,
    'bodies': None,
    'embeddings': None
}

def _strip_forwarded(text):
    # Removes common forwarded headers and repeated footers
    forwarded_pattern = re.compile(r'-{5,}.*forwarded.*-{5,}', re.IGNORECASE | re.DOTALL)
    split_text = forwarded_pattern.split(text)
    return split_text[0] if split_text else text

def _fast_summarize(text, max_tokens=130):
    summarizer = get_summarizer()

    if not text or len(text.strip()) < 30:
        return text.strip()

    # Clean forwarded content (optional, but improves quality)
    cleaned = _strip_forwarded(text)

    # Truncate overly long texts for summarizer limits (Bart supports ~1024 tokens ≈ 1200-1500 words)
    cleaned = cleaned[:3000]  # ≈ 1000–1200 tokens

    try:
        summary = summarizer(cleaned, max_length=max_tokens, min_length=30, do_sample=False)
        return summary[0]['summary_text'].strip()
    except Exception as e:
        print("Summarization failed:", e)
        return cleaned.strip()

#def _fast_summarize(text, max_sentences=3):
   # sent = re.split(r'(?<=[\.!?])\s+', text.strip())
   # return ' '.join(sent[:max_sentences])

def _ensure_fts_index():
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS email_fts;")
    cur.execute("""
      CREATE VIRTUAL TABLE email_fts
      USING fts4(subject, body, content='emails');
    """)
    cur.execute("""
      INSERT INTO email_fts(rowid, subject, body)
      SELECT id, subject, body FROM emails;
    """)
    conn.commit(); conn.close()

def _load_corpus_embeddings():
    if _ALL['ids'] is not None:
        return
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("SELECT id, subject, body FROM emails;")
    rows = cur.fetchall()
    conn.close()

    _ALL['ids']      = [r[0] for r in rows]
    _ALL['subjects'] = [r[1] for r in rows]
    _ALL['bodies']   = [r[2] for r in rows]

    print(f"🔧 Embedding {_ALL['ids'].__len__()} emails…")
    _ALL['embeddings'] = _embedder.encode(_ALL['bodies'], show_progress_bar=True)

def _load_limited_embeddings(query, top_k=50):
    """
    Load and embed top_k FTS results for semantic comparison.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Create AND-style FTS query
    terms = [t for t in query.strip().split() if t]
    fts_q = ' AND '.join(terms) if len(terms) > 1 else terms[0]

    cur.execute(f"""
      SELECT e.id, e.date_sent, e.subject, e.body, e.sender, e.receiver
        FROM email_fts
        JOIN emails e ON e.id = email_fts.rowid
       WHERE email_fts MATCH ?
       LIMIT ?
    """, (fts_q, int(top_k)))
    rows = cur.fetchall()
    conn.close()

    if rows:
        print("First row sample keys:", rows[0].keys())

    if not rows:
        return {
            'ids': [], 'date_sent': [], 'subjects': [], 'bodies': [],
            'senders': [], 'receivers': [], 'embeddings': np.array([])
        }

    ids       = [r['id'] for r in rows]
    date_sents      = [r['date_sent'] for r in rows]
    subjects  = [r['subject'] for r in rows]
    bodies    = [r['body'] for r in rows]
    senders   = [r['sender'] for r in rows]
    receivers = [r['receiver'] for r in rows]
    embs      = _embedder.encode(bodies, show_progress_bar=False)

    return {
        'ids': ids,
        'date_sents': date_sents,
        'subjects': subjects,
        'bodies': bodies,
        'senders': senders,
        'receivers': receivers,
        'embeddings': embs
    }

def _fts_search(query):
    """
    Lexical search via FTS4 on subject+body, requiring ALL tokens.
    Splits the user’s query on whitespace, then joins with AND so
    only emails containing every term (in subject or body) are returned.
    """
    # Build an AND-style FTS query: "foo bar" → "foo AND bar"
    terms    = [t for t in query.strip().split() if t]
    fts_q    = ' AND '.join(terms) if len(terms) > 1 else terms[0]

    conn     = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur      = conn.cursor()
    cur.execute(f"""
      SELECT e.id, e.subject, e.body
        FROM email_fts
        JOIN emails e ON e.id = email_fts.rowid
       WHERE email_fts MATCH ?
       LIMIT ?
    """, (fts_q, LEX_LIMIT))
    rows = cur.fetchall()
    conn.close()

    return [dict(r) for r in rows]

def _semantic_search(q):
    local_data = _load_limited_embeddings(q, top_k=100)

    # Deduplicate by body content
    seen = set()
    unique_data = {
        'ids': [], 'date_sents': [], 'subjects': [], 'bodies': [],
        'senders': [], 'receivers': []
    }

    for i in range(len(local_data['bodies'])):
        body = local_data['bodies'][i]
        if body not in seen:
            seen.add(body)
            unique_data['ids'].append(local_data['ids'][i])
            unique_data['date_sents'].append(local_data['date_sents'][i])
            unique_data['subjects'].append(local_data['subjects'][i])
            unique_data['bodies'].append(body)
            unique_data['senders'].append(local_data['senders'][i])
            unique_data['receivers'].append(local_data['receivers'][i])

    if not unique_data['bodies']:
        return []

    embs = _embedder.encode(unique_data['bodies'], show_progress_bar=False)
    qv   = _embedder.encode([q])[0].astype('float32')
    mats = np.dot(embs, qv) / (
        np.linalg.norm(embs, axis=1) * np.linalg.norm(qv)
    )
    top = np.argsort(mats)[::-1][:SEM_LIMIT]

    return [
        {
            'id': unique_data['ids'][i],
            'date_sent': unique_data['date_sents'][i],
            'subject': unique_data['subjects'][i],
            'body': unique_data['bodies'][i],
            'sender': unique_data['senders'][i],
            'receiver': unique_data['receivers'][i],
            'sim_score': float(mats[i])
        }
        for i in top
    ]
    # # _load_corpus_embeddings()
    # local_data = _load_limited_embeddings(q, top_k=50)

    # if len(local_data['embeddings']) == 0:
    #     return []  # ✅ Return empty if no results
    
    # qv = _embedder.encode([q])[0].astype('float32')
    # mats = np.dot(local_data['embeddings'], qv) / (
    #     np.linalg.norm(local_data['embeddings'], axis=1) * np.linalg.norm(qv)
    # )
    # # mats = np.dot(_ALL['embeddings'], qv) / (
    # #     np.linalg.norm(_ALL['embeddings'], axis=1) * np.linalg.norm(qv)
    # # )
    # top = np.argsort(mats)[::-1][:SEM_LIMIT]
    # return [
    #     {
    #         'id': local_data['ids'][i],
    #         'subject': local_data['subjects'][i],
    #         'body': local_data['bodies'][i],
    #         'sender': local_data['senders'][i],
    #         'receiver': local_data['receivers'][i],
    #         'sim_score': float(mats[i])
    #     }
    #     for i in top
    # ]
    
    # # return [
    # #     {'id':_ALL['ids'][i],
    # #      'subject':_ALL['subjects'][i],
    # #      'body':_ALL['bodies'][i],
    # #      'sim_score': float(mats[i])}
    # #     for i in top
    # # ]

def _summarize_and_categorize(results):
    # summaries
    for r in results:
        r['summary'] = _fast_summarize(r.get('body',''))
    # clustering
    texts = [r['summary'] for r in results]
    if len(texts) > 1:
        embs   = _embedder.encode(texts)
        k      = min(len(texts), 3)
        # km     = KMeans(n_clusters=k, random_state=0).fit(embs)
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(embs)
        labels = km.labels_
    else:
        labels = [0]*len(texts)
    for r,lab in zip(results, labels):
        r['category'] = int(lab)

    # 3. Zero-shot classification (LAZY LOAD 🔥)
    # classifier = get_classifier()
    for r in results:
        text = r.get('body', '') or r.get('summary', '')
    #     out  = classifier(text, _LABELS)
    #     r['classification'] = out['labels'][0]
    #     print(f"[CLASSIFY] {r['subject']} → {out['labels'][0]}")


        # NEW ADDITIONS:
        r['actors'] = extract_actors(text)
        r['events'] = extract_events(text)
        r['relationships'] = extract_relationships(text)

    return results

# Public API
def search_emails(query):
    """
    Run hybrid search on `query`, returning a dict:
    {
      'lexical': [ {id,subject,body,summary,category}, … ],
      'semantic':[ {id,subject,body,sim_score,summary,category}, … ]
    }
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("No emails.db found; run ingestion first.")

    # build FTS index once
    _ensure_fts_index()

    # lexical & semantic
    # lex = _fts_search(query)
    sem = _semantic_search(query)

    # on-demand enrich only these hits
    # lex = _summarize_and_categorize(lex)
    sem = _summarize_and_categorize(sem)
    sem = enrich_with_info_extraction(sem)

    return {'semantic': sem}
    # return {'lexical': lex, 'semantic': sem}

# Clean up function to call at program exit
def cleanup():
    """Call this function when the program exits to clean up resources"""
    global _embedder, _classifier
    if _embedder is not None:
        del _embedder
    if _classifier is not None:
        del _classifier