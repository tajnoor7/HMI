import sqlite3
from tqdm import tqdm
import pandas as pd

# 1️⃣ Summarization
from transformers import pipeline

# 2️⃣ Embedding & Clustering
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

DB_PATH = './db/emails.db'
SUMMARY_MODEL = 'facebook/bart-large-cnn'
EMBED_MODEL   = 'all-MiniLM-L6-v2'
NUM_CLUSTERS  = 5  # Can tune this

def add_columns():
    """Add summary & category columns if they don’t exist."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("PRAGMA table_info(emails)")
    cols = {row[1] for row in cur.fetchall()}
    if 'summary' not in cols:
        cur.execute("ALTER TABLE emails ADD COLUMN summary TEXT")
    if 'category' not in cols:
        cur.execute("ALTER TABLE emails ADD COLUMN category INTEGER")
    conn.commit()
    conn.close()

def fetch_emails():
    """Load id & body for emails that still need summaries."""
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("""
        SELECT id, body
          FROM emails
         WHERE summary IS NULL OR summary = ''
    """, conn)
    conn.close()
    return df

def persist_summary(df):
    """Write back summaries into the DB."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    for _, row in df.iterrows():
        cur.execute("UPDATE emails SET summary = ? WHERE id = ?", 
                    (row['summary'], int(row['id'])))
    conn.commit()
    conn.close()

def run_summarization():
    """Generate abstractive summaries for each email body."""
    summarizer = pipeline("summarization", model=SUMMARY_MODEL)
    df = fetch_emails()
    if df.empty:
        print("✅ No emails pending summary.")
        return

    print(f"📝 Summarizing {len(df)} emails with {SUMMARY_MODEL}…")
    summaries = []
    for text in tqdm(df['body'], total=len(df)):
        # truncate to model’s max length (adjust as needed)
        chunk = text[:1000]
        out   = summarizer(chunk, max_length=60, min_length=20, truncation=True)
        summaries.append(out[0]['summary_text'])

    df['summary'] = summaries
    persist_summary(df)
    print("✅ Summaries saved.")

def run_categorization():
    """Embed summaries (or bodies) and cluster into NUM_CLUSTERS topics."""
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT id, summary FROM emails WHERE summary IS NOT NULL", conn)
    conn.close()

    if df.empty:
        print("⚠️ No summaries found—run summarization first.")
        return

    print(f"🔍 Embedding {len(df)} summaries with {EMBED_MODEL}…")
    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(df['summary'].tolist(), show_progress_bar=True)

    print(f"🎯 Clustering into {NUM_CLUSTERS} topics…")
    km = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
    labels = km.fit_predict(embeddings)
    df['category'] = labels

    # Persist back
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    for _, row in df.iterrows():
        cur.execute("UPDATE emails SET category = ? WHERE id = ?", 
                    (int(row['category']), int(row['id'])))
    conn.commit()
    conn.close()
    print("✅ Categories saved.")