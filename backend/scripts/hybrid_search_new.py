import os
import sqlite3
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import pipeline
import torch
from utils.entity_extraction import extract_actors, extract_events, extract_relationships
from scripts.info_extraction import enrich_with_info_extraction
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import google.generativeai as genai

api_key = "AIzaSyD3L5hciawssTHrp4Ec1IHf5IObRcy8KLI"
genai.configure(api_key=api_key)

# Force single-process mode to prevent semaphore leaks
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

DB_PATH     = 'db/emails.db'
EMBED_MODEL = 'all-MiniLM-L6-v2'
LEX_LIMIT   = 10
SEM_LIMIT   = 10
SIMILARITY_THRESHOLD = 0.2  # Semantic similarity cutoff for relevance

_embedder = SentenceTransformer(EMBED_MODEL)
_summarizer = None
_classifier = None

_LABELS = [
    "legal", "financial", "project discussion",
    "human resources", "operations", "general"
]

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

_ALL = {'ids': None, 'subjects': None, 'bodies': None, 'embeddings': None}

def _strip_forwarded(text):
    forwarded_pattern = re.compile(r'-{5,}.*forwarded.*-{5,}', re.IGNORECASE | re.DOTALL)
    split_text = forwarded_pattern.split(text)
    return split_text[0] if split_text else text

def _fast_summarize(text, max_tokens=130):
    summarizer = get_summarizer()
    if not text or len(text.strip()) < 30:
        return text.strip()
    cleaned = _strip_forwarded(text)[:3000]
    try:
        summary = summarizer(cleaned, max_length=max_tokens, min_length=30, do_sample=False)
        if summary and isinstance(summary, list) and 'summary_text' in summary[0]:
            return summary[0]['summary_text'].strip()
        else:
            return cleaned.strip()  # fallback
        # return summary[0]['summary_text'].strip()
    except Exception as e:
        print("Summarization failed:", e)
        return cleaned.strip()

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

def _load_limited_embeddings(query, top_k=100):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
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
    if not rows:
        return {
            'ids': [], 'date_sents': [], 'subjects': [], 'bodies': [],
            'senders': [], 'receivers': [], 'embeddings': np.array([])
        }
    ids = [r['id'] for r in rows]
    date_sents = [r['date_sent'] for r in rows]
    subjects = [r['subject'] for r in rows]
    bodies = [r['body'] for r in rows]
    senders = [r['sender'] for r in rows]
    receivers = [r['receiver'] for r in rows]
    embs = _embedder.encode(bodies, show_progress_bar=False)
    return {
        'ids': ids, 'date_sents': date_sents, 'subjects': subjects, 'bodies': bodies,
        'senders': senders, 'receivers': receivers, 'embeddings': embs
    }

def _semantic_search(query):
    local_data = _load_limited_embeddings(query, top_k=100)
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
    qv = _embedder.encode([query])[0].astype('float32')
    mats = np.dot(embs, qv) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(qv))
    MAX_RESULTS = 5

    filtered_indices = [i for i, score in enumerate(mats) if score >= SIMILARITY_THRESHOLD]
    filtered_indices = sorted(filtered_indices, key=lambda i: mats[i], reverse=True)[:MAX_RESULTS]
    results = [
        {
            'id': unique_data['ids'][i],
            'date_sent': unique_data['date_sents'][i],
            'subject': unique_data['subjects'][i],
            'body': unique_data['bodies'][i],
            'sender': unique_data['senders'][i],
            'receiver': unique_data['receivers'][i],
            'sim_score': float(mats[i])
        }
        for i in filtered_indices
    ]
    return results

def _summarize_and_categorize(results):
    for r in results:
        r['summary'] = _fast_summarize(r.get('body',''))
    texts = [r['summary'] for r in results]
    if len(texts) > 1:
        embs = _embedder.encode(texts)
        k = min(len(texts), 3)
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(embs)
        labels = km.labels_
    else:
        labels = [0]*len(texts)
    for r, lab in zip(results, labels):
        r['category'] = int(lab)
    for r in results:
        text = r.get('body', '') or r.get('summary', '')
        r['actors'] = extract_actors(text)
        r['events'] = extract_events(text)
        r['relationships'] = extract_relationships(text)
    return results

def filter_by_project_actors(results, project_name):
    filtered = []
    pn_lower = project_name.lower()
    for r in results:
        # Only keep emails where any relationship target matches the project
        related_to_project = any(
            pn_lower in rel['target'].lower() for rel in r.get('relationships', [])
        )
        if related_to_project:
            filtered.append(r)
    return filtered

def infer_story_intent(emails):
    text = " ".join(email["summary"].lower() for email in emails)
    
    if any(kw in text for kw in ["fraud", "scam", "mislead", "bribe", "embezzle", "kickback", "phony"]):
        return "fraud"
    elif any(kw in text for kw in ["sale", "offer", "promotion", "discount", "free trial", "deal", "advertise"]):
        return "advertisement"
    elif any(kw in text for kw in ["meeting", "negotiation", "contract", "sign-off", "proposal", "update", "budget"]):
        return "business communication"
    elif any(kw in text for kw in ["threat", "warning", "lawsuit", "compliance", "legal"]):
        return "legal concern"
    else:
        return "general"

def generate_story_narrative(story_emails, project_name):
    actors = sorted(set(actor for email in story_emails for actor in email.get('actors', [])))
    summaries = "\n\n".join(f"- {email['summary']}" for email in story_emails)

    intent = infer_story_intent(story_emails)

    prompt = (
        f"You are a professional investigative journalist. Your job is to analyze internal emails "
        f"related to the project '{project_name}' and generate a report.\n\n"
        f"First, identify the overall INTENT of these emails. Then write a summary story with a title, "
        f"actor list, and a coherent narrative.\n\n"
        f"Emails:\n{summaries}\n\n"
        f"Output format:\n"
        f"Title: <short, informative title — NOT a person’s name>\n"
        f"Actors: <comma separated actors>\n"
        f"Intent: {intent}\n"
        f"Content: <coherent narrative>"
    )

    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(
        prompt,  # just the full string
        generation_config={
            "temperature": 0.3,
            "max_output_tokens": 2048
        }
    )

    # Extract generated content
    print(response, "response")
    story_text = response.text

    # Parse output, for example split by lines and extract Title, Actors, Content
    lines = story_text.split('\n')
    title, actors_line, *content_lines = lines
    title = title.replace("Title:", "").strip()
    actors_line = actors_line.replace("Actors:", "").strip()
    content = "\n".join(content_lines).replace("Content:", "").strip()

    if is_probably_bad_title(title):
        title = f"{project_name}: {intent.capitalize()} Story"

    return {
        "title": title,
        "actors": actors,
        "intent": intent,
        "content": content
    }

def is_probably_bad_title(title):
    return (
        len(title.split()) <= 2 and             # too short (e.g., a name)
        title.istitle() and                     # looks like a proper noun
        not re.search(r'\b(report|issue|update|scandal|story|fraud|deal|review|summary)\b', title.lower())
    )

def group_into_stories(results, project_name, threshold=0.75):
    if not results:
        return []
    summaries = [r['summary'] for r in results]
    summary_embs = _embedder.encode(summaries)
    sim_matrix = cosine_similarity(summary_embs)
    clusters = []
    visited = set()
    for i in range(len(results)):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in range(i+1, len(results)):
            if j in visited:
                continue
            if sim_matrix[i][j] >= threshold:
                cluster.append(j)
                visited.add(j)
        clusters.append(cluster)
    story_groups = []
    for cluster in clusters:
        emails = [results[i] for i in cluster]
        # Filter cluster emails strictly by project relationships
        # filtered_emails = filter_by_project_actors(emails, project_name)
        # if not filtered_emails and len(emails) >= 2:
        #     filtered_emails = emails

        # if not filtered_emails:
        #     continue

        story = generate_story_narrative(emails, project_name)
        story_groups.append({
            "emails": emails,
            "title": story["title"],
            "actors": story["actors"],
            "content": story["content"]
        })
    print(story_groups, 'story groups')
    return story_groups

def search_emails(query):
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("No emails.db found; run ingestion first.")
    _ensure_fts_index()
    sem = _semantic_search(query)
    sem = _summarize_and_categorize(sem)
    sem = enrich_with_info_extraction(sem)
    stories = group_into_stories(sem, query)
    return {'semantic': sem, 'stories': stories, 'query': query}

def cleanup():
    global _embedder, _classifier
    if _embedder is not None:
        del _embedder
    if _classifier is not None:
        del _classifier
