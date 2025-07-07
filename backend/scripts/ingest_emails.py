import os
import sqlite3
from datetime import datetime
from email.utils import parsedate_to_datetime
from scripts.parse_emails import parse_email

EMAIL_DIR = './INEnron'
DB_PATH   = './db/emails.db'


def create_database():
    """
    Ensure the emails table exists with the full schema, including threading,
    CC/BCC, forwarding, reply flags, in_reply_to, and thread_root.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS emails (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name           TEXT,
            message_id          TEXT UNIQUE,
            sender              TEXT,
            receiver            TEXT,
            cc                  TEXT,
            bcc                 TEXT,
            subject             TEXT,
            normalized_subject  TEXT,
            date_sent           TEXT,
            body                TEXT,
            is_reply            BOOLEAN DEFAULT 0,
            is_forwarded        BOOLEAN DEFAULT 0,
            forwarded_by        TEXT,
            forwarded_date      TEXT,
            in_reply_to         TEXT,
            thread_root         TEXT
        )
    ''')
    conn.commit()
    conn.close()


def load_and_buffer():
    """
    Phase 1: Read every file under EMAIL_DIR, parse it, and buffer in memory.
    Date parsing uses email.utils.parsedate_to_datetime to handle RFC-2822 with zones.
    """
    buf = []
    for root, _, files in os.walk(EMAIL_DIR):
        for fname in files:
            if fname.startswith('.'):
                continue
            path = os.path.join(root, fname)
            try:
                raw = open(path, 'r', encoding='utf-8', errors='ignore').read()
            except Exception as e:
                print(f"❌ READ ERROR [{fname}]: {e}")
                continue

            data = parse_email(raw)
            data['file_name'] = os.path.relpath(path, EMAIL_DIR)

            # Robust date parsing
            dt = None
            ds = data.get('date_sent')
            if ds:
                try:
                    dt = parsedate_to_datetime(ds)
                except Exception:
                    import re
                    ds_clean = re.sub(r'\s*\(.*?\)$', '', ds)
                    try:
                        dt = parsedate_to_datetime(ds_clean)
                    except Exception:
                        dt = None
            data['dt'] = dt or datetime.min

            buf.append(data)
    return buf


def compute_threads(buffered):
    """
    Build threads only when an actual “original” (no Re:/Fwd:) exists.
    Otherwise, leave replies unlinked (each is its own thread).
    """
    from collections import defaultdict

    # Group by normalized subject
    groups = defaultdict(list)
    for e in buffered:
        key = e['normalized_subject'] or e['subject'] or '<no-subj>'
        groups[key].append(e)

    for norm_subj, msgs in groups.items():
        # Sort by datetime
        msgs.sort(key=lambda x: x['dt'])

        # 1) Find a true root: subject == normalized_subject
        roots = [m for m in msgs if (m['subject'] or '').strip().lower() == norm_subj.lower()]
        if roots:
            # We have an original — pick the earliest one
            root_msg = min(roots, key=lambda m: m['dt'])
            root_id  = root_msg['message_id'] or f"auto::{root_msg['file_name']}"

            # Assign root & replies
            for m in msgs:
                # Ensure every msg has an ID
                if not m['message_id']:
                    m['message_id'] = f"auto::{m['file_name']}"

                if m is root_msg:
                    m['in_reply_to']  = None
                    m['is_reply']     = False
                else:
                    # header link if present, else direct to root
                    hdr = m.get('in_reply_to_hdr')
                    m['in_reply_to']  = hdr or root_id
                    m['is_reply']     = bool(hdr or True)

                m['thread_root'] = root_id

        else:
            # No actual original found → leave every message as its own thread
            for m in msgs:
                # ensure ID
                if not m['message_id']:
                    m['message_id'] = f"auto::{m['file_name']}"

                # No in_reply_to for pure replies
                m['in_reply_to']  = None
                m['thread_root']  = m['message_id']
                m['is_reply']     = False   # or True, depending on how want to flag this

    return buffered

def commit_buffer(buffered):
    """
    Phase 3: Write the threaded buffer into SQLite, skipping duplicate message_id errors.
    """
    # Ensure schema exists
    create_database()

    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # Optional: wipe out old data so we can re-run cleanly
    cur.execute("DELETE FROM emails")

    stats = {
        'inserted':    0,
        'dup_skipped': 0,
        'err_skipped': 0
    }

    for m in buffered:
        params = {
            'file_name'          : m.get('file_name'),
            'message_id'         : m.get('message_id'),
            'sender'             : m.get('sender'),
            'receiver'           : m.get('receiver'),
            'cc'                 : m.get('cc'),
            'bcc'                : m.get('bcc'),
            'subject'            : m.get('subject'),
            'normalized_subject' : m.get('normalized_subject'),
            'date_sent'          : m.get('date_sent'),
            'body'               : m.get('body'),
            'is_reply'           : m.get('is_reply'),
            'is_forwarded'       : m.get('is_forwarded'),
            'forwarded_by'       : m.get('forwarded_by'),
            'forwarded_date'     : m.get('forwarded_date'),
            'in_reply_to'        : m.get('in_reply_to'),
            'thread_root'        : m.get('thread_root'),
        }
        try:
            cur.execute('''
                INSERT INTO emails (
                    file_name, message_id, sender, receiver,
                    cc, bcc, subject, normalized_subject,
                    date_sent, body, is_reply,
                    is_forwarded, forwarded_by, forwarded_date,
                    in_reply_to, thread_root
                ) VALUES (
                    :file_name, :message_id, :sender, :receiver,
                    :cc, :bcc, :subject, :normalized_subject,
                    :date_sent, :body, :is_reply,
                    :is_forwarded, :forwarded_by, :forwarded_date,
                    :in_reply_to, :thread_root
                )
            ''', params)
            stats['inserted'] += 1

        except sqlite3.IntegrityError:
            # Duplicate message_id – skip it
            stats['dup_skipped'] += 1

        except Exception as e:
            print(f"❌ INSERT ERROR [{m.get('file_name')}]: {e}")
            stats['err_skipped'] += 1

    conn.commit()
    conn.close()

    # Final report
    print("\n===== COMMIT REPORT =====")
    print(f"✔️  Successfully inserted: {stats['inserted']}")
    print(f"⚠️  Duplicates skipped:      {stats['dup_skipped']}")
    print(f"❌  Other errors skipped:    {stats['err_skipped']}")
    print("==========================\n")

