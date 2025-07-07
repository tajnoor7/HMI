import re

def normalize_subject(subj):
    """
    Strip leading Re:, Fwd:, etc., and bracketed tags.
    """
    if not subj:
        return None
    # Remove any number of “Re:”, “Fwd:”, “[tag]” prefixes
    return re.sub(
        r'^(?:(?:re|fw|fwd)\s*:|\s*\[.*?\]\s*)+',
        '',
        subj,
        flags=re.IGNORECASE
    ).strip()

def parse_email(text):
    """
    Parse raw email text, extracting headers, body, reply/forward flags,
    and the raw In-Reply-To header for later threading.
    Returns a dict ready for DB insertion.
    """
    def find(pattern):
        m = re.search(pattern, text, re.MULTILINE)
        return m.group(1).strip() if m else None

    # ——— Headers ———
    message_id         = find(r'^Message-ID:\s*<(.+?)>')
    in_reply_to_hdr    = find(r'^In-Reply-To:\s*<(.+?)>')
    sender             = find(r'^From:\s*(.+)')
    receiver           = find(r'^To:\s*(.+)')
    cc                 = find(r'^Cc:\s*(.+)')
    bcc                = find(r'^Bcc:\s*(.+)')
    subject            = find(r'^Subject:\s*(.+)')
    normalized_subject = normalize_subject(subject)
    date_sent          = find(r'^Date:\s*(.+)')

    # ——— Body & metadata ———
    body = text.split('\n\n', 1)[1].strip() if '\n\n' in text else ''
    has_reply = (
        ('original message' in body.lower()) or
        (subject and subject.lower().startswith('re:'))
    )

    # ——— Forward detection ———
    fwd_pattern = (
        r'^-+\s*Forwarded by\s+(.+?)\s+on\s+'
        r'(\d{2}/\d{2}/\d{4}\s*\d{1,2}:\d{2}\s*(?:AM|PM))'
    )
    fwd_match = re.search(fwd_pattern, text, re.MULTILINE)
    is_forwarded   = bool(fwd_match)
    forwarded_by   = fwd_match.group(1).strip() if fwd_match else None
    forwarded_date = fwd_match.group(2).strip() if fwd_match else None

    return {
        'message_id':        message_id,
        'in_reply_to_hdr':   in_reply_to_hdr,
        'sender':            sender,
        'receiver':          receiver,
        'cc':                cc,
        'bcc':               bcc,
        'subject':           subject,
        'normalized_subject':normalized_subject,
        'date_sent':         date_sent,
        'body':              body,
        'has_reply':         has_reply,
        'is_forwarded':      is_forwarded,
        'forwarded_by':      forwarded_by,
        'forwarded_date':    forwarded_date
    }
