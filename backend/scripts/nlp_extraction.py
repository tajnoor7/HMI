## Not necessary to import


import spacy

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")  # can switch to 'en_core_web_trf' for higher accuracy

def extract_actors(text):
    """
    Extract named people and organizations from the email body.
    """
    doc = nlp(text)
    return list(set([ent.text for ent in doc.ents if ent.label_ in ("PERSON", "ORG")]))

def extract_relationships(text):
    """
    Extract (subject, action, object) relationships from sentences.
    """
    doc = nlp(text)
    triples = []
    for sent in doc.sents:
        for token in sent:
            if token.dep_ == "ROOT":
                subject = [w.text for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
                obj = [w.text for w in token.rights if w.dep_ in ("dobj", "pobj", "attr")]
                if subject and obj:
                    triples.append({
                        "source": subject[0],
                        "action": token.text,
                        "target": obj[0]
                    })
    return triples

def extract_events(relationships):
    """
    From the relationships, extract unique actions as 'events'.
    """
    return list(set([rel['action'] for rel in relationships]))
