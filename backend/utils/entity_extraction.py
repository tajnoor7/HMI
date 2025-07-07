# import spacy

# nlp = spacy.load("en_core_web_lg")

# def extract_actors(text):
#     doc = nlp(text)
#     actors = set()
#     for ent in doc.ents:
#         if ent.label_ == "PERSON":
#             name = ent.text.strip()
#             if len(name.split()) > 1:
#                 actors.add(name)
#     return list(actors)

# def extract_events(text):
#     # Placeholder: can use verb extraction or keyword spotting
#     doc = nlp(text)
#     events = {token.lemma_ for token in doc if token.pos_ == "VERB"}
#     return list(events)[:5]  # Top 5 verbs as events (customize this)

# def extract_relationships(text):
#     # Placeholder: relationships can be refined later
#     return []


import spacy

# nlp = spacy.load("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")

GARBAGE_NAMES = {
    'bai', 'wall', 'journal', 'chairman', 'article', 'report',
    'reuters', 'manager', 'team', 'staff', 'president',
    'subject', 'hello', 'dear', 're', 'fw', 'regards', 'sincerely',
    'thanks', 'cheers', 'employee', 'group', 'client', 'user',
    'l', 'v', 'c'
}

def clean_actor_name(name):
    name = name.strip()
    if len(name) < 3:
        return None
    if any(char in name for char in "#$%&/\\@!0123456789"):
        return None
    if name.lower() in GARBAGE_NAMES:
        return None
    if len(name.split()) < 2:  # Require full names like "John Doe"
        return None
    return name

def extract_actors(text):
    doc = nlp(text)
    actors = set()
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = clean_actor_name(ent.text)
            if name:
                actors.add(name)
    return sorted(actors)

def extract_events(text):
    doc = nlp(text)
    events = set()
    for token in doc:
        if token.pos_ == "VERB":
            lemma = token.lemma_.lower()
            if lemma not in ["be", "have", "do"]:
                events.add(lemma)
    return list(events)[:5]

def extract_relationships(text):
    doc = nlp(text)
    relationships = []
    for sent in doc.sents:
        subj = None
        verb = None
        obj = None

        for token in sent:
            if token.dep_ in ("nsubj", "nsubjpass") and token.ent_type_ == "PERSON":
                subj = token.text.strip()
            elif token.dep_ == "ROOT" and token.pos_ == "VERB":
                verb = token.lemma_.lower()
            elif token.dep_ in ("dobj", "pobj") and token.ent_type_ in ("PERSON", "ORG", "PRODUCT", "WORK_OF_ART"):
                obj = token.text.strip()

        if subj and verb and obj:
            relationships.append({
                "source": subj,
                "action": verb,
                "target": obj
            })

    return relationships

# def extract_all_entities(text):
#     actors = extract_actors(text)
#     events = extract_events(text)
#     relationships = extract_relationships(text, actors)
#     return {
#         "actors": actors,
#         "events": events,
#         "relationships": relationships
#     }