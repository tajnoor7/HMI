## Not necessary to import


from transformers import pipeline
import torch
import re

# Optional device support (CPU or GPU)
device = 0 if torch.cuda.is_available() else -1

# Named Entity Recognition model
_ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", device=device)

# Event extractor (simple rule-based placeholder)
def extract_events(text):
    events = []
    # Very naive approach using verbs after "to"
    for match in re.findall(r"\bto (\w+)", text):
        if match.lower() not in events:
            events.append(match.lower())
    return events[:3]  # limit to top 3

# Relationship extractor (placeholder for a real IE model)
def extract_relationships(text, entities):
    relationships = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            relationships.append({
                "source": entities[i]['word'],
                "action": "related to",
                "target": entities[j]['word']
            })
    return relationships[:3]  # limit for simplicity

def enrich_with_info_extraction(results):
    for r in results:
        body = r.get("body", "")
        entities = _ner(body)
        people = list({e['word'] for e in entities if e['entity_group'] == "PER"})
        r["actors"] = people
        r["events"] = extract_events(body)
        r["relationships"] = extract_relationships(body, entities)
    return results
