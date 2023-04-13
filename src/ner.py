import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

def get_entity_spacy(story):
    doc = nlp(story)
    entities = []
    for entity in doc.ents:
        if entity.label_ == "PERSON":
            entities.append(entity.text)
    return entities

def clean_entities(entities):
    clean_ent = []
    # add strings if needed
    unwanted = ['--', '\'s']
    #all lower cases
    entities = [x.lower() for x in entities]

    #remove stop words
    for entity in entities:
        x = entity.split()
        e = []
        for word in x:
            if word not in STOP_WORDS:
                for el in unwanted:
                    if el in word:
                        word = word.replace(el, '')
                e.append(word)
        clean_ent.append(' '.join(e))
    #remove duplicates
    clean_ent = list(dict.fromkeys(clean_ent))

    return clean_ent


if __name__ == "__main__":
    #path = "data/shortStories/"
    f = open("../data/shortStories/The Most Dangerous Game.txt", "r")
    text = f.read()
    entities = get_entity_spacy(text)
    rez = clean_entities(entities)
    print(rez)