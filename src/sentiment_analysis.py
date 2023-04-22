import stanza
from ner import get_entity_stanza, clean_entities

# sentiment using stanza, argument is text and list of extracted characters
def sentiment_stanza(text, characters_extracted):
    #get sentiments using stanza for each sentence in text
    nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
    doc = nlp(text)
    #create dictionary with key = characters_extracted and value empty list
    stanzaEntities = {key: [] for key in characters_extracted}

    for noSent, sent in enumerate(doc.sentences):
        # get sentecens sentiment and because stanza returns 0,1,2 so subtract -1 to get real sentiment values
        sentiment = sent.sentiment-1
        # convert object to list of dictionary
        sentence = sent.to_dict()
        #for each word in sentence check if it equals to some of the characters_extracted
        for i, word in enumerate(sentence):
            for entity in characters_extracted:
                if word['text'].lower() == entity:
                    #for key = character extracted(if present in sentence), value=that sentence sentiment
                    stanzaEntities[entity].append(sentiment)

    return stanzaEntities

if __name__ == "__main__":
    f = open("../data/shortStories/cinderella.txt", "r", encoding='utf-8')
    text = f.read()

    # NER: list of character extracted
    entitiesStanza = get_entity_stanza(text)
    rezStanza = clean_entities(entitiesStanza)
    print(rezStanza)
    ##### ['cinderwench', 'cinderella', 'king', 'y—es', 'fairy', 'queen', 'charlotte', 'prince']

    #sentiment using stanza
    sentiment = sentiment_stanza(text,rezStanza)
    print(sentiment)
    ##### {'cinderwench': [0, 0, -1], 'cinderella': [0, 1, -1, 1, 0, 1, -1, -1, 1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], 'king': [0, 0, 1, 1, -1, -1, 0, 0, 0], 'y—es': [-1], 'fairy': [0, 1, 0], 'queen': [1], 'charlotte': [0, -1], 'prince': [0, 0, 0, 0, 0]}