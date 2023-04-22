import stanza
from ner import get_entity_stanza, clean_entities


# sentiment using stanza, argument is text and list of extracted characters
def sentiment_stanza(text, characters_extracted):
    # get sentiments using stanza for each sentence in text
    nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
    doc = nlp(text)
    # create dictionary with key = characters_extracted and value empty list
    stanzaEntities = {key: [] for key in characters_extracted}
    textSentiment = []
    for noSent, sent in enumerate(doc.sentences):
        # get sentecens sentiment and because stanza returns 0,1,2 so subtract -1 to get real sentiment values
        sentiment = sent.sentiment - 1
        # convert object to list of dictionary
        sentence = sent.to_dict()
        textSentiment.append(sentiment)
        # for each word in sentence check if it equals to some of the characters_extracted
        for i, word in enumerate(sentence):
            for entity in characters_extracted:
                if word['text'].lower() == entity:
                    # for key = character extracted(if present in sentence), value=that sentence sentiment
                    stanzaEntities[entity].append(sentiment)

    # list of sentiment per character to most frequent values of list
    # for key, value in stanzaEntities.items():
    #     stanzaEntities[key] = listToValue(value)

    # return text sentiment and character sentiment
    return listToValue(textSentiment), stanzaEntities


# list of sentiments per character -> most frequent value of that list as character sentiment
def listToValue(sentimentList):
    if len(sentimentList) == 0:
        return

    counter = 0
    num = sentimentList[0]
    for i in sentimentList:
        curr_frequency = sentimentList.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i
    return num


if __name__ == "__main__":
    f = open("../data/our_data/cinderella.txt", "r", encoding='utf-8')
    text = f.read()

    # NER: list of character extracted
    entitiesStanza = get_entity_stanza(text)
    rezStanza = clean_entities(entitiesStanza)
    print(rezStanza)
    ##### ['cinderella', 'king', 'prince', 'charlotte']

    # sentiment using stanza
    textSentiment ,sentiment = sentiment_stanza(text, rezStanza)
    print(sentiment)
    ##### {'cinderella': [0, 1, -1, 1, 0, 1, -1, 0, 0, 1, 0, -1, -1, 0, 0, -1, 1, 0, 0, 0, -1, 0, 0, 1, 1, 0, 1], 'king': [0, 0, 1, 0, 0, 0, -1, 0, -1], 'prince': [0, 0, 0, 0, 0], 'charlotte': [0, -1]}
    ##### {'cinderella': 0, 'king': 0, 'prince': 0, 'charlotte': 0}
    print(textSentiment)

    # NER: list of character extracted
    char = ["cinderella", "stepmother", "sisters", "godmother", "prince", "king", "queen"]
    # sentiment using stanza
    textSentiment, sentiment = sentiment_stanza(text, char)
    print(sentiment)
    ##### {'cinderella': [0, 1, -1, 1, 0, 1, -1, 0, 0, 1, 0, -1, -1, 0, 0, -1, 1, 0, 0, 0, -1, 0, 0, 1, 1, 0, 1], 'stepmother': [0, 0], 'sisters': [-1, 1, -1, 0, 0, 1, 0, 0, -1, 0, 1, 0, 1], 'godmother': [0, 0, 0, 0, -1, 0, 1, 1, 1, 0, 0, 0, 0, 1], 'prince': [0, 0, 0, 0, 0], 'king': [0, 0, 1, 0, 0, 0, -1, 0, -1], 'queen': [1]}
    ##### {'cinderella': 0, 'stepmother': 0, 'sisters': 0, 'godmother': 0, 'prince': 0, 'king': 0, 'queen': 1}
    print(textSentiment)
