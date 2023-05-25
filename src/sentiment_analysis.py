import stanza
from ner import get_entity_stanza, clean_entities
from afinn import Afinn
from nltk.tokenize import sent_tokenize
afn = Afinn()

# sentiment using stanza, argument is text and list of extracted characters
def sentiment_one_character_stanza(doc, characters, offset):
    # create dictionary with key = characters and value empty list
    stanzaEntities = {key: [] for key in characters}

    for i in range(len(doc.sentences)):
        for character in characters:
            if is_character_in_sent(doc.sentences, i, character, offset):
                sentiment = calculate_sent_sentiment(doc.sentences, i, offset)
                stanzaEntities[character].append(sentiment)

    # list of sentiment per character to most frequent values of list
    for key, value in stanzaEntities.items():
        stanzaEntities[key] = listToValue(value)

    # return character sentiment
    return  stanzaEntities

# list of sentiments per character -> most frequent value of that list as character sentiment
def listToValue(sentimentList):
    if len(sentimentList) == 0:
        return
    d = {-1: sentimentList.count(-1) * 0.3, 0: sentimentList.count(0) * 0.2, 1: sentimentList.count(1) * 0.5}
    return max(d.items(), key=lambda n: n[1])[0]


def is_character_in_sent(sentences, i, character, offset):
    for j in range(i - offset, i + offset + 1):
        if j >= 0 and j < len(sentences):
            sent = sentences[j].to_dict()
            for word in sent:
                if character == word['text'].lower():
                    return True
    return False

def calculate_sent_sentiment(sentences, i, offset):
    sentiment = []
    for j in range(i - offset, i + offset + 1):
        if j >= 0 and j < len(sentences):
            sentiment.append(sentences[j].sentiment - 1)

    return listToValue(sentiment)

def sentiment_multiple_characters_stanza(doc, characters, offset):
    relationships = {}

    for i in range(len(characters)):
        row = {}
        for j in range(i+1, len(characters)):
            row[characters[j]] = []
            for s in range(len(doc.sentences)):
                if is_character_in_sent(doc.sentences, s, characters[i], offset) and is_character_in_sent(doc.sentences, s, characters[j], offset):
                        row[characters[j]].append(calculate_sent_sentiment(doc.sentences, s, offset))
        relationships[characters[i]] = row

    final_relationships = {}
    for key, value in relationships.items():
        final_relationships[key] = []
        for k, v in value.items():
            final_value = listToValue(v)
            if final_value is not None:
                final_relationships[key].append([k, final_value])
    return final_relationships

def eval_afinn(sentences, i, offset):
    text = ""
    if offset == 0:
        score = afn.score(sentences[i])
    else:
        for j in range(i - offset, i + offset + 1):
            if j >= 0 and j<len(sentences):
                text += sentences[j]
        score = afn.score(text)
    if score > 0:
        return 1
    elif score == 0:
        return 0
    else:
        return -1

#function that evaluates the character sentiment based on the sentence in which the character is present
#offset 0: evaluate only current sentence(contains character)
#offset x: evaluate current sentence and sentences +/- x
def sentiment_one_character_afinn(text, characters, offset):
    sentences = sent_tokenize(text)
    sentiments = {key: [] for key in characters}
    for i in range(len(sentences)):
        for character in characters:
            if character in sentences[i].lower():
                score = eval_afinn(sentences, i, offset)
                sentiments[character].append(score)

    # list of sentiment per character to most frequent values of list
    for key, value in sentiments.items():
         sentiments[key] = listToValue(value)
    return sentiments

def sentiment_multiple_characters_afinn(text, characters, offset):
    #split text into sentences
    sentences = sent_tokenize(text)
    relationships = {}

    for i in range(len(characters)):
        row = {}
        for j in range(i+1, len(characters)):
            row[characters[j]] = []
            for s in range(len(sentences)):
                if characters[i] in sentences[s].lower() and characters[j] in sentences[s].lower():
                    score = eval_afinn(sentences, s, offset)
                    row[characters[j]].append(score)
        relationships[characters[i]] = row

    final_relationships = {}
    for key, value in relationships.items():
        final_relationships[key] = []
        for k, v in value.items():
            final_value = listToValue(v)
            if final_value is not None:
                final_relationships[key].append([k, final_value])
    return final_relationships
