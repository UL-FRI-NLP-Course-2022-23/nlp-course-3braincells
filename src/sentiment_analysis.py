import stanza
from ner import get_entity_stanza, clean_entities
from afinn import Afinn
from nltk.tokenize import sent_tokenize
afn = Afinn()


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
def eval_character(sentences, i, offset):
    text = ""
    if offset == 0:
        score = afn.score(sentences[i])
    else:
        for j in range(i - offset, i + offset + 1):
            if j >= 0 and j<len(sentences):
                text += sentences[j]
        score = afn.score(text)
    return score
def sentiment_one_character_afinn(text, characters, offset):
    sentences = sent_tokenize(text)
    sentiments = {key: [] for key in characters}
    for i in range(len(sentences)):
        for character in characters:
            if character in sentences[i].lower():
                score = eval_character(sentences, i, offset)
                sentiments[character].append(score)

    # list of sentiment per character to most frequent values of list
    for key, value in sentiments.items():
            sentiments[key] = sum(value)*len(value)

    protagonist = max(sentiments, key=sentiments.get)
    antagonist = min(sentiments, key=sentiments.get)
    return (protagonist, antagonist)

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

if __name__ == "__main__":
    f = open("../data/our_data/hansel_and_gretel.txt", "r", encoding='utf-8')
    text = f.read()
    nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
    doc = nlp(text)

    #ground truth character list
    #characters =  ["cinderella", "stepmother", "sisters", "godmother", "prince", "king", "queen"]
    characters = ["hansel", "gretel", "stepmother", "father", "witch", "duck"]
    # NER: list of character extracted
    #entitiesStanza = get_entity_stanza(text)
    #characters = clean_entities(entitiesStanza)
    #print(characters)
    # ['cinderella', 'king', 'prince', 'charlotte']

    #SENTIMENT USING STANZA
    # sentiment = sentiment_one_character_stanza(doc, characters, 0)
    # print(sentiment)
    #res: {'cinderella': 0, 'stepmother': 0, 'sisters': 0, 'godmother': 0, 'prince': 0, 'king': 0, 'queen': 1}
    #USING AFINN
    print(sentiment_one_character_afinn(text,characters, 0))
    #res: {'cinderella': 1, 'stepmother': -1, 'sisters': 1, 'godmother': 1, 'prince': 1, 'king': 1, 'queen': 1}
    # print(sentiment_one_character_afinn(text,characters, 1))
    # #res: {'cinderella': 1, 'stepmother': 1, 'sisters': 1, 'godmother': 1, 'prince': 1, 'king': 1, 'queen': 1}
    # print(sentiment_one_character_afinn(text, characters, 2))
    #res: {'cinderella': 1, 'stepmother': 1, 'sisters': 1, 'godmother': 1, 'prince': 1, 'king': 1, 'queen': 1}