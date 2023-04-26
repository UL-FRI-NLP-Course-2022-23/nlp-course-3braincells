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

    counter = 0
    num = sentimentList[0]
    for i in sentimentList:
        curr_frequency = sentimentList.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i
    return num

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
    #makes a dictionary where the keys are the combinations of the characters in "john/jane" form
    # for i in range(len(characters)):
        # for j in range(i + 1, len(characters)):
            # relationships[f"{characters[i]}/{characters[j]}"] = []

    for s in range(len(doc.sentences)):
        for i in range(len(characters)):
            character_relationships = []
            for j in range(i+1, len(characters)):
                    if is_character_in_sent(doc.sentences, s, characters[i], offset) and is_character_in_sent(doc.sentences, s, characters[j], offset):
                        # relationships[f"{characters[i]}/{characters[j]}"].append(calculate_sent_sentiment(doc.sentences, s, offset))
                        character_relationships.append([characters[j], calculate_sent_sentiment(doc.sentences, s, offset)])
            relationships[characters[i]] = character_relationships

    # for key, value in relationships.items():
        # relationships[key] = listToValue(value)
    return relationships

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
    # makes a dictionary where the keys are the combinations of the characters in "john/jane" form
    for i in range(len(characters)):
        for j in range(i + 1, len(characters)):
            relationships[f"{characters[i]}/{characters[j]}"] = []

    for s in range(len(sentences)):
        for i in range(len(characters)):
            for j in range(i+1, len(characters)):
                    if characters[i] in sentences[s].lower() and characters[j] in sentences[s].lower():
                        score = eval_afinn(sentences, s, offset)
                        relationships[f"{characters[i]}/{characters[j]}"].append(score)

    for key, value in relationships.items():
         relationships[key] = listToValue(value)
    return relationships

if __name__ == "__main__":
    f = open("../data/our_data/cinderella.txt", "r", encoding='utf-8')
    text = f.read()
    nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
    doc = nlp(text)

    #ground truth character list
    characters =  ["cinderella", "stepmother", "sisters", "godmother", "prince", "king", "queen"]

    # NER: list of character extracted
    #entitiesStanza = get_entity_stanza(text)
    #characters = clean_entities(entitiesStanza)
    #print(characters)
    # ['cinderella', 'king', 'prince', 'charlotte']

    # TODO: na kraj od sekoja funckija finalniot sentiment se presmetuva taka sto gleda koj od trite mozni {-1,0,1} se pojavuva najvekje pati, treba da se proba mozda so avg sentiment
    #SENTIMENT USING STANZA
    sentiment = sentiment_one_character_stanza(doc, characters, 0)
    #res: {'cinderella': 0, 'stepmother': 0, 'sisters': 0, 'godmother': 0, 'prince': 0, 'king': 0, 'queen': 1}
    print(sentiment_multiple_characters_stanza(doc, characters, 0))
    #res: {'cinderella/stepmother': [], 'cinderella/sisters': [1, -1, 0, 0, 0, 1, 1], 'cinderella/godmother': [0, 0, 1], 'cinderella/prince': [], 'cinderella/king': [], 'cinderella/queen': [], 'stepmother/sisters': [],'stepmother/godmother': [], 'stepmother/prince': [], 'stepmother/king': [], 'stepmother/queen': [], 'sisters/godmother': [0], 'sisters/prince': [0], 'sisters/king': [], 'sisters/queen': [], 'godmother/prince': [], 'godmother/king': [0], 'godmother/queen': [], 'prince/king': [], 'prince/queen': [], 'king/queen': [1]}
    print(sentiment_multiple_characters_stanza(doc, characters, 1))
    #res: {'cinderella/stepmother': None, 'cinderella/sisters': 0, 'cinderella/godmother': 0, 'cinderella/prince': 0, 'cinderella/king': 0, 'cinderella/queen': None, 'stepmother/sisters': -1, 'stepmother/godmother': None, 'stepmother/prince': None, 'stepmother/king': None, 'stepmother/queen': None, 'sisters/godmother': 0, 'sisters/prince': 0, 'sisters/king': 0, 'sisters/queen': None, 'godmother/prince': 0, 'godmother/king': 0, 'godmother/queen': None, 'prince/king': 0, 'prince/queen': None, 'king/queen': 1}
    print(sentiment_multiple_characters_stanza(doc, characters, 2))
    #res: {'cinderella/stepmother': 0, 'cinderella/sisters': 0, 'cinderella/godmother': 0, 'cinderella/prince': 0,'cinderella/king': 0, 'cinderella/queen': None, 'stepmother/sisters': 0, 'stepmother/godmother': None,'stepmother/prince': None, 'stepmother/king': None, 'stepmother/queen': None, 'sisters/godmother': 0, 'sisters/prince': 0, 'sisters/king': 0, 'sisters/queen': None, 'godmother/prince': 0, 'godmother/king': 0,'godmother/queen': None, 'prince/king': 0, 'prince/queen': 0, 'king/queen': 1}
    #USING AFINN
    print(sentiment_one_character_afinn(text,characters, 0))
    #res: {'cinderella': 1, 'stepmother': -1, 'sisters': 1, 'godmother': 1, 'prince': 1, 'king': 1, 'queen': 1}
    print(sentiment_one_character_afinn(text,characters, 1))
    #res: {'cinderella': 1, 'stepmother': 1, 'sisters': 1, 'godmother': 1, 'prince': 1, 'king': 1, 'queen': 1}
    print(sentiment_one_character_afinn(text, characters, 2))
    #res: {'cinderella': 1, 'stepmother': 1, 'sisters': 1, 'godmother': 1, 'prince': 1, 'king': 1, 'queen': 1}
    print(sentiment_multiple_characters_afinn(text,characters, 0))
    # res: {'cinderella/stepmother': None, 'cinderella/sisters': 1, 'cinderella/godmother': 1, 'cinderella/prince': None,
    #  'cinderella/king': None, 'cinderella/queen': None, 'stepmother/sisters': None, 'stepmother/godmother': None,
    #  'stepmother/prince': None, 'stepmother/king': None, 'stepmother/queen': None, 'sisters/godmother': 0,
    #  'sisters/prince': 0, 'sisters/king': 0, 'sisters/queen': None, 'godmother/prince': None, 'godmother/king': 0,
    #  'godmother/queen': None, 'prince/king': 1, 'prince/queen': None, 'king/queen': 1}
    print(sentiment_multiple_characters_afinn(text, characters, 1))
    # res: {'cinderella/stepmother': None, 'cinderella/sisters': 1, 'cinderella/godmother': 1, 'cinderella/prince': None,
    #  'cinderella/king': None, 'cinderella/queen': None, 'stepmother/sisters': None, 'stepmother/godmother': None,
    #  'stepmother/prince': None, 'stepmother/king': None, 'stepmother/queen': None, 'sisters/godmother': 0,
    #  'sisters/prince': 1, 'sisters/king': -1, 'sisters/queen': None, 'godmother/prince': None, 'godmother/king': 0,
    #  'godmother/queen': None, 'prince/king': 1, 'prince/queen': None, 'king/queen': 1}
    print(sentiment_multiple_characters_afinn(text, characters, 2))
    # res: {'cinderella/stepmother': None, 'cinderella/sisters': 1, 'cinderella/godmother': 1, 'cinderella/prince': None,
    #  'cinderella/king': None, 'cinderella/queen': None, 'stepmother/sisters': None, 'stepmother/godmother': None,
    #  'stepmother/prince': None, 'stepmother/king': None, 'stepmother/queen': None, 'sisters/godmother': 0,
    #  'sisters/prince': 1, 'sisters/king': -1, 'sisters/queen': None, 'godmother/prince': None, 'godmother/king': -1,
    #  'godmother/queen': None, 'prince/king': 1, 'prince/queen': None, 'king/queen': 1}