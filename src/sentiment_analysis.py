import stanza
from ner import get_entity_stanza, clean_entities
from afinn import Afinn
from nltk.tokenize import sent_tokenize
afn = Afinn()
# sentiment using stanza, argument is text and list of extracted characters
def sentiment_one_character_stanza(text, characters_extracted):
    # get sentiments using stanza for each sentence in text
    nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
    doc = nlp(text)
    # create dictionary with key = characters_extracted and value empty list
    stanzaEntities = {key: [] for key in characters_extracted}
    textSentiment = []
    for noSent, sent in enumerate(doc.sentences):
        # get sentence sentiment and because stanza returns 0,1,2 subtract -1 to get real sentiment values
        sentiment = sent.sentiment-1
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
    for key, value in stanzaEntities.items():
        stanzaEntities[key] = listToValue(value)

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

def sentiment_multiple_characters_stanza(text, characters):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
    doc = nlp(text)

    relationships = {}
    #makes a dictionary where the keys are the combinations of the characters in "john/jane" form
    for i in range(len(characters)):
        for j in range(i + 1, len(characters)):
            relationships[f"{characters[i]}/{characters[j]}"] = []

    for sent in doc.sentences:
        sentence = sent.to_dict()
        words = []
        for word in sentence:
            words.append(word['text'].lower())
        for i in range(len(characters)):
            for j in range(i+1, len(characters)):
                    if characters[i] in words and characters[j] in words:
                        relationships[f"{characters[i]}/{characters[j]}"].append(sent.sentiment - 1)
    return relationships

def evaluate_afinn(score):
    if score > 0:
        return 1
    elif score == 0:
        return 0
    else:
        return -1
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
def sentiment_afinn(text, characters, offset):
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

if __name__ == "__main__":
    f = open("../data/our_data/cinderella.txt", "r", encoding='utf-8')
    text = f.read()


    #ground truth character list
    characters =  ["cinderella", "stepmother", "sisters", "godmother", "prince", "king", "queen"]
    # # NER: list of character extracted
    #entitiesStanza = get_entity_stanza(text)
    # characters = clean_entities(entitiesStanza)
    # print(characters)
    # ##### ['cinderella', 'king', 'prince', 'charlotte']
    #
    #SENTIMENT USING STANZA
    textSentiment ,sentiment = sentiment_one_character_stanza(text, characters)
    print("Stanza")
    print(sentiment)
    #res: {'cinderella': 0, 'stepmother': 0, 'sisters': 0, 'godmother': 0, 'prince': 0, 'king': 0, 'queen': 1}

    relationships = sentiment_multiple_characters_stanza(text, characters)
    print(relationships)
    #res: {'cinderella/stepmother': [], 'cinderella/sisters': [1, -1, 0, 0, 0, 1, 1], 'cinderella/godmother': [0, 0, 1], 'cinderella/prince': [], 'cinderella/king': [], 'cinderella/queen': [], 'stepmother/sisters': [], 'stepmother/godmother': [], 'stepmother/prince': [], 'stepmother/king': [], 'stepmother/queen': [], 'sisters/godmother': [0], 'sisters/prince': [0], 'sisters/king': [], 'sisters/queen': [], 'godmother/prince': [], 'godmother/king': [0], 'godmother/queen': [], 'prince/king': [], 'prince/queen': [], 'king/queen': [1]}

    #USING AFINN
    print(sentiment_afinn(text,characters, 0))
    #res: {'cinderella': 1, 'stepmother': -1, 'sisters': 1, 'godmother': 1, 'prince': 1, 'king': 1, 'queen': 1}
    print(sentiment_afinn(text,characters, 1))
    #res: {'cinderella': 1, 'stepmother': 1, 'sisters': 1, 'godmother': 1, 'prince': 1, 'king': 1, 'queen': 1}
    print(sentiment_afinn(text, characters, 2))
    #res: {'cinderella': 1, 'stepmother': 1, 'sisters': 1, 'godmother': 1, 'prince': 1, 'king': 1, 'queen': 1}
