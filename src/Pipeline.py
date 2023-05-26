import logging
logging.disable(logging.CRITICAL)
import json
import re
import argparse
import pandas as pd
import numpy as np
import spacy
import stanza
import networkx as nx
import matplotlib.pyplot as plt
from sentiment_analysis import *
from fastcoref import spacy_component
import os
from utils import characters_accuracy, relationships_accuracy
logging.getLogger('stanza').setLevel(logging.WARNING) # Set the logging level to WARNING


class Pipeline:
    def __init__(self):
        # Common characters in folktale stories..
        self.animals = ['fox', 'lion', 'tiger', 'bear', 'rabbit', 'deer', 'wolf', 'elephant', 'monkey', 'snake',
                        'zebra', 'giraffe', 'rhinoceros', 'hippopotamus', 'crocodile', 'alligator', 'jaguar', 'leopard',
                        'cheetah', 'hyena', 'buffalo', 'koala', 'kangaroo', 'panda', 'camel', 'horse', 'cow', 'sheep',
                        'goat', 'pig', 'chicken', 'duck', 'goose', 'turkey', 'parrot', 'owl', 'eagle', 'hawk', 'falcon',
                        'seagull', 'penguin', 'dolphin', 'whale', 'shark', 'octopus', 'crab', 'lobster', 'snail',
                        'spider', 'ant', 'bee', 'butterfly', 'moth', 'grasshopper', 'dragonfly', 'ladybug']
        self.relatives = ['grandmother', 'grandfather', 'mother', 'father', 'sister', 'brother', 'aunt', 'uncle',
                          'cousin', 'niece', 'nephew', 'stepmother', 'stepfather', 'stepsister', 'stepbrother',
                          'half-sister', 'half-brother', 'in-law', 'daughter-in-law', 'son-in-law', 'mother-in-law',
                          'father-in-law', 'sister-in-law', 'brother-in-law', 'granddaughter', 'grandson', 'godmother',
                          'godfather']
        self.characters = ['prince', 'princess', 'king', 'queen', 'witch', 'wizard', 'fairy', 'dragon', 'giant',
                           'dwarf', 'elf', 'mermaid', 'pirate', 'knight', 'sorcerer', 'sorceress', 'troll', 'ogre',
                           'gnome', 'vampire', 'werewolf', 'ghost', 'goblin', 'demon', 'angel', 'nymph', 'siren',
                           'centaur', 'griffin', 'phoenix', 'unicorn', 'pegasus', 'cyclops', 'minotaur', 'medusa',
                           'satyr', 'hydra', 'chimera', 'kraken']

    def _get_text(self, path):
        return "\n".join(pd.read_table(path, header=None)[0])

    def filter_characters(self, characters):
        return [re.sub(r'\bThe \b|\bthe |[\.,;:?!-]|\'s', '', character).lower() for character in characters] # Remove 'The ' and signs from text

    def extract_characters(self, path, model='spacy'):
        text = self._get_text(path)
        return np.union1d(self.filter_characters(self.rule_based_character_extraction(text)), 
                          self.filter_characters(self.model_based_character_extraction(text, model)))

    def rule_based_character_extraction(self, text, thrs=2):
        animal_matches = {}
        relative_matches = {}
        character_matches = {}

        # Extract animals from text (wolf, dog, cat, ...)
        for animal in self.animals:
            pattern = r'\b(?:the\s)?{}(?:{}s|{}es)?\b'.format(animal, animal, animal)
            matches = re.findall(pattern, text, re.IGNORECASE)
            if len(matches) > thrs:
                animal_matches[animal] = len(matches)

        # Extract relatives from text (mother, father, godmother, ...)
        for relative in self.relatives:
            pattern = r'\b(?:the\s)?(?:\w+-|\w+ )?{}\b(?:s\b)?'.format(relative)
            matches = re.findall(pattern, text, re.IGNORECASE)
            if len(matches) > thrs:
                relative_matches[relative] = len(matches)

        # Extract characters from text (king, queen, fairy, ...)
        for character in self.characters:
            pattern = r'\b(?:the\s)?((?:{}s|{}ies)|{})\b'.format(character[:-1], character[:-1], character)
            matches = re.findall(pattern, text, re.IGNORECASE)
            if len(matches) > thrs:
                character_matches[character] = len(matches)

        animal_matches = sorted(animal_matches.items(), key=lambda x: x[1], reverse=True)
        relative_matches = sorted(relative_matches.items(), key=lambda x: x[1], reverse=True)
        character_matches = sorted(character_matches.items(), key=lambda x: x[1], reverse=True)
        # Sort the matches by frequency in descending order

        # Extract all Names?
        pattern = r'\b(?:the\s)?[A-Z][a-z]+\s[A-Z][a-z]+\b'
        str_arr, int_arr = np.unique(re.findall(pattern, text), return_counts=True)
        # str_arr = np.array([words.replace('the ', '') for words in str_arr])
        matches = list(zip(str_arr[int_arr > thrs], int_arr[int_arr > thrs]))

        all_matches = np.array(animal_matches + character_matches + relative_matches + matches)
        return all_matches[:, 0] if len(all_matches) > 0 else []

    def model_based_character_extraction(self, text, model='spacy'):
        if model == 'spacy':
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            pred = np.unique([ent.text.lower() for ent in doc.ents if ent.label_ == "PERSON"])

        elif model == 'stanza':
            nlp = stanza.Pipeline(lang='en', processors='tokenize,ner', use_gpu=False)
            doc = nlp(text)
            pred = np.unique([ent.text.lower() for sent in doc.sentences for ent in sent.ents if ent.type == 'PERSON'])
        else:
            return

        return pred

    def sentiment_analysis(self, path, model, type, characters, offset):
        text = self._get_text(path)
        if model == "stanza":
            nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
            doc = nlp(text)
            if type == "relationship":
                sentiment = sentiment_multiple_characters_stanza(doc, characters, offset)
        else:
            if type == "relationship":
                sentiment = sentiment_multiple_characters_afinn(text, characters, offset)
            else:
                sentiment = sentiment_one_character_afinn(text, characters, offset)
        return sentiment

    def coreference_resolution(self, path):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("fastcoref")
        f = open(path, "r", encoding='utf-8')
        text = f.read()
        f.close()
        doc = nlp(text, component_cfg={"fastcoref": {'resolve_text': True}})
        with open("../data/coreferenced_data/coref_file.txt", 'w', encoding='utf-8') as f:
            f.write(doc._.resolved_text)

    def knowledge_graph(self, info: dict, name:str = 'Knowledge Graph', save:bool = False):
    # info is dictionary with 'Characters' and 'Relationships'
    
        color_map= {-1: 'red', 0: 'gray', 1: 'green'}    
        G = nx.MultiGraph(name=name)

        for character in info['Characters']:
            G.add_node(character)

        for character1, characters in info['Relationships'].items():
            for character2, relationship in characters:
                G.add_edge(character1, character2, value = relationship)

        edge_colors = [color_map[rel['value']] for ch1, ch2, rel in G.edges(data=True)]
        node_sizes  = [(v+1) * 500 for v in dict(G.degree()).values()]
        
        plt.figure(figsize=(10, 7))
        nx.draw_circular(G, edge_color=edge_colors, node_size=node_sizes, with_labels=True, width=1.5, font_size=14)

        plt.savefig(name+'.jpg') if save else None
        plt.axis('off')
        plt.show()


# (Example) Run pipleline in terminal using: python3 Pipeline.py --path ../data/our_data/Cinderella.txt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='CharacterInteractionPipeline',
        description='The pipeline extracts characters from the text, evaluates the interaction between the characters and builds a knowledge graph from the gathered information.',
        epilog='/')
    parser.add_argument('-p', '--path', required=True, help='Path to text')
    args = vars(parser.parse_args())

    #coreference flag
    coref = False

    print("Loading pipeline..")
    pipeline = Pipeline()

    print("Extracting characters..")
    characters = pipeline.extract_characters(args['path'], 'stanza')

    if(coref):
        print("Coreference resolution..")
        pipeline.coreference_resolution(args['path'])
        path = "../data/coreferenced_data/coref_file.txt"
    else:
        path = args['path']

    print("Sentiment analysis..")
    #possible options for model "stanza" and "afinn"
    sentiment = pipeline.sentiment_analysis(path, "stanza", "relationship", characters, 0)

    if(coref):
        if os.path.exists("../data/coreferenced_data/coref_file.txt"):
            os.remove("../data/coreferenced_data/coref_file.txt")

    # Make a dictionary that resembles the ground truth annotations (Characters, Relationships)
    info = {"Characters": characters, "Relationships": sentiment}

    print("Finished!!")

    print("Determining protagonist and antagonist..")
    protagonist, antagonist = pipeline.sentiment_analysis(path, "afinn", "character", characters, 0)
    print("Protagonist: ", protagonist)
    print("Antagonist: ", antagonist)

    print("Calculating numerical results..")
    correct, gd_number, all_charac = characters_accuracy(args['path'], characters)
    accuracy, num_gd_rel, num_rel = relationships_accuracy(args['path'], sentiment)

    print("NER results:")
    print("Correctly found characters: ", correct)
    print("Total number of found charachters: ", all_charac)
    print("Number of grountruth characters:", gd_number)

    print("Sentiment results:")
    print("Correctly found relationships between existing characters: ", accuracy)
    print("Total number of found relationships: ", num_rel)
    print("Number of grountruth relationships:", num_gd_rel)

    pipeline.knowledge_graph(info)

