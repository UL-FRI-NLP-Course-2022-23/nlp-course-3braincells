import re
import argparse
import pandas as pd
import numpy as np
import spacy


class Pipeline:
    def __init__(self):
        self.animals = ['fox', 'lion', 'tiger', 'bear', 'rabbit', 'deer', 'wolf', 'elephant', 'monkey', 'snake', 'zebra', 'giraffe', 'rhinoceros', 'hippopotamus', 'crocodile', 'alligator', 'jaguar', 'leopard', 'cheetah', 'hyena', 'buffalo', 'koala', 'kangaroo', 'panda', 'camel', 'horse', 'cow', 'sheep', 'goat', 'pig', 'chicken', 'duck', 'goose', 'turkey', 'parrot', 'owl', 'eagle', 'hawk', 'falcon', 'seagull', 'penguin', 'dolphin', 'whale', 'shark', 'octopus', 'crab', 'lobster', 'snail', 'spider', 'ant', 'bee', 'butterfly', 'moth', 'grasshopper', 'dragonfly', 'ladybug']
        self.relatives = ['grandmother', 'grandfather', 'mother', 'father', 'sister', 'brother', 'aunt', 'uncle', 'cousin', 'niece', 'nephew', 'stepmother', 'stepfather', 'stepsister', 'stepbrother', 'half-sister', 'half-brother', 'in-law', 'daughter-in-law', 'son-in-law', 'mother-in-law', 'father-in-law', 'sister-in-law', 'brother-in-law', 'granddaughter', 'grandson', 'godmother', 'godfather']
        self.characters = ['prince', 'princess', 'king', 'queen', 'witch', 'wizard', 'fairy', 'dragon', 'giant', 'dwarf', 'elf', 'mermaid', 'pirate', 'knight', 'sorcerer', 'sorceress', 'troll', 'ogre', 'gnome', 'vampire', 'werewolf', 'ghost', 'goblin', 'demon', 'angel', 'nymph', 'siren', 'centaur', 'griffin', 'phoenix', 'unicorn', 'pegasus', 'cyclops', 'minotaur', 'medusa', 'satyr', 'hydra', 'chimera', 'kraken']

    def _get_text(self, path):
        return "\n".join(pd.read_table(path, header=None)[0])

    def extract_characters(self, path):
        text = self._get_text(path)
        return self.rule_based_character_extraction(text), self.model_based_character_extraction(text)

    def rule_based_character_extraction(self, text, thrs = 2):
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

    def model_based_character_extraction(self, text, model = 'spacy'):
        nlp = spacy.load("en_core_web_sm")

        if model == 'spacy':
            nlp = spacy.load("en_core_web_sm")
        else:
            # TODO: add different models..
            return
        
        doc = nlp(text)
        pred = np.unique([ent.text.lower() for ent in doc.ents if ent.label_ == "PERSON"])
        return pred
    
    def sentiment_analysis(self, text):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='CharacterInteractionPipeline',
                    description='The pipeline extracts characters from the text, evaluates the interaction between the characters and builds a knowledge graph from the gathered information.',
                    epilog='/')

    parser.add_argument('-p', '--path', required=True, help='Path to text')
    args = vars(parser.parse_args())

    pipeline = Pipeline()
    characters = pipeline.extract_characters(args['path'])
    print(characters)

    print("TODO: rest of the pipeline..")