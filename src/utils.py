import json
#calculates how many of the ner-found characters are accurate
#correct = number of correctly found characters
#gd_numbers = number of groundtruth characters
#all_charac = number of all ner-found characters
#parametere characters represents a list of the ner-found characters
def characters_accuracy(path, characters):
    ann = json.load(open(path[:-3] + 'json'))
    gd = ann['Characters']
    gd_number = len(gd)
    all_charac = len(characters)
    correct = sum(c in gd for c in characters)
    return correct, gd_number, all_charac
#calculates how many of the relationships are accurate
#accuracy = number of correctly estimated relationships (between correct characters)
#num_gd_rel = number of ground truth relationships
#num_rel = number of found relationships
#parameter relationships is a dictionary of the found relationships
def relationships_accuracy(path, relationships):
    accuracy = 0
    ann = json.load(open(path[:-3] + 'json'))
    gd = ann['Relationships']
    gd_charac = ann['Characters']
    rel = relationships.items()
    num_gd_rel = sum(len(v) for v in gd.values())
    num_rel = sum(len(v) for v in relationships.values())
    for character1, characters in rel:
        for character2, relationship_value in characters:
            if character1 not in gd_charac or character2 not in gd_charac:
                continue
            r = gd[character1]
            for c, sent in r:
                if character2 == c and sent == relationship_value:
                    accuracy+=1
    return accuracy, num_gd_rel, num_rel
