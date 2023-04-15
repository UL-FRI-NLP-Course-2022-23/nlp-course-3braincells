import csv

header = ['name', 'characters', 'sentiment', 'protagonists', 'antagonists']
data = [
    ['hansel_and_gretel', 'hansel, gretel, witch, mother, father','hansel + gretel, hansel - witch, gretel - witch, hansel - mother, gretel - mother, father / mother', 'hansel, gretel', 'witch, mother'],
    ['little_red_cap', 'wolf, grandmother, little red cap / red cap, mother, huntsman','little red cap + grandmother, little red cap - wolf, little red cap + mother, grandmother - wolf, huntsman - wolf, huntsman + grandmother, huntsman + little red cap', 'little red cap','wolf'],
    ['cinderella', 'cinderella, mother in law, godmother / fairy, king, queen, prince / king\'s son','cinderella - mother in law, cinderella + fairy, cinderella + prince, king / queen', 'cinderella', 'mother in law'],
    ['old_sultan', 'sultan, wife, shepherd  / master, wolf, child', 'sultan + wolf, sultan - shepherd, shepherd / wife, sultan / child, wolf / child, wolf - sultan, shepherd + sultan', 'sultan', 'wolf'],
    ['rapunzel', 'husband, wife, enchantress / Dame Gothel, rapunzel, prince', 'rapunzel - enchantress / Dame Gothel, rapunzel + prince, husband - enchantress / Dame Gothel, enchantress / Dame Gothel - prince, husband + wife','rapunzel', 'enchantress / Dame Gothel']
]

with open('ann.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(data)
