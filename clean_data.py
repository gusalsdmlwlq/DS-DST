import re
import ontology


def remove_space(text, token):
        text = re.sub("\s+", " ", text)
        idx = 0
        while True:
            idx = text.find(token, idx)
            if idx == -1:
                break
            if idx>0 and text[idx-1] == ' ':
                text = text[:idx-1]+text[idx:]
            idx += 1
        return text

def clean_text(text):
    text = text.lower()
    text = re.sub("\u008e", "e", text)  # typo of cafe (this typo is also in ontology.json)
    text = re.sub("(\u2018|\u2019)", "'", text)  # typo of '
    text = re.sub("\u00a0", " ", text)  # unicode bug of whitespace
    text = re.sub("center", "centre", text)
    text = re.sub("(dontcare|dont care|do nt care|do n't care|doesn't care|does not care|doesnt care)", "don't care", text)
    text = re.sub("(not mentioned|none)", "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\s+", " ", text)
    text = re.sub("\s$", "", text)
    for token in [",", ".", "!", "?"]:
        text = remove_space(text, token)

    return text

def clean_slot_values(domain, slot, value):
    domain = domain.lower()
    slot = slot.lower()
    value = re.sub("-", " ", value)
    value = clean_text(value)
    if not value:
        value = ''
    elif domain == 'attraction':
        if slot == 'name':
            if value == 't':
                value = ''
            if value=='trinity':
                value = 'trinity college'
        elif slot == 'area':
            if value in ['ely', 'in town', 'museum', 'norwich', 'same area as hotel']:
                value = ""
            elif value in ['we']:
                value = "west"
        elif slot == 'type':
            if value in ['m', 'mus', 'musuem']:
                value = 'museum'
            elif value in ['art', 'architectural']:
                value = "architecture"
            elif value in ['churches']:
                value = "church"
            elif value in ['coll']:
                value = "college"
            elif value in ['concert', 'concerthall']:
                value = 'concert hall'
            elif value in ['night club']:
                value = 'nightclub'
            elif value in ['mutiple sports', 'mutliple sports', 'sports', 'galleria']:
                value = 'multiple sports'
            elif value in ['ol', 'science', 'gastropub', 'la raza']:
                value = ''
            elif value in ['swimmingpool', 'pool']:
                value = 'swimming pool'
            elif value in ['fun']:
                value = 'entertainment'
    elif domain == 'hotel':
        if slot == 'area':
            if value in ['east area', 'east side']:
                value = 'east'
            elif value in ['in the north', 'north part of town']:
                value = 'north'
            elif value in ['we']:
                value = "west"
        elif slot == "day":
            if value == "monda":
                value = "monday"
            elif value == "t":
                value = "tuesday"
        elif slot == 'name':
            if value in ['uni', 'university arms']:
                value = 'university arms hotel'
            elif value == 'acron':
                value = 'acorn guest house'
            elif value == 'ashley':
                value = 'ashley hotel'
            elif value == 'arbury lodge guesthouse':
                value = 'arbury lodge guest house'
            elif value == 'la':
                value = 'la margherit'
            elif value == 'no':
                value = ''
        elif slot == 'internet':
            if value == 'does not':
                value = 'no'
            elif value in ['y', 'free', 'free internet']:
                value = 'yes'
            elif value in ['4']:
                value = ''
        elif slot == 'parking':
            if value == 'n':
                value = 'no'
            elif value in ['free parking']:
                value = 'yes'
            elif value in ['y']:
                value = 'yes'
        elif slot in ['pricerange', 'price range']:
            slot = 'price range'
            if value == 'moderately':
                value = 'moderate'
            elif value in ['any']:
                value = "don't care"
            elif value in ['any']:
                value = "don't care"
            elif value in ['inexpensive']:
                value = "cheap"
            elif value in ['2', '4']:
                value = ''
        elif slot == 'stars':
            if value == 'two':
                value = '2'
            elif value == 'three':
                value = '3'
            elif value in ['4-star', '4 stars', '4 star', 'four star', 'four stars']:
                value= '4'
    elif domain == 'restaurant':
        if slot == "area":
            if value == "west part of town":
                value = "west"
            elif value == "n":
                value = "north"
            elif value in ['the south']:
                value = 'south'
        elif slot == "day":
            if value == "monda":
                value = "monday"
            elif value == "t":
                value = "tuesday"
        elif slot in ['pricerange', 'price range']:
            slot = 'price range'
            if value in ['moderately', 'mode', 'mo']:
                value = 'moderate'
            elif value in ['not']:
                value = ''
            elif value in ['inexpensive', 'ch']:
                value = "cheap"
        elif slot == "food":
            if value == "barbecue":
                value = "barbeque"
        elif slot == "time":
            if value == "7pm":
                value = "19:00"
            elif value == "1345":
                value = "13:45"
            elif value == "4pm":
                value = "16:00"
            elif value == "9":
                value = "09:00"
            elif value == "1715":
                value = "17:15"
            elif value == "1330":
                value = "13:30"
            elif value == "1430":
                value = "14:30"
            elif value == "1145":
                value = "11:45"
            elif value == "8pm":
                value = "20:00"
    elif domain == 'taxi':
        if slot == 'arriveBy':
            slot = 'arrive'
            if value == '15 minutes':
                value = ''
        elif slot == 'leaveAt':
            slot = 'leave'
            if value == "after 11:45":
                value = "11:45"
            elif value == "after 13:45":
                value = "13:45"
            elif value == "after 15:45":
                value = "15:45"
            elif value == "after 2:30":
                value = "02:30"
            elif value in ["thursday", "friday"]:
                value = ""
            value = value.replace(".", ":")
    elif domain == 'train':
        if slot == 'arriveBy':
            slot = 'arrive'
            if value == '8':
                value = '08:00'
            elif value == '1100':
                value = '11:00'
            elif value == '1545':
                value = '15:45'
            value = value.replace(".", ":")
        elif slot == 'leaveAt':
            slot = 'leave'
            if value == '1145':
                value = '11:45'
            elif value == '10':
                value = '10:00'
            elif value == '845':
                value = '08:45'
            elif value == '5:45pm':
                value = '17:45'
            elif value == '1532':
                value = '15:32'
            elif value == '1329':
                value = '13:29'
            elif value == '9':
                value = '09:00'
            elif value == '929':
                value = '09:29'
            elif value == '1545':
                value = '15:45'
            elif value == 'after 16:30':
                value = '16:30'
            elif value in ["afternoon", "after lunch"]:
                value = "12:00"
            elif value == "morning":
                value = ""
            value = value.replace(".", ":")
    if ontology.normlize_slot_names.get(slot):
        slot = ontology.normlize_slot_names[slot]

    return slot, value