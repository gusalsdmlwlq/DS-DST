# not include hospital and police
all_domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi']

# normalize slot names
normlize_slot_names = {
    "car type": "car",
    "entrance fee": "price",
    "duration": "time",
    "leaveat": 'leave',
    'arriveby': 'arrive',
    "pricerange": "price range",
    'trainid': 'id',
    'addr': "address",
    'fee': "price",
    'post': "postcode",
    'ref': 'reference',
    'ticket': 'price',
    'depart': "departure",
    'dest': "destination"
}

req_slots = {
    'taxi': ['car', 'phone'],
    'train': ['time', 'leave', 'price', 'arrive', 'id'],
    'restaurant': ['phone', 'postcode', 'address', 'price range', 'food', 'area'],
    'hotel': ['address', 'postcode', 'internet', 'phone', 'parking', 'type', 'price range', 'stars', 'area'],
    'attraction': ['price', 'type', 'address', 'postcode', 'phone', 'area']
}
all_req_slots = [
    'taxi-car',
    'taxi-phone',
    'train-time',
    'train-leave',
    'train-price',
    'train-arrive',
    'train-id',
    'restaurant-phone',
    'restaurant-postcode',
    'restaurant-address',
    'restaurant-price range',
    'restaurant-food',
    'restaurant-area',
    'hotel-address',
    'hotel-postcode',
    'hotel-internet',
    'hotel-phone',
    'hotel-parking',
    'hotel-type',
    'hotel-price range',
    'hotel-stars',
    'hotel-area',
    'attraction-price',
    'attraction-type',
    'attraction-address',
    'attraction-postcode',
    'attraction-phone',
    'attraction-area'
]
info_slots = {
    'taxi': ['leave', 'destination', 'departure', 'arrive'],
    'train': ['people', 'leave', 'destination', 'day', 'arrive', 'departure'],
    'restaurant': ['time', 'day', 'people', 'food', 'price range', 'name', 'area'],
    'hotel': ['stay', 'day', 'people', 'name', 'area', 'parking', 'price range', 'stars', 'internet', 'type'],
    'attraction': ['type', 'name', 'area']
}
all_info_slots = [
    'taxi-leave',
    'taxi-destination',
    'taxi-departure',
    'taxi-arrive',
    'train-people',
    'train-leave',
    'train-destination',
    'train-day',
    'train-arrive',
    'train-departure',
    'restaurant-time',
    'restaurant-day',
    'restaurant-people',
    'restaurant-food',
    'restaurant-price range',
    'restaurant-name',
    'restaurant-area',
    'hotel-stay',
    'hotel-day',
    'hotel-people',
    'hotel-name',
    'hotel-area',
    'hotel-parking',
    'hotel-price range',
    'hotel-stars',
    'hotel-internet',
    'hotel-type',
    'attraction-type',
    'attraction-name',
    'attraction-area'
]

gate_dict = {
    "none": 0,
    "prediction": 1,
    "don't care": 2
}

dialog_acts = {
    'attraction': ['inform', 'nooffer', 'recommend', 'request', 'select'],
    'booking': ['book', 'inform', 'nobook', 'request'],
    'hotel': ['inform', 'nooffer', 'recommend', 'request', 'select'],
    'restaurant': ['inform', 'nooffer', 'recommend', 'request', 'select'],
    'taxi': ['inform', 'request'],
    'train': ['inform', 'nooffer', 'offerbook', 'offerbooked', 'request', 'select'],
    'general': ['bye', 'greet', 'reqmore', 'welcome']
}
dialogue_acts_slots = {
    'attraction-inform': ['area', 'type', 'choice', 'postcode', 'name', 'phone', 'address', 'price', 'price', 'open'],
    'attraction-nooffer': ['area', 'type', 'name', 'choice', 'address', 'price'],
    'attraction-recommend': ['postcode', 'name', 'area', 'price', 'phone', 'address', 'type', 'choice', 'price', 'open'],
    'attraction-request': ['area', 'type', 'price', 'name'],
    'attraction-select': ['type', 'area', 'name', 'price', 'choice', 'phone', 'price', 'address'],
    'booking-book': ['reference', 'name', 'people', 'time', 'day', 'stay'],
    'booking-inform': ['stay', 'name', 'day', 'people', 'reference', 'time'],
    'booking-nobook': ['reference', 'stay', 'day', 'people', 'time', 'name'],
    'booking-request': ['time', 'day', 'people', 'stay'],
    'general-bye': [],
    'general-greet': [],
    'general-reqmore': [],
    'general-welcome': [],
    'hotel-inform': ['name', 'reference', 'type', 'choice', 'address', 'postcode', 'area', 'internet', 'parking', 'stars', 'price', 'phone'],
    'hotel-nooffer': ['type', 'price', 'area', 'stars', 'internet', 'parking', 'name', 'choice'],
    'hotel-recommend': ['name', 'price', 'area', 'stars', 'parking', 'internet', 'address', 'type', 'phone', 'postcode', 'choice'],
    'hotel-request': ['area', 'price', 'stars', 'type', 'parking', 'internet', 'name'],
    'hotel-select': ['price', 'name', 'choice', 'type', 'area', 'stars', 'parking', 'internet', 'address', 'phone'],
    'restaurant-inform': ['postcode', 'food', 'name', 'price', 'address', 'phone', 'area', 'choice', 'reference'],
    'restaurant-nooffer': ['food', 'area', 'price', 'choice', 'name'],
    'restaurant-recommend': ['area', 'price', 'name', 'food', 'address', 'phone', 'choice', 'postcode'],
    'restaurant-request': ['area', 'name', 'price', 'food'],
    'restaurant-select': ['food', 'area', 'price', 'name', 'choice', 'address'],
    'taxi-inform': ['phone', 'car', 'departure', 'destination', 'leave', 'arrive'],
    'taxi-request': ['destination', 'leave', 'departure', 'arrive'],
    'train-inform': ['arrive', 'id', 'leave', 'time', 'destination', 'price', 'departure', 'day', 'choice', 'reference', 'people'],
    'train-nooffer': ['departure', 'destination', 'leave', 'day', 'arrive', 'id', 'choice'],
    'train-offerbook': ['leave', 'id', 'people', 'arrive', 'destination', 'departure', 'day', 'time', 'price', 'choice', 'reference'],
    'train-offerbooked': ['reference', 'price', 'people', 'id', 'leave', 'departure', 'destination', 'time', 'arrive', 'day', 'choice'],
    'train-request': ['leave', 'day', 'departure', 'destination', 'arrive', 'people'],
    'train-select': ['arrive', 'leave', 'id', 'day', 'price', 'choice', 'departure', 'destination', 'people']
}

# dialogues that don't have dialog_act in data.json, but have in system_act.json
no_act_dial_id = ['PMUL4707.json', 'PMUL2245.json', 'PMUL4776.json', 'PMUL3872.json', 'PMUL4859.json']

special_tokens = ['<pad>', '<go_r>', '<unk>', '<go_b>', '<go_a>',
                            '<eos_u>', '<eos_r>', '<eos_b>', '<eos_a>', '<go_d>','<eos_d>'] # 0,1,2,3,4,5,6,7,8,9,10

eos_tokens = {
    'user': '<eos_u>', 'user_delex': '<eos_u>',
    'resp': '<eos_r>', 'resp_gen': '<eos_r>', 'pv_resp': '<eos_r>',
    'bspn': '<eos_b>', 'bspn_gen': '<eos_b>', 'pv_bspn': '<eos_b>',
    'bsdx': '<eos_b>', 'bsdx_gen': '<eos_b>', 'pv_bsdx': '<eos_b>',
    'aspn': '<eos_a>', 'aspn_gen': '<eos_a>', 'pv_aspn': '<eos_a>',
    'dspn': '<eos_d>', 'dspn_gen': '<eos_d>', 'pv_dspn': '<eos_d>',
    "trade": "<eos_b>"}