import copy
import json
import logging
import os

import numpy as np
from tqdm import tqdm

import ontology
from config import Config
from clean_data import clean_text, clean_slot_values


class DataCreator(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.train_list = open(os.path.join(data_path, "trainListFile.txt"), "r").readlines()
        self.dev_list = open(os.path.join(data_path, "valListFile.txt"), "r").readlines()
        self.test_list = open(os.path.join(data_path, "testListFile.txt"), "r").readlines()
        self.data = json.load(open(os.path.join(data_path, "data.json"), "r"))
        self.acts = json.load(open(os.path.join(data_path, "system_acts.json"), "r"))

    def create_data(self):
        data = {}
        train = {}
        dev = {}
        test = {}
        ignore_list = ["SNG1213", "PMUL0382", "PMUL0237"]
        logger.info("Processing data...")
        for dial_id, dial in tqdm(self.data.items()):
            dial_id = dial_id.split(".")[0]
            if dial_id in ignore_list:
                continue
            dialogue = {}
            goal = {}
            dial_domains = []
            for key, value in dial["goal"].items():  # process user's goal
                if key in ontology.all_domains and value != {}:
                    if value.get("reqt"):  # normalize requestable slot names
                        for idx, slot in enumerate(value["reqt"]):
                            if ontology.normlize_slot_names.get(slot):
                                value["reqt"][idx] = ontology.normlize_slot_names[slot]
                    goal[key] = value
                    dial_domains.append(key)
            if len(dial_domains) == 0:  # ignore police and hospital
                ignore_list.append(dial_id)
                continue
            dialogue["goal"] = goal
        
            dialogue["log"] = []
            acts = self.acts[dial_id]
            turn = {}
            for turn_num, turn_dial in enumerate(dial["log"]):
                meta_data = turn_dial["metadata"]
                if meta_data == {}:  # user turn
                    turn["turn_num"] = int(turn_num/2)
                    turn["user"] = clean_text(turn_dial["text"])
                else:  # system turn
                    turn["response"] = clean_text(turn_dial["text"])
                    belief = {}
                    gate = {}
                    act = {}

                    for domain in dial_domains:  # active domains of dialogue
                        for slot, value in meta_data[domain]["book"].items():  # book
                            if slot == "booked":
                                continue
                            slot, value = clean_slot_values(domain, slot, value)
                            if value != "":
                                belief["{}-{}".format(domain, slot)] = value
                                gate["{}-{}".format(domain, slot)] = ontology.gate_dict[value] if value == "don't care" else ontology.gate_dict["prediction"]
                        for slot, value in meta_data[domain]["semi"].items():  # semi
                            slot, value = clean_slot_values(domain, slot, value)
                            if value != "":
                                belief["{}-{}".format(domain, slot)] = value
                                gate["{}-{}".format(domain, slot)] = ontology.gate_dict[value] if value == "don't care" else ontology.gate_dict["prediction"]
                    turn["belief"] = belief
                    turn["gate"] = gate

                    if acts.get(str(turn["turn_num"]+1)) and type(acts.get(str(turn["turn_num"]+1))) != str:  # mapping system action
                        for domain_act, slots in acts[str(turn["turn_num"]+1)].items():
                            act_temp = []
                            for slot in slots:  # slot: [slot, value]
                                slot_, value_ = clean_slot_values(domain_act.split("-")[0], slot[0], slot[1])
                                if slot_ == "none" or value_ in  ["?", "none"]:  # general domain or request slot or parking
                                    act_temp.append(slot_)
                                else:
                                    act_temp.append("{}-{}".format(slot_, value_))
                            act[domain_act.lower()] = act_temp
                    turn["action"] = act

                    dialogue["log"].append(turn)
                    turn = {}  # clear turn
            
            data[dial_id] = dialogue

        logger.info("Processing finished.")
        logger.info("Dividing data to train/dev/test...")
        for dial_id in self.train_list:
            dial_id = dial_id.split(".")[0]
            if dial_id not in ignore_list:
                train[dial_id] = data[dial_id]
        for dial_id in self.dev_list:
            dial_id = dial_id.split(".")[0]
            if dial_id not in ignore_list:
                dev[dial_id] = data[dial_id]
        for dial_id in self.test_list:
            dial_id = dial_id.split(".")[0]
            if dial_id not in ignore_list:
                test[dial_id] = data[dial_id]
        logger.info("Dividing finished.")

        value_ontology = json.load(open(os.path.join(self.data_path, "ontology.json"), "r"))
        value_ontology_processed = {}

        logger.info("Processing ontology...")
        for domain_slot, values in value_ontology.items():
            domain = domain_slot.split("-")[0]
            slot = domain_slot.split("-")[2].lower()
            if ontology.normlize_slot_names.get(slot):
                slot = ontology.normlize_slot_names[slot]
            domain_slot = "-".join([domain, slot])
            value_ontology_processed[domain_slot] = []
            for value in values:
                _, value = clean_slot_values(domain, slot, value)
                value_ontology_processed[domain_slot].append(value)
        with open(os.path.join(data_path, "ontology_processed.json"), "w") as f:
            json.dump(value_ontology_processed, f, indent=2)
        logger.info("Ontology was processed.")

        return train, dev, test


if __name__=='__main__':
    config = Config()
    parser = config.parser
    hparams = parser.parse_args()
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    data_path = hparams.data_path
    db_paths = {
        'attraction': os.path.join(data_path, "attraction_db.json"),
        'hotel': os.path.join(data_path, "hotel_db.json"),
        'restaurant': os.path.join(data_path, "restaurant_db.json"),
        'taxi': os.path.join(data_path, "taxi_db.json"),
        'train': os.path.join(data_path, "train_db.json")
    }
    creator = DataCreator(data_path)
    train, dev, test = creator.create_data()
    
    logger.info("Saving data...")
    with open(os.path.join(data_path, "train_data.json"), "w") as f:
        json.dump(train, f, indent=2)
    with open(os.path.join(data_path, "dev_data.json"), "w") as f:
        json.dump(dev, f, indent=2)
    with open(os.path.join(data_path, "test_data.json"), "w") as f:
        json.dump(test, f, indent=2)
    logger.info("Saved.")