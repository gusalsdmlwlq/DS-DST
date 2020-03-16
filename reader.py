import os
import json
import logging
import random

import torch
from transformers import BertTokenizerFast

import ontology


class Reader:
    def __init__(self, hparams):
        self.train = {}
        self.dev = {}
        self.test = {}
        self.data_turns = {}
        self.data_path = hparams.data_path
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.batch_size = hparams.batch_size
        self.max_len = hparams.max_len
        self.max_value_len = hparams.max_value_len
        self.max_context_len = hparams.max_context_len
        self.cuda = False if hparams.no_cuda else True  # default: True

    def load_data(self, mode="train"):
        """Load train/dev/test data.
        Divide data by number of turns for batch.
        Encode user utterance & system response."""

        if mode == "train":
            train_data = json.load(open(os.path.join(self.data_path, "train_data.json"), "r"))
            dev_data = json.load(open(os.path.join(self.data_path, "dev_data.json"), "r"))
            for dial_id, dial in train_data.items():
                turns = len(dial["log"])  # number of turns
                if not self.train.get(turns):
                    self.train[turns] = {}
                self.train[turns][dial_id] = []
                for turn_dial in dial["log"]:
                    turn_dial["user"] = self.tokenizer.encode(turn_dial["user"])
                    turn_dial["response"] = self.tokenizer.encode(turn_dial["response"])
                    turn_dial["belief"], turn_dial["gate"] = self.encode_belief(turn_dial["belief"])
                    turn_dial["action"] = self.encode_action(turn_dial["action"])
                    self.train[turns][dial_id].append(turn_dial)
            for dial_id, dial in dev_data.items():
                turns = len(dial["log"]) 
                if not self.dev.get(turns):
                    self.dev[turns] = {}
                self.dev[turns][dial_id] = []
                for turn_dial in dial["log"]:
                    turn_dial["user"] = self.tokenizer.encode(turn_dial["user"])
                    turn_dial["response"] = self.tokenizer.encode(turn_dial["response"])
                    turn_dial["belief"], turn_dial["gate"] = self.encode_belief(turn_dial["belief"])
                    turn_dial["action"] = self.encode_action(turn_dial["action"])
                    self.dev[turns][dial_id].append(turn_dial)
        else:
            test_data = json.load(open(os.path.join(self.data_path, "test_data.json"), "r"))
            for dial_id, dial in test_data.items():
                turns = len(dial["log"]) 
                if not self.test.get(turns):
                    self.test[turns] = {}
                self.test[turns][dial_id] = []
                for turn_dial in dial["log"]:
                    turn_dial["user"] = self.tokenizer.encode(turn_dial["user"])
                    turn_dial["response"] = self.tokenizer.encode(turn_dial["response"])
                    turn_dial["belief"], turn_dial["gate"] = self.encode_belief(turn_dial["belief"])
                    turn_dial["action"] = self.encode_action(turn_dial["action"])
                    self.test[turns][dial_id].append(turn_dial)

    def encode_belief(self, belief):
        """Encode belief and gate.
        
        Outputs: encoded_belief, encoded_gate
            encoded_belief: List of encoded belief(values of domain-slot pairs)
            encoded_gate: List of encoded domain-slot gate(none: 0, prediction: 1, don't care: 2)

        Shapes:
            encoded_belief: [slots, value_len]
            encoded_gate: [slots]
        """

        encoded_belief = []
        encoded_gate = []
        for domain_slot in ontology.all_info_slots:
            if belief.get(domain_slot):
                encoded_belief.append(self.tokenizer.encode(belief[domain_slot]))
                if belief[domain_slot] == "don't care":
                    encoded_gate.append(ontology.gate_dict["don't care"])
                else:
                    encoded_gate.append(ontology.gate_dict["prediction"])
            else:
                encoded_belief.append(self.tokenizer.encode("none"))
                encoded_gate.append(ontology.gate_dict["none"])
        
        return encoded_belief, encoded_gate

    def encode_action(self, action):
        """Encode system action.
        
        Outputs: encoded_action
            encoded_action: Encoded concat of actions(domain-action-slot)
        """

        encoded_action = []
        for domain_act in ontology.dialogue_acts_slots.keys():
            if action.get(domain_act):
                act = " ".join(domain_act.split("-"))
                for slot_value in action[domain_act]:
                    if slot_value == "none":  # act has no slot
                        break
                    else:  # same as DAMD's action, not including value ex) [hotel] [inform] phone address
                        act += " "+slot_value.split("-")[0]
                    # elif len(slot_value.find("-")) == -1:  # act has only slot
                    #     act += " "+slot_value
                    # else:
                    #     act += " "+slot_value.split("-")[0]+" "+slot_value.split("-")[1]  # act has slot & value
                encoded_action.append(act)
        encoded_action = self.tokenizer.encode(" ".join(encoded_action))

        return encoded_action

    def make_batch(self, data):
        """Make batches and return iterator.

        Outputs: batch
            batch: Dictionary of batches
        
        Example:
            batch = {
                "dial_id" = [...]
                0: {
                    "user": [[...], ..., [...]],
                    "response": [[...], ..., [...]],
                    "belief": [[...], ..., [...]],
                    "gate": [[...], ..., [...]],
                    "action": [[...], ..., [...]]
                },
                1: {
                    "user": [[...], ..., [...]],
                    "response": [[...], ..., [...]],
                    "belief": [[...], ..., [...]],
                    "gate": [[...], ..., [...]],
                    "action": [[...], ..., [...]]
                }
            }
        """

        all_batches = []
        for turn_num, dials in data.items():
            batch = {"dial_id": []}
            for dial_id, turns in dials.items():
                if len(batch["dial_id"]) == self.batch_size:  # current batch is full
                    all_batches.append(batch)
                    batch = {"dial_id": []}
                batch["dial_id"].append(dial_id)
                for turn in turns:
                    cur_turn = turn["turn_num"]
                    if not batch.get(cur_turn):
                        batch[cur_turn] = {
                            "user": [],
                            "response": [],
                            "belief": [],
                            "gate": [],
                            "action": []
                        }
                    for key in batch[cur_turn].keys():
                        batch[cur_turn][key].append(turn[key])
            all_batches.append(batch)
        random.shuffle(all_batches)
        for batch in all_batches:
            yield batch

    def find_span(self, context, value):
        len_ = len(value)
        for idx, token in enumerate(context):
            if context[idx:idx+len_] == value:  # find value span
                return [idx, idx+len_-1]

        return [0, 0]  # not find value span

    def make_input(self, batch):
        """Make input of torch tensors.

        Outputs: inputs, contexts, spans
            inputs: List of tensors.
            contexts: List of history including user utterance and system response.
            spans: List of spans including indices of start & end of value.

        Example:
            inputs = [
                {
                    "user": tensor,  # [batch, user_len]
                    "response": tensor,  # [batch, response_len]
                    "belief": tensor,  # [batch, slots, value_len]
                    "gate": tensor,  # [batch, slots]
                    "action": tensor,  # [batch, action_len]
                },
                {
                    "user": tensor,
                    "response": tensor,
                    "belief": tensor,
                    "gate": tensor,
                    "action": tensor
                }
            ]

        Shapes:
            contexts: [batch, context_len] * turns
            spans: [batch, slots, 2] * turns
        """

        inputs = []
        turns = list(batch.keys())
        batch_size = len(batch["dial_id"])
        turns.remove("dial_id")
        turns.sort()
        contexts = []
        context_len = 0
        spans = []
        prev_resp = []
        turn_context = []
        for turn in turns:
            turn_input = {}
            # make tensor of user utterance
            user_ = torch.zeros((batch_size, self.max_len))
            max_len = 0
            for idx, user in enumerate(batch[turn]["user"]):
                len_ = len(user)
                user_[idx, :len_] = torch.tensor(user[:self.max_len])
                if len_ > max_len:
                    max_len = len_
            turn_input["user"] = user_[:, :max_len].clone().long()

            # make tensor of system response
            resp_ = torch.zeros((batch_size, self.max_len))
            max_len = 0
            for idx, resp in enumerate(batch[turn]["response"]):
                len_ = len(resp)
                resp_[idx, :len_] = torch.tensor(resp[:self.max_len])
                if len_ > max_len:
                    max_len = len_
            turn_input["response"] = resp_[:, :max_len].clone().long()

            # make tensor of belief
            belief_ = torch.zeros((batch_size, len(ontology.all_info_slots), self.max_len))
            max_len = 0
            for idx, belief in enumerate(batch[turn]["belief"]):
                for s_idx, value in enumerate(belief):
                    len_ = len(value)
                    belief_[idx, s_idx, :len_] = torch.tensor(value[:self.max_value_len])
                    if len_ > max_len:
                        max_len = len_
            turn_input["belief"] = belief_[:, :, :max_len].clone().long()

            # make tensor of gate
            turn_input["gate"] = torch.tensor(batch[turn]["gate"])

            # make tensor of action
            act_ = torch.zeros((batch_size, self.max_len))
            max_len = 0
            for idx, act in enumerate(batch[turn]["action"]):
                len_ = len(act)
                act_[idx, :len_] = torch.tensor(act[:self.max_len])
                if len_ > max_len:
                    max_len = len_
            turn_input["action"] = act_[:, :max_len].clone().long()
            
            if self.cuda:
                for key, value in turn_input.items():
                    turn_input[key] = value.cuda()

            inputs.append(turn_input)

            # make context
            if turn == 0:  # first turn
                turn_context_ = torch.zeros((batch_size, self.max_context_len))
                for idx, user in enumerate(batch[turn]["user"]):
                    turn_context.append(user[1:-1])  # remove [CLS] & [SEP]
                    if context_len < len(user[1:-1]):
                        context_len = len(user[1:-1])
                    turn_context_[idx, :len(turn_context[idx])] = torch.tensor(turn_context[idx])
                for resp in batch[turn]["response"]:
                    prev_resp.append(resp[1:-1])
                turn_context_ = turn_context_[:, :context_len]
                
                if self.cuda:
                    turn_context_ = turn_context_.cuda()
                
                contexts.append(turn_context_.clone().long())
            else:  # not first turn
                turn_context_ = torch.zeros((batch_size, self.max_context_len))
                for idx, resp in enumerate(prev_resp):
                    turn_context[idx] += resp
                prev_resp = []
                for idx, user in enumerate(batch[turn]["user"]):
                    turn_context[idx] += user[1:-1]
                    if context_len < len(turn_context[idx]):
                        context_len = len(turn_context[idx])
                    if len(turn_context[idx]) > self.max_context_len:  # cut long contexts for BERT's input
                        turn_context[idx] = turn_context[idx][-self.max_context_len:]
                    turn_context_[idx, :len(turn_context[idx])] = torch.tensor(turn_context[idx])
                for resp in batch[turn]["response"]:
                    prev_resp.append(resp[1:-1])
                turn_context_ = turn_context_[:, :min(context_len, self.max_context_len)]

                if self.cuda:
                    turn_context_ = turn_context_.cuda()

                contexts.append(turn_context_.clone().long())
            
            # make value span 
            turn_spans = torch.zeros((batch_size, len(ontology.all_info_slots), 2), dtype=torch.int64)
            for bidx, gate in enumerate(batch[turn]["gate"]):
                for idx, gate_ in enumerate(gate):
                    if gate_ == ontology.gate_dict["prediction"]:
                        turn_spans[bidx][idx] = torch.tensor(self.find_span(turn_context[bidx], batch[turn]["belief"][bidx][idx]))
            
            if self.cuda:
                turn_spans = turn_spans.cuda()
            
            spans.append(turn_spans)

        return inputs, contexts, spans


if __name__ == "__main__":
    from config import Config
    import time
    config = Config()
    parser = config.parser
    hparams = parser.parse_args()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    reader = Reader(hparams)
    start = time.time()
    logger.info("Loading data...")
    reader.load_data()
    end = time.time()
    logger.info("Loaded. {} secs".format(end-start))
    with open(os.path.join(hparams.data_path, "reader_data.json"), "w") as f:
        json.dump(reader.train, f, indent=2)
    start = time.time()
    logger.info("Making batches...")
    iterator = reader.make_batch(reader.train)
    end = time.time()
    logger.info("Making batch finished. {} secs".format(end-start))
    inputs, contexts, spans = reader.make_input(next(iterator))
    print("")
            

