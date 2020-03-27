import os
import sys
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizerFast
import numpy as np

sys.path.append("../")
import ontology
from clean_data import clean_text


class DST(nn.Module):
    def __init__(self, hparams):
        """DS-DST belief tracker.

        Args:
            hparams: hyperparameters.

        Inputs: turn_input, turn_context, turn_span, slot_idx, train
            turn_inputs: Input of a turn including user utterance, system response, belief, gate, action.
            turn_context: Stacked concat of user utterance and system response.
            turn_span: Index of value spans in context.
            slot_idx: Index of slot on ontology.
            train: Whether train or not. Default: True

        Outputs:
            loss if train
            joint_acc, slot_acc else
            loss: Sum of gate loss, span loss and value loss. 
        """
        
        super(DST, self).__init__()
        self.context_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")  # use fine-tuning
        self.context_encoder.train()
        self.value_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased").requires_grad_(False)  # fix parameter
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.hidden_size = self.context_encoder.embeddings.word_embeddings.embedding_dim  # 768
        self.linear_gate = nn.Linear(self.hidden_size, 3)  # none, don't care, prediction
        self.linear_span = nn.Linear(self.hidden_size, 2)  # start, end
        self.value_ontology = json.load(open(os.path.join(hparams.data_path, "ontology_processed.json"), "r"))
        self.gate_loss_weight = torch.tensor([0.5, 1.0, 1.0])
        self.gate_criterion = torch.nn.NLLLoss(weight=self.gate_loss_weight, reduction="none")

        self.margin = hparams.margin
        self.use_span = hparams.use_span  # default False
        self.max_context_len = hparams.max_context_len
        self.max_value_len = hparams.max_value_len

    def forward(self, turn_input, turn_context, turn_span, first_turn=False, train=True):
        """
        turn_input: {
            "user": [batch, user_len],
            "response": [batch, response_len]
            "belief": [batch, slots, value_len] => list
            "gate": [batch, slots]
            "action": [batch, action_len]
        }
        turn_context: [batch, context_len] => list
        turn_span: [bathc, slots, 2]
        """
    
        batch_size = len(turn_context)
        loss = []
        acc = []
        belief_gen = [[]] * batch_size  # belief_gen: [batch, slots, value_len]

        for slot_idx in range(len(ontology.all_info_slots)):
            slot_ = ontology.all_info_slots[slot_idx]
            slot = self.tokenizer.encode(" ".join(slot_.split("-")))  # slot: [slot_len]
            slot_len = len(slot)

            value_label = torch.zeros((batch_size, self.max_value_len), dtype=torch.int64).cuda()  # value_label: [batch, value_len]

            # # use previous belief
            # if train:
            #     teacher_forcing = 1 if np.random.rand() >= 0.5 else 0
            # else:
            #     teacher_forcing = 0

            teacher_forcing = 0

            context = torch.zeros((batch_size, self.max_context_len), dtype=torch.int64).cuda()  # context: [batch, context_len]
            max_len = 0
            for idx in range(batch_size):
                if teacher_forcing:
                    slot_value = turn_input["belief"][idx][slot_idx]  # slot_value: [value_len]
                else:
                    if first_turn:
                        slot_value = self.tokenizer.encode("none")  # first turn doesn't have previous belief
                    else:
                        slot_value = turn_input["belief_gen"][idx][slot_idx]
                
                value_label[idx, :len(turn_input["belief"][idx][slot_idx] )] = torch.tensor(turn_input["belief"][idx][slot_idx], dtype=torch.int64).cuda()

                temp = slot[:-1] + slot_value[1:] + turn_context[idx] + [self.tokenizer.sep_token_id]  # [CLS] domain slot value [SEP] context [SEP]
                context[idx, :len(temp)] = torch.tensor(temp, dtype=torch.int64).cuda()
                if max_len < len(temp):
                    max_len = len(temp)
    
            context = context[:, :max_len]
            context_mask = (context != 0)

            outputs = self.context_encoder(context, attention_mask=context_mask)[0]  # output: [batch, context_len, hidden]
            gate_output = self.linear_gate(outputs[:, 0, :])  # gate_output: [batch, 3]
            gate_output = F.log_softmax(gate_output, dim=1)

            gate_label = turn_input["gate"][:, slot_idx]  # gate_label: [batch]
            loss_gate = self.gate_criterion(gate_output, gate_label)

            if self.use_span:  # use span-based method
                span_outputs = None
                for token_idx in range(outputs.size(1)):
                    if token_idx == 0:  # [CLS]
                        continue
                    token_output = outputs[:, token_idx, :]  # token_output: [batch, hidden]
                    token_output = self.linear_span(token_output)  # token_output: [batch, 2]
                    token_output = token_output.unsqueeze(2)  # token_output: [batch, 2, 1]
                    if span_outputs is None:
                        span_outputs = token_output
                    else:
                        span_outputs = torch.cat([span_outputs, token_output], dim=2)  # span_outputs: [batch, 2, context_len]
                span_outputs = F.log_softmax(span_outputs, dim=2)

                span_label = turn_span[:, slot_idx, :].clone()  # span_label: [batch, 2]
                span_mask = (span_label == 0)  # span is zero, if there are not value spans in context
                span_label += slot_len  # update span index after concat with slot
                span_label.masked_fill_(span_mask, 0)           
                loss_span = F.nll_loss(span_outputs[:, 0, :], span_label[:, 0], reduction="none") + F.nll_loss(span_outputs[:, 1, :], span_label[:, 1], reduction="none")

            value_probs = None
            value_list = self.value_ontology[slot_] + ["none"]
            for value in value_list:
                value_output = torch.tensor([self.tokenizer.encode(value)]).cuda()
                value_output = self.value_encoder(value_output)[0]  # value_outputs: [1, value_len, hidden]
                value_prob = torch.cosine_similarity(outputs[:, 0, :], value_output[:, 0, :], dim=1).unsqueeze(dim=1)  # value_prob: [batch, 1]
                if value_probs is None:
                    value_probs = value_prob
                else:
                    value_probs = torch.cat([value_probs, value_prob], dim=1)  # value_probs: [batch, value_nums]

            # cosine similarity of true value with context
            value_mask = (value_label != 0)
            true_value_output = self.value_encoder(value_label, attention_mask=value_mask)[0]  # true_value_output: [batch, value_len, hidden]
            true_value_probs = torch.cosine_similarity(outputs[:, 0, :], true_value_output[:, 0, :], dim=1).unsqueeze(dim=1)  # true_value_prob: [batch, 1]

            acc_slot = torch.ones(batch_size).cuda()  # acc: [batch]
            
            mask = (gate_label != gate_output.argmax(dim=1))
            acc_slot.masked_fill_(mask, 0)  # fail to predict gate

            pred_value = torch.zeros_like(value_label).cuda()  # pred_value: [batch, value_len]
            value_probs_ = value_probs.argmax(dim=1)  # value_probs: [batch]
            for batch_idx in range(batch_size):
                pred = self.tokenizer.encode(value_list[value_probs_[batch_idx]])
                belief_gen[batch_idx].append(pred)
                if gate_label[batch_idx] == ontology.gate_dict["prediction"] and value_label[batch_idx, :len(pred)].tolist() != pred:
                    acc_slot[batch_idx] = 0

            acc.append(acc_slot)

            # find max cosine similarity with context except true value
            true_value_mask = torch.zeros((batch_size, len(value_list)), dtype=torch.bool).cuda()  # true_value_mask: [batch, value_nums]
            for idx in range(batch_size):
                for v_idx, v in enumerate(value_list):
                    v = self.tokenizer.encode(v)
                    if v == value_label[idx][:len(v)]:
                        true_value_mask[idx, v_idx] = True
                        break
            value_probs.masked_fill_(true_value_mask, value=-1.0)
            max_value_probs = value_probs.max(dim=1, keepdim=True)[0]  # max_value_probs: [batch, 1]
            loss_value = torch.max(torch.cat([torch.zeros_like(true_value_probs), self.margin - true_value_probs + max_value_probs], dim=1),dim=1)[0]  # loss_value: [batch]

            # loss: [batch]
            if self.use_span:
                loss_slot = loss_gate + loss_span + loss_value
            else:
                loss_slot = loss_gate + loss_value

            loss.append(loss_slot)

        loss = torch.stack(loss, dim=1).sum(dim=1)  # loss: [batch]
        acc = torch.stack(acc, dim=1)  # acc: [batch, slot]
        turn_input["belief_gen"] = belief_gen

        return loss, acc

            




    

                
                

