import os
import sys
import json
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizerFast
import numpy as np

sys.path.append("../")
import ontology
from clean_data import clean_text


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, pad_mask=None):
        # inputs: [batch, context_len, hidden] including [CLS], [SEP]
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)

        attenion_score = torch.matmul(query, key.transpose(1,2)) / np.sqrt(self.hidden_size)
        
        if pad_mask is not None:
            pad_mask = ~pad_mask.unsqueeze(dim=1)
            attenion_score.masked_fill_(pad_mask, value=-float("inf"))
        
        attenion_score = F.softmax(attenion_score, dim=2)
        attenion_score = self.dropout(attenion_score)

        output = torch.matmul(attenion_score, value)

        return output

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
        self.slot_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased").requires_grad_(False)
        self.value_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased").requires_grad_(False)  # fix parameter
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.hidden_size = self.context_encoder.embeddings.word_embeddings.embedding_dim  # 768
        self.linear_gate = nn.Linear(self.hidden_size * 2, 3)  # none, don't care, prediction
        self.linear_span = nn.Linear(self.hidden_size, 2)  # start, end
        self.value_ontology = json.load(open(os.path.join(hparams.data_path, "ontology_processed.json"), "r"))
        self.gate_loss_weight = torch.tensor([0.5, 1.0, 1.0])
        self.gate_criterion = torch.nn.NLLLoss(weight=self.gate_loss_weight, reduction="none")
        # self.context_attention = SelfAttention(self.hidden_size, hparams.dropout)
        # self.context_attention = copy.deepcopy(self.context_encoder.transformer.layer[-1])
        # self.context_attention.train()

        self.margin = hparams.margin
        self.use_span = hparams.use_span  # default False

    def forward(self, turn_input, turn_context, turn_span, train=True):
        """
        turn_input: {
            "user": [batch, user_len],
            "response": [batch, response_len]
            "belief": [batch, slots, value_len]
            "gate": [batch, slots]
            "action": [batch, action_len]
        }
        turn_context: [batch, context_len]
        turn_span: [bathc, slots, 2]
        """
    
        batch_size = turn_context.size(0)
        loss = []
        acc = []
        context = turn_context  # context: [batch, context_len]
        context_mask = (context != 0)  # False if [PAD]

        context_outputs = self.context_encoder(context, attention_mask=context_mask)[0]  # output: [batch, context_len, hidden]

        for slot_idx in range(len(ontology.all_info_slots)):
            slot_ = ontology.all_info_slots[slot_idx]
            slot = torch.tensor(self.tokenizer.encode(" ".join(slot_.split("-"))))
            slot_len = slot.size(0)
            slot = slot.expand((batch_size, slot_len)).cuda()  # slot: [batch, slot_len]
            slot_outputs = self.slot_encoder(slot)[0]  # slot_outputs: [batch, slot_len, hidden]
            
            outputs = torch.cat([context_outputs[:, 0, :], slot_outputs[:, 0, :]], dim=1)  # outputs: [batch, hidden*2]
            # outputs = self.context_attention(outputs, context_mask)[0]  # outputs: [batch, context_len, hidden]
            
            gate_output = self.linear_gate(outputs)  # gate_output: [batch, 3]
            gate_output = F.log_softmax(gate_output, dim=1)

            gate_label = turn_input["gate"][:, slot_idx]  # gate_label: [batch]
            loss_gate = self.gate_criterion(gate_output, gate_label)

            if self.use_span:  # use span-based method
                span_outputs = None
                for token_idx in range(context_outputs.size(1)):
                    if token_idx == 0:  # [CLS]
                        continue
                    token_output = context_outputs[:, token_idx, :]  # token_output: [batch, hidden]
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
                value_output = torch.tensor([self.tokenizer.encode(value)])
                value_output = value_output.cuda()
                value_output = self.value_encoder(value_output)[0]  # value_outputs: [1, value_len, hidden]
                value_prob = torch.cosine_similarity(context_outputs[:, 0, :], value_output[:, 0, :], dim=1).unsqueeze(dim=1)  # value_prob: [batch, 1]
                if value_probs is None:
                    value_probs = value_prob
                else:
                    value_probs = torch.cat([value_probs, value_prob], dim=1)  # value_probs: [batch, value_nums]

            # cosine similarity of true value with context
            value_label = turn_input["belief"][:, slot_idx, :]  # value_label: [batch, value_len], including [PAD]
            value_mask = (value_label != 0)
            true_value_output = self.value_encoder(value_label, attention_mask=value_mask)[0]  # true_value_output: [batch, value_len, hidden]
            true_value_probs = torch.cosine_similarity(context_outputs[:, 0, :], true_value_output[:, 0, :], dim=1).unsqueeze(dim=1)  # true_value_prob: [batch, 1]

            acc_slot = torch.ones(batch_size).cuda()  # acc_slot: [batch]
            
            mask = (gate_label != gate_output.argmax(dim=1))
            acc_slot.masked_fill_(mask, 0)  # fail to predict gate

            value_probs_ = value_probs.argmax(dim=1)  # value_probs: [batch]
            for batch_idx in range(batch_size):
                pred = self.tokenizer.encode(value_list[value_probs_[batch_idx]])  # pred: [value_len]
                # fail to predict value
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

            # loss_slot: [batch]
            if self.use_span:
                loss_slot = loss_gate + loss_span + loss_value
            else:
                loss_slot = loss_gate + loss_value

            loss.append(loss_slot)
        
        loss = torch.stack(loss, dim=1).sum(dim=1)  # loss: [batch]
        acc = torch.stack(acc, dim=1)  # acc: [batch, slot]

        return loss, acc

            




    

                
                

