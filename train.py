import os
import sys
import logging
import time
import random
import math
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from tqdm import tqdm

from model.dst import DST
from config import Config
from reader import Reader
import ontology


def learning_rate_schedule(global_step, max_iter, hparams):
    """Linear warmup & linear decay."""

    step = np.float32(global_step+1)
    a = hparams.max_lr / (hparams.warmup_steps - max_iter*hparams.max_epochs)
    b = hparams.max_lr - a*hparams.warmup_steps
    
    return min((hparams.max_lr - hparams.init_lr) / hparams.warmup_steps * step, a*step + b)

def train(model, reader, optimizer, writer, hparams):
    iterator = reader.make_batch(reader.train)
    t = tqdm(enumerate(iterator), total=train.max_iter, ncols=150)
    for batch_idx, batch in t:
        inputs, contexts, spans = reader.make_input(batch)
        turns = len(inputs)
        total_loss = 0
        loss_count = 0  # number of small batches in a iteration
        slot_acc = 0
        slot_count = 0
        joint_acc = 0

        # learning rate scheduling
        for param in optimizer.param_groups:
            param["lr"] = learning_rate_schedule(train.global_step, train.max_iter, hparams)

        batch_size = contexts[0].size(0)

        for turn_idx in range(turns):
            # split batches for gpu memory
            context_len = contexts[turn_idx].size(1)
            if context_len >= 450:
                small_batch_size = min(int(hparams.batch_size / 8), batch_size)
            elif context_len >= 300:
                small_batch_size = min(int(hparams.batch_size / 4), batch_size)
            elif context_len >= 200:
                small_batch_size = min(int(hparams.batch_size / 2), batch_size)
            else:
                small_batch_size = batch_size

            joint = torch.zeros((batch_size), len(ontology.all_info_slots))  # joint: [batch, slots]
            for slot_idx in range(len(ontology.all_info_slots)):
                for small_batch_idx in range(math.ceil(batch_size/small_batch_size)):
                    small_inputs = {}
                    for key, value in inputs[turn_idx].items():
                        small_inputs[key] = value[small_batch_size*small_batch_idx:small_batch_size*(small_batch_idx+1)]
                    small_contexts = contexts[turn_idx][small_batch_size*small_batch_idx:small_batch_size*(small_batch_idx+1)]
                    small_spans = spans[turn_idx][small_batch_size*small_batch_idx:small_batch_size*(small_batch_idx+1)]
                    optimizer.zero_grad()
                    loss, acc = model.forward(small_inputs, small_contexts, small_spans, slot_idx)  # acc: [batch]

                    total_loss += loss.item() * small_contexts.size(0)
                    loss_count += small_contexts.size(0)
                    slot_acc += acc.sum(dim=0).item()
                    slot_count += small_contexts.size(0)
                    joint[small_batch_size*small_batch_idx:small_batch_size*(small_batch_idx+1), slot_idx] = acc
                    
                    loss.backward()
                    optimizer.step()
                    torch.cuda.empty_cache()
            joint_acc += (joint.mean(dim=1) == 1).sum(dim=0).item()

        # for turn_idx in range(turns):
        #     for slot_idx in range(len(ontology.all_info_slots)):
        #         optimizer.zero_grad()
        #         loss = model.forward(inputs[turn_idx], contexts[turn_idx], spans[turn_idx], slot_idx)
        #         loss.backward()
        #         optimizer.step()
        #         torch.cuda.empty_cache()

        total_loss = total_loss / loss_count
        slot_acc = slot_acc / slot_count * 100
        joint_acc = joint_acc / (slot_count / len(ontology.all_info_slots)) * 100
        train.global_step += 1
        writer.add_scalar("Train/loss", total_loss, train.global_step)
        t.set_description("iter: {}, loss: {:.4f}, joint accuracy: {:.4f}, slot accuracy: {:.4f}".format(batch_idx+1, total_loss, joint_acc, slot_acc))
        time.sleep(0.1)
        # logger.info("iter: {}, loss: {}".format(batch_idx+1, loss.item()))

def validate(model, reader, hparams):
    model.eval()
    val_loss = 0
    loss_count = 0
    slot_acc = 0
    slot_count = 0
    joint_acc = 0
    with torch.no_grad():
        iterator = reader.make_batch(reader.dev)
        t = tqdm(enumerate(iterator), total=validate.max_iter, ncols=150)
        for batch_idx, batch in t:
            inputs, contexts, spans = reader.make_input(batch)
            turns = len(inputs)

            batch_size = contexts[0].size(0)

            for turn_idx in range(turns):
                # split batches for gpu memory
                context_len = contexts[turn_idx].size(1)
                if context_len >= 450:
                    small_batch_size = min(int(hparams.batch_size / 8), batch_size)
                elif context_len >= 300:
                    small_batch_size = min(int(hparams.batch_size / 4), batch_size)
                elif context_len >= 200:
                    small_batch_size = min(int(hparams.batch_size / 2), batch_size)
                else:
                    small_batch_size = batch_size

                joint = torch.zeros((batch_size), len(ontology.all_info_slots))  # joint: [batch, slots]
                for slot_idx in range(len(ontology.all_info_slots)):
                    for small_batch_idx in range(math.ceil(batch_size/small_batch_size)):
                        small_inputs = {}
                        for key, value in inputs[turn_idx].items():
                            small_inputs[key] = value[small_batch_size*small_batch_idx:small_batch_size*(small_batch_idx+1)]
                        small_contexts = contexts[turn_idx][small_batch_size*small_batch_idx:small_batch_size*(small_batch_idx+1)]
                        small_spans = spans[turn_idx][small_batch_size*small_batch_idx:small_batch_size*(small_batch_idx+1)]
                        loss, acc = model.forward(small_inputs, small_contexts, small_spans, slot_idx, train=False)
                        
                        val_loss += loss.item() * small_contexts.size(0)
                        loss_count += small_contexts.size(0)
                        slot_acc += acc.sum(dim=0).item()
                        slot_count += small_contexts.size(0)
                        joint[small_batch_size*small_batch_idx:small_batch_size*(small_batch_idx+1), slot_idx] = acc
                        torch.cuda.empty_cache()
                joint_acc += (joint.mean(dim=1) == 1).sum(dim=0).item()
            t.set_description("iter: {}".format(batch_idx+1))
            time.sleep(0.1)
    model.train()
    val_loss = val_loss / loss_count
    slot_acc = slot_acc / slot_count * 100
    joint_acc = joint_acc / (slot_count / len(ontology.all_info_slots)) * 100

    return val_loss, joint_acc, slot_acc

def save(model, optimizer, save_path):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "amp": amp.state_dict()
    }
    torch.save(checkpoint, save_path)

def load(model, optimizer, save_path):
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    amp.load_state_dict(checkpoint["amp"])

if __name__ == "__main__":
    config = Config()
    parser = config.parser
    hparams = parser.parse_args()
    
    logger = logging.getLogger("DST")
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    if not os.path.exists("log"):
        os.mkdir("log")
    log_path = os.path.join("log", "{}".format(re.sub("\s+", "_", time.asctime())))
    writer = SummaryWriter(log_dir=log_path)

    if not os.path.exists("save"):
        os.mkdir("save")
    save_path = "save/model_{}.pt".format(re.sub("\s+", "_", time.asctime()))

    random.seed(hparams.seed)
    reader = Reader(hparams)
    start = time.time()
    logger.info("Loading data...")
    reader.load_data("train")
    end = time.time()
    logger.info("Loaded. {} secs".format(end-start))

    device = torch.device("cpu" if hparams.no_cuda else "cuda")
    model = DST(hparams).to(device)    
    optimizer = Adam(model.parameters(), hparams.init_lr)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    # load saved model, optimizer
    if hparams.save_path is not None:
        load(hparams.save_path)

    train.max_iter = len(list(reader.make_batch(reader.train)))
    validate.max_iter = len(list(reader.make_batch(reader.dev)))

    train.global_step = 0
    max_joint_acc = 0
    early_stop_count = hparams.early_stop_count
    for epoch in range(hparams.max_epochs):
        logger.info("Train...")
        start = time.time()
        train(model, reader, optimizer, writer, hparams)
        end = time.time()
        logger.info("epoch: {}, {:.4f} secs".format(epoch+1, end-start))

        logger.info("Validate...")
        loss, joint_acc, slot_acc = validate(model, reader, hparams)
        logger.info("loss: {:.4f}, joint accuracy: {:.4f}, slot accuracy: {:.4f}".format(loss, joint_acc, slot_acc))
        writer.add_scalar("Val/loss", loss, epoch+1)
        writer.add_scalar("Val/joint_acc", joint_acc, epoch+1)
        writer.add_scalar("Val/slot_acc", slot_acc, epoch+1)

        if joint_acc > max_joint_acc:  # save model
            save(model, optimizer, save_path)
            logger.info("Saved to {}.".format(os.path.abspath(save_path)))
            max_joint_acc = joint_acc
            early_stop_count = hparams.early_stop_count
        else:  # ealry stopping
            if early_stop_count == 0:
                logger.info("Early stopped.")
                break
            early_stop_count -= 1
            logger.info("early stop count: {}".format(early_stop_count))
    logger.info("Training finished.")
                
