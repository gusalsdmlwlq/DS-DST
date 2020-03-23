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
from apex import amp, parallel
from tqdm import tqdm

from model.dst import DST
from config import Config
from reader import Reader
import ontology


def learning_rate_schedule(global_step, max_iter, hparams):
    """Linear warmup & linear decay."""

    step = np.float32(global_step+1)
    a = hparams.lr / (train.warmup_steps - max_iter * hparams.max_epochs)
    b = hparams.lr - a*train.warmup_steps
    
    return min(hparams.lr / train.warmup_steps * step, a*step + b)

def distribute_data(batches, num_gpus):
    distributed_data = []
    if len(batches) % num_gpus == 0:
        batch_size = int(len(batches) / num_gpus)
        for idx in range(num_gpus):
            distributed_data.append(batches[batch_size*idx:batch_size*(idx+1)])
    else:
        batch_size = math.ceil(len(batches) / num_gpus)
        expanded_batches = batches.clone()
        while True:
            expanded_batches = torch.cat([expanded_batches, batches.clone()], dim=0)
            if len(expanded_batches) >= batch_size*num_gpus:
                expanded_batches = expanded_batches[:batch_size*num_gpus:]
                break
        for idx in range(num_gpus):
            distributed_data.append(expanded_batches[batch_size*idx:batch_size*(idx+1)])
    return distributed_data

def train(model, reader, optimizer, writer, hparams):
    iterator = reader.make_batch(reader.train)

    if hparams.local_rank == 0:  # only one process prints something
        t = tqdm(enumerate(iterator), total=train.max_iter, ncols=150)
    else:
        t = enumerate(iterator)

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
            distributed_batch_size = math.ceil(batch_size / hparams.num_gpus)
            
            # split batches for gpu memory
            context_len = contexts[turn_idx].size(1)
            if context_len >= 410:
                small_batch_size = min(int(hparams.batch_size/hparams.num_gpus / 8), distributed_batch_size)
            elif context_len >= 260:
                small_batch_size = min(int(hparams.batch_size/hparams.num_gpus / 4), distributed_batch_size)
            elif context_len >= 160:
                small_batch_size = min(int(hparams.batch_size/hparams.num_gpus / 2), distributed_batch_size)
            else:
                small_batch_size = distributed_batch_size
            
            # distribute batches to each gpu
            for key, value in inputs[turn_idx].items():
                inputs[turn_idx][key] = distribute_data(value, hparams.num_gpus)[hparams.local_rank]
            contexts[turn_idx] = distribute_data(contexts[turn_idx], hparams.num_gpus)[hparams.local_rank]
            spans[turn_idx] = distribute_data(spans[turn_idx], hparams.num_gpus)[hparams.local_rank]

            joint = torch.zeros((distributed_batch_size, len(ontology.all_info_slots)))  # joint: [batch, slots]
            for slot_idx in range(len(ontology.all_info_slots)):
                for small_batch_idx in range(math.ceil(distributed_batch_size/small_batch_size)):
                    small_inputs = {}
                    for key, value in inputs[turn_idx].items():
                        small_inputs[key] = value[small_batch_size*small_batch_idx:small_batch_size*(small_batch_idx+1)]
                    small_contexts = contexts[turn_idx][small_batch_size*small_batch_idx:small_batch_size*(small_batch_idx+1)]
                    small_spans = spans[turn_idx][small_batch_size*small_batch_idx:small_batch_size*(small_batch_idx+1)]
                    optimizer.zero_grad()
                    loss, acc = model.forward(small_inputs, small_contexts, small_spans, slot_idx)  # loss: [batch], acc: [batch]

                    loss = loss.mean()

                    total_loss += loss.item() * small_contexts.size(0)
                    loss_count += small_contexts.size(0)
                    slot_acc += acc.sum(dim=0).item()
                    slot_count += small_contexts.size(0)
                    joint[small_batch_size*small_batch_idx:small_batch_size*(small_batch_idx+1), slot_idx] = acc
                    
                    # distributed training
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()

                    optimizer.step()
                    torch.cuda.empty_cache()
                    
            joint_acc += (joint.mean(dim=1) == 1).sum(dim=0).item()
            
        total_loss = total_loss / loss_count
        slot_acc = slot_acc / slot_count * 100
        joint_acc = joint_acc / (slot_count / len(ontology.all_info_slots)) * 100
        train.global_step += 1
        if hparams.local_rank == 0:
            writer.add_scalar("Train/loss", total_loss, train.global_step)
            t.set_description("iter: {}, loss: {:.4f}, joint accuracy: {:.4f}, slot accuracy: {:.4f}".format(batch_idx+1, total_loss, joint_acc, slot_acc))

def validate(model, reader, hparams):
    model.eval()
    val_loss = 0
    loss_count = 0
    slot_acc = 0
    slot_count = 0
    joint_acc = 0
    with torch.no_grad():
        iterator = reader.make_batch(reader.dev)

        if hparams.local_rank == 0:
            t = tqdm(enumerate(iterator), total=validate.max_iter, ncols=150)
        else:
            t = enumerate(iterator)

        for batch_idx, batch in t:
            inputs, contexts, spans = reader.make_input(batch)
            turns = len(inputs)

            batch_size = contexts[0].size(0)

            for turn_idx in range(turns):
                distributed_batch_size = math.ceil(batch_size / hparams.num_gpus)

                for key, value in inputs[turn_idx].items():
                    inputs[turn_idx][key] = distribute_data(value, hparams.num_gpus)[hparams.local_rank]
                contexts[turn_idx] = distribute_data(contexts[turn_idx], hparams.num_gpus)[hparams.local_rank]
                spans[turn_idx] = distribute_data(spans[turn_idx], hparams.num_gpus)[hparams.local_rank]

                joint = torch.zeros((distributed_batch_size, len(ontology.all_info_slots)))  # joint: [batch, slots]
                for slot_idx in range(len(ontology.all_info_slots)):
                    loss, acc = model.forward(inputs[turn_idx], contexts[turn_idx], spans[turn_idx], slot_idx, train=False)

                    loss = loss.mean()

                    val_loss += loss.item() * contexts[0].size(0)
                    loss_count += contexts[0].size(0)
                    slot_acc += acc.sum(dim=0).item()
                    slot_count += contexts[0].size(0)
                    joint[:, slot_idx] = acc
                    torch.cuda.empty_cache()

                joint_acc += (joint.mean(dim=1) == 1).sum(dim=0).item()

            if hparams.local_rank == 0:
                t.set_description("iter: {}".format(batch_idx+1))

    model.train()
    model.value_encoder.eval()  # fix value encoder
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
    checkpoint = torch.load(save_path, map_location = lambda storage, loc: storage.cuda(hparams.local_rank))
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    amp.load_state_dict(checkpoint["amp"])

if __name__ == "__main__":
    config = Config()
    parser = config.parser
    hparams = parser.parse_args()

    # distributed training
    hparams.distributed = False
    if 'WORLD_SIZE' in os.environ:
        hparams.distributed = int(os.environ['WORLD_SIZE']) > 1
    if hparams.distributed:
        torch.cuda.set_device(hparams.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.benchmark = True

    logger = logging.getLogger("DST")
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    if hparams.local_rank != 0:
        logger.setLevel(logging.WARNING)
    
    if hparams.local_rank == 0:
        # if not os.path.exists("log"):
        #     os.mkdir("log")
        # log_path = os.path.join("log", "{}".format(re.sub("\s+", "_", time.asctime())))
        writer = SummaryWriter()

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

    model = DST(hparams).cuda()
    optimizer = Adam(model.parameters(), hparams.lr)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    model = parallel.DistributedDataParallel(model)

    # load saved model, optimizer
    if hparams.save_path is not None:
        load(model, optimizer, hparams.save_path)
        torch.distributed.barrier()

    train.max_iter = len(list(reader.make_batch(reader.train)))
    validate.max_iter = len(list(reader.make_batch(reader.dev)))
    train.warmup_steps = train.max_iter * hparams.max_epochs * hparams.warmup_steps
    
    train.global_step = 0
    max_joint_acc = 0
    early_stop_count = hparams.early_stop_count

    for epoch in range(hparams.max_epochs):
        logger.info("Train...")
        start = time.time()
        if hparams.local_rank == 0:
            train(model, reader, optimizer, writer, hparams)
        else:
            train(model, reader, optimizer, None, hparams)
        end = time.time()
        logger.info("epoch: {}, {:.4f} secs".format(epoch+1, end-start))

        logger.info("Validate...")
        loss, joint_acc, slot_acc = validate(model, reader, hparams)
        logger.info("loss: {:.4f}, joint accuracy: {:.4f}, slot accuracy: {:.4f}".format(loss, joint_acc, slot_acc))
        if hparams.local_rank == 0:
            writer.add_scalar("Val/loss", loss, epoch+1)
            writer.add_scalar("Val/joint_acc", joint_acc, epoch+1)
            writer.add_scalar("Val/slot_acc", slot_acc, epoch+1)

        if joint_acc > max_joint_acc:  # save model
            if hparams.local_rank == 0:
                save(model, optimizer, save_path)
            torch.distributed.barrier()  # synchronize
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
                
