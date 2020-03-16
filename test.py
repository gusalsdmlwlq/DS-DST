import logging
import time
import random
import math

import torch
from apex import amp
from tqdm import tqdm

from model.dst import DST
from config import Config
from reader import Reader
import ontology


def test(model, reader, hparams):
    model.eval()
    slot_acc = 0
    slot_count = 0
    joint_acc = 0
    with torch.no_grad():
        iterator = reader.make_batch(reader.test)
        t = tqdm(enumerate(iterator), total=test.max_iter, ncols=150)
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
                        _, acc = model.forward(small_inputs, small_contexts, small_spans, slot_idx, train=False)  # acc: [batch]

                        slot_acc += acc.sum(dim=0).item()
                        slot_count += small_contexts.size(0)
                        joint[small_batch_size*small_batch_idx:small_batch_size*(small_batch_idx+1), slot_idx] = acc
                        torch.cuda.empty_cache()
                joint_acc += (joint.mean(dim=1) == 1).sum(dim=0).item()
            t.set_description("iter: {}".format(batch_idx+1))
            time.sleep(0.1)
    slot_acc = slot_acc / slot_count * 100
    joint_acc = joint_acc / (slot_count / len(ontology.all_info_slots)) * 100

    return joint_acc, slot_acc

def load(model, save_path):
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint["model"])
    amp.load_state_dict(checkpoint["amp"])

if __name__ == "__main__":
    config = Config()
    parser = config.parser
    hparams = parser.parse_args()
    if hparams.save_path is None:
        raise Exception("Save path is required. e.g) --save_path=='save/model_Sat_Mar_14_16:08:06_2020.pt'")
    save_path = hparams.save_path

    logger = logging.getLogger("DST")
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    random.seed(hparams.seed)
    reader = Reader(hparams)
    start = time.time()
    logger.info("Loading data...")
    reader.load_data("test")
    end = time.time()
    logger.info("Loaded. {} secs".format(end-start))

    device = torch.device("cpu" if hparams.no_cuda else "cuda")
    model = DST(hparams).to(device)
    model = amp.initialize(model, opt_level="O1", verbosity=0)
    
    load(model, save_path)  # load saved model, optimizer

    test.max_iter = len(list(reader.make_batch(reader.test)))
    logger.info("Test...")
    joint_acc, slot_acc = test(model, reader, hparams)
    logger.info("joint accuracy: {:.4f}, slot accuracy: {:.4f}".format(joint_acc, slot_acc))