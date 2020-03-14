import argparse


class Config:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="data/MultiWOZ_2.1", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_len", default=100, type=int)
    parser.add_argument("--max_value_len", default=20, type=int)
    parser.add_argument("--max_context_len", default=490, type=int)
    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument("--margin", default=0.5, type=float)
    parser.add_argument("--init_lr", default=1e-4, type=float)
    parser.add_argument("--max_lr", default=0.1, type=float)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--max_epochs", default=100, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--early_stop_count", default=5, type=int)
    parser.add_argument("--use_span", action="store_true", default=False)
    parser.add_argument("--save_path", default=None, type=str)