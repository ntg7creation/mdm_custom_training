import argparse
import json

def train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='train_config.json')

    args, _ = parser.parse_known_args()

    # Load from JSON
    with open(args.config, 'r') as f:
        config = json.load(f)

    for k, v in config.items():
        parser.add_argument(f'--{k}', default=v, type=type(v) if not isinstance(v, bool) else lambda x: x.lower() == 'true')

    return parser.parse_args()
