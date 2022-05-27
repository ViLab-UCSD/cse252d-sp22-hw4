import argparse
import os

parser = argparse.ArgumentParser(description='Domain Adaptation')

## Training parameters
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--dataset', default="office31", choices=["digits" , "office31"], help="Name of the dataset")
parser.add_argument('--source', type=str, nargs='?', default="amazon", choices=["amazon", "webcam", "dslr"], help="source dataset")
parser.add_argument('--target', type=str, nargs='?', default="webcam", choices=["amazon", "webcam", "dslr"],  help="target dataset")
parser.add_argument('--lr', type=float, nargs='?', default=0.03, help="target dataset")
parser.add_argument('--max_iteration', type=int, nargs='?', default=15500, help="target dataset")
parser.add_argument('--out_dir', type=str, nargs='?', default='exp_da', help="output dir")
parser.add_argument('--batch_size', type=int, default=32, help="batch size should be samples * classes")
parser.add_argument('--data_dir', type=str, default="./data", help="Path for data directory")
parser.add_argument('--total_classes', type=int, default=31, help="total # classes in the dataset")

## Testing parameters
parser.add_argument('--test_10crop', action="store_true", help="10 crop testing")
parser.add_argument('--test-iter', type=int, default=2000, help="Testing freq.")

## Architecture
parser.add_argument('--resnet', type=int, default=50, choices=[50,101,152], help="bottleneck embedding dimension")
parser.add_argument('--bn-dim', type=int, default=256, help="bottleneck embedding dimension")

## Adaptation parameters
parser.add_argument('--method', type=str, nargs='?', default='DANN', choices=['DANN', 'CDAN', 'none'])

## Loss coeffecients
parser.add_argument('--adv-loss', type=float, default=1., help="Adversarial Loss")