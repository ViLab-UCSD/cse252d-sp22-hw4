import torch
import torch.nn as nn
import model as model_no
import numpy as np
import argparse

from data_list import ImageList
import pre_process as prep

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class predictor(nn.Module):
    def __init__(self, feature_len, cate_num):
        super(predictor, self).__init__()
        self.classifier = nn.Linear(feature_len, cate_num)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.fill_(0.0)

    def forward(self, features):
        activations = self.classifier(features)
        return (activations)

class Encoder(nn.Module):
    def __init__(self, total_classes):
        super().__init__()
        self.model_fc = model_no.Resnet50Fc()
        feature_len = self.model_fc.output_num()
        self.bottleneck_0 = nn.Linear(feature_len, 256)
        self.bottleneck_0.weight.data.normal_(0, 0.005)
        self.bottleneck_0.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(self.bottleneck_0, nn.BatchNorm1d(256), nn.ReLU())
        self.classifier_layer = predictor(256, total_classes)

    def forward(self, x):
        features = self.model_fc(x)
        out_bottleneck = self.bottleneck_layer(features)
        logits = self.classifier_layer(out_bottleneck)
        return logits

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Domain Adaptation')

    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dataset', type=str, nargs='?', default='office31', help="target dataset")
    parser.add_argument('--target', type=str, nargs='?', default='dslr', help="target domain")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size should be samples * classes")
    parser.add_argument('--nClasses', type=int, default=31, help="#Classes")
    parser.add_argument('--checkpoint' , type=str, help="Checkpoint to load from.")
    parser.add_argument('--data_dir', required=True)

    args = parser.parse_args() 

    if args.dataset in ["office31"]:
        file_path = {
            "amazon": "./data/office31/amazon.txt" ,
            "webcam": "./data/office31/webcam.txt",
            "dslr": "./data/office31/dslr.txt" ,
        } 
        dataset_test = file_path[args.target]
    else:
        raise NotImplementedError

    dataset_loaders = {}

    dataset_list = ImageList(args.data_dir, open(dataset_test).readlines(), transform=prep.image_test(resize_size=256, crop_size=224))
    print("Size of target dataset:" , len(dataset_list))
    dataset_loaders["test"] = torch.utils.data.DataLoader(dataset_list, batch_size=args.batch_size, shuffle=False,
                                                          num_workers=16, drop_last=False)

    # network construction
    my_net = Encoder(args.nClasses)
    my_net = my_net.cuda()
        
    accuracy = AverageMeter()

    saved_state_dict = torch.load(args.checkpoint)
    my_net.load_state_dict(saved_state_dict, strict=True)
    my_net.eval()
    start_test = True
    iter_test = iter(dataset_loaders["test"])
    with torch.no_grad():
        for i in range(len(dataset_loaders['test'])):
            print("{0}/{1}".format(i,len(dataset_loaders['test'])) , end="\r")
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = my_net(inputs).detach()
            predictions = outputs.argmax(1)
            correct = torch.sum((predictions == labels).float())
            accuracy.update(correct/len(outputs), len(outputs))
    print_str = "\nCorrect Predictions: {}/{}".format(int(accuracy.sum), accuracy.count)
    print_str1 = '\ntest_acc:{:.4f}'.format(accuracy.avg)
    print(print_str + print_str1)