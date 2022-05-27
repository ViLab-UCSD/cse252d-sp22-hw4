import torch
import pdb
import torch.optim as optim
import torch.nn as nn
from network import Encoder, discriminatorCDAN, discriminatorDANN, AdversarialLayer
import numpy as np
from opts import parser
import os

from data_list import ImageList
import pre_process as prep
from utils import inv_lr_scheduler, test_target, loop_iterable

import time

from torch.utils.tensorboard import SummaryWriter


import numpy as np
seed=1234
torch.manual_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':

    args = parser.parse_args() 
    if args.method == "none":
        args.adv_loss = 0.

    out_dir = os.path.join("snapshot" , args.dataset , args.out_dir )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "log.txt")
    log_acc = os.path.join(out_dir, "logAcc.txt")
    print("Writing log to" , out_file)
    out_file = open(out_file, "w")
    best_file = os.path.join(out_dir, "best.txt")
    print(args)

    ##### TensorBoard & Misc Setup #####
    writer_loc = os.path.join(out_dir , 'tensorboard_logs')
    writer = SummaryWriter(writer_loc)

    if args.dataset in ["office31"]:
        file_path = {
            "amazon": "./data/office31/amazon.txt" ,
            "webcam": "./data/office31/webcam.txt",
            "dslr": "./data/office31/dslr.txt" ,
        }
        print("Source: " , args.source)
        print("Target" , args.target)
        dataset_source = file_path[args.source]
        dataset_target = file_path[args.target]
        dataset_test = file_path[args.target]

        batch_size = {"train": args.batch_size, "val": args.batch_size*4}

        out_file.write('all args = {}\n'.format(args))
        out_file.flush()

        dataset_loaders = {}

        print(dataset_source)

        dataset_list = ImageList(args.data_dir, open(dataset_source).readlines(),
                                transform=prep.image_train(resize_size=256, crop_size=224))
        
        print(f"{len(dataset_list)} source samples")

        dataset_loaders["source"] = torch.utils.data.DataLoader(dataset_list, batch_size=batch_size['train'],
                                                            shuffle=True, num_workers=8,
                                                                drop_last=True)


        dataset_list = ImageList(args.data_dir, open(dataset_target).readlines(),
                                transform=prep.image_train(resize_size=256, crop_size=224))
        dataset_loaders["target"] = torch.utils.data.DataLoader(dataset_list, batch_size=batch_size['train'], shuffle=True,
                                                            num_workers=8, drop_last=True)
        print(f"{len(dataset_list)} target samples")


        dataset_list = ImageList(args.data_dir, open(dataset_test).readlines(),
                                    transform=prep.image_test(resize_size=256, crop_size=224))
        dataset_loaders["test"] = torch.utils.data.DataLoader(dataset_list, batch_size=batch_size['val'], shuffle=False,
                                                            num_workers=8)
        print(f"{len(dataset_list)} target test samples")

        batch_iterator = zip(loop_iterable(dataset_loaders["source"]) , loop_iterable(dataset_loaders["target"]))
    else: 
        raise NotImplementedError


    # network construction
    base_network = Encoder(args.bn_dim, args.total_classes)
    base_network = base_network.cuda()

    if args.method == 'DANN':
        discriminator_net = discriminatorDANN(args.bn_dim)
    elif args.method == 'CDAN':
        discriminator_net = discriminatorCDAN(args.bn_dim, args.total_classes)
    elif args.method == "none":
        discriminator_net = None
    else:
        raise Exception('{} not implemented'.format(args.method))

    # domain discriminator
    if discriminator_net is not None:
        discriminator_net = discriminator_net.cuda()
        discriminator_net.train(True)

    # gradient reversal layer
    grl = AdversarialLayer()

    # criterion and optimizer
    criterion = {
        "classifier" : nn.CrossEntropyLoss(),
        "adversarial": nn.BCEWithLogitsLoss()
    }

    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, base_network.model_fc.parameters()), "lr": 0.1},
        {"params": filter(lambda p: p.requires_grad, base_network.bottleneck_0.parameters()), "lr": 1},
        {"params": filter(lambda p: p.requires_grad, base_network.classifier_layer.parameters()), "lr": 1},
    ]

    if discriminator_net is not None:
        optimizer_dict += [{"params": filter(lambda p: p.requires_grad, discriminator_net.parameters()), "lr": 1}]

    optimizer = optim.SGD(optimizer_dict, momentum=0.9, weight_decay=0.0005)

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])

    len_source = len(dataset_loaders["source"]) - 1
    len_target = len(dataset_loaders["target"]) - 1
    iter_source = iter(dataset_loaders["source"])
    iter_target = iter(dataset_loaders["target"])

    best_acc = 0

    len_source = len(dataset_loaders["source"]) - 1
    len_target = len(dataset_loaders["target"]) - 1
    iter_source = iter(dataset_loaders["source"])
    iter_target = iter(dataset_loaders["target"])

    with open(os.path.join(out_dir , "best.txt"), "a") as fh:
        fh.write("Best Accuracy file\n")

    start_iter=1

    start_time = time.time()
    for iter_num in range(start_iter, args.max_iteration + 1):
        base_network.train(True)
        optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=args.lr, gamma=0.001, power=0.75)
        optimizer.zero_grad()
        print("Iter:" , iter_num , end="\r")

        # import pdb; pdb.set_trace()
        source_data, target_data = next(batch_iterator)
        inputs_source, labels_source = source_data
        inputs_target , _ = target_data

        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        inputs = inputs.cuda()

        labels_source = labels_source.cuda()
        assert len(inputs_source) == len(inputs_target)
        domain_labels = torch.tensor([[1], ] * len(inputs_source)+ [[0], ] * len(inputs_target), device=torch.device('cuda:0'), dtype=torch.float)

        features, logits = base_network(inputs)
        logits_source = logits[:len(inputs_source)]
        logits_target = logits[len(inputs_source):]

        ## Classifier Loss
        classifier_loss = criterion["classifier"](logits_source, labels_source)
        writer.add_scalar("Loss/classifier_loss" , classifier_loss.detach().item(), iter_num)

        ## adversarial loss
        if discriminator_net is not None:
            domain_predicted = discriminator_net(grl.apply(features), torch.softmax(logits, dim=1).detach())
            transfer_loss = criterion["adversarial"](domain_predicted, domain_labels)
            transfer_loss = args.adv_loss*transfer_loss
            writer.add_scalar("Loss/transfer" , transfer_loss.detach().item(), iter_num)
            ## Total Loss
            total_loss = classifier_loss + transfer_loss
        else:
            total_loss = classifier_loss

        total_loss.backward()
        optimizer.step()

        # test
        test_interval = args.test_iter
        if iter_num % test_interval == 0:
            start_time = time.time()
            base_network.eval()
            test_acc = test_target(dataset_loaders, base_network)
            writer.add_scalar("Acc/test" , test_acc , iter_num)
            print_str1 = '\niter: {:05d}, test_acc:{:.4f}\n'.format(iter_num, test_acc)
            print(print_str1)

            if test_acc > best_acc:
                best_acc = test_acc
                best_model = base_network.state_dict()
                with open(os.path.join(out_dir , "best.txt"), "a") as fh:
                    fh.write("Best Accuracy : {:.4f} at iter: {:05d}\n".format(best_acc, iter_num))
                torch.save(best_model , os.path.join(out_dir , "best_model.pth.tar"))