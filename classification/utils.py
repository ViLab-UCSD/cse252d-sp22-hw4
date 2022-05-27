import torch

def loop_iterable(iterable):
    while True:
        yield from iterable

def test_target(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for data in iter_test:
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels.cuda()
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.float()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).detach() / float(all_label.size()[0])
    return accuracy

def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1
    return optimizer