import time, torch
from argparse import ArgumentTypeError


class WeightedSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices, weights) -> None:
        self.dataset = dataset
        assert len(indices) == len(weights)
        self.indices = indices
        self.weights = weights
    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]], self.weights[[i for i in idx]]
        return self.dataset[self.indices[idx]], self.weights[idx]

def train(train_loader, network, criterion, optimizer, scheduler, epoch, args, if_weighted: bool=False):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to train mode
    network.train()

    end = time.time()
    for i, contents in enumerate(train_loader):
        optimizer.zero_grad()
        if if_weighted:
            target = contents[0][1].to(args.device)
            input = contents[0][0].to(args.device)

            # compute output
            output = network(input)
            weights = contents[1].to(args.device).requires_grad_(False)
            loss = torch.sum(criterion(output, target) * weights) / torch.sum(weights)
        else:
            target = contents[1].to(args.device)
            input = contents[0].to(args.device)

            # compute output
            output = network(input)
            loss = criterion(output, target).mean()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        #optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))


def test(test_loader, network, criterion, args):
    network.no_grad = True
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    network.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.to(args.device)
        input = input.to(args.device)

        # compute output
        with torch.no_grad():
            output = network(input)

        loss = criterion(output, target).mean()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(test_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    network.no_grad = False
    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def save_checkpoint(state, path, epoch, prec):
    print("=> Saving checkpoint for epoch %d, with Prec@1 %f." % (epoch, prec))
    torch.save(state, path)
