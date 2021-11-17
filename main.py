import os, time
import torch
import torch.nn as nn
import argparse
import deepcore.nets as nets
import deepcore.datasets as datasets
import deepcore.methods as methods
from utils import train, test, str_to_bool
from torchvision import models


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--if_selection', type=str, default="True", help="if selection is preformed")
    parser.add_argument('--selection', type=str, default="uniform", help="selection method")
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--gpu', default=None, nargs="+", type=int, help='GPU id to use')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')
    parser.add_argument('--batch', '--batch-size', "-b", default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument("--train_batch", "-tb", default=None, type=int,
                        help="batch size for training, if not specified, it will equal to batch size in argument --batch")
    parser.add_argument("--selection_batch", "-sb", default=None, type=int,
                        help="batch size for selection, if not specified, it will equal to batch size in argument --batch")
    parser.add_argument('--print_freq', '-p', default=20, type=int,
                        help='print frequency (default: 20)')
    parser.add_argument("--test_interval", '-ti', default=1, type=int, help=
                        "the number of training epochs to be preformed between two test epochs； a value of 0 means no test will be run (default: 1)")
    parser.add_argument("--test_fraction", '-tf', type=float, default=1., help="proportion of test dataset used for evaluating the model (default: 1.)")
    parser.add_argument('--optimizer', default="SGD", help='optimizer to use, e.g. SGD, Adam')
    parser.add_argument("--nesterov", default=True, type=str_to_bool, help="if set nesterov")
    parser.add_argument('--fraction', default=0.1, type=float, help='fraction of data to be selected (default: 0.1)')
    parser.add_argument('--seed', default=int(time.time() * 1000) % 100000, type=int, help="random seed")
    parser.add_argument("--selection_epochs", "-se", default=40, type=int,
                        help="number of epochs whiling performing selection on full dataset")
    parser.add_argument('--selection_momentum', '-sm', default=0.9, type=float, metavar='M',
                        help='momentum whiling performing selection (default: 0.9)')
    parser.add_argument('--swd', '--selection_weight_decay', default=5e-4, type=float,
                        metavar='W', help='weight decay whiling performing selection (default: 5e-4)',
                        dest='selection_weight_decay')
    parser.add_argument('--selection_optimizer', "-so", default="SGD", help='optimizer to use whiling performing selection, e.g. SGD, Adam')
    parser.add_argument("--selection_nesterov", "-sn", default=True, type=str_to_bool, help="if set nesterov whiling performing selection")
    parser.add_argument('--selection_lr', '-slr', type=float, default=0.1, help='learning rate for selection')
    parser.add_argument("--selection_test_interval", '-sti', default=1, type=int, help=
    "the number of training epochs to be preformed between two test epochs during selection (default: 1)")
    parser.add_argument("--selection_test_fraction", '-stf', type=float, default=1.,
                        help="proportion of test dataset used for evaluating the model while preforming selection (default: 1.)")
    parser.add_argument('--balance', default=True, type=str_to_bool, help="whether balance selection is performed per class")

    args = parser.parse_args()
    args.if_selection = True if args.if_selection == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # args.device = "cpu"

    if args.train_batch is None:
        args.train_batch = args.batch
    if args.selection_batch is None:
        args.selection_batch = args.batch

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[args.dataset] \
        (args.data_path)
    args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n ' % exp)

        torch.random.manual_seed(args.seed)

        if args.dataset == "ImageNet":
            # Use torchvision models
            pass  ########
        else:
            network = nets.__dict__[args.model](channel, num_classes, im_size).to(args.device)

        if args.device == "cpu":
            print("Using CPU.")
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            network = network.cuda(args.gpu)
        elif torch.cuda.device_count() > 1:
            network = nn.DataParallel(network).cuda()

        criterion = nn.CrossEntropyLoss().to(args.device)

        optimizer = torch.optim.__dict__[args.optimizer](network.parameters(), args.lr, momentum=args.momentum,
                                                         weight_decay=args.weight_decay, nesterov=args.nesterov)

        if args.if_selection:
            #cc,ii,nnn,_,_,_,dst_pretrain,_=datasets.permutedMNIST(args.data_path)
            method = methods.__dict__[args.selection](dst_train, args, args.fraction, args.seed,
                                                      epochs=args.selection_epochs, selection_method='Entropy',
                                                      specific_model=None, balance=args.balance, network=network,optimizer=optimizer, criterion=criterion, torchvision_pretrain=False,
                                                      fraction_pretrain=.3,#dst_pretrain_dict={"channel":cc,"num_classes":nnn,"im_size":ii,"dst_train":dst_pretrain},
                                                      dst_test=dst_test,metric="euclidean")
            subset_ind = method.select()
            print(len(subset_ind))
            subset = torch.utils.data.Subset(dst_train, subset_ind)
            train_loader = torch.utils.data.DataLoader(subset, batch_size=args.train_batch, shuffle=True,
                                                       num_workers=2, pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.train_batch, shuffle=True, num_workers=2,
                                                       pin_memory=True)

        test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.train_batch, shuffle=True, num_workers=2,
                                                  pin_memory=True)

        # cosine learning rate
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.epochs)

        best_prec1 = 0.0
        args.start_epoch = 0

        for epoch in range(args.start_epoch, args.epochs):
            # train for one epoch
            train(train_loader, network, criterion, optimizer, scheduler, epoch, args)

            # evaluate on validation set
            if args.test_interval > 0 and (epoch+1) % args.test_interval == 0:
                prec1 = test(test_loader, network, criterion, args)

                # remember best prec@1 and save checkpoint
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
            '''
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)
            '''
        print('Best accuracy: ', best_prec1)


if __name__ == '__main__':
    main()
