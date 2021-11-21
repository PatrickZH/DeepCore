import os, time
import torch
import torch.nn as nn
import argparse
import deepcore.nets as nets
import deepcore.datasets as datasets
import deepcore.methods as methods
from utils import train, test, str_to_bool, WeightedSubset, save_checkpoint
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # Basic arguments
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--selection', type=str, default="uniform", help="selection method")
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--gpu', default=None, nargs="+", type=int, help='GPU id to use')
    parser.add_argument('--print_freq', '-p', default=20, type=int,
                        help='print frequency (default: 20)')
    parser.add_argument('--fraction', default=0.1, type=float, help='fraction of data to be selected (default: 0.1)')
    parser.add_argument('--seed', default=int(time.time() * 1000) % 100000, type=int, help="random seed")

    # Optimizer and scheduler
    parser.add_argument('--optimizer', default="SGD", help='optimizer to use, e.g. SGD, Adam')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')
    parser.add_argument("--nesterov", default=True, type=str_to_bool, help="if set nesterov")
    parser.add_argument("--scheduler", default="CosineAnnealingLR", type=str, help=
                        "Learning rate scheduler")
    parser.add_argument("--gamma", type=float, default=.5, help="Gamma value for StepLR")
    parser.add_argument("--step_size", type=float, default=50, help="Step size for StepLR")

    # Training
    parser.add_argument('--batch', '--batch-size', "-b", default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument("--train_batch", "-tb", default=None, type=int,
                        help="batch size for training, if not specified, it will equal to batch size in argument --batch")
    parser.add_argument("--selection_batch", "-sb", default=None, type=int,
                        help="batch size for selection, if not specified, it will equal to batch size in argument --batch")

    # Testing
    parser.add_argument("--test_interval", '-ti', default=1, type=int, help=
    "the number of training epochs to be preformed between two test epochs; a value of 0 means no test will be run (default: 1)")
    parser.add_argument("--test_fraction", '-tf', type=float, default=1.,
                        help="proportion of test dataset used for evaluating the model (default: 1.)")

    # Selecting
    parser.add_argument("--selection_epochs", "-se", default=40, type=int,
                        help="number of epochs whiling performing selection on full dataset")
    parser.add_argument('--selection_momentum', '-sm', default=0.9, type=float, metavar='M',
                        help='momentum whiling performing selection (default: 0.9)')
    parser.add_argument('--selection_weight_decay', '-swd', default=5e-4, type=float,
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

    # Checkpoint and resumption
    parser.add_argument('--save_path', "-sp", type=str, default='', help='path to save results (default: do not save)')
    parser.add_argument('--resume', '-r', type=str, default='', help="path to latest checkpoint (default: do not load)")


    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # args.device = "cpu"

    if args.train_batch is None:
        args.train_batch = args.batch
    if args.selection_batch is None:
        args.selection_batch = args.batch
    if args.save_path != "" and not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[args.dataset] \
        (args.data_path)
    args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names

    if args.resume !="":
        # Load checkpoint
        try:
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            assert {"exp", "epoch", "state_dict", "opt_dict", "best_acc1", "subset", "sel_args"} <= set(checkpoint.keys())
            assert 'indices' in checkpoint["subset"].keys()
            start_exp = checkpoint['exp']
            start_epoch = checkpoint["epoch"]
        except:
            print("=> failed to load the checkpoint, an empty one will be created")
            checkpoint = {}
            start_exp = 0
            start_epoch = 0
    else:
        checkpoint = {}
        start_exp = 0
        start_epoch = 0

    if args.save_path != "":
        checkpoint_name = "{dst}_{net}_{mtd}_exp{exp}_epoch{epc}_{dat}_".format(dst=args.dataset, net=args.model,
                                                                                mtd=args.selection,
                                                                                dat=datetime.now(), exp=start_exp,
                                                                                epc=args.epochs)
    for exp in range(start_exp, args.num_exp):
        print('\n================== Exp %d ==================\n ' % exp)

        torch.random.manual_seed(args.seed)

        if "subset" in checkpoint.keys():
            subset = checkpoint['subset']
            selection_args = checkpoint["sel_args"]
        else:
            #cc,ii,nnn,_,_,_,dst_pretrain,dst_val=datasets.MNIST(args.data_path)
            selection_args = dict(epochs=args.selection_epochs, selection_method='Entropy',
                                  specific_model=None, balance=args.balance, #network=network,optimizer=optimizer, criterion=criterion,
                                  torchvision_pretrain=False,greedy="LazyGreedy",function="FacilityLocation",
                                  fraction_pretrain=1.#,dst_pretrain_dict={"channel":cc,"num_classes":nnn,"im_size":ii,"dst_train":dst_pretrain},
                                  #dst_test=dst_test,metric="euclidean"#,dst_val=dst_val
                                    )
            method = methods.__dict__[args.selection](dst_train, args, args.fraction, args.seed, **selection_args)
            subset = method.select()
        print(len(subset["indices"]))

        ##############
        ##############
        ##############
        # Augumentation
        from torchvision import transforms
        dst_train.transform = transforms.Compose([transforms.RandomCrop(args.im_size, padding=4), transforms.RandomHorizontalFlip(), dst_train.transform])
        ##############
        ##############
        ##############

        if_weighted = "weights" in subset.keys()
        if if_weighted:
            dst_subset = WeightedSubset(dst_train, subset["indices"], subset["weights"])
        else:
            dst_subset = torch.utils.data.Subset(dst_train, subset["indices"])

        train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.train_batch, shuffle=True,
                                                   num_workers=2, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.train_batch, shuffle=True, num_workers=2,
                                                  pin_memory=True)

        network = nets.__dict__[args.model](channel, num_classes, im_size).to(args.device)

        if args.device == "cpu":
            print("Using CPU.")
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            network = network.cuda(args.gpu)
        elif torch.cuda.device_count() > 1:
            network = nn.DataParallel(network).cuda()

        if "state_dict" in checkpoint.keys():
            # Loading model state_dict
            network.load_state_dict(checkpoint["state_dict"])

        criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)

        # Optimizer
        if args.optimizer == "SGD":
            optimizer = torch.optim.SGD(network.parameters(), args.lr, momentum=args.momentum,
                                                         weight_decay=args.weight_decay, nesterov=args.nesterov)
        elif args.optimizer == "Adam":
            optimizer = torch.optim.Adam(network.parameters(), args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.__dict__[args.optimizer](network.parameters(), args.lr, momentum=args.momentum,
                                                             weight_decay=args.weight_decay, nesterov=args.nesterov)

        if "opt_dict" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["opt_dict"])

        # LR scheduler
        if args.scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.epochs, last_epoch=start_epoch - 1)
        elif args.scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader) * args.step_size, gamma=args.gamma, last_epoch=start_epoch - 1)
        else:
            scheduler = torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer, last_epoch=start_epoch - 1)


        best_prec1 = checkpoint["best_acc1"] if "best_acc1" in checkpoint.keys() else 0.0

        for epoch in range(start_epoch, args.epochs):
            # train for one epoch
            train(train_loader, network, criterion, optimizer, scheduler, epoch, args, if_weighted=if_weighted)

            # evaluate on validation set
            if args.test_interval > 0 and (epoch+1) % args.test_interval == 0:
                prec1 = test(test_loader, network, criterion, args)

                # remember best prec@1 and save checkpoint
                is_best = prec1 > best_prec1

                if is_best:
                    best_prec1 = prec1
                    if args.save_path != "":
                        save_checkpoint({"exp": exp,
                            "epoch": epoch,
                            "state_dict": network.state_dict(),
                            "opt_dict": optimizer.state_dict(),
                            "best_acc1": best_prec1,
                            "subset": subset,
                            "sel_args": selection_args},
                            os.path.join(args.save_path, checkpoint_name + "unknown.ckpt"), epoch=epoch, prec=best_prec1)

        print('Best accuracy: ', best_prec1)
        start_epoch = 0

        # Prepare for the next checkpoint
        if args.save_path != "":
            os.rename(os.path.join(args.save_path, checkpoint_name + "unknown.ckpt"), os.path.join(args.save_path, checkpoint_name + "%f.ckpt"%best_prec1))
            checkpoint_name = "{dst}_{net}_{mtd}_exp{exp}_epoch{epc}_{dat}_".format(dst=args.dataset, net=args.model, mtd=args.selection,
                            dat=datetime.now(), exp=exp + 1, epc=args.epochs)
        checkpoint = {}


if __name__ == '__main__':
    main()
