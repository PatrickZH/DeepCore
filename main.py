import os, time
import torch
import torch.nn as nn
import argparse
import nets, datasets
from utils import train, test

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--if_selection', type=str, default="True", help="if selection is preformed")
    parser.add_argument('--selection', type=str, default="Uniform", help="selection method")
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch', type=int, default=256, help='batch size for real data')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--gpu', default=None, nargs="+", type=int, help='GPU id to use.')
    parser.add_argument('--momentum', default=0.5, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        help='print frequency (default: 10)')

    args = parser.parse_args()
    args.if_selection = True if args.if_selection == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n ' % exp)

        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[args.dataset]\
            (args.data_path)
        torch.random.manual_seed(int(time.time() * 1000) % 100000)
        network = nets.__dict__[args.model](channel, num_classes).to(args.device)

        if args.device == "cpu":
            print("Using CPU.")
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            network = network.cuda(args.gpu)
        elif torch.cuda.device_count()>1:
            network = nn.DataParallel(network).cuda()

        criterion = nn.CrossEntropyLoss().to(args.device)

        optimizer = torch.optim.SGD(network.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        train_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                                   pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                                   pin_memory=True)

        for epoch in range(args.start_epoch, args.epochs):
            # train for one epoch
            train(train_loader, network, criterion, optimizer, epoch, args)

            # evaluate on validation set
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
