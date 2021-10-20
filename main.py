import os
import torch
import argparse


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

    args = parser.parse_args()
    args.if_selection = True if args.if_selection == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)





if __name__ == '__main__':
    main()

