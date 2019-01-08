import argparse


def option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
    parser.add_argument('--input_nc', type=int, default=1, help='num of input channels')
    parser.add_argument('--output_nc', type=int, default=2, help='num of output channels')
    parser.add_argument('--padding_type', type=str, default='reflect', help='padding type')
    parser.add_argument('--res_num', type=int, default=3, help='num of res block')
    parser.add_argument('--norm', type=str, default='instance', help='norm type')
    parser.add_argument('--dropout', type=bool, default=True, help='use dropout or not')
    parser.add_argument('--isTraining', type=bool, default=True, help='True for training, False for testing')
    parser.add_argument('--continue_train', type=bool, default=False, help='continue training or not')
    parser.add_argument('--dataset', type=str, default='./dataset', help='the path of dataset')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint', help='the path of saved model')
    parser.add_argument('--initial', type=str, default='normal', help='weight initializing type')
    parser.add_argument('--device', type=str, default='gpu', help='cuda device')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--img_size', type=int, default=256, help='input image size')
    opt = parser.parse_args([])
    return opt
