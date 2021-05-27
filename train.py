import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.datasets as datasets
from models import *
import google
import tarfile
from tqdm import tqdm
import pickle
import os
import argparse
import pickle
from google.cloud import storage
from torchvision import transforms, datasets


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50')
parser.add_argument('--dataset', default='imagenet', type=str,
                    help='name of the dataset (options: MNIST | CIFAR10 | CIFAR100 | SVHN, default: CIFAR10)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batchsize', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_scheduler', default=1, type=int, help='if cosine annealing or fix')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--stopcond', default=0.01, type=float,
                    help='stopping condtion based on the cross-entropy loss (default: 0.01)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--nlayers', default=50, type=int)
parser.add_argument('--width', default=64, type=int)
parser.add_argument('--seed', default=4, type=int)





if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def main():
    best_prec1 = 0
    global args
    args = parser.parse_args()
    save_dir = f'{args.arch}_{args.dataset}_{args.nlayers}_{args.width}'

    from google.cloud import storage

    ########### download train/test data/labels to the bucket

    if(args.dataset=='MNIST'):
        if "mlp" in args.arch:
            bucket_name = 'permutation-mlp'
            source_file_name = 'MNIST_Train_input_org.pkl'
            destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
            train_inputs = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST_Train_target_org.pkl'
            destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
            train_targets = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST_Test_input_org.pkl'
            destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
            test_inputs = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST_Test_target_org.pkl'
            destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
            test_targets = download_pkl(bucket_name, destination_blob_name)
        else:
            bucket_name = 'permutation-mlp'
            source_file_name = 'MNIST3d_Train_input_org.pkl'
            destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
            train_inputs = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST3d_Train_target_org.pkl'
            destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
            train_targets = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST3d_Test_input_org.pkl'
            destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
            test_inputs = download_pkl(bucket_name, destination_blob_name)

            source_file_name = 'MNIST3d_Test_target_org.pkl'
            destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
            test_targets = download_pkl(bucket_name, destination_blob_name)
    elif (args.dataset=='CIFAR10'):
        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR10_Train_input_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR10_Train_target_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_targets = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR10_Test_input_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR10_Test_target_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_targets = pickle.loads(pickle_in)
    elif (args.dataset == 'SVHN'):
        bucket_name = 'permutation-mlp'
        source_file_name = f'SVHN_Train_input_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'SVHN_Train_target_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_targets = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'SVHN_Test_input_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'SVHN_Test_target_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_targets = pickle.loads(pickle_in)
    elif (args.dataset == 'CIFAR100'):
        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR100_Train_input_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR100_Train_target_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        train_targets = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR100_Test_input_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_inputs = pickle.loads(pickle_in)

        bucket_name = 'permutation-mlp'
        source_file_name = f'CIFAR100_Test_target_org.pkl'
        destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_in = blob.download_as_string()
        test_targets = pickle.loads(pickle_in)
    # elif (args.dataset == 'ImageNet'):



    torch.manual_seed(args.seed)
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    ######## models
    nchannels, nclasses = 3, 10
    if args.dataset == 'MNIST': nchannels = 1
    if args.dataset == 'CIFAR100': nclasses = 100
    if args.dataset == 'imagenet': nclasses = 1000

    if args.nlayers == 1 and "mlp" in args.arch:
        model = MLP1_layer(n_units=args.width, n_channels=nchannels, n_classes=nclasses)
    if args.nlayers == 2 and "mlp" in args.arch:
        model = MLP2_layer(n_units=args.width, n_channels=nchannels, n_classes=nclasses)
    if args.nlayers == 4 and "mlp" in args.arch:
        model = MLP4_layer(n_units=args.width, n_channels=nchannels, n_classes=nclasses)
    if args.nlayers == 8 and "mlp" in args.arch:
        model = MLP8_layer(n_units=args.width, n_channels=nchannels, n_classes=nclasses)
    if args.nlayers == 16 and "mlp" in args.arch:
        model = MLP16_layer(n_units=args.width, n_channels=nchannels, n_classes=nclasses)

    if "resnet" in args.arch and args.nlayers == 18:
        model = ResNet18(nclasses, args.width, nchannels)
    if "resnet" in args.arch and args.nlayers == 34:
        model = ResNet34(nclasses, args.width, nchannels)
    if "resnet" in args.arch and args.nlayers == 50:
        model = ResNet50(nclasses, args.width, nchannels)

    if "vgg" in args.arch:
        model = vgg.__dict__[args.arch](nclasses)

    if "s_conv" in args.arch and args.nlayers == 2:
        model = s_conv_2layer(nchannels, args.width, nclasses)
        save_dir = f'{args.arch}_nopool_{args.dataset}_{args.nlayers}_{args.width}'
    if "s_conv" in args.arch and args.nlayers == 4:
        model = s_conv_4layer(nchannels, args.width, nclasses)
        save_dir = f'{args.arch}_nopool_{args.dataset}_{args.nlayers}_{args.width}'
    if "s_conv" in args.arch and args.nlayers == 6:
        model = s_conv_6layer(nchannels, args.width, nclasses)
        save_dir = f'{args.arch}_nopool_{args.dataset}_{args.nlayers}_{args.width}'
    if "s_conv" in args.arch and args.nlayers == 8:
        model = s_conv_8layer(nchannels, args.width, nclasses)
        save_dir = f'{args.arch}_nopool_{args.dataset}_{args.nlayers}_{args.width}'


    ### save all training arguments
    bucket_name = 'permutation-mlp'
    source_file_name = f'args.pkl'
    destination_blob_name = f'Neurips21/{save_dir}/Train/{args.seed}/{source_file_name}'
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    pickle_out = pickle.dumps(args)
    blob.upload_from_string(pickle_out)


    print(model)

    if args.cpu:
        model.cpu()
    else:
        model.cuda()

    model = nn.DataParallel(model)

    def upload_blob(bucket_name, source_file_name, destination_blob_name):
        """Uploads a file to the bucket."""
        # bucket_name = "your-bucket-name"
        # source_file_name = "local/path/to/file"
        # destination_blob_name = "storage-object-name"

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)

        print(
            "File {} uploaded to {}.".format(
                source_file_name, destination_blob_name
            )
        )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    kwargs = {'num_workers': 4, 'pin_memory': True}
    nchannels, nclasses = 3, 10
    if args.dataset == 'MNIST': nchannels = 1
    if args.dataset == 'CIFAR100': nclasses = 100
    if args.dataset == 'imagenet': nclasses = 1000

    # loading data
    if args.dataset == 'MNIST':
        #### get mnist: solution 3
        os.system("wget www.di.ens.fr/~lelarge/MNIST.tar.gz")
        os.system("tar -xvzf MNIST.tar.gz")

        from torchvision.datasets import MNIST
        from torchvision import transforms

        normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
        train_loader = torch.utils.data.DataLoader(
            MNIST(root='./', download=False, transform=transforms.Compose([transforms.ToTensor(),
                                                                           transforms.Resize((32, 32)),
                                                                           normalize,
                                                                           ]), train=True),
            batch_size=args.batchsize, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            MNIST(root='./', download=False, transform=transforms.Compose([transforms.ToTensor(),
                                                                           transforms.Resize((32, 32)),
                                                                           normalize,
                                                                           ]), train=False),
            batch_size=args.batchsize, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    else:
        train_dataset = load_data('train', args.dataset, args.save_dir, nchannels)
        val_dataset = load_data('val', args.dataset, args.save_dir, nchannels)

        train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, **kwargs)
        val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, **kwargs)



    # ###########upload train/test data/labels to the bucket
    # train_inputs, train_targets = save_dataset(train_loader)
    # test_inputs, test_targets = save_dataset(val_loader)
    #
    #
    # bucket_name = 'permutation-mlp'
    # source_file_name = f'ImageNet_Train_input_org.pkl'
    # destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
    # storage_client = storage.Client()
    # bucket = storage_client.bucket(bucket_name)
    # blob = bucket.blob(destination_blob_name)
    # pickle_out = pickle.dumps(train_inputs)
    # blob.upload_from_string(pickle_out)
    #
    # bucket_name = 'permutation-mlp'
    # source_file_name = f'ImageNet_Train_target_org.pkl'
    # destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
    # storage_client = storage.Client()
    # bucket = storage_client.bucket(bucket_name)
    # blob = bucket.blob(destination_blob_name)
    # pickle_out = pickle.dumps(train_targets)
    # blob.upload_from_string(pickle_out)
    #
    # bucket_name = 'permutation-mlp'
    # source_file_name = f'ImageNet_Test_input_org.pkl'
    # destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
    # storage_client = storage.Client()
    # bucket = storage_client.bucket(bucket_name)
    # blob = bucket.blob(destination_blob_name)
    # pickle_out = pickle.dumps(test_inputs)
    # blob.upload_from_string(pickle_out)
    #
    # bucket_name = 'permutation-mlp'
    # source_file_name = f'ImageNet_Test_target_org.pkl'
    # destination_blob_name = f'Neurips21/dataset_pkl/{source_file_name}'
    # storage_client = storage.Client()
    # bucket = storage_client.bucket(bucket_name)
    # blob = bucket.blob(destination_blob_name)
    # pickle_out = pickle.dumps(test_targets)
    # blob.upload_from_string(pickle_out)
    #################################################################### end save to pickle
    criterion = nn.CrossEntropyLoss()
    if args.cpu:
        criterion = criterion.cpu()
    else:
        criterion = criterion.cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    stats = {}
    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)
        if args.lr_scheduler == 1:
            lr_scheduler.step()

        if epoch == 0:
            torch.save(model.state_dict(), 'model_0.th')
            bucket_name = 'permutation-mlp'
            source_file_name = 'model_0.th'
            destination_blob_name = f'Neurips21/{save_dir}/Train/{args.seed}/{source_file_name}'
            upload_blob(bucket_name, source_file_name, destination_blob_name)
        # train for one epoch
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        if epoch % 9 == 0:
            torch.save(model.state_dict(), 'model_' + str(epoch) + '.th')
            bucket_name = 'permutation-mlp'
            source_file_name = 'model_' + str(epoch) + '.th'
            destination_blob_name = f'Neurips21/{save_dir}/Train/{args.seed}/{source_file_name}'
            upload_blob(bucket_name, source_file_name, destination_blob_name)

        train_accuracy = evaluate_model(model, train_inputs, train_targets)
        test_accuracy = evaluate_model(model, test_inputs, test_targets)
        print(train_acc, train_accuracy)
        print("train_loss, train_accuracy, test_accuracy", train_loss, train_accuracy, test_accuracy)
        add_element(stats, 'train_acc', train_accuracy['top1'])
        add_element(stats, 'test_acc', test_accuracy['top1'])

        #### save best
        is_best = test_accuracy['top1'] > best_prec1
        best_prec1 = max(test_accuracy['top1'], best_prec1)
        if is_best:
            print(is_best, epoch, test_accuracy['top1'], best_prec1)
            torch.save(model.state_dict(), 'model_best.th')
            bucket_name = 'permutation-mlp'
            source_file_name = 'model_best.th'
            destination_blob_name = f'Neurips21/{save_dir}/Train/{args.seed}/{source_file_name}'
            upload_blob(bucket_name, source_file_name, destination_blob_name)



        # add_element(stats, 'train_acc', train_acc)
        # add_element(stats, 'test_acc', prec1)
        # #### save best
        # is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)
        # if is_best:
        #     print(is_best, epoch, prec1, best_prec1)
        #     torch.save(model.state_dict(), 'model_best.th')
        #     bucket_name = 'permutation-mlp'
        #     source_file_name = 'model_best.th'
        #     destination_blob_name = f'Neurips21/{save_dir}/Train/{args.seed}/{source_file_name}'
        #     upload_blob(bucket_name, source_file_name, destination_blob_name)

        from google.cloud import storage
        bucket_name = 'permutation-mlp'
        source_file_name = f'stats.pkl'
        destination_blob_name = f'Neurips21/{save_dir}/Train/{args.seed}/{source_file_name}'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        pickle_out = pickle.dumps(stats)
        blob.upload_from_string(pickle_out)

        if train_loss < args.stopcond or epoch == args.epochs:
            break

def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if args.cpu == False:
            input = input.to(device)
            # print(input.shape)
            target = target.to(device)
        if args.half:
            input = input.half()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    return top1.avg, losses.avg

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if args.cpu == False:
            input = input.to(device)
            target = target.to(device)

        if args.half:
            input = input.half()

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batchsize = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batchsize))
    return res

def calc_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate_model(model, inputs, targets):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for data, target in zip(inputs, targets):
            # input = data.to(device).view(data.size(0), -1)
            input = data.to(device)
            target = target.to(device)
            # print(input.shape)

            # compute output
            output = model(input)

            # measure accuracy and record loss
            acc1 = calc_accuracy(output, target, topk=(1,))[0]
            top1.update(acc1[0], input.size(0))
        # results = dict(top1=top1.avg, loss=losses.avg, batch_time=batch_time.avg)
        results = dict(top1=top1.avg)

    return {key: float(val) for key, val in results.items()}



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batchsize = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batchsize))
    return res

def add_element(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

def download_blob(bucket_name, source_blob_name, destination_file_name,
                  blob_path_prefix=""):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path_prefix + source_blob_name)
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    try:
        blob.download_to_filename(destination_file_name)
    except google.api_core.exceptions.NotFound as e:
        os.remove(destination_file_name)
        print(e)
        raise FileNotFoundError

def download_pkl(bucket_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    pickle_in = blob.download_as_string()
    return  pickle.loads(pickle_in)


def load_data(split, dataset_name, datadir, nchannels):
    ## https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
    if dataset_name == 'MNIST':
        normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
    elif dataset_name == 'SVHN':
        normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
    elif dataset_name == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    elif dataset_name == 'CIFAR100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    elif dataset_name == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if dataset_name == 'imagenet':
        tr_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        val_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    else:
        tr_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
        val_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])

    if dataset_name == "imagenet":
        imagenet_folder = "Neurips21/small_imagenet"
        imagenet_tarfile = "small_imagenet.tar.gz"
        file = os.path.join(imagenet_folder, imagenet_tarfile)
        if not os.path.isfile(file):
            print("Downloading imagenet...")
            download_blob("permutation-mlp", file, file)
            print("done.")
        if not os.path.isdir(os.path.join(imagenet_folder, split)):
            print("Unpacking...")
            with tarfile.open(file) as tar:
                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers()), desc="Unpacking - "):
                    tar.extract(member, path=imagenet_folder)
            print("done.")
        dataset = ImageFolder(os.path.join(imagenet_folder, split), transform=tr_transform)

    else:
        get_dataset = getattr(datasets, dataset_name)
        if dataset_name == 'SVHN':
            if split == 'train':
                dataset = get_dataset(root=datadir, split='train', download=True, transform=tr_transform)
            else:
                dataset = get_dataset(root=datadir, split='test', download=True, transform=val_transform)
        else:
            if split == 'train':
                dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
            else:
                dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)

    return dataset

def save_dataset(loader):
    device = torch.device('cuda')
    # switch to evaluate mode

    inputs = []
    targets = []
    with torch.no_grad():
        for data, target in loader:
            # input = data.to(device).view(data.size(0), -1)
            input = data.to(device)
            target = target.to(device)
            inputs.append(input)
            targets.append(target)



    return inputs, targets
if __name__ == '__main__':
    main()