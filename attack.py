
from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from Dataset import ImageSet
from Model import Classifier, Conv_4x4_8x32
import numpy as np
import random
import torchvision.utils as vutils
import os, shutil
from utils import TV

# 以test_images作为训练集

def main():
    cuda = "cuda:1"
    layer = "relu7_all_white_64/"
    flag = "relu7"
    mode = "train"
    step = 0.001
    end_epoch = 20000
    device = torch.device(cuda)
    torch.manual_seed(2)
    torch.cuda.manual_seed(2)
    torch.cuda.manual_seed_all(2)
    np.random.seed(2)
    random.seed(2)

    os.makedirs('../ImagResult/blackbox/'+mode+'/'+layer, exist_ok=True)
    os.makedirs('../ModelResult/blackbox/'+mode+'/'+layer, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = ImageSet(root="../Dataset/test_images", transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    test_images1 = ImageSet(root="../Dataset/test_images", transform=transform)
    test_loader1 = torch.utils.data.DataLoader(test_images1, batch_size=128, shuffle=True)
    test_images2 = ImageSet(root="../Dataset/train_images", transform=transform)
    test_loader2 = torch.utils.data.DataLoader(test_images2, batch_size=128, shuffle=True)
    classifier = Classifier(nc=1, ndf=128, nz=530).to(device)
    inversion = Conv_4x4_8x32(nc=1, ngf=128, nz=128).to(device)

    optimizer = optim.Adam(inversion.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True)

    # 定义Train函数
    def train(classifier, inversion, device, data_loader, optimizer, epoch):
        classifier.eval()
        inversion.train()

        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                prediction = classifier(data, relu=7)
            reconstruction = inversion(prediction)
            # blackbox
            # with torch.no_grad():
            #     grad = torch.zeros_like(reconstruction)
            #     num = 0
            #     for j in range(1, 50):
            #         random_direction = torch.randn_like(reconstruction)
            #
            #         new_pic1 = reconstruction + step * random_direction
            #         new_pic2 = reconstruction - step * random_direction
            #
            #         target1 = classifier(new_pic1, relu=2)
            #         target2 = classifier(new_pic2, relu=2)
            #
            #         loss1 = F.mse_loss(target1, prediction)
            #         loss2 = F.mse_loss(target2, prediction)
            #
            #         num = num + 2
            #         grad = loss1 * random_direction + grad
            #         grad = loss2 * -random_direction + grad
            #
            #     grad = grad / (num * step)
            #     # grad = grad.squeeze(dim=0)
            # #loss_TV = 3*TV(reconstruction)
            # #loss_TV.backward(retain_graph=True)
            # reconstruction.backward(grad)
            # optimizer.step()
            reconstruction_prediction = classifier(reconstruction, relu=7)
            loss_TV = TV(reconstruction)
            loss_mse = F.mse_loss(reconstruction_prediction, prediction)
            loss = loss_mse + loss_TV
            loss.backward()
            optimizer.step()

        print(' Train epoch {} '.format(epoch))


    # test
    def test(classifier, inversion, device, data_loader):
        classifier.eval()
        inversion.eval()
        mse_loss = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                prediction = classifier(data, relu=7)
                reconstruction = inversion(prediction)
                mse_loss += F.mse_loss(reconstruction, data, reduction='sum').item()

        mse_loss /= len(data_loader.dataset) * 64 * 64
        # print('\nTest inversion model on test set: Average MSE loss: {:.4f}\n'.format(mse_loss))
        return mse_loss

    # record
    def record(classifier, inversion, device, data_loader, epoch, msg, num, loss):
        classifier.eval()
        inversion.eval()

        plot = True
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                prediction = classifier(data, relu=7)
                reconstruction = inversion(prediction)

                truth = data[0:num]
                inverse = reconstruction[0:num]
                out = torch.cat((inverse, truth))

                vutils.save_image(out, '../ImagResult/blackbox/'+mode+'/'+layer+'/{}_{}_{:.4f}.png'.format(msg.replace(" ", ""), epoch, loss), normalize=False)
                if epoch != end_epoch-1:
                    vutils.save_image(reconstruction[0], '../ImagResult/blackbox/'+mode+'/'+layer + '/final_inverse.png', normalize=False)
                    vutils.save_image(data[0], '../ImagResult/blackbox/'+mode+'/'+layer + '/origin.png',normalize=False)
                if epoch == end_epoch-1:
                    vutils.save_image(reconstruction[0], '../ImagResult/blackbox/'+mode+'/'+layer + '/final_epoch.png',
                                  normalize=False)
                break
    # Load classifier
    path = '../ModelResult/classifier/classifier_32.pth'


    checkpoint = torch.load(path, map_location=cuda)
    classifier.load_state_dict(checkpoint['model'])


    # Load inversion
    path = '../ModelResult/blackbox/'+layer+'/inversion.pth'
    best_mse_loss = 0.0600
    begin_epoch = 1

    try:
        checkpoint = torch.load(path, map_location={'cuda:0': cuda})
        inversion.load_state_dict(checkpoint['model'])
        begin_epoch = checkpoint['epoch']
        best_mse_loss = checkpoint['best_mse_loss']
        print("=> loaded inversion checkpoint '{}' (epoch {}, best_mse_loss {:.4f})".format(path, epoch, best_mse_loss))
    except:
        print("=> load inversion checkpoint '{}' failed".format(path))

    target_mse_loss = best_mse_loss - 0.0005

    for epoch in range(begin_epoch, end_epoch):
        train(classifier, inversion, device, train_loader, optimizer, epoch)
        mse_loss = test(classifier, inversion, device, train_loader)

        if mse_loss < target_mse_loss:
            target_mse_loss = mse_loss - 0.0005
            best_mse_loss = mse_loss
            state = {
                'epoch': epoch,
                'model': inversion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mse_loss': best_mse_loss
            }
            torch.save(state, '../ModelResult/blackbox/'+mode+'/'+layer+'/inversion.pth')
            print('\nTest inversion model on test set: Average MSE loss: {}_{:.4f}\n'.format(epoch, mse_loss))
            record(classifier, inversion, device, test_loader1, epoch, flag+"_same", 32, mse_loss)
            record(classifier, inversion, device, test_loader2, epoch, flag+"_differ", 32, mse_loss)
        if epoch == end_epoch-1 :
            state = {
                'epoch': epoch,
                'model': inversion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mse_loss': best_mse_loss
            }
            torch.save(state, '../ModelResult/blackbox/'+mode+'/'+layer+'/final_inversion.pth')
            record(classifier, inversion, device, test_loader1, epoch, flag + "_same", 32, mse_loss)
            record(classifier, inversion, device, test_loader2, epoch, flag+"_differ", 32, mse_loss)

main()