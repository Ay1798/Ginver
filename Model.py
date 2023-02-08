from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, nc, ndf, nz):
        super(Classifier, self).__init__()

        self.nc = nc
        self.ndf = ndf
        self.nz = nz

        self.conv11 = nn.Conv2d(nc, ndf, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(ndf)
        self.relu11 = nn.ReLU(True) # sp1 128*64*64

        self.conv12 = nn.Conv2d(ndf, ndf, 3, 1, 1)
        self.bn12 = nn.BatchNorm2d(ndf)
        self.pool1 = nn.MaxPool2d(2, 2, 0)
        self.relu12 = nn.ReLU(True) #sp2 128*32*32


        # 32*32

        self.conv21 = nn.Conv2d(ndf, 2*ndf, 3, 1, 1)
        self.bn21 = nn.BatchNorm2d(2*ndf)
        self.relu21 = nn.ReLU(True)

        self.conv22 = nn.Conv2d(2*ndf, 2*ndf, 3, 1, 1)
        self.bn22 = nn.BatchNorm2d(2*ndf)
        self.pool2 = nn.MaxPool2d(2, 2, 0)
        self.relu22 = nn.ReLU(True) # sp3 2*ndf*16*16


        # 16*16

        self.conv31 = nn.Conv2d(2*ndf, 4*ndf, 3, 1, 1)
        self.bn31 = nn.BatchNorm2d(4*ndf)
        self.relu31 = nn.ReLU(True)

        self.conv32 = nn.Conv2d(4*ndf, 4*ndf, 3, 1, 1)
        self.bn32 = nn.BatchNorm2d(4*ndf)
        self.pool3 = nn.MaxPool2d(2, 2, 0)
        self.relu32 = nn.ReLU(True) # 4*ndf*8*8


        # 8*8
        self.conv4 = nn.Conv2d(4*ndf, 8*ndf, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(8*ndf)
        self.pool4 = nn.MaxPool2d(2, 2, 0)
        self.relu4 = nn.ReLU(True) # sp4 8*ndf*4*4
        # 4*4

        self.fc1 = nn.Linear(8 * ndf * 4 * 4, 5 * nz)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(5 * nz, nz)

    # release为true，则输出的是softmax的结果，若为false则是log_softmax的结果, softmax为true，则输出softmax之前的结果，对应添加log+c处理，否则输出softmax之前的处理结果
    def forward(self, x, new_idea=0, release=False, conv=-1, relu=-1, softmax=True, fc=-1):


        if new_idea==1:
            x = self.fc2(x)
            return F.softmax(x, dim=1)

        x = x.view(-1, 1, 64, 64)

        # 一个block 这之后变成32*32
        x = self.conv11(x)
        if conv == 1:
            return x
        x = self.bn11(x)
        x = self.relu11(x)
        if relu == 1:
            return x

        x = self.conv12(x)
        if conv == 2:
            return x

        x = self.bn12(x)
        x = self.pool1(x)
        x = self.relu12(x)
        if relu == 2:
            return x

        # block 结束

        # 一个block 这之后变成16*16
        x = self.conv21(x)
        if conv == 3:
            return x
        x = self.bn21(x)
        x = self.relu21(x)
        if relu == 3:
            return x

        x = self.conv22(x)
        if conv == 4:
            return x
        x = self.bn22(x)
        x = self.pool2(x)
        x = self.relu22(x)
        if relu == 4:
            return x
        # block 结束

        # 一个block 这之后变成8*8
        x = self.conv31(x)
        if conv == 5:
            return x
        x = self.bn31(x)
        x = self.relu31(x)
        if relu == 5:
            return x

        x = self.conv32(x)
        if conv == 6:
            return x
        x = self.bn32(x)
        x = self.pool3(x)
        x = self.relu32(x)
        if relu == 6:
            return x
        # block 结束

        x = self.conv4(x)
        if conv == 7:
            return x
        x = self.bn4(x)
        x = self.pool4(x)
        x = self.relu4(x)
        if relu == 7:
            return x

        x = x.view(-1, 8*self.ndf*4*4)
        x = self.fc1(x)
        if fc == 1:
            return x

        x = self.drop(x)
        x = self.fc2(x)
        if fc == 2:
            return x

        if not softmax:
            return x

        if release:
            return F.softmax(x, dim=1)
        else:
            return F.log_softmax(x, dim=1)


class Conv_64x64(nn.Module):
    def __init__(self, nc, ngf, nz):
        super(Conv_64x64, self).__init__()
        self.nc = nc  # 3
        self.ngf = ngf  # 128
        self.nz = nz  # 530

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),

            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.view(-1, self.nz, 64, 64)
        x = self.decoder(x)
        return x


class Conv_32x32_128(nn.Module):
    def __init__(self, nc, ngf, nz):
        super(Conv_32x32_128, self).__init__()
        self.nc = nc  # 3
        self.ngf = ngf  # 128
        self.nz = nz  # 530

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 2*128, 4, 2, 1),
            nn.BatchNorm2d(2*128),
            nn.Tanh(),

            nn.Conv2d(2*128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),

            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.view(-1, self.nz, 64, 64)
        x = self.decoder(x)
        return x



class Conv_16x16_2x16(nn.Module):
    def __init__(self, nc, ngf, nz):
        super(Conv_16x16_2x16, self).__init__()
        self.nc = nc  # 3
        self.ngf = ngf  # 128
        self.nz = nz  # 530

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2*16, 4*128, 4, 2, 1),
            nn.BatchNorm2d(4*128),
            nn.Tanh(),

            nn.Conv2d(4 * 128, 4*128, 3, 1, 1),
            nn.BatchNorm2d(4*128),
            nn.Tanh(),

            nn.ConvTranspose2d(4 * 128, 2 * 128, 4, 2, 1),
            nn.BatchNorm2d(2 * 128),
            nn.Tanh(),

            nn.Conv2d(2*128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),

            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.view(-1, self.nz, 64, 64)
        x = self.decoder(x)
        return x

class Conv_16x16_2x128(nn.Module):
    def __init__(self, nc, ngf, nz):
        super(Conv_16x16_2x128, self).__init__()
        self.nc = nc  # 3
        self.ngf = ngf  # 128
        self.nz = nz  # 530

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2*128, 4*128, 4, 2, 1),
            nn.BatchNorm2d(4*128),
            nn.Tanh(),

            nn.Conv2d(4 * 128, 4*128, 3, 1, 1),
            nn.BatchNorm2d(4*128),
            nn.Tanh(),

            nn.ConvTranspose2d(4 * 128, 2 * 128, 4, 2, 1),
            nn.BatchNorm2d(2 * 128),
            nn.Tanh(),

            nn.Conv2d(2*128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),

            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.view(-1, self.nz, 64, 64)
        x = self.decoder(x)
        return x


class Conv_4x4_8x128(nn.Module):
    def __init__(self, nc, ngf, nz):
        super(Conv_4x4_8x128, self).__init__()
        self.nc = nc  # 3
        self.ngf = ngf  # 128
        self.nz = nz  # 530

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8*128, 8*128, 4, 2, 1),
            nn.BatchNorm2d(8*128),
            nn.Tanh(), # 8*128*8*8

            nn.Conv2d(8 * 128, 8*128, 3, 1, 1),
            nn.BatchNorm2d(8*128),
            nn.Tanh(), # 4*128*8*8

            nn.ConvTranspose2d(8 * 128, 4 * 128, 4, 2, 1),
            nn.BatchNorm2d(4 * 128),
            nn.Tanh(), # 4*128*16*16

            nn.Conv2d(4 * 128, 4 * 128, 3, 1, 1),
            nn.BatchNorm2d(4 * 128),
            nn.Tanh(),  # 4*128*16*16

            nn.ConvTranspose2d(4 * 128, 2 * 128, 4, 2, 1),
            nn.BatchNorm2d(2 * 128),
            nn.Tanh(),  # 2*128*32*32

            nn.Conv2d(2 * 128, 2 * 128, 3, 1, 1),
            nn.BatchNorm2d(2 * 128),
            nn.Tanh(),  # 2*128*32*32

            nn.ConvTranspose2d(2 * 128, 128, 4, 2, 1),
            nn.BatchNorm2d( 128),
            nn.Tanh(),  # 128*64x64

            nn.Conv2d( 128,  128, 3, 1, 1),
            nn.BatchNorm2d( 128),
            nn.Tanh(),  # 128*64x64

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),

            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.view(-1, self.nz, 64, 64)
        x = self.decoder(x)
        return x



class Conv_4x4_8x128_1019(nn.Module):
    def __init__(self, nc, ngf, nz):
        super(Conv_4x4_8x128_1019, self).__init__()
        self.nc = nc  # 3
        self.ngf = ngf  # 128
        self.nz = nz  # 530

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8*128, 8*128, 4, 2, 1),
            nn.BatchNorm2d(8*128),
            nn.Tanh(), # 8*128*8*8

            nn.ConvTranspose2d(8 * 128, 4 * 128, 4, 2, 1),
            nn.BatchNorm2d(4 * 128),
            nn.Tanh(), # 4*128*16*16


            nn.ConvTranspose2d(4 * 128, 2 * 128, 4, 2, 1),
            nn.BatchNorm2d(2 * 128),
            nn.Tanh(),  # 2*128*32*32

            nn.ConvTranspose2d(2 * 128, 128, 4, 2, 1),
            nn.BatchNorm2d( 128),
            nn.Tanh(),  # 128*64x64


            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.view(-1, self.nz, 64, 64)
        x = self.decoder(x)
        return x


# classifier_64专用
class Conv_4x4_8x64(nn.Module):
    def __init__(self, nc, ngf, nz):
        super(Conv_4x4_8x64, self).__init__()
        self.nc = nc  # 3
        self.ngf = ngf  # 128
        self.nz = nz  # 530

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8*64, 8*128, 4, 2, 1),
            nn.BatchNorm2d(8*128),
            nn.Tanh(), # 8*128*8*8

            nn.Conv2d(8 * 128, 8*128, 3, 1, 1),
            nn.BatchNorm2d(8*128),
            nn.Tanh(), # 4*128*8*8

            nn.ConvTranspose2d(8 * 128, 4 * 128, 4, 2, 1),
            nn.BatchNorm2d(4 * 128),
            nn.Tanh(), # 4*128*16*16

            nn.Conv2d(4 * 128, 4 * 128, 3, 1, 1),
            nn.BatchNorm2d(4 * 128),
            nn.Tanh(),  # 4*128*16*16

            nn.ConvTranspose2d(4 * 128, 2 * 128, 4, 2, 1),
            nn.BatchNorm2d(2 * 128),
            nn.Tanh(),  # 2*128*32*32

            nn.Conv2d(2 * 128, 2 * 128, 3, 1, 1),
            nn.BatchNorm2d(2 * 128),
            nn.Tanh(),  # 2*128*32*32

            nn.ConvTranspose2d(2 * 128, 128, 4, 2, 1),
            nn.BatchNorm2d( 128),
            nn.Tanh(),  # 128*64x64

            nn.Conv2d( 128,  128, 3, 1, 1),
            nn.BatchNorm2d( 128),
            nn.Tanh(),  # 128*64x64

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),

            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.view(-1, self.nz, 64, 64)
        x = self.decoder(x)
        return x

# classifier_32专用
class Conv_4x4_8x32(nn.Module):
    def __init__(self, nc, ngf, nz):
        super(Conv_4x4_8x32, self).__init__()
        self.nc = nc  # 3
        self.ngf = ngf  # 128
        self.nz = nz  # 530

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8*32, 8*128, 4, 2, 1),
            nn.BatchNorm2d(8*128),
            nn.Tanh(), # 8*128*8*8

            nn.Conv2d(8 * 128, 8*128, 3, 1, 1),
            nn.BatchNorm2d(8*128),
            nn.Tanh(), # 4*128*8*8

            nn.ConvTranspose2d(8 * 128, 4 * 128, 4, 2, 1),
            nn.BatchNorm2d(4 * 128),
            nn.Tanh(), # 4*128*16*16

            nn.Conv2d(4 * 128, 4 * 128, 3, 1, 1),
            nn.BatchNorm2d(4 * 128),
            nn.Tanh(),  # 4*128*16*16

            nn.ConvTranspose2d(4 * 128, 2 * 128, 4, 2, 1),
            nn.BatchNorm2d(2 * 128),
            nn.Tanh(),  # 2*128*32*32

            nn.Conv2d(2 * 128, 2 * 128, 3, 1, 1),
            nn.BatchNorm2d(2 * 128),
            nn.Tanh(),  # 2*128*32*32

            nn.ConvTranspose2d(2 * 128, 128, 4, 2, 1),
            nn.BatchNorm2d( 128),
            nn.Tanh(),  # 128*64x64

            nn.Conv2d( 128,  128, 3, 1, 1),
            nn.BatchNorm2d( 128),
            nn.Tanh(),  # 128*64x64

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),

            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.view(-1, self.nz, 64, 64)
        x = self.decoder(x)
        return x



# classifier_16专用
class Conv_4x4_8x16(nn.Module):
    def __init__(self, nc, ngf, nz):
        super(Conv_4x4_8x16, self).__init__()
        self.nc = nc  # 3
        self.ngf = ngf  # 128
        self.nz = nz  # 530

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8*16, 8*128, 4, 2, 1),
            nn.BatchNorm2d(8*128),
            nn.Tanh(), # 8*128*8*8

            nn.Conv2d(8 * 128, 8*128, 3, 1, 1),
            nn.BatchNorm2d(8*128),
            nn.Tanh(), # 4*128*8*8

            nn.ConvTranspose2d(8 * 128, 4 * 128, 4, 2, 1),
            nn.BatchNorm2d(4 * 128),
            nn.Tanh(), # 4*128*16*16

            nn.Conv2d(4 * 128, 4 * 128, 3, 1, 1),
            nn.BatchNorm2d(4 * 128),
            nn.Tanh(),  # 4*128*16*16

            nn.ConvTranspose2d(4 * 128, 2 * 128, 4, 2, 1),
            nn.BatchNorm2d(2 * 128),
            nn.Tanh(),  # 2*128*32*32

            nn.Conv2d(2 * 128, 2 * 128, 3, 1, 1),
            nn.BatchNorm2d(2 * 128),
            nn.Tanh(),  # 2*128*32*32

            nn.ConvTranspose2d(2 * 128, 128, 4, 2, 1),
            nn.BatchNorm2d( 128),
            nn.Tanh(),  # 128*64x64

            nn.Conv2d( 128,  128, 3, 1, 1),
            nn.BatchNorm2d( 128),
            nn.Tanh(),  # 128*64x64

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),

            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.view(-1, self.nz, 64, 64)
        x = self.decoder(x)
        return x

# 1x1
class Conv_1x1_5x530(nn.Module):
    def __init__(self, nc, ngf, nz):
        super(Conv_1x1_5x530, self).__init__()
        self.nc = nc  # 3
        self.ngf = ngf  # 128
        self.nz = nz  # 530

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(5*530, 128 * 8, 4, 1, 0),
            nn.BatchNorm2d(128 * 8),
            nn.Tanh(),

            nn.ConvTranspose2d(8*128, 8*128, 4, 2, 1),
            nn.BatchNorm2d(8*128),
            nn.Tanh(), # 8*128*8*8

            nn.Conv2d(8 * 128, 8*128, 3, 1, 1),
            nn.BatchNorm2d(8*128),
            nn.Tanh(), # 4*128*8*8

            nn.ConvTranspose2d(8 * 128, 4 * 128, 4, 2, 1),
            nn.BatchNorm2d(4 * 128),
            nn.Tanh(), # 4*128*16*16

            nn.Conv2d(4 * 128, 4 * 128, 3, 1, 1),
            nn.BatchNorm2d(4 * 128),
            nn.Tanh(),  # 4*128*16*16

            nn.ConvTranspose2d(4 * 128, 2 * 128, 4, 2, 1),
            nn.BatchNorm2d(2 * 128),
            nn.Tanh(),  # 2*128*32*32

            nn.Conv2d(2 * 128, 2 * 128, 3, 1, 1),
            nn.BatchNorm2d(2 * 128),
            nn.Tanh(),  # 2*128*32*32

            nn.ConvTranspose2d(2 * 128, 128, 4, 2, 1),
            nn.BatchNorm2d( 128),
            nn.Tanh(),  # 128*64x64

            nn.Conv2d( 128,  128, 3, 1, 1),
            nn.BatchNorm2d( 128),
            nn.Tanh(),  # 128*64x64

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),

            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 5*530, 1, 1)
        x = self.decoder(x)
        return x

# resnet 专用
class Inversion(nn.Module):
    # nc 要生成的图片通道 ngf 中间值 nz 输入的通道数
    def __init__(self, nc, ngf, nz):
        super(Inversion, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.nz = nz

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(nz, 8*ngf, 4, 2, 1),
            nn.BatchNorm2d(8*ngf),
            nn.Tanh(),

            nn.ConvTranspose2d(8 * ngf, 4 * ngf, 4, 2, 1), #28*28
            nn.BatchNorm2d(4 * ngf),
            nn.Tanh(),

            nn.ConvTranspose2d(4 * ngf, 2 * ngf, 4, 2, 1), #56*56
            nn.BatchNorm2d(2 * ngf),
            nn.Tanh(),

            nn.ConvTranspose2d(2 * ngf, ngf, 4, 2, 1), #112*112
            nn.BatchNorm2d(ngf),
            nn.Tanh(),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1), #224*224
            nn.Sigmoid()
        )

    def forward(self, x):
        # 这个nz应该是1024
        x = x.view(-1, self.nz, 7, 7)
        return self.decoder(x)
