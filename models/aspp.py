import torch


class ASPP(torch.nn.Module):
    '''
    '''
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        filters = int(out_channels / 8)

        p = 1
        k = 3
        d = 1
        s = 2
        self.conv1 = torch.nn.Conv2d(in_channels, filters, k, s, p, d)

        p = 2
        k = 3
        d = 2
        s = 2
        self.conv2 = torch.nn.Conv2d(in_channels, filters, k, s, p, d)

        p = 4
        k = 3
        d = 4
        s = 2
        self.conv3 = torch.nn.Conv2d(in_channels, filters, k, s, p, d)

        p = 6
        k = 3
        d = 6
        s = 2
        self.conv4 = torch.nn.Conv2d(in_channels, filters, k, s, p, d)

        p = 8
        k = 3
        d = 8
        s = 2
        self.conv5 = torch.nn.Conv2d(in_channels, filters, k, s, p, d)
        
        p = 12
        k = 3
        d = 12
        s = 2
        self.conv6 = torch.nn.Conv2d(in_channels, filters, k, s, p, d)

        p = 18
        k = 3
        d = 18
        s = 2
        self.conv7 = torch.nn.Conv2d(in_channels, filters, k, s, p, d)

        p = 24
        k = 3
        d = 24
        s = 2
        self.conv8 = torch.nn.Conv2d(in_channels, filters, k, s, p, d)
    
    def forward(self, x):
        out_1 = self.conv1(x)
        out_2 = self.conv2(x)
        out_3 = self.conv3(x)
        out_4 = self.conv4(x)
        out_5 = self.conv5(x)
        out_6 = self.conv6(x)
        out_7 = self.conv7(x)
        out_8 = self.conv8(x)

        out = torch.cat((out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8), axis=1)

        return out
