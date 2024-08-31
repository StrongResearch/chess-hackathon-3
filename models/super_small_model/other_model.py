##### AlphaZero implementation #####
class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(19, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, s):
        s = s.view(-1, 19, 8, 8)  # batch_size x channels x board_x x board_y
        s = F.relu(self.conv1(s))
        #s = F.relu(self.bn1(self.conv1(s)))
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out
    
class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        # value head 
        self.conv = nn.Conv2d(256, 1, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8*8, 64)
        self.fc2 = nn.Linear(64, 3)
        
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(128)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(8*8*128, 1858)
    
    def forward(self,s):
        #v = F.relu(self.bn(self.conv(s))) # value head
        #v = v.view(-1, 8*8)  # batch_size X channel X height X width
        #v = F.relu(self.fc1(v))
        #v = self.logsoftmax(self.fc2(v)).exp()
        #v = F.tanh(self.fc2(v))
        '''
        p = F.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, 8*8*128)
        p = self.fc(p)
        p = F.tanh(p) 
        #p = self.logsoftmax(p).exp()
        '''
        p = F.relu(self.conv1(s)) # policy head
        p = p.view(-1, 8*8*128)
        p = self.fc(p)
        p = F.tanh(p) 
        #p = self.logsoftmax(p).exp()
        return p, None
    
class ZeroNet(nn.Module):
    def __init__(self, num_res_blocks=19):
        super(ZeroNet, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv = ConvBlock()
        for block in range(num_res_blocks):
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock()
    
    def forward(self,s):
        s = self.conv(s)
        #for block in range(self.num_res_blocks):
        #    s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s