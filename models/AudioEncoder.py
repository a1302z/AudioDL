import torch
import torch.nn as nn

"""
First version of audio conv model
Needs samples with size > 15346
"""
class ConvAudioEncoderV0(nn.Module):
    def __init__(self):
        super(ConvAudioEncoder, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(2, 64, 1024, stride=16, dilation=15),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, 32),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, 32),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, 32, stride=4, dilation=3),
            nn.LeakyReLU(),
            nn.Conv1d(128, 128, 4),
            nn.LeakyReLU(),
            nn.Conv1d(128, 128, 4),
            nn.LeakyReLU(),
            nn.Conv1d(128, 256, 32, stride=2, dilation=1),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 3),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 3),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 16, stride=1, dilation=1),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 3),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 3),
            nn.LeakyReLU(),
            nn.Conv1d(256, 512, 8, stride=1, dilation=1),
            nn.LeakyReLU(),
            nn.Conv1d(512, 512, 3),
            nn.LeakyReLU(),
            nn.Conv1d(512, 512, 3),
            nn.LeakyReLU(),
            nn.Conv1d(512, 512, 8, stride=1, dilation=1),
            nn.LeakyReLU(),
            nn.Conv1d(512, 1024, 4, stride=1, dilation=1),
            nn.LeakyReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2),
            nn.Softmax(),
        )
        
    def forward(self, x):
        x = self.convs(x)
        x = x.mean(dim=2)
        x = self.classifier(x)
        return x

class convs_features(nn.Module):
    def __init__(self):
        super(convs_features, self).__init__()
        self.mp = nn.MaxPool1d(2, stride=2)
        self.a0 = nn.Sequential(
            nn.Conv1d(64, 128, 17, padding=8),
            nn.BatchNorm1d(128),
        )
        self.a = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        self.b0 = nn.Sequential(
            nn.Conv1d(128, 256, 17, padding=8),
            nn.BatchNorm1d(256),
        )
        self.b = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        )
        self.c = nn.Sequential(
            nn.Conv1d(256, 512, 5, padding=2),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Conv1d(512, 1024, 3, padding=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
        )   
    def forward(self, x):
        x = self.mp(x)
        x = self.a0(x)
        x = self.a(x)+x
        x = self.mp(x)
        x = self.b0(x)
        x = self.b(x)+x
        x = self.mp(x)
        x = self.c(x)
        return x

class ConvAudioEncoderV1(nn.Module):
    def __init__(self, classes=8):
        super(ConvAudioEncoderV1, self).__init__()
        self.classes = classes
        self.convs_low_freq = nn.Sequential(
            nn.Conv1d(1, 64, 32, dilation=8),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(64, 64, 4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(64, 64, 3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            convs_features(),
        )
        self.convs_mid_freq = nn.Sequential(
            nn.Conv1d(1, 64, 32, dilation=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(3, stride=3),
            nn.Conv1d(64, 64, 4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(64, 64, 3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            convs_features(),
        )
        self.convs_high_freq = nn.Sequential(
            nn.Conv1d(1, 64, 32), 
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(64, 64, 4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(64, 64, 3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            convs_features(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classify = nn.Sequential(
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.classes),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x):
        hf = self.convs_high_freq(x)
        mf = self.convs_mid_freq(x)
        lf = self.convs_mid_freq(x)
        #hf = self.convs_features(hf)
        #mf = self.convs_features(mf)
        #lf = self.convs_features(lf)
        #print('hf size: {:s}\nmf size: {:s}\nlf size: {:s}'.format(str(hf.size()), str(mf.size()), str(lf.size())))
        x = torch.cat([hf, mf, lf], dim=2)
        x = hf
        x = self.pool(x).squeeze(2)
        x = self.classify(x)
        #print(fts.size())
        return x
    
if __name__ == '__main__':
    import torchsummary
    
    #"""
    model = ConvAudioEncoderV1()
    if torch.cuda.is_available():
        model = model.cuda()
    print(torchsummary.summary(model,(1, 400_000)))
    #"""
    
    
    #import torchvision
    #print(torchsummary.summary(torchvision.models.resnet34(pretrained=False).cuda(), (3, 1000, 1000)))
    
    
    
        