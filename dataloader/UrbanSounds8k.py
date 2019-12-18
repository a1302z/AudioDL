import os
import csv
import tqdm
import torchaudio
from torch.utils.data import Dataset as dset



class UrbanSound(dset):
    def __init__(self, root='data/UrbanSound8K/', train=True, train_split=10, limit=None, overfit=None):
        super(UrbanSound).__init__()
        self.root = root
        self.train= train
        self.train_split = train_split
        self.limit = limit
        self.num_classes = 0
        with open(os.path.join(root, 'metadata', 'UrbanSound8K.csv')) as csvfile:
            csvreader = csv.DictReader(csvfile, delimiter=',')
            #print(next(csvreader))
            self.samples = []
            for row in tqdm.tqdm(csvreader):
                path = os.path.join(root, 'audio', 'fold{:d}'.format(int(row['fold'])), row['slice_file_name'])
                classid = int(row['classID'])
                self.samples.append((path, classid))
                self.num_classes = max(self.num_classes, classid)
        if overfit:
            self.samples = self.samples[:overfit+1]
        if train_split:
            if train:
                idcs = [i for i in range(len(self.samples)) if i%train_split != 0]
            else:
                idcs = [i for i in range(len(self.samples)) if i%train_split == 0]
            self.samples = [self.samples[i] for i in idcs]
        
        
    def get_num_classes(self):
        return self.num_classes+1
            
        
        
    def __getitem__(self, i):
        s = self.samples[i]
        if type(s) is tuple:
            file = torchaudio.load(s[0])[0][0].unsqueeze(0)
            if self.limit:
                _, l = file.size()
                if l > self.limit:
                    l = l//2
                    file = file[:, l-self.limit:l+self.limit]
            return file, s[1]
        else:
            raise NotImplementedError('No multiindexing allowed')
            #files = [torchaudio.load(x[0])[0] for x in s]
            #return files, [x[1] for x in s]
        
    
    def __len__(self):
        return len(self.samples)
    
"""def collate_audio(batch_list):
    files, targets = batch_list
    files = torch.stack(files)
    targets = torch.Tensor(targets)
    print(files.size())
    print(targets.size())
    return files, targets
    
"""    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ds = UrbanSound()
    L = len(ds)
    min_size = int(1e10) ## should be big enough
    for i, x in tqdm.tqdm(enumerate(ds), total=len(ds), leave=False):
        min_size = min(min_size, x[0].size()[1])
        if x[0].size()[0] != 1:
            print('Not 1 channels but {:d} instead: Index = {:d}'.format(x[0].size()[0], i))
    print(min_size)
    plt.figure()
    plt.plot(ds[0][0].t().numpy())
    plt.show()
    
    #collate_audio(ds[10:15])