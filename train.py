import argparse
import tqdm
import torch
import numpy as np
import models.AudioEncoder as AE
from dataloader.UrbanSounds8k import UrbanSound

def model_accuracy(model, dataset, num_classes):
    with torch.no_grad():
        correct, incorrect = [0]*num_classes, [0]*num_classes
        class_dist = [0]*num_classes
        for data, target in tqdm.tqdm(dataset, total=len(train_set), leave=False, desc='model accuracy'):
            out = model(data.unsqueeze(0).to(device))
            class_id = torch.argmax(out).to('cpu')
            if class_id == target:
                correct[target] += 1
            else:
                incorrect[target] += 1
            class_dist[target] += 1
        total_correct, total_incorrect = sum(correct), sum(incorrect)
        print('Accuracy: Correct: {:d} \t Incorrect: {:d}\t Rate: {:.2f}'.format(total_correct, total_incorrect, float(total_correct)/float(total_correct+total_incorrect)))
        per_class_rates = []
        for i in range(num_classes):
            si = correct[i]+incorrect[i]
            if si > 0:
                per_class_rates.append(correct[i]/float(si))
            else:
                per_class_rates.append(None)
        print('Class distribution: {:s}'.format(str(class_dist)))
        print('Per class accuracy: {:s} \t Average: {:.2f}'.format(str(per_class_rates), np.mean([i for i in per_class_rates if i])))


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, type=str, choices=['UrbanSound8k', ], help='Choose dataset to train on')
parser.add_argument('--no_val', action='store_false', help='Refrain from doing validation')
parser.add_argument('--overfit', default=None, type=int, help='Test model by overfitting to number of samples')
args = parser.parse_args()

## Dataset init
num_classes = None
if args.dataset == 'UrbanSound8k':
    train_set = UrbanSound(train=True, limit=200_000, overfit=args.overfit, train_split=10 if args.no_val else None)
    val_set = UrbanSound(train=False, limit=200_000, overfit=args.overfit, train_split=10 if args.no_val else None)
    num_classes = train_set.get_num_classes()
else:
    raise NotImplementedError('Dataset unkown')
print('{:d} classes'.format(num_classes))
dataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=2)
    
## Model init
model = AE.ConvAudioEncoderV1(classes=num_classes)

## CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

## optimizer
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

epochs = 5
L = len(train_set)

with tqdm.tqdm(range(epochs), total=epochs, desc='Training', bar_format="{postfix[0]} {postfix[1][value]:>2.3f}",
          postfix=["Train loss", dict(value=0)]) as t:
    for i in range(epochs):
        model.train()
        for data, target in tqdm.tqdm(dataloader, total=L, desc='Epoch {:d}'.format(i), leave=False):
            optim.zero_grad()
            output = model(data.to(device))
            train_loss = loss_fn(output, target.to(device))
            train_loss.backward()
            optim.step()
            t.postfix[1]['value'] = train_loss.item()
            t.update()
        if args.no_val:
            model.eval()
            for data, target in tqdm.tqdm(val_set, total=len(val_set), desc='Validation {:d}'.format(i), leave=False):
                val_loss = 0
                with torch.no_grad():
                    output = model(data.unsqueeze(0).to(device))
                    target = torch.tensor(target, dtype=torch.long).unsqueeze(0).to(device)
                    val_loss += loss_fn(output, target)
            val_loss /= float(len(val_set))
            print('Validation loss after epoch {:d}: {:.3f}'.format(i, val_loss))    
            
## Test model
model_accuracy(model, train_set, num_classes)

## Save model
torch.save(model.state_dict(), 'logs/model.pth.tar')
        
