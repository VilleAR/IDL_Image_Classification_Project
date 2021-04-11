import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision
import Datakiller
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
import random
import pandas as pd 
from torch.autograd import Variable
import matplotlib.pyplot as plt 
from PIL import Image 

DATA_PATH = '../'
TRAIN_DATA = 'train'
TEST_DATA = 'test'
TRAIN_IMG_FILE = 'imstrain.txt'
TEST_IMG_FILE = 'imsval.txt'
TRAIN_LABEL_FILE = 'labelstrain.txt'
TEST_LABEL_FILE = 'labelsval.txt'
NUM_CLASSES=14
KERNEL_SIZE=3

def f1_score(y_pred:torch.Tensor, y_true:torch.Tensor, is_training=False) -> torch.Tensor:

    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-12
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    acc=(tp+tn)/(tp+tn+fp+fn)
    return f1, acc

#--- model ---
class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, padding=1, kernel_size=KERNEL_SIZE)
        self.bn1 = nn.BatchNorm2d(num_features = 12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(in_channels=12 , out_channels=20 , kernel_size=KERNEL_SIZE, stride=1,padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=12 , out_channels=32 , kernel_size=KERNEL_SIZE, stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(in_features= 32*64*64, out_features=NUM_CLASSES)
        self.sig=nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = F.dropout(x)
        #x = self.conv2(x)
        #x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = x.view(-1, 32*64*64)
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.fc(x)
        #x = self.sig(x)
        return x 


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model=CNN().to(device)
cp=torch.load("checkpoint.pt")
model.load_state_dict(cp)
def seed_worker(worker_id):


    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated



def visualize_model(model, num_images=6):
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])
                plt.pause(2)
                if images_so_far == num_images:                 
                    return
BATCH_SIZE_TEST=1
DATA_PATRICK = '../testimages'
test_transform = transforms.Compose([                                        
                                        #transforms.Resize((224,224)),
                                        transforms.ToTensor() # scales the pixel values to the range [0,1]
                                        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])
test_set  = datasets.ImageFolder(DATA_PATRICK,  transform=test_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE_TEST, shuffle=False, worker_init_fn=seed_worker)
model.eval()
dset_train = Datakiller.Datakiller(DATA_PATH, TRAIN_DATA, TRAIN_IMG_FILE, TRAIN_LABEL_FILE, test_transform)
train_loader = torch.utils.data.DataLoader(dataset=dset_train, batch_size=BATCH_SIZE_TEST, shuffle=True, worker_init_fn=seed_worker)

class_names=['baby', 'bird','car','clouds','dog','female','flower','male','night','people','portrait','river','sea','tree']
    # Get a batch of training data
inputs, classes = next(iter(train_loader))

    # Make a grid from batch
out = torchvision.utils.make_grid(inputs)
classes=classes.tolist()
#imshow(out, title=[class_names[x] for x in classes[0]])


preds=[]
with torch.no_grad():
    z=20
    freq = {}
    for data, target in test_loader:
        if z!=0:
            output=model(data)
            output=torch.sigmoid(output)
            i=0
            #print(output)
            v=torch.round(output) 
            v=v.tolist()
            preds.append(v)
            if z>1:
                z=z-1
                print(v)
        else:
            break

with open('predictions.txt', 'w') as f:
    for item in preds:
        f.write("%s\n" % item)
        pass

