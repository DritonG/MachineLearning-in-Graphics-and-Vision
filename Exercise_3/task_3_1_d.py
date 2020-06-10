import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF
from pathlib import Path
# TODO: declare your model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            #nn.MaxPool2d(2,2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            #nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(32*28*28, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 10, bias=True),
        )
    def forward(self, x):
        out = self.layer1(x)
        out = out.view(-1, 32*28*28)  # Flatten
        out = self.layer2(out)
        return out

model = Model()

#TODO
# load model variables in evaluation mode

data_folder = Path("./models")

probability = torch.nn.Softmax(dim=0)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

checkpoint = torch.load(data_folder)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()


for i in range(10):

    image = Image.open('img0'+ str(i) +'.png')
    x = TF.to_tensor(image)
    input_matrix = x.unsqueeze_(0)
    label_network_prediction = model(input_matrix)

    # TODO:
    #1. Find probability
    #2. Find labels
    prediction = None
    print('network prediction for img',i,' : ',prediction)
