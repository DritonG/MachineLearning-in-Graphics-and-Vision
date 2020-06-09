import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF

# TODO: declare your model
class Model(nn.Module):
    pass


model = Model()

#TODO
# load model variables in evaluation mode

probability = torch.nn.Softmax(dim=0)

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
