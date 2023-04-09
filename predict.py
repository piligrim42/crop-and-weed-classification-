
#Using Pre-trained model ('my_model.pt') to predict plants. 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#Neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32*56*56, 128)
        self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(-1, 32*56*56)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


net = Net()

#Loading the saved model
net.load_state_dict(torch.load('my_model.pt'))

#Loading the image and applying the same transformations used for the training data
image = Image.open("test_ctn2.jpg")
image = transform(image).unsqueeze(0)


#Predicting the label for the image
with torch.no_grad():
    output = net(image)
    _, predicted = torch.max(output.data, 1)
    if predicted.item() == 0:
        print("This plant is classified as a Crop.")
    else:
        print("This plant is classified as a Weed.")
