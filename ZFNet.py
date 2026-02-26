import torch 
import torch.nn as nn

class ZFNet(nn.Module): 
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2), 

            # Layer 2 
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=2), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), 

            # Layer 3 
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding="same"), 
            nn.ReLU(inplace=True), 

            # Layer 4 
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=2, stride=1, padding="same"), 
            nn.ReLU(inplace=True), 

            # Layer 5 
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding="same"), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2) 
        )

        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256*6*6, out_features=4096), 
            nn.ReLU(inplace=True), 
            nn.Dropout(0.5),

            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(in_features=4096, out_features=1000)
        )

    def forward(self, x): 
        x = self.feature_extractor(x) 
        x = self.fully_connected(x) 
        return x 