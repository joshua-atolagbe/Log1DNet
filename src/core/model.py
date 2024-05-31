from torch import nn 

class Log1DNetv3(nn.Module):

    def __init__(self, n_features:int=4, batch_size:int=64, n_outputs:int=2):
        super(Log1DNetv3, self).__init__()
        self.features = n_features
        self.bs = batch_size
        self.outputs = n_outputs

        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, batch_size, kernel_size=1),
            nn.BatchNorm1d(batch_size),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.AvgPool1d(1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(batch_size, batch_size*2, kernel_size=1),
            nn.BatchNorm1d(batch_size*2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AvgPool1d(1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(batch_size*2, batch_size*4, kernel_size=1),
            nn.BatchNorm1d(batch_size*4),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(batch_size*4, batch_size, kernel_size=1),
            nn.BatchNorm1d(batch_size),
            nn.ReLU(),
        )
        self.flat = nn.Flatten()
        self.layer = nn.Sequential(
            nn.Linear(batch_size, batch_size),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(batch_size, 2)
        )

    def forward(self, x):
        x = x.reshape(self.bs, self.features, 1)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.flat(out)
        out = self.layer(out)
        out = self.output(out)

        return out