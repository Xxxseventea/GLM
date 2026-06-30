from torch import nn


class MlpHead(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(.1),
            nn.Linear(hid_dim, out_dim, bias=True),
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        # x shape: [b t d] where t means the number of views
        x1 = self.model(x[:,21//2])
        x = self.act(x1)
        return x1,x