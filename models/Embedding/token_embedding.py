##

from torch import nn
class FeatureShrink(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=19, out_features=16)

    def forward(self, STAFea):
        return self.linear(STAFea)
