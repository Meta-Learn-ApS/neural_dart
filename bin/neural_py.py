import torch
import torch.nn as nn
import json


def save(net: nn.Module):
    with open("test.json", "w") as file:
        model = str([module for module in net.model.modules()][1:]).replace("True", "true").replace("False", "false").replace("in_features=", "").replace("bias=", "").replace("out_features=", "").replace("), ", "),  ").replace(", ", ",")[1:-1].split(", ")
        parameters = {key: value.numpy().tolist() for key, value in dict(net.state_dict()).items()}
        file.write(json.dumps({"model": model, "parameters": list(parameters.values())}))


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 25),
            nn.ReLU(),
            nn.Linear(25, 3)
        )

    def forward(self, x):
        return self.model(x)


net = Network()


params = list(net.parameters())
# print(len(params))
# for param in params:
#     print(param)


input = torch.tensor([0.0, -1.1, 2.2, 3.3, 4.4, -5.5, 6.6, 7.7, 8.8, 9.9])
out = net(input)
# print(input.grad)
# print(out.grad)
print(list(out.detach().numpy()))


save(net)
# net.zero_grad()
# out.backward(torch.randn(1, 2))
