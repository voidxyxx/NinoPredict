import torch as t


class BasicBlock(t.nn.Module):
    """
    Resnet v1
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = t.nn.BatchNorm3d

        self.conv1 = t.nn.Conv3d(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1))
        self.bn1 = norm_layer(planes)
        self.relu = t.nn.ReLU(inplace=True)
        self.conv2 = t.nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Nino3DNet(t.nn.Module):
    def __init__(self, layers, in_channels=4, channel_expansion=1, output_dim=1, norm_layer=None):
        super(Nino3DNet, self).__init__()

        if norm_layer is None:
            norm_layer = t.nn.BatchNorm3d
        self._norm_layer = norm_layer
        self.inplanes = 64 * channel_expansion

        self.conv1 = t.nn.Conv3d(in_channels,
                                 self.inplanes,
                                 kernel_size=(1, 7, 7),
                                 stride=(1, 1, 3),
                                 padding=(0, 3, 3))
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = t.nn.ReLU(inplace=True)
        self.maxpool = t.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(64 * channel_expansion, layers[0])
        self.layer2 = self._make_layer(128 * channel_expansion, layers[1], stride=2)
        self.layer3 = self._make_layer(256 * channel_expansion, layers[2], stride=2)
        self.avgpool = t.nn.AdaptiveAvgPool3d(1)
        self.fc = t.nn.Linear(256 * channel_expansion, output_dim)

        for m in self.modules():
            if isinstance(m, t.nn.Conv2d):
                t.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (t.nn.BatchNorm2d, t.nn.GroupNorm)):
                t.nn.init.constant_(m.weight, 1)
                t.nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        if stride != 1 or self.inplanes != planes:
            downsample = t.nn.Sequential(
                t.nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride),
                self._norm_layer(planes))
        else:
            downsample = None

        layers = [BasicBlock(self.inplanes, planes, stride, downsample, self._norm_layer)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, norm_layer=self._norm_layer))
        return t.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = t.flatten(out, 1)
        out = self.fc(out)

        return out


def _nino3d(layers, in_channels=4, channel_expansion=1,  out_dim=36):
    return Nino3DNet(layers, in_channels, channel_expansion, out_dim)


if __name__ == '__main__':
    from data_loader import *
    import time
    import matplotlib.pyplot as plt
    from score import *

    # train_set = Nino3DDatasetTrain()
    # print('data loaded')
    # lr = 1e-3
    # epochs = 300
    # batch_size = 4
    # weight_decay = 1e-3
    # if t.cuda.is_available():
    #     print("CUDA in use")
    #     device = t.device('cuda')
    # else:
    #     device = t.device('cpu')
    #
    # model = _nino3d([2, 2, 2], 4, 1, 36).to(device)
    # optimizer = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # lr_scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    #
    # criterion = t.nn.MSELoss()
    # train_loss_list = []
    # for epoch in range(epochs):
    #     model.train()
    #     data_loader = t.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    #     total_loss = 0.0
    #     for idx, (src, tgt) in enumerate(data_loader):
    #         start_time = time.time()
    #         src, tgt = src.to(device), tgt.to(device)
    #         pred = model(src)
    #         optimizer.zero_grad()
    #         loss = criterion(tgt.squeeze(), pred)
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += criterion(tgt.squeeze(1)[:, 12:], pred[:, 12:]).item() * src.size(0)
    #
    #         if (idx + 1) % 100 == 0:
    #             print("Epoch {:d}: {:d}/{:d} \t train loss: {:.6f} \t lr: {:.6f} \t time cost: {:.3f}".format(
    #                 epoch + 1, (idx + 1) * src.size(0), len(data_loader.dataset),
    #                 loss.item(), optimizer.param_groups[0]['lr'], time.time() - start_time
    #             ))
    #     lr_scheduler.step(total_loss)
    #     train_loss_list.append(total_loss / len(data_loader.dataset))

    model = t.load("Conv3d.pth")
    device = t.device('cuda')
    model.eval()
    test_set = Nino3DDatasetTest()
    test_loader = t.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)
    score = 0.0
    count = 0
    for src, tgt in test_loader:
        src, tgt = src.to(device), tgt.to(device)
        pred = model(src)
        score += get_score(tgt.squeeze(1).cpu().detach().numpy()[:, 12:], pred.cpu().detach().numpy()[:, 12:]) * len(tgt)
        count += tgt.shape[0]
    t.save(model, "Conv3d.pth")
    print(score/count)

    # plt.figure()
    # plt.plot(list(range(1, epochs + 1)), train_loss_list)
    # plt.savefig('3d.png')
