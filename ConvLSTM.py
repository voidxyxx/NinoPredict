"""
Reference: github.com/spacejake/convLSTM.pytorch/convlstm.py
"""

import torch as t


class ConvLSTMCell(t.nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = t.nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, x, prev_state):
        h_prev, c_prev = prev_state
        combined = t.cat((x, h_prev), dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = t.split(combined_conv, self.hidden_dim, dim=1)

        i = t.sigmoid(cc_i)
        f = t.sigmoid(cc_f)
        o = t.sigmoid(cc_o)
        g = t.tanh(cc_g)

        c_cur = f * c_prev + i * g
        h_cur = o * t.tanh(c_cur)

        return h_cur, c_cur

    def init_hidden(self, batch_size, cuda=True):
        state = (t.autograd.Variable(t.zeros(batch_size, self.hidden_dim, self.height, self.width)),
                 t.autograd.Variable(t.zeros(batch_size, self.hidden_dim, self.height, self.width)))
        if cuda:
            state = (state[0].cuda(), state[1].cuda())
        return state


class ConvLSTM(t.nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False,
                 bias=False, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.cell_list = t.nn.ModuleList(cell_list)

    def forward(self, input, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input = input.permute(1, 0, 2, 3, 4)

        if hidden_state is None:
            hidden_state = self.get_init_states(batch_size=input.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input.size(1)
        cur_layer_input = input

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for s in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, s, :, :, :],
                                                    prev_state=[h, c])
                output_inner.append(h)

            layer_output = t.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        layer_output = layer_output_list[-1]
        if not self.batch_first:
            layer_output = layer_output.permute(1, 0, 2, 3, 4)

        return layer_output, last_state_list

    def get_init_states(self, batch_size, cuda=True):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, cuda))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


from CNN3d import _nino3d
class NinoConvLSTM(t.nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, out_dim,
                 batch_first=False, bias=False, return_all_layers=False):
        super(NinoConvLSTM, self).__init__()
        self.conv_lstm = ConvLSTM(input_size,
                                  input_dim,
                                  hidden_dim,
                                  kernel_size,
                                  num_layers,
                                  batch_first=batch_first,
                                  bias=bias,
                                  return_all_layers=return_all_layers)
        expansion = (hidden_dim[-1] // 64) * 2 if hidden_dim[-1] >= 64 else 1
        self.conv = _nino3d([2, 2, 2], hidden_dim[-1], expansion, out_dim)

    def forward(self, x):
        out, _ = self.conv_lstm(x)
        out = out.permute(0, 2, 1, 3, 4)
        out = self.conv(out)
        return out


if __name__ == '__main__':
    height = 24
    width = 72
    channels = 4
    model = ConvLSTM(input_size=(height, width),
                     input_dim=channels,
                     hidden_dim=[64, 64, 128],
                     kernel_size=(3, 3),
                     num_layers=3,
                     batch_first=True,
                     bias=True,
                     return_all_layers=False).cuda()
    data = t.rand((16, 12, 4, 24, 72)).cuda()

    from data_loader import *
    import time
    import matplotlib.pyplot as plt
    from score import *

    train_set = Nino3DDatasetTrain()
    print('data loaded')
    lr = 1e-3
    epochs = 100
    batch_size = 16
    weight_decay = 1e-3
    if t.cuda.is_available():
        print("CUDA in use")
        device = t.device('cuda')
    else:
        device = t.device('cpu')

    # model = NinoConvLSTM((24, 72), 4, [64, 64], kernel_size=(3, 3),
    #                      num_layers=2,
    #                      out_dim=36,
    #                      batch_first=True,
    #                      bias=True,
    #                      return_all_layers=False
    #                      ).cuda()
    model = NinoConvLSTM((24, 72), 4, [64, 64], kernel_size=(3, 3),
                         num_layers=2,
                         out_dim=24,
                         batch_first=True,
                         bias=True,
                         return_all_layers=False
                         ).cuda()
    optimizer = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    criterion = t.nn.MSELoss()
    train_loss_list = []
    for epoch in range(epochs):
        model.train()
        data_loader = t.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        total_loss = 0.0
        for idx, (src, tgt) in enumerate(data_loader):
            start_time = time.time()
            src, tgt = src.to(device), tgt.to(device)
            src = src.permute(0, 2, 1, 3, 4)
            pred = model(src)
            optimizer.zero_grad()
            # loss = criterion(tgt.squeeze(), pred)
            loss = criterion(tgt.squeeze()[:, 12:], pred)
            loss.backward()
            optimizer.step()
            # total_loss += criterion(tgt.squeeze(1)[:, 12:], pred[:, 12:]).item() * src.size(0)
            total_loss += criterion(tgt.squeeze(1)[:, 12:], pred).item() * src.size(0)

            if (idx + 1) % 100 == 0:
                print("Epoch {:d}: {:d}/{:d} \t train loss: {:.6f} \t lr: {:.6f} \t time cost: {:.3f}".format(
                    epoch + 1, (idx + 1) * src.size(0), len(data_loader.dataset),
                    loss.item(), optimizer.param_groups[0]['lr'], time.time() - start_time
                ))
        lr_scheduler.step(total_loss)
        train_loss_list.append(total_loss / len(data_loader.dataset))

    model.eval()
    test_set = Nino3DDatasetTest()
    test_loader = t.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)
    score = 0.0
    count = 0
    for src, tgt in test_loader:
        src, tgt = src.to(device), tgt.to(device)
        src = src.permute(0, 2, 1, 3, 4)
        pred = model(src)
        # score += get_score(tgt.squeeze(1).cpu().detach().numpy()[:, 12:], pred.cpu().detach().numpy()[:, 12:]) * len(
        #     tgt)
        score += get_score(tgt.squeeze(1).cpu().detach().numpy()[:, 12:],
                           pred.cpu().detach().numpy()) * len(tgt)
        count += tgt.shape[0]
    print(score / count)
    t.save(model, "ConvLSTM_minmax.pth")

    # plt.figure()
    # plt.plot(list(range(1, epochs + 1)), train_loss_list)
    # plt.savefig('ConvLSTM.png')

