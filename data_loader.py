import numpy as np
import torch as t


def get_nino_data(mode='soda'):
    """
    :param mode: from ['soda', 'CMIP5', 'CMIP5', 'CMIP6', 'CMIP6']
    :return: data_input: array of [length, 4, 24, 72]
             data_label: array of [length, 1]
    """
    if mode == 'soda':
        data_dict = np.load('data/Data_connect_' + mode + '.npy', allow_pickle=True).tolist()
        tgt = data_dict['label_' + mode]  # (1, 1200)
        sst = data_dict['sst_' + mode]
        t300 = data_dict['t300_' + mode]
        ua = data_dict['ua_' + mode]
        va = data_dict['va_' + mode]
    else:
        data_dict1 = np.load('data/Data_connect_' + mode + '_1.npy', allow_pickle=True).tolist()
        data_dict2 = np.load('data/Data_connect_' + mode + '_2.npy', allow_pickle=True).tolist()
        mode = 'smip'
        tgt = data_dict1['label_' + mode].reshape(1, -1)
        sst = data_dict1['sst_' + mode].reshape(1, -1, 24, 72)
        t300 = data_dict1['t300_' + mode].reshape(1, -1, 24, 72)
        ua = data_dict2['ua_' + mode].reshape(1, -1, 24, 72)
        va = data_dict2['va_' + mode].reshape(1, -1, 24, 72)
    src = np.concatenate((sst, t300, ua, va), axis=0)  # (4, 1200, 24, 72)
    return src, tgt


class Nino3DDataset(t.utils.data.Dataset):
    def __init__(self, mode='soda'):
        super(Nino3DDataset, self).__init__()
        try:
            self.src, self.tgt = get_nino_data(mode)
        except EOFError:
            print("data mode should be 'soda', 'CMIP5', 'CMIP6', check dataset params")
            exit(1)

    def __getitem__(self, index):
        return t.tensor(self.src[:, index:index+12, :, :]).float(), t.tensor(self.tgt[:, index:index+36]).float()

    def __len__(self):
        return self.tgt.shape[1] - 35


class Nino3DDatasetTrain(t.utils.data.Dataset):
    def __init__(self):
        super(Nino3DDatasetTrain, self).__init__()
        try:
            src1, tgt1 = get_nino_data('CMIP5')
            src2, tgt2 = get_nino_data('CMIP6')
            self.src = np.concatenate((src1, src2), axis=1)
            self.tgt = np.concatenate((tgt1, tgt2), axis=1)
        except EOFError:
            print("data mode should be 'soda', 'CMIP5', 'CMIP6', check dataset params")
            exit(1)

    def __getitem__(self, index):
        return t.tensor(self.src[:, index:index+12, :, :]).float(), t.tensor(self.tgt[:, index:index+36]).float()

    def __len__(self):
        return self.tgt.shape[1] // 12

class Nino3DDatasetTest(t.utils.data.Dataset):
    def __init__(self):
        super(Nino3DDatasetTest, self).__init__()
        try:
            self.src, self.tgt = get_nino_data('soda')
        except EOFError:
            print("data mode should be 'soda', 'CMIP5', 'CMIP6', check dataset params")
            exit(1)

    def __getitem__(self, index):
        return t.tensor(self.src[:, index:index + 12, :, :]).float(), t.tensor(self.tgt[:, index:index + 36]).float()

    def __len__(self):
        return self.tgt.shape[1] // 12


def get_loader(mode, batch_size, shuffle):
    # If every epoch uses different shuffle state, dataset may be reloaded several times
    return t.utils.data.DataLoader(Nino3DDataset(mode), batch_size=batch_size, shuffle=shuffle)


if __name__ == '__main__':
    dataset = Nino3DDatasetTest()
    print(len(dataset))

