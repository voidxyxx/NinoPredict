import numpy as np
import torch as t


"""
CMIP5:  (4, 27180, 24, 72)
sst_min:  -16.549121856689453 sst_max:  10.198975563049316
t300_min:  -11.912577629089355 t300_max:  9.318188667297363
ua_min:  -22.261333465576172 ua_max:  14.382851600646973
va_min:  -17.876197814941406 va_max:  17.425098419189453
label_min:  -3.5832221508026123 label_max:  4.138188362121582

CMIP6:  (4, 28560, 24, 72)
sst_min:  -16.549121856689453 sst_max:  10.198975563049316
t300_min:  -11.912577629089355 t300_max:  9.318188667297363
ua_min:  -22.261333465576172 ua_max:  14.382851600646973
va_min:  -17.876197814941406 va_max:  17.425098419189453
label_min:  -3.5832221508026123 label_max:  3.911078691482544
"""

sst_min, sst_max = -16.549, 10.199
t300_min, t300_max = -11.913, 9.318
ua_min, ua_max = -22.261, 14.382
va_min, va_max = -17.876, 17.425
label_min, label_max = -3.583, 4.138188362121582


def fullfill(data):
    if np.isnan(data).any():
        nans, x = np.isnan(data), lambda z: z.nonzero()[0]
        data[nans] = np.interp(x(nans), x(~nans), data[~nans])
    return data


def get_nino_data(mode='soda'):
    """
    :param mode: from ['soda', 'CMIP5', 'CMIP6']
    :return: data_input: array of [length, 4, 24, 72]
             data_label: array of [length, 1]
    """
    if mode == 'soda':
        data_dict = np.load('data/Data_connect_' + mode + '.npy', allow_pickle=True).tolist()
        tgt = fullfill(data_dict['label_' + mode])  # (1, 1200)
        sst = fullfill(data_dict['sst_' + mode])
        t300 = fullfill(data_dict['t300_' + mode])
        ua = fullfill(data_dict['ua_' + mode])
        va = fullfill(data_dict['va_' + mode])

    else:
        data_dict1 = np.load('data/Data_connect_' + mode + '_1.npy', allow_pickle=True).tolist()
        data_dict2 = np.load('data/Data_connect_' + mode + '_2.npy', allow_pickle=True).tolist()
        mode = 'smip'
        tgt = fullfill(data_dict1['label_' + mode]).reshape(1, -1)
        sst = fullfill(data_dict1['sst_' + mode]).reshape(1, -1, 24, 72)
        t300 = fullfill(data_dict1['t300_' + mode]).reshape(1, -1, 24, 72)
        ua = fullfill(data_dict2['ua_' + mode]).reshape(1, -1, 24, 72)
        va = fullfill(data_dict2['va_' + mode]).reshape(1, -1, 24, 72)
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
            src = np.concatenate((src1, src2), axis=1)
            tgt = np.concatenate((tgt1, tgt2), axis=1)
            for i in range(4):
                src[i] = -1 + 2 * (src[i] - np.mean(src[i], axis=0)) / (np.std(src[i], axis=0) + 1e-8)
            self.src = src
            self.tgt = -1 + 2 * (tgt - np.mean(tgt)) / (np.std(tgt) + 1e-8)
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
            src, tgt = get_nino_data('soda')
            src1, tgt1 = get_nino_data('CMIP5')
            src2, tgt2 = get_nino_data('CMIP6')
            src12 = np.concatenate((src1, src2), axis=1)
            tgt12 = np.concatenate((tgt1, tgt2), axis=1)
            for i in range(4):
                src[i] = -1 + 2 * (src[i] - np.mean(src12[i], axis=0)) / (np.std(src12[i], axis=0) + 1e-8)
            self.src = src
            self.tgt = -1 + 2 * (tgt - np.mean(tgt12)) / (np.std(tgt12) + 1e-8)
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
    src_5, tgt_5 = get_nino_data('CMIP5')
    print('CMIP5: ', src_5.shape)
    print('sst_min: ', np.min(src_5[0]), 'sst_max: ', np.max(src_5[0]))
    print('t300_min: ', np.min(src_5[1]), 't300_max: ', np.max(src_5[1]))
    print('ua_min: ', np.min(src_5[2]), 'ua_max: ', np.max(src_5[2]))
    print('va_min: ', np.min(src_5[3]), 'va_max: ', np.max(src_5[3]))
    print('label_min: ', np.min(tgt_5), 'label_max: ', np.max(tgt_5))
    src_6, tgt_6 = get_nino_data('CMIP6')
    print('CMIP6: ', src_6.shape)
    print('sst_min: ', np.min(src_6[0]), 'sst_max: ', np.max(src_6[0]))
    print('t300_min: ', np.min(src_6[1]), 't300_max: ', np.max(src_6[1]))
    print('ua_min: ', np.min(src_6[2]), 'ua_max: ', np.max(src_6[2]))
    print('va_min: ', np.min(src_6[3]), 'va_max: ', np.max(src_6[3]))
    print('label_min: ', np.min(tgt_6), 'label_max: ', np.max(tgt_6))
    src_s, tgt_s = get_nino_data('soda')
    print('SODA: ', src_6.shape)
    print('sst_min: ', np.min(src_s[0]), 'sst_max: ', np.max(src_s[0]))
    print('t300_min: ', np.min(src_s[1]), 't300_max: ', np.max(src_s[1]))
    print('ua_min: ', np.min(src_s[2]), 'ua_max: ', np.max(src_s[2]))
    print('va_min: ', np.min(src_s[3]), 'va_max: ', np.max(src_s[3]))
    print('label_min: ', np.min(tgt_s), 'label_max: ', np.max(tgt_s))

