from loader import load_config, load_data
import numpy as np
import os
class CF_Dataset(object):
    def __init__(self, train_file, test_file, addnoise, noise_scale = 0.0):
        if (not (os.path.isfile(train_file))):
            raise Exception('No Such Training File')
        if (not (os.path.isfile(test_file))):
            raise Exception('No Such Test File')
        self.train_x, self.train_t, self.train_y = load_data(train_file)
        if (addnoise):
            self.train_y += np.random.normal(0, noise_scale, size = self.train_y.shape)
        self.test_x, self.test_t, self.test_y = load_data(test_file)
        self.x = self.train_x
        self.t = self.train_t
        self.y = self.train_y
    def getBasicInfo(self):
        return self.x.shape[0], self.t.shape[1], self.x.shape[1]
    def switch(self, mode):
        mode = mode.lower()
        if (mode[:5] == 'train'):
            self.x = self.train_x
            self.t = self.train_t
            self.y = self.train_y
        elif (mode[:4] == 'test'):
            self.x = self.test_x
            self.t = self.test_t
            self.y = self.test_y
    def getData(self):
        return self.x, self.t, self.y