import pickle

class EverythingExceptModel(object):
    def __init__(self, train_loss, test_loss, train_acc, test_acc, fig_train, fig_test):
        self.train_loss = train_loss
        self.test_loss = test_loss
        self.train_acc = train_acc
        self.test_acc = test_acc
        self.fig_train = fig_train
        self.fig_test = fig_test
    def save(self,path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

