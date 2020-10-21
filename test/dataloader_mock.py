class DataLoaderMock(object):
    def __init__(self, data_loader):
        self._data_loader = data_loader

    def __len__(self):
        return 1

    def __iter__(self):
        for batch in self._data_loader:
            yield batch
            break
