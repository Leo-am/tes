import numpy as np



# need open 
class _Hdf5DS:
    """
    wrapper for DS in hdf5 file
    """
    def __init__(
            self, h, name, ds_shape, dtype, index=None, index_shape=None
    ):
        self.h = h
        self.file_ds = h.create_dataset(name, ds_shape, dtype=dtype)
        self.i = 0
        self.index = None
        if index is not None:
            self.index = _Hdf5Index(self.h, index, index_shape)

    def append(self, data):
        if data is None:
            return None
        d = data.view(self.file_ds.dtype)
        l = len(d)
        start = self.i
        self.file_ds[start: start + l] = d
        self.i += l
        # print(start, self.i)
        entry = Slice(start, self.i)
        if self.index is not None:
            self.index.append(entry)
        return entry

    def __getitem__(self, item):
        """
        :param item:
        :return: If has index return data referenced by index[item]
                 else return data[item]
        """
        if self.index is not None:
            return self.file_ds[self.index[item]]
        else:
            return self.file_ds[item]

    def __len__(self):
        """

        :return: if has index return len(index) else len(ds)
        """
        if self.index is not None:
            return len(self.index)
        return len(self.file_ds)


class _Hdf5Index(_Hdf5DS):
    def __init__(self, h, name, shape, ):
        """
        Index for a Hdf5DS entrys are np.dtype((1,), dtype=index_dt)
        returns slices into the Hdf5DS
        :param h:
        :param shape:
        :param name:
        """
        super().__init__(h, name, shape, dtype=Slice.dtype)

    def __getitem__(self, item):
        """
        warning abuses slices! if item is a slice returns a slice for the
        underlying Hdf5DS that goes from index[item.start][0] to
        index[item.stop][1] ignoring step

        If item is int return a slice from index[item][0] to index[item][1]
        :param item:
        :return:
        """
        if item >= self.i:
            raise IndexError(
                '{} out of range for index length {}'.format(item, self.i)
            )
        if isinstance(item, int):
            entry = self.file_ds[item]
            return slice(entry[0], entry[1])
        if isinstance(item, slice):
            return slice(
                self.file_ds[item.start][0], self.file_ds[item.stop][1]
            )
        raise NotImplementedError(
            'index of type:{} are not implemented'.format(type(item))
        )

    def append(self, data):
        if isinstance(data, Slice):
            self.file_ds[self.i] = np.array(
                [data.start, data.stop], dtype='u8'
            ).view(Slice.dtype)
            self.i += 1
        else:
            raise NotImplementedError('type:{} not implemented for index entry')


class _Hdf5Structure:
    def __init__(
            self, h, ds_names, shapes, ds_dtypes, index_shape, data_path='data',
            index_path='index'
    ):
        """
        multiple idexed data sets
        DS will be stored  @data/names and index @index/names

        data is appended as a tuple with elements corresponding to each ds
        access though index only

        :param h: h5py file handle
        :param shapes: dataset shape
        :param ds_names: sequence of names for each ds stored @ data/name
        :param ds_dtypes: sequence of np.dtypes for each DS
        :param index_shape: name for index DS stored at index/index_name
        """
        # print(ds_names, shapes, ds_dtypes, index_shape)
        self.hdf5 = h
        self.names = ds_names
        ds = [
            _Hdf5DS(
                h, '{}/{}'.format(data_path, name), shape, dtype,
                index='{}/{}'.format(index_path, name), index_shape=index_shape
            ) for name, dtype, shape in zip(ds_names, ds_dtypes, shapes)
        ]
        self.ds = tuple(ds)

    def __len__(self):
        return len(self.ds[0].index)

    def append(self, data_tuple):
        """
        append data to each DS in structure, update index if it exists

        :param data_tuple:
        :return: tuple of regions added.
        """
        if (
                    not isinstance(data_tuple, tuple) or len(data_tuple) !=
                    len(self.names)
        ):
            raise AttributeError(
                'Expected data tuple of length {}'.format(len(self.names))
            )

        return [ds.append(data) for ds, data in zip(self.ds, data_tuple)]

    def __getitem__(self, item):
        """

        :param item: index entrys to return
        :return: tuple with data from each dataset
        """
        return tuple([ds[item] for ds in self.ds])
