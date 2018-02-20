import numpy as np
from .base import VhdlEnum


class Value(VhdlEnum):
    zero = 0
    filtered_signal = 1
    filtered_area = 2
    filtered_extrema = 3
    slope_signal = 4
    slope_area = 5
    slope_extrema = 6
    raw_signal = 7
    raw_area = 8
    raw_extrema = 9
    pulse_area = 10
    pulse_length = 11
    rise_time = 12


class Trigger(VhdlEnum):
    disabled = 0
    clock = 1
    pulse_threshold_rising = 2
    pulse_threshold_falling = 3
    filtered_zero = 4
    slope_zero = 5
    raw_zero = 6
    slope_threshold_rising = 7
    cfd_high_rising = 8
    cfd_low_rising = 9
    max_slope = 10
    slope_zero_rising = 11
    slope_zero_falling = 12


class Qualifier(VhdlEnum):
    disabled = 0
    all = 1
    valid_peak = 2
    above_area = 3
    above_pulse = 4
    will_go_above_pulse = 5
    armed = 6
    will_arm = 7
    valid_peak0 = 8
    valid_peak1 = 9
    valid_peak2 = 10


header_dt = np.dtype([
    ('size', np.uint16), ('last_bin', np.uint16),
    ('lowest_value', np.int32), ('reserved', np.uint16),
    ('most_frequent', np.uint16), ('flags', np.uint32),
    ('total', np.uint64), ('start_time', np.uint64),
    ('stop_time', np.uint64)
])


class Distribution:
    """
    wrapper for transmitted zmq frame representing a MCA distribution
    """
    def __init__(self, data, buffer=True):
        self.data = data
        self.buffer = buffer
        # these should just be views on the frame
        # not clear if the HDF5 version is copying
        # if buffer:
        self.header = np.frombuffer(self.data[:40], dtype=header_dt)[0]
        self.counts = np.frombuffer(
            self.data[40:], dtype=np.dtype(np.uint32)
        )
        # else:
        #     self.header = self.data[:40].view(header_dt)[0]
        #     self.counts = self.data[40:].view(np.uint32)

    @property
    def most_frequent(self):
        return self.header['most_frequent']

    # flags
    @property
    def channel(self):
        return np.bitwise_and(self.header['flags'], 0x00000007)

    @property
    def bin_width(self):
        return 2**np.right_shift(
            np.bitwise_and(self.header['flags'], 0x000000F8), 3
        )

    @property
    def trigger(self):
        return Trigger(
            np.right_shift(
                np.bitwise_and(self.header['flags'], 0x00000F00), 8
            )
        )

    @property
    def value(self):
        return Value(
            np.right_shift(np.bitwise_and(self.header['flags'], 0x0000F000), 12)
        )

    @property
    def qualifier(self):
        return Qualifier(
            np.right_shift(np.bitwise_and(self.header['flags'], 0x000F0000), 16)
        )

    @property
    def bins(self):
        return self.counts[1:-1]

    @property
    def underflow(self):
        return self.counts[0]

    @property
    def overflow(self):
        return self.counts[-1]

    @property
    def lowest_value(self):
        return self.header['lowest_value']

    @property
    def total(self):
        return self.header['total']

    @property
    def start_time(self):
        return self.header['start_time']

    @property
    def stop_time(self):
        return self.header['stop_time']

    @property
    def last_bin(self):
        return self.header['last_bin']

    def __repr__(self):
        return (
            'Distribution: channel:{:} value:{:}, trigger:{:}, qualifier:{:}'
            .format(
                self.channel, str(self.value), str(self.trigger),
                str(self.qualifier)
            )
        )




