from enum import Enum
import numpy as np
import logging

_logger = logging.getLogger(__name__)


class VhdlEnum(int, Enum):
    def __str__(self):
        return self.name.replace("_", " ")

    def __repr__(self):
        return self.name

    def select(self):
        return pow(2, self.value)


class Detection(VhdlEnum):  # Event type
    peak = 0
    area = 1
    pulse = 2
    trace = 3
    tick = 4

    @staticmethod
    def from_flags(flags):
        if np.bitwise_and(flags[1], 0x02):
            return Detection(4)
        else:
            return Detection(
                np.right_shift(np.bitwise_and(flags[1], 0x0C), 2))


class EventType(VhdlEnum):
    peak = 0
    area = 1
    pulse = 2
    dot_product = 3
    tick = 4
    average = 5

    @staticmethod
    def from_data(data):
        """
        :param data: at least the first 6 bytes of the event
        :return: EventType
        """
        detection = Detection.from_flags(data[4:6])
        if detection == Detection.trace:
            # print(detection)
            if (
                    TraceType(data[3] & 0xC0 >> 6) in
                    (TraceType.dot_product_trace, TraceType.dot_product)
            ):
                return EventType.dot_product
            else:
                return EventType.pulse
        else:
            return EventType(detection.value)

    @staticmethod
    def max_peaks(data):
        """

        :param data: bytes must be pulse type of event
        :return:
        """
        if EventType.has_samples(data):
            max_peaks = (data[3] & 0x0F) - 2
            if EventType.has_dp(data):
                max_peaks -= 1
        else:
            max_peaks = data[0:2].view('u2')
        return max_peaks

    @staticmethod
    def has_samples(data):
        return (
            (Detection.from_flags(data[4:6]) == Detection.trace) and
            (TraceType((data[3] & 0xC) >> 6) != TraceType.dot_product)
        )

    @staticmethod
    def has_dp(data):
        t = TraceType((data[3] & 0xC) >> 6)
        return (
            (Detection.from_flags(data[4:6]) == Detection.trace) and
            ((t == TraceType.dot_product) or (t == TraceType.dot_product_trace))
        )

    @staticmethod
    def pulse_fmt(data):
        max_peaks = EventType.max_peaks(data)
        if (max_peaks > 0) and (max_peaks <= 9):
            return (
                pulse_header_fmt + [('peaks', pulse_rise_fmt, (max_peaks,))]
            )
        else:
            raise AttributeError('peak_count must be between 1 and 8 inclusive')

    @staticmethod
    def dtype(data):
        t = EventType.from_data(data)
        if t == EventType.peak:
            return np.dtype(rise_fmt)
        elif t == EventType.area:
            return np.dtype(area_fmt)
        elif t == EventType.tick:  # tick
            return np.dtype(tick_fmt)
        elif t == EventType.pulse:
            return np.dtype(EventType.pulse_fmt(data))
        else:
            return np.dtype(EventType.pulse_fmt(data) + dot_product_fmt)


class TraceType(VhdlEnum):
    single = 0
    average = 1
    dot_product = 2
    dot_product_trace = 3


class Signal(VhdlEnum):
    none = 0
    raw = 1
    filtered = 2
    slope = 3


def event_dt(detection_type, trace_type, max_peaks):
    if detection_type == Detection.pulse:
        fmt = (
            pulse_header_fmt + [('peaks', pulse_rise_fmt, (max_peaks + 1,))]
        )
    else:
        raise NotImplementedError

    return np.dtype(fmt)


# FIXME add traces
def event_fmt(flags, size, offset=None):
    print(flags, size, offset)
    event = Detection.from_flags(flags)
    if event is Detection.tick:
        return np.dtype(tick_fmt)
    elif event is Detection.area:
        return np.dtype(area_fmt)
    elif event is Detection.peak:
        return np.dtype(rise_fmt)
    # elif event is DetectionType.trace:
    #     trace_fmt = (
    #         trace_header_fmt + [('peaks', pulse_peak_fmt, (int(offset)-2,))] +
    #         [('trace', np.int16, (int(size-offset)*4,))]
    #     )
    #     print(trace_fmt)
    #     return np.dtype(trace_fmt)
    elif event is Detection.pulse:
        pulse_fmt = (
            pulse_header_fmt + [('rises', pulse_rise_fmt, (int(size) - 2,))]
        )
        return pulse_fmt


class Height(VhdlEnum):
    peak_height = 0
    cfd_high = 1
    cfd_height = 2
    slope_max = 3


class Timing(VhdlEnum):
    pulse_threshold = 0
    slope_threshold = 1
    cfd_low = 2
    max_slope = 3


class Payload(VhdlEnum):
    rise = 0
    area = 1
    pulse = 2
    trace = 3
    tick = 4
    mca = 5
    average_trace = 6
    dot_product = 7
    dot_product_trace = 8
    unknown = 9
    unknown_ethertype = 10
    bad_ethernet = 11
    sequence_error = 12

    # add from index

    @staticmethod
    def from_frame(frame):
        ether_type = frame[12:14].view('>u2')[0]

        if not np.array_equal(
                frame[:12], [0x5A, 1, 2, 3, 4, 5, 0xDA, 1, 2, 3, 4, 5]
        ):
            return Payload.bad_ethernet
        elif ether_type == 0x88B6:
            return Payload.mca
        elif ether_type == 0x88B5:
            return Payload.from_field(frame[22:24].view('>u2')[0])
        else:
            return Payload.unknown_ethertype

    @staticmethod
    def from_field(field):
        event_type = field & 0xFF
        trace_type = (field & 0xFF00) >> 8
        # print(
        #     'event_type:0x{:04X} trace_type:0x{:04X}'
        #     .format(event_type, trace_type)
        # )
        if event_type & 2:
            return Payload.tick
        detection = (event_type & 0x0C) >> 2
        if detection != 3:
            return Payload(detection)
        else:
            if trace_type == 0:
                return Payload.trace
            if trace_type == 1:
                return Payload.average_trace
            if trace_type == 2:
                return Payload.dot_product
            if trace_type == 3:
                return Payload.dot_product_trace

    def event_dt(self, size=None, length=None):
        if self == Payload.rise:
            return np.dtype(rise_fmt)
        if self == Payload.area:
            return np.dtype(area_fmt)
        if self == Payload.pulse:
            if size < 3:
                raise RuntimeError('pulse with size < 3')
            if size == 3:  # one rise slot
                return np.dtype(pulse_header_fmt + pulse_rise_fmt)
            return np.dtype(
                pulse_header_fmt +
                [('rises', pulse_rise_fmt, (int(size) - 2,))])
        if self == Payload.tick:
            return np.dtype(tick_fmt)
        if self == Payload.dot_product:
            if size < 4:
                raise RuntimeError('dot_product with size < 4')
            if size == 4:  # one rise slot
                return np.dtype(
                    pulse_header_fmt + pulse_rise_fmt + dot_product_fmt
                )
            return np.dtype(
                pulse_header_fmt +
                [('rises', pulse_rise_fmt, (int(size) - 3,))] +
                dot_product_fmt
            )


def lookup(value, enum):
    if isinstance(value, str):
        return enum[value]
    else:
        return enum(value)

pulse_rise_fmt = [
    ('height', 'i2'), ('minima', 'i2'), ('rise_time', 'u2'), ('peak_time', 'u2')
]

# offset is the time since the first minima to the timing point
pulse_header_fmt = [
    ('size', 'u2'), ('flags', '(4,)u1'), ('pulse_time', 'u2'),
    ('area', 'i4'), ('pulse_length', 'u2'), ('time_offset', 'u2')
]

average_fmt = [
    ('size', 'u2'), ('flags', '(4,)u1'), ('pulse_time', 'u2'),
    ('multipulse_count', 'u4'), ('multipeak_count', 'u4')
]

# TODO deprecate
trace_header_fmt = [
        ('size', np.uint16), ('tflags0', np.uint8), ('tflags1', np.uint8),
        ('flags0', np.uint8), ('flags1', np.uint8), ('time', np.uint16),
        ('area', np.uint32), ('length', np.uint16), ('offset', np.uint16)
]

rise_fmt = [
    ('height', 'i2'), ('minima', 'i2'), ('flags', '(2,)u1'), ('time', 'u2')
]

area_fmt = [('area', 'i4'), ('flags', '(2,)u1'), ('time', 'u2')]

tick_fmt = [
    ('period', 'u4'), ('eflags', '(2,)u1'), ('time', 'u2'),
    ('timestamp', 'u8'),
    ('overflow', 'u1'), ('error', 'u1'), ('cfd_error', 'u1'), ('res2', 'u1'),
    ('events_lost', 'u4')
]

dot_product_fmt = [('dot_product', 'u8')]
