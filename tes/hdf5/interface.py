import logging

import h5py
import numpy as np

from ..base import (
    Detection, Height, Timing, Payload, pulse_rise_fmt, EventType,
    pulse_header_fmt, rise_fmt, average_fmt,
    area_fmt, tick_fmt, dot_product_fmt,
    TraceType, Signal
)
from ..mca import Distribution

from .storage import (
    _PacketData, _import, _region, _protocol_dt, _frame_index_dt, _DS
)

_logger = logging.getLogger(__name__)


class EventFlags:
    def __init__(self, data):
        """

        :param data: 2 byte array
        """
        self.data = data

    @property
    def peak_number(self):
        return self.data[0] & 0xF0 >> 8

    @property
    def rel2min(self):
        return self.data[0] & 0x08 >> 3

    @property
    def channel(self):
        return self.data[0] & 0x07

    @property
    def timing(self):
        return Timing(self.data[1] & 0xC0 >> 6)

    @property
    def height(self):
        return Height(self.data[1] & 0x30 >> 4)

    @property
    def detection(self):
        return Detection.from_flags(self.data)

    def __str__(self):
        return (
                'detection:{!r} peaks:{} rel2min:{} channel:{} timing:{!r} ' +
                'height:{!r} '
            ).format(
                self.detection, self.peak_number, self.rel2min, self.channel,
                self.timing, self.height
            )

    def __repr__(self):
        return str(self)


class Pulse:
    def __init__(self, data):
        self._bytestream = data
        flags = EventFlags(data[4:6])
        if flags.detection == Detection.trace:
            trace_flags = TraceFlags(data[2:4])
        else:
            trace_flags = None
        self._trace_flags = trace_flags
        self._flags = flags
        rel_time = data[6:8].view('u2')[0]
        self._relative_timestamp = rel_time
        if trace_flags is not None and trace_flags.type != TraceType.average:
            self._area = int(data[8:12].view('u4')[0])
            self._length = int(data[12:14].view('u2')[0])
            time_offset = int(data[14:16].view('u2')[0])
            self._time_offset = time_offset
            # if trace_flags is not None:
            #     peak_count = trace_flags.offset - 2
            # else:
            #     peak_count = data[0:2].view(np.uint16)[0] - 2
            # print(peak_count)

            self._peaks = PulsePeaks(
                data[16:16+(self._flags.peak_number+1)*8], 0
            )
            self._multipeaks = None
            self._multipulses = None
            if (
                    trace_flags.type == TraceType.dot_product or
                    trace_flags.type == TraceType.dot_product_trace
            ):
                o = (trace_flags.offset-1)*8
                self._dot_product = data[o:o+8].view(np.uint64)[0]
            else:
                self._dot_product = None

        else:
            self._area = None
            self._length = None
            self._time_offset = None
            self._peaks = None
            self._multipulses = data[8:12].view(np.uint32)
            self._multipeaks = data[12:16].view(np.uint32)
            self._dot_product = None

    def __getattr__(self, item):
        if hasattr(self._peaks, item):
            return getattr(self._peaks, item)
        else:
            raise AttributeError(
                '{} has no {} field'.format(self._ptype(), item)
            )

    def __str__(self):
        s = 'flags:0x{:02X}{:02X} trace_flags:0x{:02X}{:02X}\n'
        s += 'relative_timestamp:{} area:{} length{}\n'
        s += 'peaks:{}'
        return s.format(
            self.flags.data[0], self.flags.data[1],
            self.trace_flags.data[0], self.trace_flags.data[1],
            self._relative_timestamp, self._area, self._length, len(self.peaks)
        )

    def __repr__(self):
        return str(self)

    def _ptype(self):
        if self._trace_flags is None:
            return 'pulse'
        return str(self.trace_flags.type)

    @property
    def flags(self):
        return self._flags

    @property
    def trace_flags(self):
        if self._trace_flags is None:
            raise AttributeError(
                '{} has no trace_flags field'.format(self._ptype())
            )
        return self._trace_flags

    @property
    def area(self):
        if self._area is None:
            raise AttributeError(
                '{} has no area field'.format(self._ptype())
            )
        return self._area

    @property
    def length(self):
        if self._length is None:
            raise AttributeError(
                '{} has no length field'.format(self._ptype())
            )
        return self._length

    @property
    def relative_timestamp(self):
        if self._relative_timestamp is None:
            raise AttributeError(
                '{} has no relative_timestamp field'.format(self._ptype())
            )
        return self._relative_timestamp

    @property
    def time_offset(self):
        if self._time_offset is None:
            raise AttributeError(
                '{} has no time_offset field'.format(self._ptype())
            )
        return self._time_offset

    @property
    def peaks(self):
        if self._peaks is None:
            raise AttributeError(
                '{} has no peaks field'.format(self._ptype())
            )
        return self._peaks

    @property
    def dot_product(self):
        if self._dot_product is None:
            raise AttributeError(
                '{} has no dot_product field'.format(self._ptype())
            )
        return self._dot_product


class PulsePeaks:
    def __init__(self, data, time_offset):
        self._dtype = np.dtype(pulse_rise_fmt)
        _data = data.view(np.dtype(pulse_rise_fmt))
        self._data = _data
        for p in _data:
            p['time'] -= time_offset

    def __iter__(self):
        for p in self._data:
            yield p

    def __getitem__(self, item):
        return self._data[item]

    def __getattr__(self, item):
        if item in self._dtype.names:
            return self._data[:][item]
        else:
            raise AttributeError

    def __len__(self):
        return len(self._data)

    def __str__(self):
        s = ''
        for p in self:
            for n in self._dtype.names:
                s += (n + ':{} ').format(p[n])
            s += '\n'
        return s

    def __repr__(self):
        return str(self)


"""FPGA frame header data type"""
protocol_header_dt = np.dtype(
    [
        ('sequence', 'u2'),
        ('protocol_sequence', 'u2'),
        ('event_size', 'u2'),
        ('type_field', '>u2')
    ]
)


class ProtocolHeader:
    def __init__(self, data):
        self._data = data.view(protocol_header_dt)

    def __getattr__(self, item):
        if item in protocol_header_dt.names:
            return self._data[item][0]
        else:
            raise AttributeError('ProtocolHeader has no {}'.format(item))

    @property
    def payload_type(self):
        return Payload.from_field(self.type_field)

    def __repr__(self):
        return self._data.__repr__()


class Rises:
    def __init__(self, peaks):
        self.peaks = peaks
        # print(peaks.dtype)
        self.dtype = peaks.dtype

    def __getattr__(self, item):
        dtype = self.dtype
        if item in dtype.names:
            if len(self.peaks) == 1:
                return self.peaks[0][item]
            else:
                return self.peaks[item]
        raise AttributeError('object has no {}'.format(item))

    def __repr__(self):
        return '{}'.format(self.peaks)

    @property
    def fields(self):
        return self.dtype.names


class TickFlags:
    @property
    def tick_lost(self):
        return bool(self.data[4] & 0x10)

    @property
    def loss(self):
        return bool(self.data[4] & 0x20)

    @property
    def mux_full(self):
        return bool(self.data[4] & 0x40)


class TraceFlags:

    @property
    def multi_peak(self):
        return (self.data[2] & 0x04) != 0

    @property
    def multi_pulse(self):
        return (self.data[2] & 0x02) != 0

    @property
    def trace_stride(self):
        return self.data[2] & 0x1F

    @property
    def trace_type(self):
        return TraceType((self.data[3] & 0xC0) >> 6)

    @property
    def trace_signal(self):
        return Signal((self.data[3] & 0x30) >> 4)

    @property
    def trace_offset(self):
        return self.data[3] & 0x0F


class Flags:
    @property
    def peak_count(self):
        return (self.data[4] & 0xF0) >> 4

    @property
    def rel2min(self):
        return (self.data[4] & 0x08) >> 3

    @property
    def channel(self):
        return self.data[4] & 0x07

    @property
    def timing_type(self):
        return Timing((self.data[5] & 0xC0) >> 6)

    @property
    def height_type(self):
        return Height((self.data[5] & 0x30) >> 4)


class Events:
    """
    contiguous eventstream region of homogeneous events
    """
    def __init__(self, h, region=None, payload_type=None):
        self.h = h
        self.region = region
        self._payload_type = payload_type

    @property
    def payload_type(self):
        return

    # need to establish homogeneity
    # the return an iterator when not, modify get item
    def __getitem__(self, item):
        print('event _get_item_')
        return self._data.np_data[item]

    def __getattr__(self, item):
        # print('events getattr', item)
        if hasattr(self._data, item):
            return getattr(self._data, item)
        raise AttributeError(
            '{} has no {}'.format(self.__class__.__name__, item)
        )

    def __repr__(self):
        return 'implement me'


class Event:
    def __init__(self, data):
        self.data = data
        # print(data)
        # print(len(data))
        dtype = self._dtype()
        # print(dtype)
        self.dtype = dtype
        self.np_data = data.view(dtype)

    @property
    def new_window(self):
        return bool(self.data[5] & 0x01)

    @property
    def fields(self):
        s = self.dtype.names
        if hasattr(self, 'peaks'):
            s += self.peaks.fields
        return s

    @property
    def max_peaks(self):
        # print(self.data,self.data[5])
        if self.data[5] & 0x02:
            return None
        detection = Detection((self.data[5] & 0x0C) >> 2)
        if detection == Detection.trace:
            mp = (self.data[3] & 0x0F) - 2
            trace_type = TraceType((self.data[3] & 0xC0) >> 6)
            if (
                trace_type == TraceType.dot_product_trace or
                trace_type == TraceType.dot_product
            ):
                mp -= 1
        elif detection == Detection.pulse:
            mp = int(self.data[:2].view('u2')[0]/8 - 2)
        else:
            return None
        return mp

    @property
    def event_type(self):
        if self.data[5] & 0x02:
            return EventType.tick
        detection = Detection((self.data[5] & 0x0C) >> 2)
        if detection == Detection.trace:
            trace_type = TraceType((self.data[3] & 0xC0) >> 6)
            if trace_type == TraceType.average:
                return EventType.average
            elif (
                trace_type == TraceType.dot_product or
                trace_type == TraceType.dot_product_trace
            ):
                return EventType.dot_product
            else:
                return EventType.pulse
        return EventType(detection.value)

    def _pulse_fmt(self):
        max_peaks = self.max_peaks
        # print(max_peaks)
        if (max_peaks > 0) and (max_peaks <= 9):
            return (
                pulse_header_fmt + [('peaks', pulse_rise_fmt, (max_peaks,))]
            )
        else:
            raise AttributeError('peak_count must be between 1 and 8 inclusive')

    def _dtype(self):
        t = self.event_type
        # print(t)
        if t == EventType.peak:
            return np.dtype(rise_fmt)
        elif t == EventType.area:
            return np.dtype(area_fmt)
        elif t == EventType.tick:  # tick
            return np.dtype(tick_fmt)
        elif t == EventType.average:
            return np.dtype(average_fmt)
        elif t == EventType.pulse:
            return np.dtype(self._pulse_fmt())
        else:
            return np.dtype(self._pulse_fmt() + dot_product_fmt)

    def __getattr__(self, item):
        dtype = self.dtype
        if (item == 'peaks') and ('peaks' in dtype.names):
            return Rises(
                self.np_data['peaks'][0][:min(self.peak_count, self.max_peaks)]
            )

        if item == 'time':
            # print('Event time')
            if 'peaks' in dtype.names:
                # print('peak_time')
                # FIXME only uses first peak
                return (
                    # self.np_data['peaks']['peak_time'][:, 0] - self.time_offset
                    self.np_data['peaks']['peak_time'][0] - self.time_offset
                    + self.pulse_time
                )
            else:
                return self.np_data['time']

        if item in dtype.names:
            # if len(self.np_data) == 1:
            #     return self.np_data[0][item]
            # else:
            return self.np_data[item]
        elif 'peaks' in dtype.names:
            peaks = self.np_data['peaks'][0]
            if item in peaks.dtype.names:
                return peaks[item]
        raise AttributeError(
            '{} object has no {}'.format(self.__class__.__name__, item))

    def __repr__(self):
        return '{} {!r}(s)'.format(len(self.np_data), self.event_type)


class Frames:
    """
    An iterable contiguous region of frames
    The protocol headers are array addressable
    """
    def __init__(self, h, region=None, payload_types=None, homogeneous=None):
        self.h = h
        self._payload_types = payload_types
        self._homogeneous = homogeneous
        if region is None:
            self._len = len(h['index/frame'])
            self.region = _region(0, self._len)
            return
        if region[1] < 0:
            region[1] = len(h['index/frame']) - region[1]
        if region[0] < 0:
            region[0] = len(h['index/frame']) - region[0]
        self.region = region
        self._len = region[1] - region[0]

    def __len__(self):
        return self._len

    def _tick_mask(self):
        return np.bitwise_and(self.event_type[:, 1], 0x02) != 0

    def _mca_mask(self):
        return np.bitwise_and(self.event_type[:, 1], 0x40) != 0

    def _event_mask(self):
        return np.logical_and(
            np.logical_not(self._tick_mask()), np.logical_not(self._mca_mask())
        )

    @property
    def payload_types(self):
        if self._payload_types is None:
            self._payload_types = np.zeros((len(self),), 'i1')
            self._payload_types.fill(-1)
            tick = np.bitwise_and(self.event_type[:, 1], 0x02) != 0
            mca = np.bitwise_and(self.event_type[:, 1], 0x40) != 0
            self._payload_types[tick] = Payload.tick.value
            self._payload_types[mca] = Payload.mca.value
            detection = np.right_shift(
                np.bitwise_and(self.event_type[:, 1], 0x0C), 2
            )
            event = np.logical_and(np.logical_not(tick), np.logical_not(mca))
            trace_event = np.logical_and(detection == 3, event)
            non_trace = np.logical_and(detection != 3, event)
            trace_type = np.bitwise_and(self.event_type[:, 0], 0x03)
            single_trace = np.logical_and(trace_type == 0, trace_event)
            other_trace = np.logical_and(trace_type != 0, trace_event)
            self._payload_types[non_trace] = detection[non_trace]
            self._payload_types[single_trace] = Payload.trace.value
            self._payload_types[other_trace] = trace_type[other_trace] + 5
        return self._payload_types

    @property
    def homogeneous(self):
        if self._homogeneous is None:
            event_types = self.payload_types[self._event_mask()]
            hom = len(
                event_types[np.where(event_types != event_types[0])]
            ) == 0
            self._homogeneous = hom
        return self._homogeneous

    def __getattr__(self, item):
        if item in _frame_index_dt.names:
            return (
                self.h['index/frame'][self.region[0]:self.region[1]][item]
            )
        raise AttributeError('no attribute {}'.format(item))

    def __getitem__(self, item):
        if isinstance(item, int):
            if item >= self._len:
                raise IndexError(
                    '{} out of range ({}-{})'.format(item, 0, self._len-1)
                )
            return Frames(self.h,
                _region(self.region[0] + item, self.region[0] + item + 1)
            )

        if isinstance(item, slice):
            start = self.region[0][0] + item.start
            slen = item.stop - item.start
            return Frames(self.h, _region(start, start+slen))

        # if item == 'type':
        #     return self.h['index/type']

    def __iter__(self):
        for i in range(self.region[0][0], self._len):
            yield self[i]

    def __repr__(self):
        if self._len == 1:
            return (
                'event_size:{}, payload_type:{}'
                .format(
                    self.event_size[0], self.event_type[0][0],
                    self.event_type[0][1]
                )
            )
        return '{} frames'.format(self._len)


class Frame:
    def __init__(self, h, i):
        self.h = h
        self.i = i
        self.header = h['data/protocol'][i].view(_protocol_dt)
        pt = Payload(h['index/type'][i])
        self.payload_type = pt
        region = h['index/frame'][i][0]
        if pt < 4 or (5 < pt < 9):  # stored in eventstream
            self.payload = h['data/eventstream'][region[0]:region[1]]
            return
        if pt == 4:  # stored in tickstream
            self.payload = h['data/tickstream'][region[0]:region[1]]
            return
        if pt < 12:  # unidentified
            self.payload = (
                h['data/unidentified'][region[0]:region[1]]
            )
            return

    def __getattr__(self, item):
        if item in _protocol_dt.names:
            return self.header[item]
        else:
            raise AttributeError('Frame has no {}'.format(item))

    def __str__(self):
        # return 'frame'
        return (
            'frame:{} type:{!r} frame_seq:0x{:04X} protocol_seq:0x{:04X} '
            'payload:{} bytes'
        ).format(
            self.i, self.payload_type, self.frame_seq, self.protocol_seq,
            len(self.payload)
        )

    def __repr__(self):
        return str(self)


class Traces:
    """
    handle indexing, iteration etc for traces in the HDF5 file.
    """
    def __init__(self, h, region=None, homogeneous=None):
        self.h = h
        self.index = _DS(h, 'index/trace')
        self.frame_index = _DS(h, 'index/frame')
        self.homogeneous = homogeneous
        if region is None:
            self._len = len(self.index)
            self.region = _region(0, self._len)
            return
        if region[1] < 0:
            region[1] = len(self.index) - region[1]
        if region[0] < 0:
            region[0] = len(self.index) - region[0]
        self.region = region
        self._len = region[1] - region[0]

    def __getitem__(self, item):
        """
        :param item: index currently expects int.
        :return: Trace object
        """
        if isinstance(item, int):  # single trace
            length = len(self.index)
            if item >= length:
                raise IndexError('{} out of range (0-{})'.format(item, length))
            start_frame = self.index[item]['start']
            stop_frame = self.index[item]['stop']-1
            return (
                self.h['data/eventstream']
                [
                    self.frame_index[start_frame['payload']]:
                    self.frame_index[stop_frame['payload']+start_frame['length']]-1
                ]
            )
        elif isinstance(item, slice):
            print(item)
            return item
            # raise NotImplementedError()

    def __len__(self):
        return len(self.h['/index/trace'])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
        raise StopIteration

    def __repr__(self):
        return '{} traces'.format(len(self))


class Trace(TraceFlags, Flags, Event):
    def __init__(self, event, samples):
        super().__init__(event)
        self._samples = samples

    @property
    def samples(self):
        return self._samples

    def __len__(self):
        return len(self._samples)

    def __repr__(self):
        return (
            'trace:{} samples, event type:{!r}'
            .format(len(self._samples), self.event_type)
        )


# add superclass for iterable regions, ticks and frames
# ticks should subclass frames?
class Ticks:
    def __init__(self, h, region=None):
        self.h = h
        if region is None:
            self._len = len(h['index/tick'])
            self.region = _region(0, self._len)
            return
        if region['stop'] < 0:
            region['stop'] = len(h['index/tick']) + region['stop']
        self.region = region
        self._len = region['stop'][0] - region['start'][0]

    def __getitem__(self, item):
        if isinstance(item, int):
            if item >= self._len:
                raise IndexError(
                    '{} out of range ({}-{})'.format(item, 0, self._len-1)
                )
            return Tick(self.h, int(self.region['start'][0] + item))

        if isinstance(item, slice):
            start = self.region[0][0]+item.start
            slen = item.stop-item.start
            return Ticks(self.h, _region(start, start+slen))

    @property
    def homogeneous(self):
        return (
            len(
                np.bitwise_and(
                    self.h['index/tick']['event_type'][:, 1]
                    [self.region[0]:self.region[1]],
                    0x80
                ).nonzero()[0]
            ) == 0
        )

    @property
    def payload_type(self):
        if not self.homogeneous:
            RuntimeError('the tick range is not homogeneous')
        return self.frames[0].payload_type

    @property
    def frames(self):
        first_entry = self.h['index/tick'][self.region[0]]
        last_entry = self.h['index/tick'][self.region[1]-1]
        first_frame = first_entry['first_frame']
        last_frame = last_entry['first_frame'] + last_entry['frames']
        return Frames(self.h, _region(first_frame, last_frame))

    @property
    def events(self):
        if not self.homogeneous:
            raise RuntimeError(
                'events for this tick range are not homogeneous, '
                'extracting events requires iteration over frames'
            )
        # entries = self.h['index/tick'][self.region[0][0]:self.region[0][1]]
        # r = _region(entries['events'][0][0], entries['events'][-1][1])
        dt = self.payload_type.event_dt(size=self.frames[0].header['event_size'])
        return self.h['data/eventstream'][
            self.h['index/tick'][0]['events'][0]:
            self.h['index/tick'][-1]['events'][1]
        ].view(dt)

    def __len__(self):
        return self._len

    def __repr__(self):
        return '{} ticks'.format(len(self))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class Tick(TickFlags, Event):
    def __init__(self, data, h, i, frames, events):
        super().__init__(data)
        self.h = h
        self.i = i
        self.frames = Frames(h, frames)
        # FIXME send Frames object instead of region?
        if events.start == events.stop:
            self.events = None
        else:
            self.events = Events(h, frames, events)
        self._frame_region = frames
        self._event_region = events

    def __repr__(self):
        error_mask = np.bitwise_or(
            np.bitwise_or(self.data[16], self.data[17]), self.data[18]
        )
        if self.events_lost != 0:
            return (
                'tick, events lost:{}, error channels:{:08b}'
                .format(self.events_lost, error_mask)
            )
        else:
            return 'tick, no events lost'


average_header_dt = np.dtype('(8,)u1')
region_dt = np.dtype([('start', 'u8'), ('stop', 'u8')])
# index has two regions one for header part other for data part of frame
# for traces data is the samples header is the pulse or average header
index_dt = np.dtype([('header', region_dt), ('data', region_dt)])
tick_index_dt = np.dtype(
    [
        ('header', region_dt), ('event', region_dt), ('trace', region_dt),
        ('dist', region_dt)
    ]
)


def _make_index(header_region, data_region):
    i = np.zeros((1,), dtype=index_dt)
    i[0][0] = header_region
    i[0][1] = data_region
    return i


def _make_tick_index(header_region, event_region, trace_region, dist_region):
    i = np.zeros((1,), dtype=index_dt)
    i[0][0] = header_region
    i[0][1] = event_region
    i[0][2] = trace_region
    i[0][3] = dist_region
    return i


# return slices for index entry


# need to get all event frames between ticks and all trace pulses
# need an event_type as well as frame_type

class Hdf5:
    def __init__(self, hdf5_filename='TES.hdf5', data=None):
        """ if data is not none import and open, else open
            TODO: proper docstring.
        """
        if data is not None:
            if not isinstance(data, _PacketData):
                data = _PacketData(data)
            _import(data, hdf5_filename)
            del data
        self.h = h5py.File(hdf5_filename, 'r+')
        self.frames = Frames(self.h, slice(0, len(self.h['/index/frame'])))
        self.traces = Traces(self.h)
        self.ticks = Ticks(self.h)
        self.events = Events(self.h)
        self.distributions = Hdf5.Distributions(self)

    def close(self):
        self.h.close()

    class Distributions:
        def __init__(self, hdf5):
            self.h = hdf5.h

        def __getitem__(self, item):
            region = self.h['/index/mca'][item]
            return Distribution(
                self.h['/data/mca'][region[0]:region[1]]
            )
