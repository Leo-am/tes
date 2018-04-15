import logging
from collections import namedtuple
import zmq
import numpy as np
from numpy import logical_and as and_l, logical_or as or_l

from .mca import Distribution
import os
from tes.base import (
    fidx_dt, tidx_dt, tick_dt, pulse_fmt, payload_ref_dt, Payload,
    rise_fmt, area_fmt, av_trace_fmt, dot_product_fmt, pulse_rise_fmt
)


_logger = logging.getLogger(__name__)


class FpgaStats(
    namedtuple(
        'FpgaOutputStats', 'frames dropped bad tick mca trace event types'
    )
):
    """
    Class holding FPGA ethernet output statistics
    """
    __slots__ = ()

    def __repr__(self):
        s = (
                'Ethernet frames:{} - dropped:{}, invalid:{}\n'
                .format(self.frames, self.dropped, self.bad) +
                'Ticks:{}, '.format(self.tick) +
                'MCA headers:{}, '.format(self.mca) +
                'Trace headers:{}\n'.format(self.trace) +
                'Events:{}'.format(self.event)
        )
        if self.event != 0:
            s += ', {!r}'.format([
                Payload(n) for n in range(8)
                if self.types & (1 << n)
            ])
        return s


def fpga_stats(time=1.0):
    """
    Diagnostic statistics for the FPGA ethernet output.

    :param float time: time to capture statistics for
    :return: FpgaStats object that subclasses namedtuple.
    """

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.connect('tcp://smp-qtlab11.instrument.net.uq.edu.au:55554')
    sock.send_multipart(
        [
            bytes('{}'.format(time), encoding='utf8'),
        ]
    )
    r = sock.recv_multipart()
    if int(r[0]) == 1:
        print(r)
        raise RuntimeError(
            'Malformed request'
        )

    return FpgaStats(
        frames=int(r[1]), dropped=int(r[2]), bad=int(r[3]), tick=int(r[4]),
        mca=int(r[5]), trace=int(r[6]), event=int(r[7]), types=int(r[8])
    )


class CaptureResult(namedtuple(
    'CaptureResult', 'ticks events traces mca frames dropped invalid'
)):
    __slots__ = ()

    def __repr__(self):
        s = (
                'Frames captured:{} - dropped:{}, invalid:{}\n'
                .format(self.frames, self.dropped, self.invalid) +
                'Ticks:{}, '.format(self.ticks) +
                'MCA histograms:{}, '.format(self.mca) +
                'Traces:{}, '.format(self.traces) +
                'Events:{}'.format(self.events)
        )
        return s


def capture(
        filename, measurement, ticks=10, events=0, write_mode=1,
        conversion_mode=0, capture_mode=1
):
    """
    Capture FPGA output as a collection of data and index files.

    :param filename:
    :param measurement:
    :param ticks:
    :param events:
    :param write_mode:
    :param conversion_mode:
    :param capture_mode:
    :return:
    """

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.connect('tcp://smp-qtlab11.instrument.net.uq.edu.au:55555')

    sock.send_multipart(
        [
            bytes(filename, encoding='utf8'),
            bytes(measurement, encoding='utf8'),
            bytes('{}'.format(ticks), encoding='utf8'),
            bytes('{}'.format(events), encoding='utf8'),
            bytes('{}'.format(write_mode), encoding='utf8'),
            bytes('{}'.format(conversion_mode), encoding='utf8'),
            bytes('{}'.format(capture_mode), encoding='utf8')
        ]
    )

    r = sock.recv_multipart()
    if int(int(r[0])) != 0:
        if int(r[0]) == 1:
            msg = 'Malformed request'
        elif int(r[0]) == 2:
            if ticks == 0 and events == 0:
                msg = 'file error: status requested and file does not exist'
            else:
                msg = 'file error: file exists and overwiting not enabled'
        elif int(r[0]) == 3:
            msg = 'bad path to file'
        elif int(r[0]) == 4:
            msg = 'initialisation error durining capture or conversion'
        elif int(r[0]) == 5:
            msg = 'error while writing file'
        elif int(r[0]) == 6:
            msg = 'error while converting to hdf5'
        elif int(r[0]) == 7:
            msg = 'error writing stats or deleting files after conversion'
        else:
            msg = 'unknown error'
        raise RuntimeError(msg)

    if int(r[6]) != 0:
        print('WARNING {} frames dropped'.format(int(r[6])))

    return CaptureResult(
        ticks=int(r[1]), events=int(r[2]), traces=int(r[3]), mca=int(r[4]),
        frames=int(r[5]), dropped=int(r[6]), invalid=int(r[7])
    )


def read_mca(n):
    """
    Capture MCA histograms

    :param int n: number of histograms to capture
    :return: List of tes.mca.Distributions.
    """
    context = zmq.Context.instance()
    sock = context.socket(zmq.SUB)
    sock.connect('tcp://smp-qtlab11.instrument.net.uq.edu.au:55565')
    sock.subscribe('')
    dists = []
    for i in range(n):
        dists.append(Distribution(sock.recv()))
    return dists


def av_trace(timeout=30):
    """
    Capture an average trace.

    :param timeout: Timeout value in seconds.
    :return:
    """

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.connect('tcp://smp-qtlab11.instrument.net.uq.edu.au:55556')
    sock.send_multipart(
        [
            bytes('{}'.format(timeout), encoding='utf8'),
        ]
    )
    r = sock.recv_multipart()
    if int(r[0]) == 2:
        raise RuntimeError(
            'Time out while waiting for average trace.'
        )
    if int(r[0]) == 1:
        raise RuntimeError(
            'Invalid request.'
        )

    return r[1]


def _memmap_data(path, file, dtype=np.uint8):
    if os.stat(path + file).st_size == 0:
        return None
    else:
        return np.memmap(path + file, dtype=dtype)


class CaptureData:

    def __init__(self, path):

        # memmap the files
        self.edat = _memmap_data(path, 'edat', dtype=np.uint8)
        self.fidx = _memmap_data(path, 'fidx', dtype=fidx_dt)
        self.tidx = _memmap_data(path, 'tidx', dtype=tidx_dt)
        self.tdat = _memmap_data(path, 'tdat', dtype=tick_dt)
        self.midx = _memmap_data(path, 'midx', dtype=payload_ref_dt)
        self.mdat = _memmap_data(path, 'mdat', dtype=np.uint8)
        self.ridx = _memmap_data(path, 'ridx', dtype=payload_ref_dt)
        self.bdat = _memmap_data(path, 'bdat', dtype=np.uint8)

        self._frame_payload_types = self.fidx['type'] & 0x0F
        self.event_types = set(
            self._frame_payload_types[self._frame_payload_types < 7]
        )

        self._sequence_error_mask = self.fidx['type'] & 0x80 != 0
        self.sequence_errors = len(self._sequence_error_mask.nonzero()[0])

        self.has_frame_errors = (
            len(
                (self._frame_payload_types == Payload.bad_frame).nonzero()[0]
            ) != 0 or self._sequence_error_mask[0] != 0
        )
        self.has_traces = self.ridx is not None and len(self.ridx) != 0

        # mask for fidx indicating frames carrying multiple events
        pf_mask = (
            or_l(self._frame_payload_types < 3, self._frame_payload_types == 5)
        )
        pf = self.fidx[pf_mask]
        pf_lengths = pf['length']//(pf['event_size']*8)
        event_count = np.sum(pf_lengths, 0)
        if self.has_traces:
            event_count += np.uint64(len(self.ridx))
        self.event_count = event_count

        # find max_rises
        pulse_mask = self.fidx['type'] == Payload.pulse
        pulse_sizes = set(self.fidx['event_size'][pulse_mask])
        max_rises = 0
        for s in pulse_sizes:
            max_rises = max(max_rises, s-2)

        if self.has_traces:
            trace_sizes = set(self.ridx['length'])
            for s in trace_sizes:
                i = np.where(self.ridx['length'] == s)[0]
                max_rises = max(
                    max_rises,
                    self._trace_specs_from_payload(
                        self.ridx[i[0]]['start']
                    )[0]-2  # offset-2
                )
        # print('max_rises', max_rises)
        # return

        # homogeneous data sets are generated when each channel produces events
        # of the same type and size, in this case edat is a contiguous array.
        # Otherwise the transport frames need to be iterated over to extract the
        # fields.
        event_fields = {}
        tick_idx = np.zeros(len(self.tidx), np.uint64)
        if len(self.fidx['changed'].nonzero()[0]) == 0:
            # The event stream is homogeneous, edat has a single dtype
            event_frame_idxs = np.where(self._frame_payload_types < 7)[0]
            self._homogeneous = self._dtype_from_frame(
                event_frame_idxs[0], full=True
            )  # The single dtype for edat
            data = self.edat.view(self._homogeneous)
            # record fields to the event_fields dict
            for field in self._homogeneous.names:
                if field == 'time':
                    event_fields[field] = np.array(data[field], copy=True)
                else:
                    event_fields[field] = data[field]

            # create tick_idx that contains the indices in edat, viewed as the
            # homogeneous type, that immediately follow tick events.
            for t in range(0, len(self.tidx)):
                f_start = self.tidx[t]['start']  # first frame in tick
                f_last = self.tidx[t]['stop']    # last frame in tick
                p_start = self.fidx[f_start]['payload']  # start payload range
                p_stop = p_start+self.fidx[f_last]['length']
                if self.has_traces:  # homogeneous, must be all traces
                    # indices of traces in this payload range
                    # fixme this won't work when there are not traces between ticks
                    i = np.where(and_l(self.ridx['start'] >= p_start,
                                       self.ridx['start'] < p_stop))[0]
                    if t < len(tick_idx)-1:
                        tick_idx[t] = i[-1]+1
                else:
                    tick_idx[t] = p_start//self._homogeneous.itemsize

        else:
            self._homogeneous = None

            # extract common fields from frames to form contiguous arrays
            # extract rises as contiguous array with max_rises per event
            # TODO generalise to more than two channels and optimise performance
            print('Non homogeneous capture, collating data from frames')
            type_fields = [self._event_fields(p) for p in self.event_types]
            common_fields = set(type_fields[0].keys())
            for field in type_fields[:-1]:
                common_fields = common_fields & set(field.keys())

            # allocate contiguous arrays for collating the fields
            for field in common_fields:
                if field == 'rise':
                    event_fields[field] = np.zeros(
                        (event_count, max_rises), np.dtype(pulse_rise_fmt)
                    )
                else:
                    for t in type_fields:
                        if field in t:
                            event_fields[field] = np.zeros(
                                event_count, t[field][0]
                            )
                            break

            # trace_mask True means idx points to ridx else fidx
            trace_mask = np.zeros(self.event_count, np.bool)
            idx = np.zeros(self.event_count, np.uint64)
            t_i = 0
            e_i = 0
            r_i = 0
            # adjust time field to account for tick events
            for f in range(len(self.fidx)):
                f_type = self.fidx[f]['type'] & 0x0F
                if f_type == Payload.tick:
                    tick_idx[t_i] = e_i if e_i < event_count else event_count
                    t_i += 1
                elif f_type in [0, 1, 2, 5]:
                    # frame carrying multiple events
                    f_data = self._frame_event_data(f)
                    stop = e_i+len(f_data)
                    for field in event_fields:
                        event_fields[field][e_i:stop] = f_data[field]
                    idx[e_i:stop] = f
                    e_i += len(f_data)
                elif f_type in [3, 4, 6] and self.fidx[f]['type'] & 0x40:
                    # trace_header
                    if self.ridx[r_i][0] == self.fidx[f][0]:  # good trace
                        f_data = self._frame_event_data(f)
                        for field in event_fields:
                            event_fields[field][e_i] = f_data[field]
                        trace_mask[e_i] = True
                        idx[e_i] = r_i
                        e_i += 1
                        r_i += 1

            self._idx = idx
            self._trace_mask = trace_mask

        event_fields['time'][0] = 2**16-1
        # print(tick_idx[-10:])
        # print(len(tick_idx))
        # print(len(self.tidx))
        for t in range(1, len(self.tidx)):
            time0 = min(
                int(self.tdat['time'][t])+event_fields['time'][tick_idx[t]],
                2**16-1
            )
            event_fields['time'][tick_idx[t]] = time0

        self._fields = event_fields
        self._tick_idx = tick_idx

    def _is_event_field(self, field):
        if '_fields' in self.__dict__.keys():
            return field in self.__dict__['_fields']
        else:
            return False

    def __getattr__(self, item):
        if self._is_event_field(item):
            return self.__dict__['_fields'][item]
        else:
            raise AttributeError('no attribute or field {}'.format(item))

    def __setattr__(self, key, value):
        if self._is_event_field(key):
            raise AttributeError('event fields are not writable')
        else:
            super().__setattr__(key, value)

    def __repr__(self):
        return 'some data'

    def _frame_type(self, frame):
        """

        :param int frame: index
        :return: Payload, sequence_error, is_header
        """
        return (
            Payload(self._frame_payload_types[frame]),
            self.fidx[frame]['type'] & 0x80 != 0,
            self.fidx[frame]['type'] & 0x40 != 0
        )

    def _trace_specs_from_payload(self, payload):
        eflags = self.edat[
                    payload + np.uint64(5):payload + np.uint64(6)
                 ][0] & 0x0F

        if (eflags & 0x0C) >> 2 != 3:
            raise RuntimeError(
                'payload does not represent a trace carrying samples'
            )

        trace_type = (eflags & 0xC0) >> 6
        if trace_type == 2:
            raise RuntimeError(
                'payload does not represent a trace carrying samples'
            )

        offset = self.edat[
                    payload + np.uint64(3):payload + np.uint64(4)
                 ][0] & 0x0F
        byte_length = self.edat[payload:payload+np.uint64(2)].view(np.uint16)[0]

        return offset, byte_length, Payload(trace_type+3)

    @staticmethod
    def _trace_header_fmt(offset, length, payload_type):
        sample_fmt = [('samples', '({},)i2'.format((length-offset*8)//2))]
        if payload_type == Payload.single_trace:
            return pulse_fmt(offset-2)+sample_fmt
        if payload_type == Payload.average_trace:
            return av_trace_fmt+sample_fmt
        if payload_type == Payload.dot_product_trace:
            return pulse_fmt(offset-2)+dot_product_fmt+sample_fmt

    def _trace_specs_from_frame(self, frame, full=True):
        payload = self.fidx[frame]['payload']
        offset, full_length, payload_type = (
            self._trace_specs_from_payload(payload)
        )

        if full:
            return offset, full_length, payload_type
        else:
            return offset, self.fidx[frame]['length'], payload_type

    def _trace_dtype_from_payload(self, payload):
        offset, length, trace_payload = (
            self._trace_specs_from_payload(payload)
        )
        return np.dtype(
            self._trace_header_fmt(offset, length, trace_payload)
        )

    def _trace_dtype_from_frame(self, frame, full=True):
        """
        The compound numpy.dtype for the trace in frame.

        :param int frame: index of frame
        :param bool full: return dtype for full trace else just the current
                          frame.
        :return: numpy.dtype
        :raises: AttributeError when Full is True and frame is not a header.
        """

        if self.fidx[frame]['type'] & 0x40 != 0:  # header frame
            offset, length, trace_payload = (
                self._trace_specs_from_frame(frame, full=full)
            )

            return np.dtype(
                self._trace_header_fmt(offset, length, trace_payload)
            )
        else:
            if full:
                raise AttributeError(
                    'Frame at index:{} is not a header'.format(frame)
                )
            return np.dtype(np.int16)

    def _trace(self, ridx):
        payload = self.ridx[ridx][0]
        offset, length, trace_payload = self._trace_specs_from_payload(payload)
        dtype = self._trace_header_fmt(offset, length, trace_payload)
        # print(dtype.itemsize)
        return self.edat[payload:payload+self.ridx[ridx][1]].view(dtype)[0]

    def _dtype_from_frame(self, frame, full=False):
        p = Payload(self.fidx[frame]['type'] & 0x0F)
        if p in [3, 4, 6]:
            return self._trace_dtype_from_frame(frame, full=full)
        if p == Payload.rise:
            return np.dtype(rise_fmt)
        if p == Payload.area:
            return np.dtype(area_fmt)
        if p == Payload.pulse:
            return np.dtype(pulse_fmt(self.fidx[frame]['event_size']-2))
        if p == Payload.dot_product:
            return np.dtype(
                pulse_fmt(self.fidx[frame]['event_size']-2)+dot_product_fmt
            )

        raise NotImplementedError(
            'payload type {} not yet implemented'.format(p)
        )

    def _frame_event_data(self, frame):
        """
        frame return frame payload with correct view.

        :param frame: frame index.
        :return: ndarray with appropriate view.
        """

        f_type = self.fidx[frame]['type'] & 0x0F
        # print(frame, Payload(f_type))
        if f_type > 7:
            raise NotImplementedError(
                'Not implemented for {!r} frames.', Payload(f_type)
            )

        start = self.fidx[frame]['payload']
        stop = self.fidx[frame]['payload']+self.fidx[frame]['length']

        if f_type == Payload.tick:
            return self.tdat.view(np.uint8)[start:stop].view(tick_dt)

        return (
            self.edat[start:stop].view(
                self._dtype_from_frame(frame, full=False)
            )
        )

    @property
    def homogeneous(self):
        return self._homogeneous is not None

    @staticmethod
    def _event_fields(payload):
        if payload == Payload.rise:
            return np.dtype(rise_fmt).fields
        if payload == Payload.area:
            return np.dtype(area_fmt).fields
        if payload == Payload.pulse:
            return np.dtype(pulse_fmt(1)).fields
        if payload == Payload.single_trace:
            return np.dtype(pulse_fmt(1)).fields
        if payload == Payload.average_trace:
            return np.dtype(av_trace_fmt).fields
        if payload == Payload.average_trace:
            return np.dtype(av_trace_fmt).fields
        if payload == Payload.dot_product:
            return np.dtype(pulse_fmt(1)+dot_product_fmt).fields
        if payload == Payload.dot_product:
            return np.dtype(pulse_fmt(1)+dot_product_fmt).fields
        raise NotImplementedError(
            'Not implemented for {}'.format(Payload(payload))
        )

