import logging
from os import stat

import h5py
import numpy as np
from platform import system


from tes.hdf5storage import _Hdf5Structure

import pcapng
from tes.base import (
    Detection, Height, Timing, Payload, pulse_rise_fmt, EventType,
    pulse_header_fmt, rise_fmt, average_fmt,
    area_fmt, tick_fmt, dot_product_fmt,
    TraceType, Signal
)
from tes.mca import Distribution

_logger = logging.getLogger(__name__)

# start=0 end=0 indicates null reference
_region_dt = np.dtype([('start', 'i8'), ('stop', 'i8')])


def _region(start, stop):
    r = np.zeros((1,), _region_dt)[0]
    r['start'] = start
    r['stop'] = stop
    return r


def _isnull(region):
    if region['start'] == 0 and region['stop'] == 0:
        return True
    return False


_frame_index_dt = np.dtype([
    ('payload', 'u8'), ('length', 'u4'), ('event_size', 'u2'),
    ('event_type', '(2,)u1')
])

_tick_index_dt = np.dtype([
    ('first_frame', 'u8'), ('frames', 'u4'), ('event_size', 'u2'),
    ('event_type', '(2,)u1')
])

_frame_ref_dt = np.dtype([
    ('start', 'u8'), ('stop', 'u8')  # stop is the last frame +1
])

_protocol_dt = np.dtype([
    ('frame_seq', 'u2'), ('protocol_seq', 'u2'), ('event_size', 'u2'),
    ('event_type', '(2,)u1')
])

_tick_dt = np.dtype(tick_fmt)


class _DS:
    """
    wrapper for DS in hdf5 file
    """

    def __init__(self, h, name, shape=None, dtype=None, create=False):
        self.h = h
        if create:
            self.ds = h.create_dataset(name, shape, dtype)
        else:
            self.ds = h[name]
        self.i = 0
        self.dtype = self.h[name].dtype

    def append(self, data):
        """
        appeds data to the _DS
        :param data:
        :return: the _Region appended
        """
        if data is None:
            return None

        if isinstance(data, int):
            entry = np.zeros((1,), self.dtype)
            entry[0] = data
            data = entry

        d = data.view(self.dtype)
        start = self.i
        length = d.size
        self.ds[start: start + length] = d
        self.i += length
        return _region(start, self.i)[0]

    def __getitem__(self, item):
        """
        :param item:
        """
        if isinstance(item, np.ndarray):
            if item.dtype == _region_dt:
                return self.ds[item['start']:item['stop']]
        return self.ds[item]

    def __len__(self):
        """
        :return: len(ds)
        """
        return len(self.ds)


class _FrameData:
    """
    Iterator over ethernet frames.

    data is either a structured ndarray with integer fields 'data' and last
    where last < 0 marks the last byte in the packet
    (this type is returned from VHDL simulation).

    Or a file, either .pcapng packet capture or a capture file from
    tes-0mq server v1.
    """

    def __init__(self, data):
        if isinstance(data, np.ndarray):
            if 'data' not in data.dtype.names or 'last' not in data.dtype.names:
                raise AttributeError(
                    'PacketData:ndarray missing required fields'
                )
            else:
                _logger.debug('PacketData: from ndarray')
                self.data = data
                self.type = 'ndarray'
                self.file_info = None
            return

        if isinstance(data, str):
            self.file_info = stat(data)
            file_len = self.file_info.st_size

            mmap = np.memmap(
                data, dtype=np.uint8, mode='r', shape=(file_len,)
            )
            tes0mq_type = np.array_equal(
                mmap[40:40 + 12], [90, 1, 2, 3, 4, 5, 218, 1, 2, 3, 4, 5]
            )
            pcapng_type = np.array_equal(mmap[:4], [0x0A, 0x0D, 0x0D, 0x0A])

            if pcapng_type:
                _logger.info('PacketData:from file {:} (pcapng)'.format(data))
                self.type = 'pcapng'
                self.data = data
                del mmap
                return
            elif tes0mq_type:
                self.type = 'tes-0mq'
                self.data = mmap
            else:
                raise AttributeError('PacketData:unidentifiable data type')

    def __iter__(self):
        if self.type == 'ndarray':
            lasts = np.where(self.data['last'] < 0)[0] + 1
            if not lasts.size:
                return

            # for some unknown reason the first byte is duplicated by xsim
            start = 0
            for last in lasts:
                yield np.array(
                    self.data['data'][start:last], dtype=np.uint8, copy=True
                )
                start = last

        elif self.type == 'pcapng':
            with open(self.data, 'rb') as pcap:
                s = pcapng.FileScanner(pcap)
                for block in s:
                    if isinstance(block, pcapng.blocks.EnhancedPacket):
                        yield np.frombuffer(
                            block.packet_data, dtype=np.uint8
                        )
                return
        else:  # tes-0mq
            p = 40
            data_len = len(self.data)
            #             print(data_len)
            while p < data_len:
                try:
                    frame_len = (self.data[p + 14:p + 16].view('<u2'))[0]
                except:
                    print(p, data_len)
                yield self.data[p:p + frame_len]
                p += frame_len
            return

    def __getitem__(self, item):
        if self.type == 'ndarray':
            lasts = np.where(self.data['last'] < 0)[0] + 1

            if item + 1 >= len(lasts):
                raise IndexError

            start = lasts[item]
            end = lasts[item + 1]

            return self.data[start:end]
        else:
            raise NotImplementedError(
                'Indexing not implemented for {} files'.format(self.type)
            )


class _PacketData:
    """
    Deprecated
    data is either a structured ndarray with integer fields 'data' and last
    where last < 0 marks the last byte in the packet.
    or a file, either .pcapng packet capture or the raw bytestream
    """

    def __init__(self, data):
        if isinstance(data, np.ndarray):
            if 'data' not in data.dtype.names or 'last' not in data.dtype.names:
                raise AttributeError(
                    'PacketData:ndarray missing required fields'
                )
            else:
                _logger.debug('PacketData: from ndarray')
                self.data = data
                self.type = 'ndarray'

        elif isinstance(data, str):
            file_info = stat(data)
            file_len = file_info.st_size

            mmap = np.memmap(data, dtype=np.uint8, mode='r', shape=(file_len,))
            raw_type = np.array_equal(
                mmap[:12], [90, 1, 2, 3, 4, 5, 218, 1, 2, 3, 4, 5]
            )
            pcapng_type = np.array_equal(mmap[:4], [0x0A, 0x0D, 0x0D, 0x0A])
            del mmap

            if raw_type:
                _logger.info('PacketData:from file {:} (raw)'.format(data))
                self.type = 'raw'
            elif pcapng_type:
                _logger.info('PacketData:from file {:} (pcapng)'.format(data))
                self.type = 'pcapng'
            else:
                raise AttributeError('PacketData:unidentifiable data type')
            self.data = data

    def __iter__(self):
        if self.type == 'ndarray':
            lasts = np.where(self.data['last'] < 0)[0] + 1

            if not lasts.size:
                raise StopIteration

            # for some unknown reason the first byte is duplicated by xsim
            start = 0

            for last in lasts:
                yield np.array(
                    self.data['data'][start:last], dtype=np.uint8, copy=True
                )
                start = last

        elif self.type == 'raw':
            raise NotImplementedError
        elif self.type == 'pcapng':
            with open(self.data, 'rb') as pcap:
                s = pcapng.FileScanner(pcap)
                for block in s:
                    if isinstance(block, pcapng.blocks.EnhancedPacket):
                        yield np.frombuffer(block.packet_data, dtype=np.uint8)
                raise StopIteration

    def __getitem__(self, item):
        if self.type == 'ndarray':
            lasts = np.where(self.data['last'] < 0)[0] + 1

            if item + 1 >= len(lasts):
                raise IndexError

            start = lasts[item]
            end = lasts[item + 1]

            return self.data[start:end]
        else:
            raise NotImplementedError


def _mca_handler(frame, frame_i, data_bytes, index_i, log=None):
    head = True
    data_bytes = 0
    index_bytes = 0
    mca_seq = 0
    while True:
        frame_len = frame[14:16].view('<u2')[0]
        payload_len = frame_len - 24
        data_bytes += payload_len
        protocol_seq = frame[18:20].view(np.uint16)[0]
        if head and protocol_seq != 0:
            if log:
                log.warning('Partial mca found in frame:{}')
        else:
            pass


def _import(data, hdf5_filename='TES.hdf5'):
    """
    import PacketData into a structured HDF5 file.
    data is either _PacketData or something that can be handled by _PacketData

    File structure:
    ############################################################################
    /data           main data store.
    ----------------------------------------------------------------------------
                    protocol        frame protocol header array.
                    tickstream      payload byte array for tickstream.
                    eventstream     payload byte array for eventstream.
                    unidentified    payload byte array for unid frames.
    ----------------------------------------------------------------------------
    /index          indices
    ----------------------------------------------------------------------------
                    frame               payload data range, end tick index.

                    type                frame type byte array.

                    sequence_errors     index of frames with seq errors.

                    tick                frame index range since prev tick,
                                        eventstream range since prev tick.

                    tick_flags          bit 0:error 1:not homogeneous

                    trace               eventstream array range,
                                        end tick index.

                    trace_type          frame type byte array.
    """

    _logger.info('import:pass 1 - calculating sizes for hdf5 pre-allocation')
    # byte counts
    eventstream_bytes = 0
    tickstream_bytes = 0
    mcastream_bytes = 0

    frame_i = 0
    tick_i = 0
    trace_i = 0
    mca_i = 0
    unidentified_i = 0

    sequence = None
    sequence_errors = 0

    trace = False
    trace_error = False
    trace_part = False
    trace_done = False
    trace_start = 0
    trace_bytes_remaining = 0
    trace_seq = None

    # pass 1
    for frame in data:
        payload_type = Payload.from_frame(frame)
        frame_len = frame[14:16].view('<u2')[0]
        payload_len = frame_len - 24
        if payload_type == Payload.bad_ethernet:
            # store in unidentified
            _logger.warning(
                "import:Bad ethernet header in frame:{}".format(frame_i)
            )
            eventstream_bytes += frame_len
            frame_i += 1
            unidentified_i += 1
            continue

        seq = frame[16:18].view(np.uint16)[0]
        protocol_seq = frame[18:20].view(np.uint16)[0]
        frame_i += 1

        if sequence is not None:
            if seq != np.uint16(sequence + 1):
                _logger.warning(
                    (
                        'import: Sequence error in frame {}, ' +
                        'got 0x{:04X} expected 0x{:04X}'
                    ).format(frame_i, seq, (sequence + 1))
                )
                sequence_errors += 1
        sequence = seq

        if payload_type == Payload.unknown_ethertype:
            _logger.warning(
                "import:Unknown ethertype for frame:{}".format(frame_i)
            )
            eventstream_bytes += frame_len
            unidentified_i += 1
        elif payload_type == Payload.mca:
            # _logger.warning('mca frame:{} skipped'.format(frame_i))
            continue
        elif (
                payload_type == Payload.trace or
                payload_type == Payload.average_trace or
                payload_type == Payload.dot_product_trace
        ):
            # contains samples
            if protocol_seq == 0:  # header frame
                trace = True
                trace_error = False
                trace_part = False
                trace_start = eventstream_bytes
                trace_seq = 0
                trace_bytes_remaining = frame[24:26].view('u2')[0]

            # print(trace_bytes_remaining, payload_len)
            eventstream_bytes += payload_len
            trace_bytes_remaining -= payload_len
            trace_done = trace_bytes_remaining == 0

            if not trace:  # header frame missing
                if not trace_part:
                    _logger.warning(
                        (
                            'import:partial trace found starting frame ' +
                            '{} with trace sequence:0x{:04X}'
                        ).format(frame_i, protocol_seq)
                    )
                continue
                # trace_part = True  # only warn once

            if trace_bytes_remaining < 0:
                _logger.critical(
                    (
                        'import:trace length error at frame {}, ' +
                        'bytes remaining:{}'
                    ).format(frame_i, trace_bytes_remaining)
                )
                RuntimeError('Trace length error')

            if trace_done:
                if not (trace_error or trace_part):
                    trace_i += 1
                    _logger.debug(
                        'import:trace complete bytes:{}'
                        .format(eventstream_bytes - trace_start)
                    )
                    # else:
                    #     eventstream_bytes = trace_start

            if protocol_seq != np.uint16(trace_seq):
                _logger.warning(
                    (
                        'import:trace sequence error in frame {}, ' +
                        'got 0x{:04X} expected 0x{:04X}').format(
                        frame_i, trace_seq, np.uint16(protocol_seq)
                    )
                )
            trace_seq += 1
                # trace_error = True

        elif payload_type == Payload.tick:
            tick_i += 1
            tickstream_bytes += frame_len - 24
            _logger.debug('import:tick frame:{}'.format(frame_i))
        else:
            # normal detection event types
            eventstream_bytes += payload_len

    _logger.info(
        'import:pass 1 - found {} data bytes in {} frames, sequence errors:{}'
        ', bad frames:{}.'
        .format(
            int(eventstream_bytes + mcastream_bytes + tickstream_bytes),
            frame_i, sequence_errors, unidentified_i
        )
    )
    _logger.info('import:pass 1 - ticks:{}, traces:{}'.format(tick_i, trace_i))
    _logger.info(
        'import:pass 1 - tickstream:{}, eventstream:{}, mcastream:{}'
        .format(tick_i * 24, eventstream_bytes, mcastream_bytes)
    )
    _logger.info('import:pass 2 - importing data')

    # create hdf5 file and import
    # there is a bug in OsX python and open can return OSError
    # (file temp unavail) can't use context manager
    # dirty work around
    if system() == 'Darwin':
        for i in range(10):
            try:
                hdf5 = h5py.File('test.hdf5', 'w')
            except OSError:
                if i == 9:
                    raise
                continue
            break
        else:
            hdf5 = h5py.File('test.hdf5', 'w')
    # with h5py.File(hdf5_filename, 'w') as hdf5:
        # sequence_errors = 0
        # tick_frames = 0
        # tick_events = 0

    frame_index = _DS(
        hdf5, 'index/frame', (frame_i,), _frame_index_dt, create=True
    )
    eventstream = _DS(
        hdf5, 'data/eventstream', (eventstream_bytes,), np.uint8,
        create=True
    )
    tick_index = _DS(
        hdf5, 'index/tick', (tick_i,), _tick_index_dt, create=True
    )
    tickstream = _DS(
        hdf5, 'data/tickstream', (tick_i,), np.dtype(tick_fmt), create=True
    )
    trace_index = _DS(
        hdf5, 'index/trace', (trace_i,), _frame_ref_dt, create=True
    )

    sequence = None
    trace = False

    # tick flags
    # tick_time = 0
    # tick_start_frame = 0
    tick_type = None  # the type of events in this tick
    tick_size = None  # the size of events in this tick
    # the eventstream for this tick is all the same type and size
    new_tick = True
    tick_homogeneous = True
    tick_start_frame = 0

    trace_error = False
    trace_start = 0
    trace_bytes_remaining = 0
    trace_seq = None

    for f in data:
        frame_entry = np.zeros((1,), _frame_index_dt)[0]
        trace_entry = np.zeros((1,), _frame_ref_dt)[0]

        frame_len = f[14:16].view('<u2')[0]
        frame = f[:frame_len]
        payload_len = frame_len - 24
        # frame = np.array(f[:frame_len], copy=True)
        payload_type = Payload.from_frame(frame)
        protocol = frame[16:24].view(_protocol_dt)[0]

        frame_entry['payload'] = eventstream.i
        frame_entry['length'] = payload_len
        frame_entry['event_size'] = protocol['event_size']
        frame_entry['event_type'] = protocol['event_type']

        # if last_frame_event_size is None:
        #     last_frame_event_size = protocol['event_size']
        #     last_frame_event_type = protocol['event_type']

        if sequence is not None:
            if protocol['frame_seq'] != np.uint16(sequence + 1):
                frame_entry[0]['event_type'][1] |= 0x10
                if payload_type != Payload.mca:
                    tick_homogeneous = False

        sequence = protocol['frame_seq']
        # print(protocol)

        if (
            payload_type == Payload.bad_ethernet or
            payload_type == Payload.unknown_ethertype
        ):
            frame_entry['event_type'][1] = 0x20
            frame_entry['event_type'][0] = 0
            frame_entry['length'] = frame_len
            tick_homogeneous = False
            sequence = None
            trace = False
            mca = False
            eventstream.append(frame)
            frame_index.append(frame_entry)
            # unidentified_i += 1
            continue

        if payload_type == Payload.mca:
            # FIXME implement
            continue

        # mca and error frames have been handled
        # time = int(frame[30:32].view(np.uint16)[0])
        if payload_type == Payload.tick:
            frame_entry['payload'] = tickstream.i
            tickstream.append(frame[24:])
        else:  # non-tick event
            eventstream.append(frame[24:])
            # if (
            #     protocol['event_type'].view('u2') !=
            #     last_frame_event_type.view('u2')
            #     or
            #     protocol['event_size'] != last_frame_event_size
            # ):
            #     frame_entry['event_type'][1] |= 0x80  # non homogeneous

            if tick_type is None:
                tick_type = protocol['event_type']
                tick_size = protocol['event_size']
            else:
                if (
                            tick_type.view('u2') !=
                            protocol['event_type'].view('u2')
                ):
                    tick_homogeneous = False
                if tick_size != protocol['event_size']:
                    tick_homogeneous = False

            if new_tick:  # update time
                tick_start_frame = frame_index.i
                #     etime = tick_time + time
                #     if etime > 655365:
                #         etime = 655365
                #     # print(frame[30:32].view(np.uint16)[0], time)
                #     frame[30:32] = np.uint16(etime)
                new_tick = False

        # frame_index.append(frame_entry)
        # mca and error frames have been handled, must be an event frame.

        if (
            payload_type == Payload.trace or
            payload_type == Payload.average_trace or
            payload_type == Payload.dot_product_trace
        ):
            # events that contain sequence records
            # print(protocol['protocol_seq'], trace_seq)
            if protocol['protocol_seq'] == 0:  # header frame
                trace = True
                trace_done = False
                trace_error = False
                trace_start = frame_index.i
                trace_bytes_remaining = (
                    frame[24:26].view('u2')[0] - payload_len
                )
                # print('header', trace_start, trace_bytes_remaining)
                trace_seq = 1
            elif (
                    not trace_error and (
                        trace_seq is None or
                        protocol['protocol_seq'] != np.uint16(trace_seq)
                    )
                 ):
                # print('trace seq error')
                trace_error = True
                tick_homogeneous = False
                trace = False
                trace_seq = None
            else:
                trace_seq = trace_seq + 1
                trace_bytes_remaining -= payload_len
                trace_done = trace_bytes_remaining == 0
                # print('nonheader', trace_seq, trace_bytes_remaining)

            if trace_bytes_remaining < 0:
                raise RuntimeError('Trace length mismatch')
                # trace_error = True
                # tick_error = True
                # trace = False

            if not trace:  # no header yet
                # print('no trace')
                trace_done = False
                trace_error = True
                tick_homogeneous = False
                trace = False

            if trace_done and not trace_error:
                trace_entry['start'] = trace_start
                trace_entry['stop'] = frame_index.i + 1
                # print('trace done', trace_start, frame_index.i)
                trace_index.append(trace_entry)
                trace = False

        elif payload_type == Payload.tick:

            tick_entry = np.zeros((1,), _tick_index_dt)[0]

            if not tick_homogeneous:
                tick_entry['event_type'][1] = 0x80
                tick_entry['event_type'][0] = 0
            else:
                tick_entry['event_size'] = tick_size
                tick_entry['event_type'] = tick_type

            tick_entry['first_frame'] = tick_start_frame
            tick_entry['frames'] = frame_index.i + 1 - tick_start_frame
            tick_index.append(tick_entry)

            tick_start_frame = 0
            new_tick = True

        frame_index.append(frame_entry)

    hdf5.close()
