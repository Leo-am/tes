import logging
import numpy as np
from os import path as ospath
from os import stat
from pickle import Pickler, Unpickler
from collections import namedtuple
from datetime import datetime
from pcapng import FileScanner
import pcapng
from .mca import Distribution, header_dt as mca_header_dt
from .base import (
    Detection, Height, Timing, Payload, pulse_rise_fmt, pulse_header_fmt,
    rise_fmt, area_fmt, tick_fmt, event_fmt, trace_header_fmt
)
import zmq
import h5py

_logger = logging.getLogger(__name__)

DEFAULT_REPO_PATH = 'TES_project/fpga_ise/'
File = namedtuple('File', ['filename', 'dtype', 'is_list', 'is_sliceable'])


class McaStream:
    def __init__(self, stream64):
        self.lasts = np.where(stream64['clk'] < 0)


class EventStream:
    # NOTE copies stream64 to bytestream.
    # Expects stream64 to contain only one PayloadType
    def __init__(self, file, project=None, testbench=None, tool='PlanAhead',
                 repo=DEFAULT_REPO_PATH, root='c:/'):
        if project is not None and testbench is not None:
            path = (
                root + repo + project + '/' + tool + '/' + project + '.sim/' +
                testbench + '/'
            )
        else:
            path = './'
        # print(path + file)
        stream_dt = np.dtype([('data', 'u8'), ('last', 'i4')])
        if ospath.isfile(path + file):
            stream64 = np.fromfile(path + file, stream_dt)
        else:
            print('{:s} does not exist'.format(path + file))
            raise AttributeError('file {:s} not found'.format(path + file))

        self.last = (np.where(stream64['last'] < 0)[0]+1)*8
        self.bytestream = np.copy(stream64['data'].byteswap()).view(np.uint8)
        del stream64

    def __iter__(self):
        for i in range(len(self.last)):
            if i == 0:
                yield self.bytestream[:self.last[i]]
            else:
                yield self.bytestream[self.last[i-1]:self.last[i]]

    def __getitem__(self, i):
        if i == 0:
            return self.bytestream[:self.last[0]]
        else:
            return self.bytestream[self.last[i-1]:self.last[i]]

    @property
    def events(self):
        if self._event_dt is None:
            return None
        else:
            return self.bytestream.view(self._event_dt)

    class Flags:
        def __init__(self, flags):
            self.peak_count = np.right_shift(np.bitwise_and(flags[0], 0xF0), 4)
            self.height_rel2min = np.bitwise_and(flags[0], 0x08) != 0
            self.channel = np.bitwise_and(flags[0], 0x07)
            self.timing = Timing.from_int(
                np.right_shift(np.bitwise_and(flags[1], 0xC0), 6))
            self.height = Height.from_int(
                np.right_shift(np.bitwise_and(flags[1], 0x30), 4))
            self.type = Detection.from_int(
                np.right_shift(np.bitwise_and(flags[1], 0x000C), 2))
            self.tick = np.bitwise_and(flags[1], 0x02) != 0
            self.new_window = np.bitwise_and(flags[1], 0x01) != 0

        def __repr__(self):
            return (
                'peak count:{:d}\nheight_rel2min:{:}\n{:}\n{:}\n' +
                'channel:{:d}\n{:}\ntick:{:}\nnew_window:{:}'
                .format(
                    self.peak_count, self.height_rel2min, self.height,
                    self.timing, self.channel, self.type, self.tick,
                    self.new_window
                )
            )


class Data:
    def __init__(self, fileset, channels, project, testbench, tool='PlanAhead',
                 repo=DEFAULT_REPO_PATH):
        self._fileset = fileset
        self.channels = channels
        self.project = project
        self.testbench = testbench
        self.tool = tool
        self.creation_date = datetime.now()

        for file in fileset:
            fileinfo = fileset[file]
            if fileinfo.is_list:
                data = []
                for c in range(0, channels):
                    # print('{:s}{:d}'.format(simfile.filename, c))
                    data.append(
                        self.fromfile(
                            '{:s}{:d}'.format(fileinfo.filename, c),
                            fileinfo.dtype, project, testbench, tool, repo
                        )
                    )
            else:
                data = self.fromfile(fileinfo.filename, fileinfo.dtype, project,
                                     testbench, tool, repo)

            setattr(self, file, data)

    @staticmethod
    def fromfile(file, dt, project=None, testbench=None, tool='PlanAhead',
                 repo=DEFAULT_REPO_PATH, root='c:/'):
        if project is not None and testbench is not None:
            path = (
                root + repo + project + '/' + tool + '/' + project + '.sim/' +
                testbench + '/'
            )
        else:
            path = './'
        # print(path + file)
        if ospath.isfile(path + file):
            return np.fromfile(path + file, dt)
        else:
            print('{:s} does not exist'.format(path + file))
            return None

    # def _parse_settings(self):
    #     registers = dict()
    #     if not hasattr(self, 'settings') and not hasattr(self, 'mca_settings'):
    #         return None
    #
    #     if hasattr(self, 'settings'):
    #         registers['baseline'] = dict()
    #         registers['baseline']['offset'] = self.settings[0]
    #         registers['baseline']['subtraction'] = self.settings[1] != 0
    #         registers['baseline']['time_constant'] = self.settings[2]
    #         registers['baseline']['threshold'] = self.settings[3]
    #         registers['baseline']['count_threshold'] = self.settings[4]
    #         registers['generic'] = dict()
    #         registers['baseline']['average_order'] = self.settings[5]
    #         registers['capture'] = dict()
    #         registers['capture']['cfd_relative'] = self.settings[6] != 0
    #         registers['capture']['constant_fraction'] = self.settings[7]
    #         registers['capture']['pulse_threshold'] = self.settings[8]
    #         registers['capture']['slope_threshold'] = self.settings[9]
    #         registers['capture']['pulse_area_threshold'] = self.settings[10]
    #         registers['capture']['height_type'] = HeightType.from_int(self.settings[11])
    #         registers['capture']['threshold_rel2min'] = self.settings[12] != 0
    #         registers['capture']['trigger_type'] = TriggerType.from_int(self.settings[13])
    #         registers['capture']['event_type'] = PayloadType.from_int(self.settings[14])
    #         registers['capture']['height_rel2min'] = self.settings[15] != 0
    #
    #     if hasattr(self, 'mca_settings'):
    #         registers['mca'] = self._parse_mcasettings()
    #
    #     return registers

    # def _parse_mcasettings(self):
    #     if hasattr(self, 'mca_settings'):
    #         mca_registers = dict()
    #         mca_registers['channel'] = self.mca_settings[0]
    #         mca_registers['bin_n'] = self.mca_settings[1]
    #         mca_registers['last_bin'] = self.mca_settings[2]
    #         mca_registers['lowest_value'] = self.mca_settings[3]
    #         mca_registers['value'] = McaValueType.from_int(self.mca_settings[4])
    #         mca_registers['trigger'] = McaTriggerType.from_int(self.mca_settings[5])
    #         mca_registers['ticks'] = self.mca_settings[6]
    #         return mca_registers
    #     else:
    #         return None

    def save(self, filename=None):
        if filename is None:
            filename = '{:s}-{:}.pickle'.format(self.testbench,
                                                datetime.now().date())
        fp = open(filename, 'wb')
        pickler = Pickler(fp, protocol=-1)
        pickler.dump(self)

    @staticmethod
    def load(filename):
        fp = open(filename, 'rb')
        unpickler = Unpickler(fp)
        return unpickler.load()

    # time slice returns nested class Slice
    def slice(self, bounds):
        return self.Slice(self, bounds)

    def region(self, point, pre, length):
        return self.slice((point - pre, point - pre + length))

    # quick and dirty proxy to handle slices
    class Slice:

        def _slice(self, attr, array):
            if array is None:
                return None

            if 'index' in self._data._fileset[attr].dtype.fields:
                sliced = array[
                         np.searchsorted(array['index'], self.bounds[0]):
                         np.searchsorted(array['index'], self.bounds[1])
                         ]
            else:
                sliced = array[self.bounds[0]:self.bounds[1]]
            return sliced

        def _apply_bounds(self, attr):

            data = getattr(self._data, attr)  # get array from Data instance

            if self._bounds == 'all' or not self._data._fileset[
                attr].is_sliceable:
                return data
            else:
                if self._data._fileset[attr].is_list:
                    sliced = []
                    for array in data:
                        sliced.append(self._slice(attr, array))
                else:
                    sliced = self._slice(attr, data)

                self._sliced[attr] = sliced
                return sliced

        def __init__(self, data, bounds='all'):
            self._bounds = bounds  # slice bounds -- (a,b) yields a numpy [a:b] type slice
            self._data = data  # points to the Data instance
            self._sliced = dict()  # arrays that are already sliced

        def __getattr__(self, attr):

            if attr not in self._data._fileset:
                return getattr(self._data, attr)

            if attr in self._sliced:  # already sliced
                return self._sliced[attr]

            return self._apply_bounds(attr)

        @property
        def bounds(self):
            return self._bounds

        @bounds.setter
        def bounds(self, value):
            self._bounds = value
            # rest stored slices
            self._sliced = dict()


class Packet:
    def __init__(self, stream):
        self.bytes = stream
        self.ethertype = stream[12:14].view(np.uint16).byteswap()[0]
        self.length = stream[14:16].view(np.uint16)[0]
        self.payload = stream[24:]

        # print(self.bytes)
        if self.ethertype == 0x88B5:
            if np.bitwise_and(self.bytes[23], 0x02):
                self.event_type = Detection.tick
            else:
                self.event_type = Detection(
                    np.right_shift(np.bitwise_and(self.bytes[23], 0x0C), 2))
        elif self.ethertype == 0x88B6:
            self.event_type = None
        else:
            print('Unknown ethertype:{:X}'.format(self.ethertype))
            self.event_type = None

        self.frame_sequence = self.bytes[16:18].view(np.uint16)[0]
        self.protocol_sequence = self.bytes[18:20].view(np.uint16)[0]
        self.event_size = self.bytes[20:22].view(np.uint16)[0]

    def __repr__(self):
        if self.event_type is None:
            if self.ethertype == 0x88B6:
                pname = 'MCA'
            else:
                pname = 'UNKNOWN'
        else:
            pname = str(self.event_type)
        return 'ethertype:{:04X} length:{:d} Payload:{:s} frame:{:d} ' \
               'protocol:{:d}' \
            .format(
                self.ethertype, self.length, pname, self.frame_sequence,
                self.protocol_sequence
            )

    @property
    def events(self):
        if self.event_type == Payload.tick:
            return self.payload.view(np.dtype(tick_fmt))
        elif self.event_type == Payload.peak:
            return self.payload.view(np.dtype(rise_fmt))
        elif self.event_type == Payload.area:
            return self.payload.view(np.dtype(area_fmt))
        elif self.event_type == Payload.pulse:
            # FIXME check this
            fmt = pulse_header_fmt + [('peaks', (np.dtype(pulse_rise_fmt), 1))]
            # print(fmt)
            return self.payload.view(np.dtype(fmt))
        #FIXME change to trace
        elif self.event_type == Payload.trace:
            return self.payload.view(np.dtype(trace_header_fmt))
        else:
            return None


class PacketStream:
    # stream can be a filename of a pcapng file
    # otherwise a np array from simulation data
    # NOTE copies stream['data'] to bytestream
    def __init__(self, stream):

        self.packets = []
        self._distributions = None
        self._traces = None
        self._events = None
        self._eventstream = None

        if isinstance(stream, str):
            with open(stream, 'rb') as pcap:
                s = FileScanner(pcap)
                for block in s:
                    if isinstance(block, pcapng.blocks.EnhancedPacket):
                        packet_data = np.frombuffer(
                            block.packet_data, dtype=np.uint8
                        )
                        packet_length = packet_data[14:16].view(np.uint16)[0]
                        self.packets.append(Packet(packet_data[:packet_length]))
        else:
            lasts = np.where(stream['last'] < 0)[0] + 1

            if not lasts.size:
                self.bytes = None
                self.packets = None
                return

            self.bytes = np.uint8(stream['data'])
            start = 0

            for last in lasts:
                self.packets.append(Packet(self.bytes[start:last]))
                start = last

    def __iter__(self):
        for i in range(len(self.packets)):
            yield self.packets[i]

    def __getitem__(self, i):
        if i not in range(len(self.packets)):
            raise IndexError
        else:
            return self.packets[i]

    def __repr__(self):
        return repr(self.packets)

    @property
    def distributions(self):
        if self._distributions is None:
            last_seq = None
            buffer = None
            self._distributions = []
            if self.packets is None:
                return None
            for packet in self.packets:
                if packet.ethertype == 0x88B6:
                    # print(packet)
                    if last_seq is None:
                        if packet.protocol_sequence == 0:  # header frame
                            header = np.frombuffer(
                                packet.payload[:40], mca_header_dt
                            )[0]
                            # print(header)
                            numcounts = (header['last_bin']+1)
                            buffer = np.zeros((numcounts*4+40,), np.uint8)
                            pointer = 0
                        else:
                            continue
                    else:
                        if packet.protocol_sequence != last_seq + 1:
                            print('MCA sequence number:{:d} missing'.format(
                                last_seq + 1))
                            last_seq = None
                            continue

                    start = pointer
                    pointer += len(packet.bytes)-24
                    last_seq = packet.protocol_sequence
                    # print(start, pointer, len(buffer))
                    buffer[start:pointer] = packet.bytes[24:]
                    if pointer == len(buffer):
                        self._distributions.append(Distribution(buffer))
                        last_seq = None

        return self._distributions

    @property
    def traces(self):
        if self._traces is None:
            last_seq = None
            buffer = None
            self._traces = []
            if self.packets is None:
                return None
            for packet in self.packets:
                if packet.ethertype == 0x88B5 and \
                                packet.event_type == Payload.trace:
                    # print(packet)
                    if last_seq is None:
                        if packet.protocol_sequence == 0:  # header frame
                            header = np.frombuffer(
                                packet.payload[:16],
                                np.dtype(trace_header_fmt)
                            )[0]
                            print(header)
                            buffer = np.zeros((header['size']*8,), np.uint8)
                            pointer = 0
                        else:
                            continue
                    else:
                        if packet.protocol_sequence != last_seq + 1:
                            print(
                                'Trace event sequence number:{:d} missing'
                                .format( last_seq + 1)
                            )
                            last_seq = None
                            continue

                    start = pointer
                    pointer += len(packet.payload)
                    last_seq = packet.protocol_sequence
                    print(start, pointer, len(buffer))
                    buffer[start:pointer] = packet.payload
                    if pointer == len(buffer):
                        self._traces.append(Trace(buffer))
                        last_seq = None

        return self._traces

    @property
    def events(self):
        if self._events is None:
            self._events = []
            for p in self.packets:
                if p.ethertype == 0x88B5 and p.event_type != Payload.trace:
                    self._events.append((p.event_type, p.events))
        return self._events

    @property
    def eventstream(self):
        if self._eventstream is None:
            self._eventstream = []
            timestamp = 0
            event_list = self.events
            for etupple in event_list:
                event_type = etupple[0]
                events = etupple[1]
                #print(etupple)
                if event_type == Detection.tick:
                    for tick in events:
                        if timestamp != -1:
                            if tick['time'] != 0xFFFF:
                                timestamp += tick['time']
                                if timestamp != tick['timestamp']:
                                    print('timestamp sync error')
                            else:
                                timestamp = tick['timestamp']
                        else:
                            if tick['time'] != 0xFFFFF:
                                print('timestamp sync error')
                            timestamp = tick['timestamp']
                        self._eventstream.append((timestamp, event_type, tick))
                else:
                    for event in events:
                        # print('not tick', event_type)
                        if timestamp != -1:
                            if event['time'] == 0xFFFF:
                                timestamp = -1
                            else:
                                timestamp += event['time']
                        self._eventstream.append((timestamp, event_type, event))
        return self._eventstream


class Trace:
    def __init__(self, buffer):

        self.dtype = event_fmt(
            buffer[5:6][0], buffer[:2].view(np.uint16)[0],
            np.bitwise_and(buffer[3:4], 0x0F)[0]
        )
        self._view = buffer.view(self.dtype)[0]


class EventFlags:

    def __init__(self, flags):
        self.flags = flags
        self._tick = np.bitwise_and(flags, 0x0002) != 0

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if self.tick:
            return (
                'Event:tick lost:{:} new window:{:}'
                .format(self.tick_lost, self.new_window)
            )
        else:
            return (
                'Event:{:} channel:{:} peak_number:{:} cfd_rel2min:{:}\n'
                .format(
                    str(self.type), self.channel, self.peak_number,
                    self.cfd_rel2min
                ) +
                'height:{:} timing:{:} new window:{:}'
                .format(
                    str(self.height), str(self.timing), str(self.new_window)
                )
            )

    @property
    def new_window(self):
        return np.bitwise_and(self.flags, 0x0001) != 0

    @property
    def tick(self):
        return self._tick

    @property
    def tick_lost(self):
        if not self._tick:
            return None
        return np.bitwise_and(self.flags, 0x0010) != 0

    @property
    def type(self):
        if self._tick:
            return Detection.tick
        else:
            return Detection(
                np.right_shift(np.bitwise_and(self.flags, 0x000C), 2)
            )

    @property
    def height(self):
        if self._tick:
            return None
        return Height(
            np.right_shift(np.bitwise_and(self.flags, 0x0030), 4)
        )

    @property
    def timing(self):
        if self._tick:
            return None
        return Timing(
            np.right_shift(np.bitwise_and(self.flags, 0x00C0), 6)
        )

    @property
    def channel(self):
        if self._tick:
            return None
        return np.right_shift(np.bitwise_and(self.flags, 0x0700), 8)

    @property
    def cfd_rel2min(self):
        if self._tick:
            return None
        return np.bitwise_and(self.flags, 0x0800) != 0

    @property
    def peak_number(self):
        if self._tick:
            return None
        return np.right_shift(np.bitwise_and(self.flags, 0xF000), 12)


class Client:
    def __init__(
            self,
            server_ip='tcp://smp-qtlab-tes.staff.science.uq.edu.au',
            event_port=23000,
            mca_port=23001
    ):
        context = zmq.Context()
        self.event = context.socket(zmq.REQ)
        self.mca = context.socket(zmq.SUB)
        print('connecting to server at {:}'.format(server_ip))
        self.event.connect('{:}:{:}'.format(server_ip, event_port))
        self.mca.connect('{:}:{:}'.format(server_ip, mca_port))

    def mca_frame(self):
        self.mca.subscribe('')
        print('subscribed')
        d = Distribution(self.mca.recv())
        print('recieved')
        self.mca.unsubscribe('')
        return d

    def capture(self, ticks):
        self.event.send_multipart(
            [bytes('{:}'.format(ticks), 'utf-8'), b'test.raw']
        )
        print(self.event.recv_multipart())


# convert a raw capture file or pcapng file to a HDF% file containg only events
# and perform some post proccessing
def _to_hdf5(infile, hdf5_file):
    file_info = stat(infile)
    file_len = file_info.st_size

    mmap = np.memmap(infile, dtype=np.uint8, mode='r', shape=(file_len,))
    raw = np.array_equal(mmap[:12], [90, 1, 2, 3, 4, 5, 218, 1, 2, 3, 4, 5])
    if not raw:
        _logger.info(
            'to_hdf5:assuming {:} is a .pcapng capture file'.format(infile)
        )
        del mmap
    else:
        _logger.info(
            'to_hdf5:{:} detected as a native capture file'.format(infile)
        )

    with h5py.File(hdf5_file, "w") as hdf5:

        packets = hdf5.create_group('packets')
        _logger.info(
            'to_hdf5:pass 1 - getting sizes for hdf5 pre-allocation'
        )
        # get sizes to pre-allocate
        byte_count = 0
        packet_count = 0

        if not raw:  # assume pcapng
            with open(infile, 'rb') as pcap:
                s = pcapng.FileScanner(pcap)
                for block in s:
                    if isinstance(block, pcapng.blocks.EnhancedPacket):
                        packet_data = np.frombuffer(
                            block.packet_data, dtype=np.uint8
                        )
                        ether_type = packet_data[12:14].view('>u2')[0]

                        if ether_type == 0x88b5:
                            byte_count += (
                                packet_data[14:16].view(np.uint16)[0] - 24
                            )
                            packet_count += 1
        else:
            # raw file captured by python client
            # will only be event packets ethertype=0x88b5
            pointer = 0
            while pointer < file_len:
                if not np.array_equal(
                        mmap[pointer:pointer + 14],
                        [90, 1, 2, 3, 4, 5, 218, 1, 2, 3, 4, 5, 0x88, 0xB5]
                ):
                    raise RuntimeError('corrupt native capture file')
                plen = (mmap[pointer + 14:pointer + 16].view(np.uint16)[0])
                byte_count += (plen - 24)
                packet_count += 1
                pointer = pointer + plen

        packets.attrs['count'] = packet_count
        packets.attrs['bytes'] = byte_count
        _logger.info(
            'to_hdf5:found {:} bytes in {:} frames'
            .format(byte_count, packet_count)
        )
        # allocate hdf5 data sets
        _logger.info('to_hdf5:pass 2 - importing to hdf5')
        byte_chunk = 64 * 1024
        packet_chunk = 16 * 1024

        # if byte_count >= byte_chunk:
        #     packets.create_dataset('bytestream', (byte_count,), dtype=np.uint8,
        #                            chunks=(64 * 1024,))
        # else:
        packets.create_dataset('bytestream', (byte_count,), dtype=np.uint8)

        # if packet_count >= packet_chunk:
        #     packets.create_dataset('sequence', (packet_count,), dtype=np.uint16,
        #                            chunks=(16 * 1024,))
        #     packets.create_dataset('event_type', (packet_count,),
        #                            dtype=np.uint8, chunks=(16 * 1024,))
        #     packets.create_dataset('range', (packet_count, 2), dtype=np.uint64,
        #                            chunks=(16 * 1024, 2))
        # else:
        packets.create_dataset('sequence', (packet_count,), dtype=np.uint16)
        packets.create_dataset('fsequence', (packet_count,), dtype=np.uint16)
        packets.create_dataset('event_type', (packet_count,), dtype=np.uint8)
        packets.create_dataset('event_size', (packet_count,), dtype=np.uint8)
        packets.create_dataset('range', (packet_count, 2), dtype=np.uint64)

        bs = packets['bytestream']
        seq = packets['sequence']
        fseq = packets['fsequence']
        etype = packets['event_type']
        esize = packets['event_size']
        prange = packets['range']  # packet[i] = bs[range[i,0]:range[i,1]]

        # import packets into HDF5
        start = 0
        packet = 0
        if not raw:  # assume pcapng
            with open(infile, 'rb') as pcap:
                s = pcapng.FileScanner(pcap)
                last_seq = None
                seq_errors = 0
                frame_number = 0
                for block in s:
                    if isinstance(block, pcapng.blocks.EnhancedPacket):
                        frame_number += 1
                        packet_data = np.frombuffer(
                            block.packet_data, dtype=np.uint8
                        )
                        ether_type = packet_data[12:14].view('>u2')[0]
                        if ether_type == 0x88b5:
                            plen = packet_data[14:16].view(np.uint16)[0]
                            end = start + plen - 24
                            bs[start:end] = packet_data[24:plen]
                            prange[packet] = [start, end]
                            seq[packet] = packet_data[18:20].view(np.uint16)[0]
                            this_seq = packet_data[16:18].view(np.uint16)[0]
                            fseq[packet] = this_seq
                            if last_seq is not None:
                                if np.uint16(last_seq+1) != this_seq:
                                    seq_errors += 1
                                    _logger.debug(
                                        ('sequence error frame:{:d} expected ' +
                                         '0x{:04X} got 0x{:04X}')
                                        .format(
                                            frame_number, np.uint16(last_seq+1),
                                            this_seq
                                        )
                                    )
                            last_seq = this_seq

                            etype[packet] = packet_data[23:24]
                            esize[packet] = packet_data[20:21]
                            packet += 1
                            start = end
                        elif ether_type != 0x88b6:
                            _logger.warning(
                                'to_hdf5:Unknown ethertype {:04X} ' +
                                'for packet {:}'.format(ether_type, packet)
                            )
                packets['sequence'].attrs['errors'] = seq_errors
                _logger.info('to_hdf5:{:} sequence errors'.format(seq_errors))

        else:
            pointer = 0
            while pointer < file_len:
                plen = mmap[pointer + 14:pointer + 16].view(np.uint16)[0]
                end = start + plen - 24
                bs[start:end] = mmap[pointer + 24:pointer + plen]
                prange[packet] = [start, end]
                seq[packet] = mmap[pointer + 18:pointer + 20].view(np.uint16)[0]
                etype[packet] = mmap[pointer + 23:pointer + 24]
                esize[packet] = mmap[pointer + 20:pointer + 21]
                start = end
                packet += 1
                pointer += plen
            del mmap

        _logger.info('to_hdf5:post processing {:}'.format(hdf5_file))

        # index ticks
        pr = prange

        et = etype[...]
        tick_mask = np.tile(0x02, (etype.len(),))
        tick_packets = np.bitwise_and(et, tick_mask).nonzero()[0]
        tick_grp = hdf5.create_group('tick')
        tick_grp['packets'] = tick_packets

        tick_dt = np.dtype(tick_fmt)
        tick_count = 0
        for tp in tick_packets:
            ticks = bs[pr[tp, 0]:pr[tp, 1]].view(tick_dt)
            tick_count += len(ticks)

        tick_grp.attrs['count'] = tick_count
        _logger.info('to:hdf5:{:} ticks found'.format(tick_count))

        # TODO make this a region ref?
        tick_grp.create_dataset('range', (tick_count, 2),
                                dtype=np.uint64)  # bytestream range for this tick
        tick_grp.create_dataset('event_packets', (tick_count, 2),
                                dtype=np.uint64)  # packet indexes
        tick_grp.create_dataset('events_lost', (tick_count,), dtype=np.uint32)
        tick_grp.create_dataset('homogeneous', (tick_count,), dtype=np.bool)

        trange = tick_grp['range']
        tlost = tick_grp['events_lost']
        tevents = tick_grp['event_packets']

        total_lost = 0
        homogeneous = True
        pindex = 0
        last_packet = tick_packets[pindex]
        tindex = 0
        while tindex < tick_count:
            packet = tick_packets[pindex]
            pstart = pr[packet, 0]
            pend = pr[packet, 1]
            tick_packet = bs[pstart:pend]
            ticks = tick_packet.view(tick_dt)
            offset = 0
            tick_len = 24

            for tick in ticks:
                trange[tindex] = [pstart + offset, pstart + offset + tick_len]
                tlost[tindex] = tick['events_lost']
                total_lost += tick['events_lost']
                bounds = [last_packet + 1, packet-1]
                if bounds[1] >= bounds[0]:
                    tevents[tindex] = bounds
                    _logger.debug(
                        'to_hdf5:tick index {:} event packet bounds {:}'
                        .format(tindex, bounds)
                    )
                    event_types = etype[bounds[0]:bounds[1]]
                    if len(event_types) > 1:
                        h = np.array_equal(
                            event_types,
                            np.tile(event_types[0], len(event_types))
                        )
                        if not h:
                            homogeneous = False
                        tick_grp['homogeneous'][tindex] = h
                    else:
                        tick_grp['homogeneous'][tindex] = True

                else:
                    tevents[tindex] = [packet, packet]
                    # empty (no events) is homogeneous
                    tick_grp['homogeneous'][tindex] = True

                offset += tick_len
                tindex += 1

            last_packet = packet
            pindex += 1

        total_lost = sum(tlost)
        _logger.info('to_hdf5:{:} events lost'.format(total_lost))
        tlost.attrs['total'] = total_lost
        tick_grp['homogeneous'].attrs['all'] = homogeneous
        if homogeneous:
            _logger.info('to_hdf5:The event stream is homogeneous')
        else:
            _logger.info('to_hdf5:The event stream is inhomogeneous')


class Hdf5:
    def __init__(self, hdf5file, infile=None):
        if infile:
            _to_hdf5(infile, hdf5file)
        self.h = h5py.File(hdf5file, 'r')
        self.bs = self.h['packets/bytestream']
        self.pr = self.h['packets/range']
        self.ep = self.h['tick/event_packets']
        self.tp = self.h['tick/packets']
        self.et = self.h['packets/event_type']
        self.es = self.h['packets/event_size']
        self.fseq = self.h['packets/fsequence']
        self.seq = self.h['packets/sequence']
        self.th = self.h['tick/homogeneous']
        self._trace_ranges = None

    def __repr__(self):
        return (
            'ticks:{:} events lost:{:} sequence errors:{:} homogeneous:{:}'
            .format(
                len(self), self.events_lost, self.sequence_errors,
                self.homogeneous
            )
        )

    def __len__(self):
        return self.h['tick'].attrs['count']

    def __getitem__(self, item):
        if item < 0 or item >= len(self):
            raise IndexError
        return self._Tick(item, self)

    def __iter__(self):
        for i in range(1, len(self)):
            yield self._Tick(i, self)

    @property
    def homogeneous(self):
        return self.h['tick/homogeneous'].attrs['all']

    @property
    def events_lost(self):
        return self.h['tick/events_lost'].attrs['total']

    @property
    def sequence_errors(self):
        return self.h['packets/sequence'].attrs['errors']

    @property
    def frame_count(self):
        return len(self.pr)

    @property
    def frames(self):
        return self._Frames(self)

    def close(self):
        if self.h:
            self.h.close()

    class _Frames:
        def __init__(self, hdf5):
            self.h = hdf5

        def __iter__(self):
            for i in range(len(self.h.pr)):
                yield Hdf5.Frame(self.h, i)

        def __getitem__(self, item):
            if item < 0 or item >= len(self.h.pr):
                raise IndexError
            return Hdf5.Frame(self.h, item)

    class _Packet:
        def __init__(self, i, tick):
            self.t = tick
            self.index = i
            prange = self.t._p_range  # range for event packets
            h = self.t.h
            start = h.pr[prange[0]+i][0]
            stop = h.pr[prange[0]+i][1]
            self.bytestream = h.bs[int(start):int(stop)+1]

    class _Tick:
        """Tick event from FPGA"""

        def __init__(self, i, hdf5):
            self.h = hdf5
            self.index = i
            self._p_range = self.h.ep[self.index]
            p_i = self.h.tp[i]
            dt = np.dtype(tick_fmt)
            data = self.h.bs[self.h.pr[p_i, 0]:self.h.pr[p_i, 1]].view(dt)[0]
            for n in dt.names:
                self.__setattr__(n, data[n])

        def __len__(self):
            return int(self._p_range[1])-int(self._p_range[0])+1

        @property
        def bytestream(self):
            return self.h.bs[
                     self.h.pr[self._p_range[0]][0]:
                     self.h.pr[self._p_range[-1]][-1]
                   ]

        @property
        def event_types(self):
            if self._p_range[0] == self._p_range[1]:
                return None
            return self.h.et[int(self._p_range[0]):int(self._p_range[1])+1]

        @property
        def events(self):
            """returns a numpy structured array of detection events since the
            previous tick.
            """

            if not self.h.th[self.index]:
                raise NotImplementedError(
                    'Tick is not homogeneous -- contains multiple event types')

            p_range = self._p_range
            if p_range[0] == p_range[1]:
                return None
            etype = self.h.et[p_range[0]]
            esize = self.h.es[p_range[0]]
            self.dtype = np.dtype(event_fmt(etype, esize))
            start = self.h.pr[self.h.ep[self.index, 0]][0]
            end = self.h.pr[self.h.ep[self.index, 1]][1]
            return self.h.bs[start:end].view(self.dtype)

        @property
        def channel(self):
            return np.bitwise_and(
                self.events['flags0'], np.tile(0x3, len(self.events))
            )

        @property
        def homogeneous(self):
            return self.h.th[self.index]

    class Frame:
        def __init__(self, hdf5, i):
            self.i = i
            self.h = hdf5

        @property
        def payload_type(self):
            if np.bitwise_and(0x02, self.h.et[self.i]):
                return Payload.tick
            return Payload(
                np.right_shift(np.bitwise_and(0x0C, self.h.et[self.i]), 2)
            )

        def __len__(self):
            l = self.h.pr[self.i][1]-self.h.pr[self.i][0]
            return l

        @property
        def sequence(self):
            return self.h.fseq[self.i]

        @property
        def protocol_sequence(self):
            return self.h.seq[self.i]

        @property
        def bytestream(self):
            start = self.h.pr[self.i][0]
            stop = self.h.pr[self.i][1]
            return self.h.bs[int(start):int(stop)]

        def __repr__(self):
            return 'payload:{!r} length:{} frame:{} protocol:{}'.format(
                self.payload_type, len(self), self.sequence,
                self.protocol_sequence
            )

        @property
        def events(self):
            if self.payload_type == Payload.tick:
                return self.bytestream.view(np.dtype(tick_fmt))
            elif self.payload_type == Payload.peak:
                return self.bytestream.view(np.dtype(rise_fmt))
            elif self.payload_type == Payload.area:
                return self.bytestream.view(np.dtype(area_fmt))
            elif self.payload_type == Payload.pulse:
                # FIXME check this
                fmt = pulse_header_fmt + [
                    ('peaks', (np.dtype(pulse_rise_fmt), 1))
                ]
                # print(fmt)
                return self.bytestream.view(np.dtype(fmt))
            else:
                return None

    class _Traces:
        def __init__(self, hdf5):
            self.h = hdf5
            if hdf5._trace_ranges is None:
                pass

        def __iter__(self):
            for i in range(len(self.h.pr)):
                yield Hdf5.Frame(self.h, i)

        def __getitem__(self, item):
            if item < 0 or item >= len(self.h.pr):
                raise IndexError
            return Hdf5.Frame(self.h, item)
