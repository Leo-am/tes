import serial
import serial.tools.list_ports
from tes.base import Height, Timing, Detection, lookup
# from collections import namedtuple
import logging
import numpy as np
from functools import partial
import zmq
import h5py

from . import mca

_logger = logging.getLogger(__name__)

__serial__ = None


def __serial_open__(port):
    global __serial__
    if port is not None:
        if __serial__ is None:
            _logger.info('opening serial port on {:}'.format(port))
            __serial__ = serial.serial_for_url(port, baudrate=115200, timeout=2)
    else:
        _logger.info('Opening dummy serial port')


def __serial_close__():
    global __serial__
    if __serial__ is not None:
        _logger.info('Closing serial port')
        __serial__.close()
        __serial__ = None


def __serial_read__(address):
    __serial__.write(
        b'00000000' + bytes('{:08X}02\n'.format(address), encoding='utf8'))
    _logger.debug('reading address:{:08X}'.format(address))
    return _get_response()


def __serial_write__(data, address):
    __serial__.write(
        bytes('{:08X}{:08X}01\n'.format(data, address), encoding='utf8'))
    _logger.debug('writing {:08X} to {:08X}'.format(data, address))
    return _get_response()


def _get_response(socket):
    # l = __serial__.readline()
    l = socket.recv()
    if len(l) == 0:
        raise serial.SerialTimeoutException
    _logger.debug('response {:}'.format(l[:-1]))
    resp = int(chr(l[-3]), 16)
    if resp != 0:
        axi_bits = resp & 3
        err = (resp & 0xC) >> 2
        non_hex = (err & 2) != 0
        bad_length = (err & 1) != 0
        if axi_bits == 3:
            axi = 'DECERR'
        elif axi_bits == 2:
            axi = 'SLVERR'
        elif axi_bits == 0:
            axi = 'OKAY'
        else:
            axi = 'UNKNOWN'
        raise SerialRegisterError(non_hex, bad_length, axi)
    else:
        if len(l[:-3]) == 0:
            return
        data = np.frombuffer(
            bytearray.fromhex(l[:-3].decode('utf8')), np.uint32
        ).byteswap()[0]
        return data


class SerialRegisterError(Exception):
    def __init__(self, non_hex=False, bad_length=False, axi='OKAY'):
        self.non_hex = non_hex
        self.bad_length = bad_length
        self.axi = axi

    def __str__(self):
        return (
            'register access error - non_hex:{} bad_length:{} AXI:{}\n'.format(
                self.non_hex, self.bad_length, self.axi) +
            'AXI:DECERR indicates a non-existent address\n' +
            'AXI:SLVERR typically indicates an access violation' +
            'eg. writing a read-only address'
        )


class RegisterError(AttributeError):
    pass


# The reg_info name tuple describes how to read and write the register.
# bit field is a tuple (mask, shift)
# when reading the mask is bitwise anded with the register contents and the
# result is shifted right by shift, with the reverse process on writing.
# Address is the 32 bit register address.
# strobe is a boolean indicating a strobe register. FIXME why is this needed?
# output_transform is a function called on read, after mask and shift, or None.
# input_transform is a function called on write, before mask and shift, or None.
# reg_info = namedtuple('reg_info', ['address', 'bit_field', 'strobe',
#                                    'output_transform', 'input_transform'])

# TODO make field a named tuple

class RegInfo:
    def __init__(
        self, address, field, strobe, output_transform, input_transform
    ):
        self.address = address
        self.field = field  # (mask, shift)
        self.strobe = strobe
        self.output_transform = output_transform
        self.input_transform = input_transform


# transforms
def _cpu_version(data):
    y = 2016 + ((data & 0xF0000000) >> 30)
    m = (data & 0x0F000000) >> 24
    d = (data & 0x00FF0000) >> 16
    h = (data & 0x0000FF00) >> 8
    mi = data & 0x000000FF
    return '{:04d}-{:02d}-{:02d} {:02d}:{:02d}'.format(y, m, d, h, mi)


def _from_onehot(value):
    b = np.log2(value)
    # print(value, b, np.modf(b))

    if np.modf(b)[0] != 0:
        raise ArithmeticError('{:} is not one-hot'.format(b))
    return int(b)


def _to_onehot(value, bits=8):
    v = int(value)
    if v > bits-1:
        raise AttributeError('value must be <= {:}'.format(bits-1))
    return 2 ** v


def _to_cf(cf):
    if (cf < 0) or (cf > 1):
        raise AttributeError('Constant fraction must be between 0 and 0.5')
    cfi = int(cf*2**17)
    if abs(cf-(np.float(cfi)/2**17)) > abs(cf-(np.float(cfi+1)/2**17)):
        return cfi+1
    return np.uint32(cfi)


def _from_cf(value):
    return np.float(value)/2**17

event_lookup = partial(lookup, enum=Detection)
height_lookup = partial(lookup, enum=Height)
timing_lookup = partial(lookup, enum=Timing)
value_lookup = partial(lookup, enum=mca.Value)
trigger_lookup = partial(lookup, enum=mca.Trigger)
qualifier_lookup = partial(lookup, enum=mca.Qualifier)

# maps
mca_map = {
    'lowest_value': RegInfo(0x10000004, (), False, np.int32, None),
    'ticks': RegInfo(0x10000008, (), False, None, None),
    'value': RegInfo(
        0x10000002, (0x0000000F, 0), False, mca.Value, value_lookup
    ),
    'trigger': RegInfo(
        0x10000002, (0x000000F0, 4), False, mca.Trigger, trigger_lookup
    ),
    'channel': RegInfo(0x10000002, (0x00000700, 8), False, np.uint8, None),
    'bin_n': RegInfo(0x10000002, (0x0000F800, 11), False, np.uint8, None),
    'last_bin': RegInfo(0x10000002, (0xFFFF0000, 16), False, np.uint16, None),
    'update': RegInfo(0x10002000, (0x00000002, 0), True, None, None),
    'update_on_completion': RegInfo(
        0x10002000, (0x00000001, 0), True, None, None),
    'qualifier': RegInfo(
        0x10000800, (0x0000000F, 0), False, mca.Qualifier, qualifier_lookup
    )
}

global_map = {
    'cpu_version': RegInfo(0x10000000, (), False, _cpu_version, None),
    'hdl_version': RegInfo(0x10000001, (), False,
                            lambda x: 'sha-1 {:07X}'.format(x), None),
    'channels': RegInfo(0x10800000, (0x000000FF, 0), False, np.uint8, None),
    'adc_chips': RegInfo(0x10800000, (0x0000FF00, 8), False, np.uint8, None),
    'fmc': RegInfo(0x10800000, (0x00010000, 16), False, bool, None),
    'fmc_power': RegInfo(0x10800000, (0x00020000, 17), False, bool, None),
    'ad9510_status': RegInfo(0x10800000, (0x00040000, 18), False, bool, None),
    'mmcm_locked': RegInfo(0x10800000, (0x00080000, 19), False, bool, None),
    'iodelay_ready': RegInfo(0x10800000, (0x00100000, 20), False, bool, None),
    'tick_period': RegInfo(0x10000020, (), False, np.uint32, None),
    'tick_latency': RegInfo(0x10000040, (), False, np.uint32, None),
    'adc_enable': RegInfo(0x10000080, (0x000000FF, 0), False, np.uint8, None),
    'channel_enable': RegInfo(
        0x10000100, (0x000000FF, 0), False, np.uint8, None
    ),
    'window': RegInfo(0x10000400, (), False, np.uint16, None),
    'fmc_internal_clk': RegInfo(
        0x10000200, (0x00000001, 0), False, bool, None
    ),
    'vco_power_en': RegInfo(0x10000200, (0x00000002, 1), False, bool, None),
    'mtu': RegInfo(0x10000010, (), False, np.uint16, None)
}

baseline_map = {
    'offset': RegInfo(0x00000040, (), False, np.int16, None),
    'time_constant': RegInfo(0x00000080, (), False, np.uint32, None),
    'threshold': RegInfo(0x00000100, (), False, np.uint16, None),
    'count_threshold': RegInfo(0x00000200, (), False, np.uint32, None),
    'new_only': RegInfo(0x00000400, (0x00000001, 0), False, bool, None),
    'subtraction': RegInfo(0x00000400, (0x00000002, 1), False, bool, None)
}

cfd_map = {
    'fraction': RegInfo(0x00000008, (0x0001FFFF, 0), False, _from_cf, _to_cf),
    'rel2min': RegInfo(0x00000008, (0x80000000, 31), False, np.bool, None),
}

event_map = {
    'packet': RegInfo(0x00000001, (0x00000003, 0), False, Detection, event_lookup),
    'timing': RegInfo(
        0x00000001, (0x0000000C, 2), False, Timing, timing_lookup
    ),
    'max_peaks': RegInfo(0x00000001, (0x000000F0, 4), False, np.uint8, None),
    'height': RegInfo(
        0x00000001, (0x00000300, 8), False, Height, height_lookup
    ),
    'pulse_threshold': RegInfo(0x00000002, (), False, np.uint32, None),
    'slope_threshold': RegInfo(0x00000004, (), False, np.int32, None),
    'area_threshold': RegInfo(0x00000010, (), False, np.uint32, None),
}

channel_map = {
    'version': RegInfo(0x00000000, (), False, _cpu_version, None),
    'adc_select': RegInfo(
        0x00000800, (0x000000FF, 0), False, _from_onehot, _to_onehot
    ),
    'invert': RegInfo(0x00000800, (0x00000100, 8), False, bool, None),
    'delay': RegInfo(0x00000020, (), False, np.uint16, None),
    'event_enable': RegInfo(0x10000100, (0x000000FF, 0), False, np.uint8, None)
}

adc_map = {
    'enable': RegInfo(0x10000080, (0x000000FF, 0), False, bool, None),
    'pattern': RegInfo(0x20000062, (), False, np.uint8, None),
    'pattern_low': RegInfo(0x20000051, (), False, np.uint8, None),
    'pattern_high': RegInfo(0x20000052, (), False, np.uint8, None),
    'gain': RegInfo(0x20000055, (0x000000F0, 4), False, np.uint8, None)
}


# FIXME make this the adc address transform
def _adc_spi_transform(address, field, channel):
    # map spi to ADCs
    if ((address & 0xFF000000) >> 24) == 0x20:
        spi_address = np.uint32(address & 0x00000FF)
        if channel % 2:  # B channel
            spi_address += 0x13
        mosi_mask = np.uint32(pow(2, channel // 2) << 8)
        spi = 0x20000000 | mosi_mask | spi_address
        _logger.debug(
            'spi_address:{:02X} mosi:{:02X} spi:{:08X}'
            .format(spi_address, mosi_mask, spi)
        )
        return spi, ()

    return address, (_to_onehot(channel), channel)


def _channel_address_transform(address, field, channel):
    return (address & 0xF0FFFFFF) | (channel << 24), field


class _RegBlock(object):
    def __init__(self, reg_map, read, write, address_transform=None):
        object.__setattr__(self, 'reg_map', reg_map)
        object.__setattr__(self, 'read', read)
        object.__setattr__(self, 'write', write)
        object.__setattr__(self, 'address_transform', address_transform)

    def __getattribute__(self, name):
        reg_map = object.__getattribute__(self, 'reg_map')
        read = object.__getattribute__(self, 'read')
        write = object.__getattribute__(self, 'write')
        transform = object.__getattribute__(self, 'address_transform')

        _logger.debug('get {:}'.format(name))
        if name in reg_map:
            field = reg_map[name].field
            address = reg_map[name].address
            if transform is not None:
                address, field = transform(address, field)

            # strobe must have a non empty bit_field tuple
            # FIXME this should generate an exception
            if reg_map[name].strobe:
                try:
                    write(field[0], address)
                except Exception:
                    logging.exception(
                        'Exception while attempting to write strobe at {:08X}'
                        .format(address)
                    )
                    raise
                return

            try:
                data = read(address)
            except Exception:
                logging.exception(
                    'Exception while attempting to read {:08X}'.format(address)
                )
                raise

            _logger.debug('Read {:08X} from {:08X}'.format(data, address))
            if len(field):
                data = np.right_shift(
                    np.bitwise_and(data, field[0]), field[1]
                )
                _logger.debug(
                    'mask:{:08X} shift:{:} result:{:}'
                    .format(field[0], field[1], data)
                )
            if reg_map[name].output_transform is not None:
                return reg_map[name].output_transform(data)

            return data
        else:
            # raise AttributeError('no register')
            return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        reg_map = object.__getattribute__(self, 'reg_map')
        read = object.__getattribute__(self, 'read')
        write = object.__getattribute__(self, 'write')
        transform = object.__getattribute__(self, 'address_transform')
        _logger.debug('set {:} to {:}'.format(name, value))
        if name in reg_map:
            field = reg_map[name].field
            address = reg_map[name].address
            input_transform = reg_map[name].input_transform

            if transform is not None:
                address, field = transform(address, field)

            # strobe must have a non empty bit_field tuple
            if reg_map[name].strobe:
                try:
                    write(field[0], address)
                except Exception:
                    _logger.exception(
                        'Exception writing strobe at {:08X}'.format(address)
                    )
                return

            if len(field):
                try:
                    data = read(address)
                except Exception:
                    _logger.exception(
                        'Exception reading fields at {:08X}'.format(address)
                    )
                    raise
                if input_transform is None:
                    new_value = (data & ~field[0]) | \
                                ((int(value) << field[1]) & field[0])
                else:
                    new_value = (
                        (data & ~field[0]) |
                        ((int(input_transform(value)) << field[1]) & field[0])
                    )

                _logger.debug(
                    'Address:{:08x} (field mask:{:08X} shift:{:})'
                    .format(address, field[0], field[1])
                )
            else:
                new_value = np.uint32(int(value))

            # print('new_value {:08X}'.format(new_value))

            try:
                written = write(new_value, address)
            except Exception:
                _logger.exception(
                    'Exception writing to {:08X}'.format(address)
                )
                raise

            if written is not None:
                _logger.debug(
                    'response from write to {:08X} is {:08X}'
                    .format(address, written)
                )
            else:
                _logger.debug(
                    'no response writing to {:08X}'.format(address)
                )
        else:
            # raise AttributeError('there is no {:} register'.format(name))
            object.__setattr__(self, name, value)


def _read(address, socket=None):
    _logger.debug('reading address:{:08X}'.format(address))
    socket.send(
        b'00000000' + bytes('{:08X}02\n'.format(address), encoding='utf8'))
    return _get_response(socket)


def _write(data, address, socket=None):
    _logger.debug('writing {:08X} to {:08X}'.format(data, address))
    socket.send(
        bytes('{:08X}{:08X}01\n'.format(data, address), encoding='utf8')
    )
    return _get_response(socket)


def _flush(socket=None):
    socket.send(b'flush')


class Registers:
    def __new__(cls, ip):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        _logger.info('connecting to server at {:}'.format(ip))
        socket.connect(ip)

        inst = _RegBlock(
            global_map,
            partial(_read, socket=socket),
            partial(_write, socket=socket)
        )

        inst.flush = partial(_flush, socket=socket)

        inst.mca = _RegBlock(
            mca_map,
            partial(_read, socket=socket),
            partial(_write, socket=socket)
        )
        inst.channel = []
        inst.adc = []
        dsp_channels = inst.channels
        adc_channels = inst.adc_chips*2

        _logger.info(
            'Opening register connection to FPGA - HDL version:{:}, '
            'Main cpu software version {:}, '
            'DSP channels:{:}, ADC channels:{:}'
            .format(
                inst.hdl_version, inst.cpu_version, dsp_channels, adc_channels
            )
        )
        # validation
        # input_map['select'].input_transform = \
        #     partial(_to_onehot, bits=adc_channels)

        for c in range(dsp_channels):
            address_transform = partial(_channel_address_transform, channel=c)
            channel = _RegBlock(
                channel_map,
                partial(_read, socket=socket),
                partial(_write, socket=socket),
                address_transform
            )
            channel.cfd = _RegBlock(
                cfd_map,
                partial(_read, socket=socket),
                partial(_write, socket=socket),
                address_transform
            )
            channel.event = _RegBlock(
                event_map,
                partial(_read, socket=socket),
                partial(_write, socket=socket),
                address_transform
            )
            channel.baseline = _RegBlock(
                baseline_map,
                partial(_read, socket=socket),
                partial(_write, socket=socket),
                address_transform
            )
            inst.channel.append(channel)

        for c in range(adc_channels):
            address_transform = partial(_adc_spi_transform, channel=c)
            adc = _RegBlock(
                adc_map,
                partial(_read, socket=socket),
                partial(_write, socket=socket),
                address_transform
            )
            inst.adc.append(adc)

        return inst
