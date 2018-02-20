import serial
import serial.tools.list_ports
from tes.base import Height, Timing, Detection, lookup
import logging
import numpy as np
from functools import partial
import yaml

from . import mca

_logger = logging.getLogger(__name__)


class SerialRegisterError(Exception):
    def __init__(self, non_hex=False, bad_length=False, axi='OKAY'):
        self.non_hex = non_hex
        self.bad_length = bad_length
        self.axi = axi

    def __str__(self):
        return (
            'register access error - non_hex:{} bad_length:{} AXI:{}\n'
            .format(self.non_hex, self.bad_length, self.axi) +
            'AXI:DECERR indicates a non-existent address\n' +
            'AXI:SLVERR typically indicates an access violation' +
            'eg. writing a read-only address'
        )


class RegisterError(AttributeError):
    pass


def _fstr(field):
    return '(0x{:08X},{:})'.format(*field) if field else '()'


def _mstr(field, address):
    return '0x{:08X} {:}'.format(address, _fstr(field))


def _indices(index, size):
    """return iterable containing register channels to read/write"""
    _logger.debug(
        '_indices:index {:}={:}'.format(type(index), index)
    )
    if isinstance(index, tuple):
        return index
    if isinstance(index, slice):
        return range(*index.indices(size))
    if type(index) is int:
        if index < 0 or index >= size:
            raise IndexError
        return (index,)

    raise NotImplementedError(
        'Indexing with {:} not implemented', format(index)
    )


class RegInfo:
    """descriptor mediating FPGA register access.
       the owning instance must have two methods;
       read(address) -> register value
       write(address, value) -> value written or None
    """
    def __init__(
        self, address, field, strobe, output_transform, input_transform,
        loadable, doc=None
    ):
        """

        :param address: 32 bit FPGA register address
        :param field: tuple (32 bit mask, shift)
        :param strobe: boolean indicating a strobe value
        :param output_transform: callable called on the register value after
               reading
        :param input_transform: callable called to on value before writing
        :param doc: docstring for this register
        """

        self.address = address
        self.field = field  # (mask, shift)
        self.strobe = strobe
        self.output_transform = output_transform
        self.input_transform = input_transform
        self.loadable = loadable
        self.__doc__ = doc

    def __get__(self, instance, owner):
        _logger.debug(
            'RegInfo.__get__ instance:{:} owner:{:}'
            .format(instance.__class__.__name__, owner.__name__)
        )
        if not instance:
            return self

        read = instance.read
        write = instance.write
        if hasattr(instance, 'transform'):  # FIXME use isinstance
            transform = instance.transform
            indices = _indices(instance._index, instance.size)
            _logger.debug('RegInfo.__get__ indices:{:}'.format(indices))
            values = []
            for c in indices:
                address, field = transform(self.address, self.field, channel=c)
                values.append(
                    self._get_reg(address, field, read, write)
                )
            if len(values) == 1:
                return values[0]
            return values
        else:
            return self._get_reg(self.address, self.field, read, write)

    def _get_reg(self, address, field, read, write):
        # strobe must not have an empty field tuple
        if self.strobe:
            if not field:
                raise AttributeError(
                    '_get_reg:strobe with empty field, bad reg_map'
                )
            write(field[0], address)
            return

        data = read(address)
        _logger.debug(
            '_get_reg:Reading 0x{:08X} returned 0x{:X}'.format(address, data)
        )

        if len(field):
            # data = np.right_shift(np.bitwise_and(data, field[0]), field[1])
            data = (data & field[0]) >> field[1]
            _logger.debug(
                '_get_reg:extract field{:} -> 0x{:}'.format(_fstr(field), data)
            )
        if self.output_transform is not None:
            tdata = self.output_transform(data)
            _logger.debug(
                '_get_reg:transform output 0x{:X} -> {:}'.format(data, tdata)
            )
            return tdata

        return int(data)

    def __set__(self, instance, values):
        _logger.debug(
            'RegInfo.__set__ instance:{:}'.format(instance.__class__.__name__)
        )
        read = instance.read
        write = instance.write
        if hasattr(instance, 'transform'):
            transform = instance.transform
            indices = _indices(instance._index, instance.size)
            _logger.debug(
                'RegInfo.__set__ indices {:} to {:}'.format(indices, values)
            )

            if type(values) is str:  # name of an enum
                len_values = 1
                values = (values,)
            else:
                try:
                    len_values = len(values)
                except TypeError:
                    len_values = 1
                    values = (values,)

            if len_values != len(indices) and len_values != 1:
                raise IndexError(
                    'Cannot broadcast {:} values to {:} indices'
                    .format(len_values, len(indices))
                )

            for i in range(len(indices)):
                address, field = transform(
                    self.address, self.field, channel=indices[i]
                )
                if len_values == 1:
                    self._set_reg(values[0], address, field, read, write)
                else:
                    self._set_reg(values[i], address, field, read, write)
        else:
            self._set_reg(values, self.address, self.field, read, write)

    def _set_reg(self, value, address, field, read, write):
        # strobe must have a non empty bit_field tuple
        _logger.debug(
            '_set_reg:{:} to 0x{:}'.format(_mstr(field, address), value)
        )
        input_transform = self.input_transform
        if self.strobe:
            if not field:
                raise AttributeError('strobe with empty field, bad reg_map')
            write(field[0], address)
            return

        if field:
            old_value = read(address)

            if input_transform:
                tvalue = int(input_transform(value))
                _logger.debug(
                    '_set_reg:input_transform {:} -> 0x{:08X}'
                    .format(value, tvalue)
                )
                value = tvalue
            new_value = (
                (old_value & ~field[0]) | ((int(value) << field[1]) & field[0])
            )
            _logger.debug(
                '_set_reg:address(0x{:08X}) from 0x{:08X} to 0x{:08X}'
                .format(address, old_value, new_value)
            )
        else:
            new_value = int(value)
            _logger.debug(
                '_set_reg:address(0x{:X}) to 0x{:08X}'
                .format(address, new_value)
            )

        written = write(new_value, address)

        if written is not None:
            _logger.debug(
                '_set_reg:response from write to 0x{:08X} is 0x{:08X}'
                .format(address, written)
            )
        else:
            _logger.debug(
                '_set_reg:no response writing to 0x{:08X}'.format(address)
            )


# IO transforms
def _cpu_version(data):
    y = 2016 + ((data & 0xF0000000) >> 30)
    m = (data & 0x0F000000) >> 24
    d = (data & 0x00FF0000) >> 16
    h = (data & 0x0000FF00) >> 8
    mi = data & 0x000000FF
    return '{:04d}-{:02d}-{:02d} {:02d}:{:02d}'.format(y, m, d, h, mi)


def _from_onehot(value):
    if value == 0:
        return 0

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
    if abs(cf-(cfi/2**17)) > abs(cf-((cfi+1)/2**17)):
        return cfi+1
    return cfi


def _from_cf(value):
    return value/2**17

event_lookup = partial(lookup, enum=Detection)
height_lookup = partial(lookup, enum=Height)
timing_lookup = partial(lookup, enum=Timing)
value_lookup = partial(lookup, enum=mca.Value)
trigger_lookup = partial(lookup, enum=mca.Trigger)
qualifier_lookup = partial(lookup, enum=mca.Qualifier)

# register maps
# address, field, strobe, output_transform, input_transform, loadable
mca_map = {
    'lowest_value': RegInfo(
        0x10000004, (), False, None, None, True,
        doc="""
        signed:32 bit
        Values < lowest value are placed in the underflow bin.

        """
    ),
    'ticks': RegInfo(
        0x10000008, (), False, None, None, True,
        doc="""
        unsigned:32 bit
        The number of tick_periods to accumulate statistics over.

        """
    ),
    'value': RegInfo(
        0x10000002, (0x0000000F, 0), False, mca.Value, value_lookup, True,
        doc="""
        unsigned:4 bit
        The value to collect statistics for.

        Returns tes.mca.Value enum, name or value can be used as input.
        """
    ),
    'trigger': RegInfo(
        0x10000002, (0x000000F0, 4), False, mca.Trigger, trigger_lookup, True,
        doc="""
        unsigned: 4 bit
        Values are only included when the selected trigger is True.

        Returns tes.mca.Trigger enum, name or value can be used as input.
        """
    ),
    'channel': RegInfo(
        0x10000002, (0x00000700, 8), False, None, None, True,
        doc="""
        unsigned:3 bit
        The channel to capture statistics from.

        """
    ),
    'bin_n': RegInfo(
        0x10000002, (0x0000F800, 11), False, None, None, True,
        doc="""
        unsigned:5 bit
        The width of histogram bins is 2**bin_n.

        """
    ),
    'last_bin': RegInfo(
        0x10000002, (0x3FFF0000, 16), False, None, None, True,
        doc="""
        unsigned:14 bit
        The bin used for overflows and the last bin in the histogram.

        """
    ),
    'update': RegInfo(
        0x10002000, (0x00000002, 0), True, None, None, False,
        doc="""
        strobe
        Update the MCA register settings on next tick if the buffer is free.

        """
    ),
    'update_on_completion': RegInfo(
        0x10002000, (0x00000001, 0), True, None, None, False,
        doc="""
        strobe
        Update the MCA register settings on after the current buffer has
        completed its ticks.

        """
    ),
    'qualifier': RegInfo(
        0x10000800, (0x0000000F, 0), False, mca.Qualifier, qualifier_lookup,
        True,
        doc="""
        unsigned:4 bit
        Qualifier for trigger, this must also be true for the value to be
        included in the histogram.

        Returns tes.mca.Qualifier enum, name or value can be used as input.
        """
    )
}

global_map = {
    'central_cpu': RegInfo(
        0x10000000, (), False, _cpu_version, None, False,
        doc="""
        Read only
        Date and time central cpu code was assembled.

        type:str
        """
    ),
    'channel_cpu': RegInfo(
        0x00000000, (), False, _cpu_version, None, False,
        doc="""
        Date and time the channel cpu code was assembled.

        Read only
        type:str
        """
    ),
    'hdl_version': RegInfo(
        0x10000001, (), False, lambda x: 'sha-1 {:07X}'.format(x), None, False,
        doc="""
        Short SHA-1 of HDL code commit.

        Read only
        type:str
        """
    ),
    'channel_count': RegInfo(
        0x10800000, (0x000000FF, 0), False, None, None, False,
        doc="""
        Number of processing channels.

        Read only
        type:np.uint8
        """
    ),
    'adc_chips': RegInfo(
        0x10800000, (0x0000FF00, 8), False, None, None, False,
        doc="""
        Number of dual channel ADC chips.

        Read only
        type:np.uint8
        """
    ),
    'fmc': RegInfo(
        0x10800000, (0x00010000, 16), False, bool, None, False,
        doc="""
        FMC card present.

        Read only
        type:bool
        """
    ),
    'fmc_power': RegInfo(
        0x10800000, (0x00020000, 17), False, bool, None, False,
        doc="""
        FMC card powered up.

        Read only
        type:bool
        """
    ),
    'ad9510_status': RegInfo(
        0x10800000, (0x00040000, 18), False, bool, None, False,
        doc="""
        AD9510 clock generator chip status pin.

        type:bool
        """
    ),
    'mmcm_locked': RegInfo(
        0x10800000, (0x00080000, 19), False, bool, None, False,
        doc="""
        FPGA MMCM locked to FMC clock.

        type:bool
        """
    ),
    'iodelay_ready': RegInfo(
        0x10800000, (0x00100000, 20), False, bool, None, False,
        doc="""
        FPGA iodelay controller for ADC inputs is initialised.

        Read only
        type:bool
        """
    ),
    'tick_period': RegInfo(
        0x10000020, (), False, None, None, True,
        doc="""
        4ns clocks between tick events.

        type:np.uin32
        """
    ),
    'tick_latency': RegInfo(
        0x10000040, (), False, None, None, True,
        doc="""
        Maximum number of 4ns clocks to wait after a tick before
        flushing the event buffer to the next tick.

        type:np.uint32
        """
    ),
    # 'adc_enable': RegInfo(
    #     0x10000080, (0x000000FF, 0), False, np.uint8, None,
    #     doc="""
    #     Enable the ADC channels corresponding to the set bits,
    #     disabled channels are in low power mode.
    #
    #     type:np.uint8
    #     """
    # ),
    # 'event_enable': RegInfo(
    #     0x10000100, (0x000000FF, 0), False, np.uint8, None,
    #     doc="""
    #     Enable events for the processing channels corresponding to the
    #     set bits.
    #
    #     type:np.uint8
    #     """
    # ),
    'window': RegInfo(
        0x10000400, (), False, None, None, True,
        doc="""
        Coincidence window, see new window bit in event_flags.

        type:np.uint16
        """
    ),
    'fmc_internal_clk': RegInfo(
        0x10000200, (0x00000001, 0), False, bool, None, False,
        doc="""
        FMC108 internal clock pin.

        type:bool
        """
    ),
    'vco_power_en': RegInfo(
        0x10000200, (0x00000002, 1), False, bool, None, False,
        doc="""
        AD9510 clock generator VCO power enable pin. See data sheet.

        type:bool
        """
    ),
    'mtu': RegInfo(
        0x10000010, (), False, None, None, False,
        doc="""
        Maximum size of Ethernet frames.

        type:np.uint16
        """
    )
}

baseline_map = {
    'offset': RegInfo(
        0x00000040, (), False, None, None, True,
        doc="""
        Subtracted from ADC signal.

        type:np.int16
        """
    ),
    'time_constant': RegInfo(
        0x00000080, (), False, None, None, True,
        doc="""
        4ns clocks between base MCA buffer swaps.

        type:np.uint32
        """
    ),
    'threshold': RegInfo(
        0x00000100, (), False, None, None, True,
        doc="""
        Values of ADC-offset > threshold are not included in the
        baseline estimate.

        type:np.uint16
        """
    ),
    'count_threshold': RegInfo(
        0x00000200, (), False, None, None, True,
        doc="""
        Baseline MCAs most frequent bin is included in the
        baseline estimate when its count > count_threshold.

        type:np.int32
        """
    ),
    'new_only': RegInfo(
        0x00000400, (0x00000001, 0), False, bool, None, True,
        doc="""
        When true the baseline MCAs most frequent bin is included in the
        baseline estimate when it is different from the previous most
        frequent bin. Otherwise it is included whenever its count changes.

        type:bool
        """
    ),
    'subtraction': RegInfo(
        0x00000400, (0x00000002, 1), False, bool, None, True,
        doc="""
        When true the baseline estimate is subtracted from the ADC signal.

        type:bool
        """
    )
}

cfd_map = {
    'fraction': RegInfo(
        0x00000008, (0x0001FFFF, 0), False, _from_cf, _to_cf, True,
        doc="""
        The constant fraction.

        type:float
        """
    ),
    'rel2min': RegInfo(
        0x00000008, (0x80000000, 31), False, bool, None, True,
        doc="""
        When true, the constant fraction threshold are calculated from
        the difference between the peak and the preceding minima. Otherwise
        from the peak height alone.

        type:bool
        """
    ),
}

event_map = {
    'enable': RegInfo(
        0x10000100, (), False, bool, None, True,
        doc="""
        Enable events from this channel.

        type:bool
        """
    ),
    'packet': RegInfo(
        0x00000001, (0x00000003, 0), False, Detection, event_lookup, True,
        doc="""
        The type of event packet generated by this channel.

        type:tes.base.Event enum, input can also be a str name from the
        enumeration.
        """
    ),
    'timing': RegInfo(
        0x00000001, (0x0000000C, 2), False, Timing, timing_lookup, True,
        doc="""
        The point at which a peak is timestamped.

        type:tes.base.timing enum, input can also be a str name from the
        enumeration.
        """
    ),
    'max_peaks': RegInfo(
        0x00000001, (0x000000F0, 4), False, None, None, True,
        doc="""
        The maximum number of peaks that a pulse event can record is
        max_peaks+1. This sets the length of pulse events.

        type:np.uint8 Lower 4 bits.
        """
    ),
    'height': RegInfo(
        0x00000001, (0x00000300, 8), False, Height, height_lookup, True,
        doc="""
        The value placed in the height field of events.

        type:tes.base.Height enum, input can also be a str name from the
        enumeration.
        """
    ),
    'pulse_threshold': RegInfo(
        0x00000002, (0x0000FFFF, 0), False, None, None, True,
        doc="""
        For a peak to be produce an event it maxima must be
        greater than or equal to pulse threshold and the peak detector
        must be armed. See slope threshold.

        type:np.int16
        """
    ),
    'slope_threshold': RegInfo(
        0x00000004, (0x0000FFFF,0), False, None, None, True,
        doc="""
        When slope makes a positive going crossing of slope_threshold
        the peak detector is armed and remains armed until the slope
        makes a negative going zero crossing.

        type:np.uint16
        """
    ),
    'area_threshold': RegInfo(
        0x00000010, (), False, None, None, True,
        doc="""
        The area of a pulse must be greater that or equal to
        area_threshold to produce an area or pulse event.

        type:np.uint32
        """
    )
}

channel_map = {
    # 'version': RegInfo(
    #     0x00000000, (), False, _cpu_version, None, False,
    #     doc="""
    #     Date and time the channel cpu code was assembled.
    #
    #     Read only
    #     type:str
    #     """
    # ),
    'adc_select': RegInfo(
        0x00000800, (0x000000FF, 0), False, _from_onehot, _to_onehot, True,
        doc="""
        Select the ADC channel this channel processes.

        type:np.uint8
        """
    ),
    'invert': RegInfo(
        0x00000800, (0x00000100, 8), False, bool, None, True,
        doc="""
        Multiply the ADC input by -1.

        type:bool
        """
    ),
    'delay': RegInfo(
        0x00000020, (), False, None, None, True,
        doc="""
        The ADC input is delayed by delay*4ns.

        type:np.uint16
        """
    )
    # 'event_enable': RegInfo(
    #     0x10000100, (0x000000FF, 0), False, bool, None,
    #     doc="""
    #     Enable events from this channel.
    #
    #     type:bool
    #     """
    # )
}

adc_map = {
    'enable': RegInfo(
        0x10000080, (), False, bool, None, True,
        doc="""
        Sets/clears the adc_enable bit for this ADC channel.
        See adc_enable.

        type:bool
        """
    ),
    'pattern': RegInfo(
        0x20000062, (), False, None, None, False,
        doc="""
        Set the ADC test pattern. See the ADS62P49 data sheet.

        type:np.uint8
        """
    ),
    'pattern_low': RegInfo(
        0x20000051, (), False, None, None, False,
        doc="""
        See the ADS62P49 data sheet.

        type:np.uint8
        """
    ),
    'pattern_high': RegInfo(
        0x20000052, (), False, None, None, False,
        doc="""
        See the ADS62P49 data sheet.

        type:np.uint8
        """
    ),
    'gain': RegInfo(
        0x20000055, (0x000000F0, 4), False, None, None, False,
        doc="""
        ADC gain. See the ADS62P49 data sheet.

        type:np.uint8
        """
    )
}


# address transforms alter the RegInfo address and field so that that they
# address the # register field for required channel
def _adc_spi_transform(address, field, channel=0):
    # map spi to ADCs
    if ((address & 0xFF000000) >> 24) == 0x20:  # it's a SPI address
        spi_address = address & 0x00000FF
        if channel % 2:  # B channel on the chip
            spi_address += 0x13
        mosi_mask = pow(2, channel // 2) << 8
        spi = 0x20000000 | mosi_mask | spi_address
        _logger.debug(
            (
                '_adc_spi_transform:chip address:0x{:02X} MOSI:0x{:02X} -> ' +
                'SPI address:0x{:08X}'
            ).format(spi_address, mosi_mask, spi)
        )
        taddress, tfield = spi, ()
    else:
        taddress, tfield = address, (_to_onehot(channel), channel)

    _logger.debug(
        '_adc_spi_transform:channel={:} {:} -> {:}'
            .format(channel, _mstr(field, address), _mstr(tfield, taddress))
    )
    return taddress, tfield


def _channel_transform(address, field, channel=0):
    if address & 0x10000000:  # its an enable
        taddress, tfield = address, (_to_onehot(channel), channel)
    else:
        taddress, tfield = (address & 0xF0FFFFFF) | (channel << 24), field

    _logger.debug(
        '_channel_transform:channel={:} {:} -> {:}'
            .format(channel, _mstr(field, address), _mstr(tfield, taddress))
    )
    return taddress, tfield


# transport functions
def _get_response(socket):
    if socket is None:  # in direct serial mode
        l = __serial__.readline()
    else:
        l = socket.recv()

    if len(l) == 0:
        raise serial.SerialTimeoutException
    _logger.debug('_get_response {:}'.format(l[:-1]))
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
            bytearray.fromhex(l[:-3].decode('utf8')), np.uint32).byteswap()[0]
        return data


def _dummy_read(address):
    _logger.debug('_dummy_read:address:{:08X}'.format(address))
    return 0


def _dummy_write(value, address):
    _logger.debug('_dummy_write:{:08X} to {:08X}'.format(value, address))


# zmq read and write methods for talking to the register server
def _zmq_read(address, socket=None):
    _logger.debug('_zmq_read:address:{:08X}'.format(address))
    socket.send(
        b'00000000' + bytes('{:08X}02\n'.format(address), encoding='utf8'))
    return _get_response(socket)


def _zmq_write(data, address, socket=None):
    _logger.debug('_zmq_write:{:08X} to {:08X}'.format(data, address))
    socket.send(
        bytes('{:08X}{:08X}01\n'.format(data, address), encoding='utf8')
    )
    return _get_response(socket)


def _flush(socket=None):
    socket.send(b'flush')


# metaclass
class RegisterMap(type):
    """
    metaclass for register access classes
    Used to inject the reg_map descriptors at class creation
    Is there a better way?
    """
    @classmethod
    def __prepare__(mcs, name, bases, **kwargs):
        return kwargs['reg_map'].copy()

    def __new__(mcs, name, bases, attrs, **kwargs):
        return type.__new__(mcs, name, bases, attrs)

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        cls.reg_map = kwargs['reg_map']
        super().__init__(name, bases, attrs)


# base classes
class RegisterGroup(metaclass=RegisterMap, reg_map={}):
    """base class for a register map"""
    def __init__(self, read, write):
        """

        :param read: callable, read(address) -> value
        :param write: callable, write(value, address) -> response or None.
        """
        self.read = read
        self.write = write

    def __iter__(self):
        for reg, desc in getattr(self, 'reg_map').items():
            if not desc.strobe:
                yield reg, getattr(self, reg)


class RegisterGroupCollection(RegisterGroup, reg_map={}):
    """base class for a register map with channel indexing"""

    def __init__(self, read, write, size, transform):
        """

        :param read: callable, read(address) -> value
        :param write: callable, write(value, address) -> response or None.
        :param size:  number of channels
        :param transform:
        callable transform(address, field, channel) -> address, field
        transforms address and field to access the register for channel.
        """
        self.read = read
        self.write = write
        self.size = size
        self.transform = transform
        #  Used internally for indexing.
        self._index = None
        super().__init__(read, write)

    def __getitem__(self, index):
        _logger.debug(
            '{:}.__getitem__({:})'.format(self.__class__.__name__, index)
        )
        self._index = index
        return self

    def __get__(self, instance, owner):
        _logger.debug(
            '{:}.__get__ instance:{:} owner:{:}'
            .format(
                self.__class__.__name__, instance.__class__.__name__,
                owner.__name__
            )
        )
        if not instance:
            return self
        self._index = slice(None, None, None)
        return self

    def __set__(self, instance, value):
        _logger.debug(
            '{:}.__set__ instance:{:}'
            .format(self.__class__.__name__, instance.__class__.__name__)
        )
        self._index = slice(None, None, None)
        return self


# Concrete classes for the register groups
class EventRegisters(RegisterGroupCollection, reg_map=event_map):
    """Register group controlling event output.

    See help(EventRegisters) for details.
    """


class ChannelRegisters(RegisterGroupCollection, reg_map=channel_map):
    """Register group controlling input to the processing channels.

    See help(ChannelRegisters) for details.
    """


class BaselineRegisters(RegisterGroupCollection, reg_map=baseline_map):
    """Register group controlling the baseline correction process.

    See help(BaselineRegisters) for details.
    """


class AdcRegisters(RegisterGroupCollection, reg_map=adc_map):
    """Register group controlling the ADC chips (ADS62P49).

    See help(AdcRegisters) for details.
    """


class McaRegisters(RegisterGroup, reg_map=mca_map):
    """Register group controlling the MCA.

    See help(McaRegisters) for details.
    """


class CfdRegisters(RegisterGroupCollection, reg_map=cfd_map):
    """Register group controlling the CFD process.

    See help(CfdRegisters) for details.
    """

# Want to be able to assign dicts
# how to assign to the root?
#


class _GroupIterator:
    def __init__(self, root):
        self.root = root

    def generator(self):
        yield 'root', self.root
        for attr_name in dir(self.root):
            attr = getattr(self.root, attr_name)
            if type(type(attr)) is RegisterMap:
                yield attr_name, attr
        return

    def __iter__(self):
        return self.generator()


class Registers(RegisterGroup, reg_map=global_map):
    """
    Client for reading and writing the internal FPGA control registers.

    Registers are arranged in functional groups and accessed through an
    instance of the Registers class. Let r be an instance of Registers
    then r.regname  references the regname register while r.groupname.regname
    the regname register of the  groupname group. Some groups support
    indexing to reference a register for a particular channel. Slicing and
    fancy indexing are supported while ommiting indexing is equvalent  to
    referencing ALL channels.

    For example:
    r.groupname[0].regname refers to the regname register of the groupname group
    for channel 0. While r.groupname.regname refers to the same register for all
    channels. Therefore, r.groupname.regname will return a list containing the
    value of the register for each channel, r.groupname.regname = value will set
    the for all channels to the same value and
    r.groupname.regname = [value0, value1, ..., valuen] will broadcast the list
    of values to the appropriate channel.

    Groups without indexing:
    No groupnme - accesses a general register. See help(Registers).
    mca - accesses the registers controlling the MCA. See help(McaRegisters).

    Groups supporting indexing:
    channel controls input to the processing channels. See help(ChannelRegisters).
    event  controls event output. See help(EventRegisters).
    baseline  controls the baseline process. See help(BaselineRegisters).
    cfd  controls the constant fraction process. See help(CfdRegisters).
    adc  controls the ADC chips. See help(AdcRegisters).

    The the number of channels in the adc group is twice the value of the
    general register adc_chips while the number of channels in all other
    groups is the value of the general register channel_count.

    """

    mca = McaRegisters(None, None)
    channel = ChannelRegisters(None, None, None, _channel_transform)
    event = EventRegisters(None, None, None, _channel_transform)
    baseline = BaselineRegisters(None, None, None, _channel_transform)
    cfd = CfdRegisters(None, None, None, _channel_transform)
    adc = AdcRegisters(None, None, None, _adc_spi_transform)

    def __init__(self, port=None, read=_dummy_read, write=_dummy_write):
        super().__init__(read, write)
        self.read = read
        self.write = write
        #FIXME assign transport during class creation
        Registers.channel.read = read
        Registers.channel.write = write
        Registers.event.read = read
        Registers.event.write = write
        Registers.baseline.read = read
        Registers.baseline.write = write
        Registers.cfd.read = read
        Registers.cfd.write = write
        Registers.adc.read = read
        Registers.adc.write = write
        Registers.mca.read = read
        Registers.mca.write = write

        self.all = _GroupIterator(self)

        if read is _dummy_read:
            Registers.channel.size = 8
            Registers.event.size = 8
            Registers.baseline.size = 8
            Registers.cfd.size = 8
            Registers.adc.size = 8
        else:
            raise NotImplementedError

    def _group_gen(self):
        yield 'root', self
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if type(type(attr)) is RegisterMap:
                yield attr_name, attr
        return

    def _full_dict(self):
        return {reg: dict(value) for (reg, value) in self._group_gen()}

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write(yaml.dump(self._full_dict()))

# direct serial connection FIXME add module switch to make these work again
__serial__ = None  # module global serial port handle


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
