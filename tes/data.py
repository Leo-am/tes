import logging
from collections import namedtuple
import zmq

from .base import Payload

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
