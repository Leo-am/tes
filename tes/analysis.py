"""
Calibration and modeling
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy import (
    bitwise_and as and_b, logical_and as and_l, logical_or as or_l,
    logical_not as not_l, bitwise_or as or_b
)
from scipy import signal
from scipy import stats
from scipy.optimize import brentq
from numba import jit
from collections import namedtuple
import qutip as qt
from lmfit import Model


@jit
def maxima(f, thresh=10):
    """
    Find local maxima of f using rising zero crossings of the gradient of f.

    :param f: function to find maxima for
    :param thresh: Only return a maxima if f[maxima]>thresh

    :return: array of maxima.
    :Notes: Used to find the maxima of the smoothed measurement histogram.
    """
    g = np.gradient(f)
    pos = g > 0
    xings = (pos[:-1] & ~pos[1:]).nonzero()[0] + 1  # indices of the maxima
    return xings[np.where(f[xings] > thresh)]


Guess = namedtuple(
    'Guess', 'hist f bin_c max_i thresholds'
)


def guess_thresholds(data, bins=16000, win_length=200, maxima_threshold=10):
    """
    Guess the photon number thresholds using a smoothed histogram.

    :param ndarray data: the measurement data.
    :param bins: argument to numpy.histogram.
    :param int win_length: Hanning window length used for smoothing.
    :param int maxima_threshold: passed to maxima function.

    :return: (hist, f, bin_c, max_i, thresholds) as a named
            tuple. Where hist is the histogram
            generated by np.histogram(), f is the smoothed histogram, bin_c is a
            ndarray of bin centers, max_i are the indices of the maxima in f,
            and thresholds is a ndarray of threshold guesses.

    :notes: The smoothing is achieved by convolving the Hanning window with a
            histogram generated using numpy.histogram. The guess still needs
            some human sanity checks, especially multiple maxima per photon
            peak. Use plot_guess() and adjust window and maxima_threshold to
            remove them.
    """

    hist, edges = np.histogram(data, bins=bins)
    bin_w = edges[1] - edges[0]
    bin_c = edges[:-1] + bin_w / 2

    # smooth the histogram
    win = signal.hann(win_length)
    f = signal.convolve(hist, win, mode='same') / sum(win)

    max_i = maxima(f, thresh=maxima_threshold)  # indices of the maxima
    m = [0.0] + list(bin_c[max_i])
    t = [m[i - 1] + (m[i] - m[i - 1]) / 2 for i in range(2, len(m))]
    thresholds = np.array([0.0] + t + [m[-1] + (m[-1] - t[-1])*1.10])

    return Guess(hist, f, bin_c, max_i, thresholds)


def plot_guess(hist, f, x, max_i, init_t, figsize=None):
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()
    ax = fig.add_axes([0.12, 0.12, 0.87, 0.87])
    ax.plot(
        x[hist.nonzero()[0]], hist[hist.nonzero()[0]], 's',
        markersize=3, markerfacecolor='none', mew=0.5,
        label='histogram bin', color='gray'
    )
    ax.plot(x, f, 'k', lw=1, label='smoothed histogram')
    ax.plot(
        x[max_i], f[max_i], 'o', label='maxima', markerfacecolor='none', mew=1,
        markersize=5, color='r'
    )
    for t in init_t:
        plt.plot([t, t], [0, max(hist)], ':k', lw=0.5)

    ax.set_xlabel('Energy (abitrary units)')
    ax.set_ylabel('count')
    # ax.set_title('Histogram and initial threshold guess')
    plt.legend()
    return fig


# @jit
def refactor_time(events, tdat, fidx, tidx):
    # TODO generalise
    # remove incorporate tick relative times
    #     events = edat.view(event_dt)
    event_time = events['time']
    event_i = 0
    event_time[0] = 0xFFFF
    # tidx[0] is the first tick period ending with tick 1
    for i in range(len(tidx) - 1):
        # i+1 is the tick number
        changed = fidx[
                  tidx[i]['start'] + 1:tidx[i]['stop'] + 1]['changed'
        ].nonzero()[0]
        if len(changed):
            raise NotImplementedError(
                'tick:{} is not homogeneous'.format(i)
            )
        event_count = int(
            (
                    fidx[tidx[i]['stop']]['payload'] +
                    fidx[tidx[i]['stop']]['length'] -
                    fidx[tidx[i]['start']]['payload']
            ) / (fidx[tidx[i]['start']]['event_size'] * 8)
        )
        #         print(event_count)
        event_i += event_count
        time = int(tdat[i + 1]['time']) + int(event_time[event_i])
        #     print(event_i)
        if time > 0xFFFF:
            time = 0xFFFF
        event_time[event_i] = time
    return event_time


def normalised_pdf(x, params, dist=stats.gamma):
    """
    Convenience function, calculate normalised pdf at each x for dist.

    :param union[numpy.ndarray, int] x: calculate the pdf a each x
    :param iterable params: fitted distribution parameters
    :param dist: type of distribution (see scipy.stats)
    :return: ndarray containing pdf(x)
    """
    return dist.pdf(x, *params[:-1])


def scaled_pdf(x, params, dist=stats.gamma):
    """
    Convenience function, calculate pdf at each x for dist, normalisation is
    given by params[-1].

    :param union[numpy.ndarray, int] x: calculate the pdf a each x
    :param iterable params: fitted distribution parameters
    :param dist: type of distribution (see scipy.stats)
    :return: ndarray containing pdf(x)
    """
    return params[-1]*dist.pdf(x, *params[:-1])


# FIXME this is actually the expectation step
def maximisation_step(param_list, normalise=False, dist=stats.gamma):
    """
    Calculate the thresholds for a mixture model that define the data
    partitions.

    :param list param_list: list of parameters for the distributions in the
           mixture model, returned by expectation_maximisation().
    :param bool normalise: calculate thresholds based on a mixture of
           normalised pdf's.
    :param dist: type of distribution
    :return: ndarray of threshold values.
    """
    t = [0]
    for m in range(1, len(param_list)):
        left = param_list[m - 1]
        right = param_list[m]

        if normalise:
            pdf = normalised_pdf
        else:
            pdf = scaled_pdf

        def f(x):
            return pdf(x, left, dist) - pdf(x, right, dist)

        t.append(brentq(f, dist.median(*left[:-1]), dist.median(*right[:-1])))
    return np.array(t)


class MixtureModel(
    namedtuple(
        'MixtureModel', 'param_list thresholds zero_loc log_likelihood '
        'converged dist'
     )
):
    """
    Subclass of namedtuple used to represent a mixture model.

    fields:
        :param_list: list of parameters for each distribution in the mixture.
        :thresholds: the intersection of neighbouring distributions in the
                     mixture.
        :zero_loc: Fixed location parameter used to fit the first distribution,
                   or None if the location parameter was fitted.
        :log_likelihood: the log of the likelihood that the model could produce
                         the data used to construct it.
        :converged: boolean indicating that the expectation maximisation
                    algorithm terminated normally when fitting the model.
        :dist: the type of distribution forming the components of the mixture,
               (see scipy.stats).

    """

    __slots__ = ()

    def save(self, filename):
        """
        save model as .npz file.

        :param filename: filename to save as, excluding extension"
        :return: None
        """
        np.savez(
            filename,
            param_list=self.param_list,
            thresholds=self.thresholds,
            zero_loc=self.zero_loc,
            log_likelihood=self.log_likelihood,
            converged=self.converged,
            dist=self.dist.name
        )

    @staticmethod
    def load(filename):
        """
        load model from .npz file.

        :param filename: filename excluding extension"
        :return: instance of MixtureModel.
        """
        d = np.load(filename+'.npz')
        return MixtureModel(
            d['param_list'],
            d['thresholds'],
            d['zero_loc'],
            d['log_likelihood'],
            d['converged'],
            getattr(stats, str(d['dist']))
        )

    def _eval(self, x, d=None, func='pdf'):
        """
        evaluate the function of the distribution(s) selected by d at x.

        :param x: value(s) where to the function is evaluated.
        :param d: selects element(s) of param_list that parameterise the
                  distributions. If None all distributions in param_list are
                  used.
        :param str func: the name of the distribution  function to evaluate
                             (see scipy.stats.rv_continuous)
        :return: ndarray shape (len(d), len(x)) containing the pdf(s) or a
                 single value if only 1 point in 1 pdf is selected.
        """

        if d is None:
            p_list = self.param_list
        else:
            p_list = self.param_list[d]

        try:
            xl = len(x)
        except TypeError:
            xl = 1

        f = getattr(self.dist, func)

        # print(p_list.shape)
        if p_list.shape[0] == 1 or len(p_list.shape) == 1:
            return f(x, *p_list[:-1])
        else:
            f_evals = np.zeros((len(p_list), xl))
            i = 0
            for p in p_list:
                f_evals[i, :] = f(x, *p[:-1])
                i += 1

            return f_evals

    def pdf(self, x, d=None):
        """
        evaluate the pdf(s) of the distribution(s) selected by d at x.

        :param x: value(s) where to the function is evaluated.
        :param d: selects element(s) of param_list that parameterise the
                  distributions. If None all distributions in param_list are
                  used.
        :return: ndarray shape (len(d), len(x)) containing the pdf(s) or a
                 single value if only 1 point in 1 distribution is selected.
        """
        return self._eval(x, d, func='pdf')

    def cdf(self, x, d=None):
        """
        evaluate the pdf(s) of the distribution(s) selected by d at x.

        :param x: value(s) where to the function is evaluated.
        :param d: selects element(s) of param_list that parameterise the
                  distributions. If None all distributions in param_list are
                  used.
        :return: ndarray shape (len(d), len(x)) containing the pdf(s) or a
                 single value if only 1 point in 1 distribution is selected.
        """
        return self._eval(x, d, func='cdf')


@jit
def expectation_maximisation(
    data, initial_thresholds, zero_loc=None, dist=stats.gamma, tol=1,
    max_iter=30, verbose=True, normalise=False
):
    """
    Fit a mixture of distributions to data.

    :param ndarray data: the data to fit.
    :param initial_thresholds: initial thresholds that divide data into
           individual distributions in the mixture.
    :param int or None zero_loc: if not None fixes the location parameter
           of the first distribution in the mixture.
    :param dist: the type of distribution to use (see scipy.stats)
    :param float tol: termination tolerance for change in log_likelihood.
    :param int max_iter: max iterations of expectation maximisation to use.
    :param bool verbose: print progress during optimisation.
    :param bool normalise: passed to maximise step, thresholds are calculated
           using the intersection of the normalised distributions.

    :return: (param_list, zero_loc, log_likelihood, converged) as a named tuple.
            Where param_list is a list of parameter values for distribution in
            the mixture, zero_loc is the arg used to fit, log_likelihood is the
            log_likelihood of the fit, converged is a bool indicating normal
            termination.

    :note: Uses The Expectation maximisation algorithm with hard assignment of
           responsibilities.

    """

    last_threshold = initial_thresholds[-1]
    data = data[data <= last_threshold]
    initial_thresholds = initial_thresholds[:-1]

    i = 1
    if verbose:
        print(
            'Starting expectation maximisation, {} {} distributions'
            .format(len(initial_thresholds), dist.name)
        )
    if verbose:
        print('Expectation step:{}'.format(i))
    param_list = expectation_step(
        data, initial_thresholds, zero_loc=zero_loc, dist=dist
    )
    if verbose:
        print('Maximisation step:{}'.format(i))
    new_thresh = np.array(
        maximisation_step(param_list, dist=dist, normalise=normalise)
    )
    ll = mixture_model_ll(data, param_list, dist=dist)
    if verbose:
        print('Threshold changes:{!r}'.format(initial_thresholds - new_thresh))
        print('log likelihood:{}'.format(ll))

    converged = False
    initial_thresholds = new_thresh
    while i < max_iter:
        i += 1
        if verbose:
            print('Expectation step:{}'.format(i))
        new_param_list = expectation_step(
            data, initial_thresholds, zero_loc=zero_loc, dist=dist
        )
        if verbose:
            print('Maximisation step:{}'.format(i))
        new_thresh = np.array(
            maximisation_step(new_param_list, dist=dist, normalise=normalise)
        )
        new_ll = mixture_model_ll(data, new_param_list, dist=dist)
        if verbose:
            print('Threshold changes:{!r}'.format(initial_thresholds - new_thresh))
            print('log likelihood:{} change:{}'.format(new_ll, ll-new_ll))

        # terminate if tol met or ll goes down. FIXME is this appropriate?
        if abs(new_ll - ll) <= tol or new_ll < ll:
            if new_ll > ll:
                ll = new_ll
                param_list = new_param_list
            converged = True
            break

        ll = new_ll
        param_list = new_param_list
        initial_thresholds = new_thresh

    if verbose:
        if converged:
            print('Converged')
        else:
            print('Maximum iterations reached')
    thresholds = list(
        maximisation_step(param_list, normalise=True, dist=dist)
    ) + [last_threshold]

    return MixtureModel(
        param_list, np.array(thresholds), zero_loc, ll, converged, dist
    )


def mixture_model_ll(data, param_list, dist=stats.gamma):
    """
    Calculate the log likelihood of a mixture model.

    :param ndarray data: the data used to construct the model.
    :param param_list: list of parameters for each distribution in the model.
    :param dist: the type of distribution to use (see scipy.stats).
    :return: the log likelihood.
    """

    ll = np.empty((len(param_list), len(data)), dtype=np.float64)
    for i in range(len(param_list)):
        p = param_list[i][:-1]
        ll[i, :] = dist.pdf(data, *p) * param_list[i][-1]
    return np.sum(np.log(np.sum(ll, 0)))


# FIXME this is actually the maximisation step
def expectation_step(
        data, thresholds, zero_loc=None, dist=stats.gamma, verbose=True
):
    """
    Fit distributions to the data partitioned by thresholds.

    :param data: the data to model.
    :param thresholds: thresholds that divide data into separate distributions.
    :param zero_loc: fix location parameter for first distribution.
    :param dist: type of distribution to fit (scipy.stats).
    :param bool verbose: Print fitted distribution parameters.
    :return: list of parameters for each distribution.
    """
    param_list = []
    #     print('len thresholds',len(thresholds))
    for i in range(len(thresholds)):
        part = partition(data, thresholds, i)
        if zero_loc is not None and i == 0:
            fit = dist.fit(part, floc=zero_loc)
        else:
            fit = dist.fit(part)
        if verbose:
            print('{} distribution:{} params:{}'.format(dist.name, i, fit))
        param_list.append(list(fit) + [len(part) / len(data)])
    return param_list


@jit
def partition(data, thresholds, i):
    """
    Return the ith partition of the data as defined by thresholds.

    :param ndarray data: the data to partition.
    :param iterable thresholds: the threshold values that partition data.
    :param int i: the partition to return, the ith partition is defined as
                  threshold[i] < data <= threshold[i+1]. The last partition,
                  when i = len(thresholds-1), is defined as
    :return: ndarray of the data in the partition.
    :note: This is used to assign a hard responsibility in the expectation
           maximisation algorithm used to fit data to a mixture model.
    """

    if i == len(thresholds) - 1:
        return data[data > thresholds[i]]
    else:
        return data[
            np.bitwise_and(data > thresholds[i], data <= thresholds[i + 1])
        ]


def plot_mixture_model(
        model, data=None, x=None, normalised=False, bins=16000, figsize=None
):
    """
    Plot a mixture model optionally including a histogram of the modeled data.
    if no data is None x must not be None.

    :param MixtureModel model: the mixture model.
    :param ndarray data: the data to histogram.
    :param ndarray x: the x values to plot.
    :param bool normalised: normalise the model distributions.
    :param bins: passed to numpy.histogram().
    :param figsize: passed to matplotlib.pyplot.figure().
    :return: the figure handle.
    """

    if data is None and x is None:
        raise AttributeError('Either data or x must be supplied')

    if figsize is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)

    ax = fig.add_axes([0.12, 0.12, 0.87, 0.84])
    if data is not None:
        hist, edges = np.histogram(data, bins=bins)
        w = edges[1] - edges[0]
        c = edges[:-1] + w / 2
        # x = np.linspace(edges[0], edges[-1], bins)
        max_p = max(hist)
        s = sum(hist) * w
        if not normalised:
            ax.plot(
                c[hist.nonzero()[0]],
                hist[hist.nonzero()[0]],
                's', color='darkgray', markerfacecolor='none', mew=0.5,
                markersize=3, label='bin count'
            )
        else:
            s = 1
    else:
        c = x
        s = 1

    pdfs = np.zeros((len(model.param_list), len(c)), np.float64)

    for i in range(len(model.param_list)):
        if normalised:
            pdfs[i, :] = normalised_pdf(c, model.param_list[i])
        else:
            pdfs[i, :] = scaled_pdf(c, model.param_list[i])*s

        if i == 0:
            ax.plot(c, pdfs[i, :], lw=2, label='noise')
        elif i != 1:
            ax.plot(c, pdfs[i, :], lw=2, label='{} photons'.format(i))
        else:
            ax.plot(c, pdfs[i, :], lw=2, label='{} photon'.format(i))

    if data is None or normalised:
        max_p = np.max(pdfs[1:])
        ax.set_xlim([-10, max(data)])

    if data is None:
        ax.set_xlim([-10, max(x)])
    else:
        ax.set_xlim([-10, max(data)])

    ax.set_ylim([0, max_p*1.1])

    ax.plot(c, np.sum(pdfs, 0), 'k', lw=0.5, label='model')

    if not normalised:
        thresh = np.array(
            list(maximisation_step(
                model.param_list, normalise=False, dist=model.dist
            )) + [model.thresholds[-1]]
        )
    else:
        thresh = model.thresholds

    for t in thresh:
        ax.plot([t, t], [0, max_p], ':k', lw=1)

    ax.set_xlabel('area')
    if normalised:
        ax.set_ylabel('probability')
    else:
        ax.set_ylabel('count')
    fig.legend()
    return fig


"""
timing analysis
"""


@jit
def xcor(s1, s2):
    """
    Count correlations between timestamp sequences.

    :param ndarray s1: Monotonically increasing timestamp sequence.
    :param ndarray s2: Monotonically increasing timestamp sequence.
    :return: correlation count.

    :notes: iterates over s1 and performs a linear search of  s2 for the s1
            timestamp.
    """
    i2 = 0
    c = 0
    for i in range(len(s1)):
        while i2 < len(s2) and s1[i] > s2[i2]:
            i2 += 1
        if i2 < len(s2):
            c += s1[i] == s2[i2]
    return c


@jit
def x_correlation(s1, s2, r):
    """
    Cross correlation of timestamp sequences s1 and s2 over the delay range r.

    :param ndarray s1: Monotonically increasing timestamp sequence.
    :param ndarray s2: Monotonically increasing timestamp sequence.
    :param r: range of delays added to s1
    :return: ndarray representing the cross correlation function.
    """
    x_corr = np.zeros((len(r),), np.uint64)
    for l in r:
        x_corr[l - r[0]] = xcor(s1 + l, s2)
    return x_corr


@jit
def drive_correlation(abs_time, mask, data, thresholds, r, verbose=False):
    """
    Calculate the temporal cross-correlation between a channel measuring a
    heralding signal, ie the laser drive pulse, and a channel detecting photons.
    The correlation is calculated for the different photon numbers determined by
    the thresholds parameter.


    :param ndarray abs_time: absolute timestamp sequence.
    :param ndarray mask: boolean mask that identifies the entries in abs_time
                         belonging to the photon channel, other entries are
                         assumed to be the heralding channel.
    :param ndarray data: energy measurement data for the photon channel.
    :param thresholds: thresholds that partition data into photon number.
    :param r: the range of heralding channel delays to cross correlate.
    :param bool verbose: Print progress message as each threshold is processed.
    :return: list of ndarrays representing the cross correlation.

    :note: abs_time[mask] and data must be the same length.
           TODO speed up the algorithm.
    """

    xc = []
    for t in range(len(thresholds)):
        if verbose:
            if t == 0:
                print('Cross correlating Noise partition')
            elif t == len(thresholds)-1:
                print('Cross correlating {}+ photon partition'.format(t))
            else:
                print('Cross correlating {} photon partition'.format(t))

        if t == len(thresholds)-1:
            xc.append(
                x_correlation(
                    abs_time[not_l(mask)],
                    abs_time[mask][data > thresholds[t]],
                    r
                )
            )
        else:
            xc.append(
                x_correlation(
                    abs_time[not_l(mask)],
                    abs_time[mask]
                    [and_b(data > thresholds[t], data <= thresholds[t + 1])],
                    r
                )
            )

    return xc


@jit
def window(i, abs_time, low, high):
    """
    get offsets from the current index i in abs_time that are in the relative
    time window defined by low and high.

    :param int i: current abs_time index.
    :param ndarray abs_time: absolute times.
    :param low: low end of relative window.
    :param high: high end of relative window.
    :return: (low_o, high_o) coincident times are abs_time[low_o:i+high_o]
    """

    length = len(abs_time)
    low_o = 0  # low index offset
    high_o = 0  # high index offset
    now = abs_time[i]
    low_t = now + low
    high_t = now + high

    if low_t < now:
        while (i + low_o >= 0) and abs_time[i + low_o] >= low_t: low_o -= 1;
    else:
        while (low_o + i < length) and abs_time[i + low_o] < low_t: low_o += 1;

    if high_t < now:
        while (i + high_o >= 0) and abs_time[i + high_o] > high_t: high_o -= 1;
    else:
        while (high_o + i < length) and abs_time[i + high_o] <= high_t: high_o += 1;
    return low_o, high_o  # offset indexs marking abs_times in the window


def coincidence(abs_time, mask, low, high):
    """
    Find coincidences between two channels returning indices of partner events.

    :param ndarray abs_time: absolute times.
    :param ndarray mask: ndarray of bool that identifies the channels in
                         abs_time. For abs_times where mask is False, time t is
                         coincident if abs_time+low <= t <= abs_time+high. If
                         the mask is True t is coincident if
                         abs_time-high <= t <= abs_time-low.
    :param float low: the low side of the coincidence window.
    :param float high:  the high side of the coincidence whindw.
    :return: (coinc, coinc_mask) where coinc is a ndarray of indices of the
             coincident event in the other channel. When more than one event is
             found in the window the negated value of the first index is
             entered.
             coinc_mask is a ndarray of bool indicating where exactly one
             event was found in the window.
    """

    coinc = np.zeros(len(abs_time), np.int32)
    coinc_mask = np.zeros(len(abs_time), np.bool)
    for t in range(len(abs_time)):
        if t % 10000000 == 0:
            print(t)
        if not mask[t]:
            low_o, high_o = window(t, abs_time, low, high)
            o = low_o
        else:
            low_o, high_o = window(t, abs_time, -high, -low)
            o = high_o
        coinc_length = high_o - low_o
        if high_o and low_o:
            if coinc_length == 1:
                coinc[t] = t + o
                coinc_mask[t] = True
            elif coinc_length > 1:
                coinc[t] = -(t + o)
                coinc_mask[t] = False
    return coinc, coinc_mask


Counts = namedtuple(
    'Counts', 'count noise vacuum'
)


def count(
        data, measurement_model, vacuum=False, coinc_mask=None, herald_mask=None
):
    """

    :param ndarray data: measurement data.
    :param MixtureModel measurement_model: model created using
                                           expectation_maximisation.
    :param bool vacuum: estimate vacuum terms using coinc_mask and herald_mask.
    :param coinc_mask: ndarray of bool indicating coincidence.
    :param herald_mask: ndarray of bool indicating which events are in the
                        heralding channel.
    :return: Counts(count, noise, vacuum). Where count is an ndarray of counts
             for each photon number
             and len(count)=len(measurement_model.thresholds). noise is a
             ndarray the same shape as count and counts
             events not correlated with the herald. noise is only
             valid when vacuum is True. vacuum replicates the parameter value
             and indicates that count[0] and noise contain valid values.
    """

    if vacuum and (coinc_mask is None or herald_mask is None):
        raise AttributeError(
            'Neither coinc_mask or herald_mask can be None when vacuum=True'
        )

    if herald_mask is not None:
        data_mask = not_l(herald_mask)
    else:
        data_mask = None

    counts = []
    noise = []

    # thresholds
    for t in range(1, len(measurement_model.thresholds)):
        counts.append(
            len(
                and_l(
                    (data > measurement_model.thresholds[t - 1]),
                    (data <= measurement_model.thresholds[t])
                ).nonzero()[0]
            )
        )
        if vacuum:
            noise.append(
                len(
                    and_l(
                        data[not_l(coinc_mask)[data_mask]] >
                        measurement_model.thresholds[t-1],
                        data[not_l(coinc_mask)[data_mask]] <=
                        measurement_model.thresholds[t]
                    ).nonzero()[0]
                )
            )
        else:
            noise.append(0)

    # last threshold overflow
    counts.append(len((data > measurement_model.thresholds[-1]).nonzero()[0]))

    if vacuum:
        noise.append(
            len(
              (
                  data[not_l(coinc_mask)[data_mask]] >
                  measurement_model.thresholds[-1]
              ).nonzero()[0]
            )
        )
    else:
        noise.append(0)

    # counting vacuum
    # uncorrelated heralds + correlated data <= the first threshold
    if vacuum:
        counts[0] = (
                len((not_l(coinc_mask)[herald_mask]).nonzero()[0]) +
                len(
                    (data[coinc_mask[data_mask]] <=
                     measurement_model.thresholds[1]).nonzero()[0]
                )
        )

    return Counts(np.array(counts), np.array(noise), vacuum)


Povm = namedtuple(
    'povm', 'elements vacuum'
)


def povm_elements(measurement_model, counts):
    """
    estimate the elements of the system povm from the measurement model and the
    counting data.

    :param array like data: measurement data.
    :param mixturemodel measurement_model: model created using
                                           expectation_maximisation.
    :param Counts counts: the count data returned by count().
    :return: (elements count vacuum). where counts is a ndarray with
             shape (len(n)) containing the count for each measurement outcome
             in n. elements is a ndarray with
             shape(len(n), len(measurement_model.thresholds)) the first index is
             the fock state, the second is the outcome. The contents of elements
             is the probability of the outcome when measuring fock state.

    :notes: POVM elements is indexed by (fock state, measurement outcome).
            The last threshold in the measurement model marks the boundary of
            the data that was used to create the model, it is *not* altered by
            the expectation maximisation algorithm and is essentially a guess.
            The second last threshold is the boundary of the overflow bin,
            this last photon number bin is counts detections of >=
            len(measurement_model.param_list)-2 photons.
            TODO expand and clarify description.
    """

    num_elements = len(measurement_model.thresholds)
    # cumulative density functions at the thresholds, the last value is not used
    cdf = measurement_model.cdf(measurement_model.thresholds)

    elements = np.zeros((num_elements, num_elements))

    # the majority of the POVM elements are given by the difference in the CDFs
    # at the threshold values
    elements[1:-1, :-1] = cdf[1:, 1:]-cdf[1:, :-1]

    # vacuum terms
    if counts.vacuum:
        elements[0, 0] = (counts.count[0]-counts.noise[1])/counts.count[0]
        elements[0, 1] = counts.noise[1]/counts.count[0]
    else:
        elements[0, 0] = 0
        elements[0, 1] = 0

    # Adjust the first elements, which should not have a proceeding threshold.
    elements[1:-1, 0] = cdf[1:, 1]

    # Adjust the last elements to account for the truncation.
    elements[:, -2] = 1-np.sum(elements[:, :-2], 1)

    # c = np.array(counts.count[:-1])
    # c[-1] += counts.count[-1]

    return Povm(elements, counts.vacuum)


def fit_state_least_squares(
        counts, povm, max_outcome=None, vacuum=True, thermal=True, N=150
):

    def qutip_d_thermal(n, nbar, alpha, N):
        # n [low, high]
        t = qt.thermal_dm(N, nbar)
        D = qt.displace(N, alpha)
        s = D * t * D.dag()

        #         print('n in model',n)
        state = [
                    (qt.states.fock(N, num) *
                     qt.states.fock(N, num).dag() * s).tr()
                    for num in np.arange(0, N)
                ]
        # print('state sum',np.sum(state))

        # print('state.shape', state.shape)
        num_elements = min(povm.shape[1], n[1]-n[0]+1)
        # print('num elements', num_elements, 'N', N)
        truncated_state = np.real(state[n[0]:num_elements + 1])
        truncated_state[-1] += np.real(np.sum(state[num_elements + 1:]))
        # print('nbar', nbar, 'alpha', alpha)
        # print('sum state', np.sum(state[n[0]:]))
        # print('sum t state', np.sum(truncated_state[n[0]:]))
        # print(
        #     'trunc test',
        #     np.real(np.sum(state[n[0]:]))-np.sum(truncated_state[n[0]:])
        # )
        if num_elements < povm.shape[0]:
            truncated_povm = povm[:, n[0]:num_elements]
            truncated_povm[-1] += np.sum(povm[num_elements:])
        else:
            truncated_povm = povm
        outcomes = np.zeros_like(truncated_povm)
        #         print('outcomes',outcomes.shape)
        for e in range(num_elements):
            outcomes[e, :] += truncated_state[:-1] * truncated_povm[e, :]
        out = np.sum(outcomes, 0)
        out[-1] += truncated_state[-1]
        # if n[0] != 0:
        #     out = out / np.sum(out)
        #       print('out',out.shape)
        return out

    n = np.zeros(2, np.uint32)
    if vacuum:
        n[0] = 0
    else:
        n[0] = 1

    if max_outcome is not None:
        n[1] = max_outcome
        c = counts[n[0]:n[1] + 1]
        c[-1] += np.sum(counts[n[1] + 1:])
    else:
        n[1] = povm.shape[0]+1
        c = counts[n[0]:]

    print('fit c shape', c.shape)

    state_model = Model(qutip_d_thermal)
    # alpha = 0.1
    pars = state_model.make_params(nbar=0.1, alpha=0.1, N=N)
    pars['nbar'].set(min=0, max=0.5)
    pars['alpha'].set(min=0, max=10)
    pars['nbar'].set(vary=thermal)
    if not thermal:
        pars['nbar'].set(value=0.0)
    pars['N'].set(vary=False)
    #     print('n',n)
    #     pars['povm_elements'].set(vary=False)
    fit = state_model.fit(c / np.sum(c), pars, n=n)
    print('fit n', n)
    print('fit c', c)
    print('fit sum(c)', np.sum(c))
    print('fit best_fit', fit.best_fit)
    print('fit sum(best_fit)', np.sum(fit.best_fit))
    print('fit best_fit counts', fit.best_fit * np.sum(c))
    return c, n, fit


def _resize_vector(a, max_index, copy=True):
    if max_index+1 > len(a):
        raise AttributeError('max_index must be <= len(a)')
    if max_index+1 == len(a):
        return a
    if copy:
        o = np.copy(a[:max_index+1])
    else:
        o = a[:max_index+1]
    o[-1] += np.sum(a[max_index+1:])
    return o


def displaced_thermal(max_photon_number, nbar, alpha, N=100):
    # n [low, high]
    t = qt.thermal_dm(N, nbar)
    D = qt.displace(N, alpha)
    s = D * t * D.dag()

    state = [
        (qt.states.fock(N, num)*qt.states.fock(N, num).dag()*s).tr()
        for num in np.arange(0, N)
    ]

    truncated_state = _resize_vector(np.real(state), max_photon_number)

    return truncated_state #/np.sum(truncated_state)


def outcome_probabilities(state, povm, max_photon_number):

    if max_photon_number is None:
        max_photon_number = len(povm.elements) - 2
    elif max_photon_number > len(povm.elements) - 2:
            raise RuntimeError(
                'max_photon_number must be at least 2 less than the number of '
                'POVM elements ({})'.format(len(povm.elements) - 2)
            )

    outcomes = np.zeros((max_photon_number+1, max_photon_number+1))

    for i in range(max_photon_number+1):
        outcomes[i, :] = _resize_vector(
            state*povm.elements[i, :], max_photon_number
        )
    return outcomes


def neg_log_likelihood(x, povm, counts, max_photon_number):
    # x = (nbar, alpha)
    # args = (povm, counts, max_photon_number)
    #     povm = args[0]
    #     counts = args[1]
    #     max_photon_number = args[2]
    if max_photon_number is None:
        max_photon_number = len(povm.elements) - 2
    if max_photon_number > len(povm.elements) - 2:
        raise AttributeError(
            'max_photon_number must be <= len(povm.elements)-2'
        )
    c = _resize_vector(counts.count, max_photon_number)
    # print(c, c.shape)
    state = displaced_thermal(len(povm.elements) - 1, *x, N=100)
    # print(state, state.shape)
    outcomes = outcome_probabilities(state, povm, max_photon_number)
    # print(np.sum(outcomes, 0), outcomes.shape)

    if len(np.sum(outcomes, 0).nonzero()[0]) != outcomes.shape[1]:
        ll = -np.inf
    else:
        ll = np.sum(np.log(np.sum(outcomes, 0))*c)
    return -ll


def neg_log_likelihood2(x, povm, counts, max_photon_number):
    # x = (nbar, alpha)
    # args = (povm, counts, max_photon_number)
    #     povm = args[0]
    #     counts = args[1]
    #     max_photon_number = args[2]
    if max_photon_number is None:
        max_photon_number = len(povm.elements) - 2
    if max_photon_number > len(povm.elements) - 2:
        raise AttributeError(
            'max_photon_number must be <= len(povm.elements)-2'
        )
    c = _resize_vector(counts.count, max_photon_number)
    # print(c, c.shape)
    state = displaced_thermal(max_photon_number, *x, N=100)
    # print(state, state.shape)
    # outcomes = outcome_probabilities(state, povm, max_photon_number)
    # print(np.sum(outcomes, 0), outcomes.shape)

    if len(state.nonzero()[0]) != len(state):
        print('inf')
        ll = -np.inf
    else:
        ll = np.sum(np.log(state)*c)
    return -ll


def plot_state_fit(
        max_photon_number, nbar, alpha, counts, povm=None, thermal=True,
        figsize=None
):
    width = 0.46   # the width of the bars
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()

    ax = fig.add_axes([0.12, 0.12, 0.84, 0.83])

    if max_photon_number is None:
        max_photon_number = povm.elements.shape[1]-2

    state = displaced_thermal(povm.elements.shape[1]-1, nbar, alpha, N=150)
    outcomes = outcome_probabilities(state, povm, max_photon_number)

    # model_counts = np.zeros_like(outcomes)
    # for i in range(model_counts.shape[0]):
    #     model_counts[i, :] = outcomes[i, :]*counts.count[i]
    model_counts = _resize_vector(
        np.sum(outcomes*sum(counts.count), 0), max_photon_number
    )


    # print('model_count', model_counts.shape, model_counts)
    c = _resize_vector(counts.count, max_photon_number)
    # print(c.shape, c)
    fock = np.arange(0, max_photon_number+1)
    # print(fock.shape, fock)

    ax.bar(fock-width/2, c, width=width, color='r', label='data')
    ax.bar(
        fock+width/2, model_counts, width=width, color='b', label='model'
    )
    # chi2 = sum(np.power(c - model_counts, 2) / model_counts)
    ax.set_ylabel('count')
    ax.set_xlabel('Fock state')
    ax.set_xticks(fock)
    xlabels = [r'$\vert {} \rangle$'.format(f) for f in fock]
    xlabels[-1] = r'$\vert {}\rangle{}$'.format(fock[-1], '+')
    ax.set_xticklabels(xlabels)

    ax.legend(frameon=False)
    # plt.plot(n,nc,'.')
    # plt.plot(n,r.best_fit,'o',markerfacecolor='none')
    # ax.set_ylim(0,11000000)
    ty = 0.8
    tx = 0.01
    ls = 0.05
    ax.text(tx, ty, r'Displaced thermal state', transform=ax.transAxes)
    # ax.text(
    #     tx, ty - ls, r'$\alpha={:.4f}\pm{:.4f}$'
    #         .format(
    #         state_fit.params['alpha'].value, state_fit.params['alpha'].stderr
    #     ),
    #     transform=ax.transAxes, horizontalalignment='left'
    # )
    # ax.text(
    #     tx, ty - 2 * ls, r'${}={:.4f}\pm{:.4f}$'
    #         .format(
    #         r'\bar{n}', state_fit.params['nbar'].value,
    #         state_fit.params['nbar'].stderr
    #     ),
    #     transform=ax.transAxes, horizontalalignment='left'
    # )
    # ax.text(
    #     tx, ty - 3 * ls, r'$\chi^2/{}={:.3f}$'
    #         .format(
    #         state_fit.nfree - 1, chi2 / (state_fit.nfree - 1)
    #     ),
    #     transform=ax.transAxes, horizontalalignment='left'
    # )
    ax.legend(frameon=False)
    return fig
