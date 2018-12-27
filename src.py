import numpy as np
import scipy
import scipy.ndimage
import os
import sys
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.fftpack as fft
import scipy.signal
import scipy.interpolate
import six
import math
from scipy.io import wavfile
import inspect
import warnings

##################################################################################################
#DONE
class Deprecated(object):
    '''A dummy class to catch usage of deprecated variable names'''
    def __repr__(self):
        return '<DEPRECATED parameter>'


def rename_kw(old_name, old_value, new_name, new_value, version_deprecated, version_removed):
    '''Handle renamed arguments.
    '''
    if isinstance(old_value, Deprecated):
        return new_value
    else:
        stack = inspect.stack()
        dep_func = stack[1]
        caller = stack[2]

        warnings.warn_explicit("{:s}() keyword argument '{:s}' has been renamed to '{:s}' in "
                               "version {:}."
                               "\n\tThis alias will be removed in version "
                               "{:}.".format(dep_func[3],
                                             old_name, new_name,
                                             version_deprecated, version_removed),
                               category=DeprecationWarning,
                               filename=caller[1],
                               lineno=caller[2])

        return old_value


##################################################################################################
class ParameterError(Exception):
    '''Exception class for mal-formed inputs'''
    pass
##################################################################################################
def onset_strength(y=None, sr=22050, S=None, lag=1, max_size=1,
                   detrend=False, center=True,
                   feature=None, aggregate=None,
                   centering=None, cached_spec=None, y_start=None,
                   **kwargs):
    """Compute a spectral flux onset strength envelope.


    .. [1] Bck, Sebastian, and Gerhard Widmer.
           "Maximum filter vibrato suppression for onset detection."
           16th International Conference on Digital Audio Effects,
           Maynooth, Ireland. 2013.

    """

    odf_all,SS = onset_strength_multi(y=y,
                                   sr=sr,
                                   S=S,
                                   lag=lag,
                                   max_size=max_size,
                                   detrend=detrend,
                                   center=center,
                                   feature=feature,
                                   aggregate=aggregate,
                                   channels=None, cached_spec=cached_spec, y_start=y_start,
                                   **kwargs)

    return odf_all[0],SS
##################################################################################################

def onset_strength_multi(y=None, sr=22050, S=None, lag=1, max_size=1,
                         detrend=False, center=True, feature=None,
                         aggregate=None, channels=None, cached_spec=None, y_start=None, **kwargs):
    """Compute a spectral flux onset strength envelope across multiple channels.

    """

    if feature is None:
        feature = melspectrogram
        kwargs.setdefault('fmax', 22050.0)

    if aggregate is None:
        aggregate = np.mean

    if lag < 1 or not isinstance(lag, int):
        raise ParameterError('lag must be a positive integer')

    if max_size < 1 or not isinstance(max_size, int):
        raise ParameterError('max_size must be a positive integer')

    # First, compute mel spectrogram
    if S is None:
        S,SS = np.abs(feature(y=y, sr=sr, cached_spec=cached_spec, y_start=y_start, **kwargs))
	
        # Convert to dBs
        S = power_to_db(S)
    # Retrieve the n_fft and hop_length,
    # or default values for onsets if not provided
    n_fft = kwargs.get('n_fft', 2048)
    hop_length = kwargs.get('hop_length', 512)


    # Ensure that S is at least 2-d
    S = np.atleast_2d(S)

    # Compute the reference spectrogram.
    # Efficiency hack: skip filtering step and pass by reference
    # if max_size will produce a no-op.
    if max_size == 1:
        ref_spec = S
    else:
        ref_spec = scipy.ndimage.maximum_filter1d(S, max_size, axis=0)

    # Compute difference to the reference, spaced by lag
    onset_env = S[:, lag:] - ref_spec[:, :-lag]

    # Discard negatives (decreasing amplitude)
    onset_env = np.maximum(0.0, onset_env)

    # Aggregate within channels
    pad = True
    if channels is None:
        channels = [slice(None)]
    else:
        pad = False

    onset_env = sync(onset_env, channels,
                          aggregate=aggregate,
                          pad=pad,
                          axis=0)

    # compensate for lag
    pad_width = lag
    if center:
        # Counter-act framing effects. Shift the onsets by n_fft / hop_length
        pad_width += n_fft // (2 * hop_length)

    onset_env = np.pad(onset_env, ([0, 0], [int(pad_width), 0]),
                       mode='constant')

    # remove the DC component
    if detrend:
        onset_env = scipy.signal.lfilter([1.0, -1.0], [1.0, -0.99],
                                         onset_env, axis=-1)

    # Trim to match the input duration
    if center:
        onset_env = onset_env[:, :S.shape[1]]

    return onset_env, SS
##################################################################################################
def get_window(window, Nx, fftbins=True):
    '''Compute a window function.
    '''
    if six.callable(window):
        return window(Nx)

    elif (isinstance(window, (six.string_types, tuple)) or
          np.isscalar(window)):
        # TODO: if we add custom window functions in librosa, call them here

        return scipy.signal.get_window(window, Nx, fftbins=fftbins)

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)

        raise ParameterError('Window size mismatch: '
                             '{:d} != {:d}'.format(len(window), Nx))
    else:
        raise ParameterError('Invalid window specification: {}'.format(window))
##################################################################################################
def melspectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, 
                   power=2.0,cached_spec=None,y_start=None, **kwargs):
    """Compute a Mel-scaled spectrogram.
    """

    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length,
                            power=power,cached_spec=cached_spec,y_start=y_start)

    # Build a Mel filter
    mel_basis = mel(sr, n_fft, **kwargs)

    return np.dot(mel_basis, S), S
    
#############################################################################################
def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=True):
    """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins
    """

    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)))

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    freqs = mel_frequencies(n_mels + 2,
                            fmin=fmin,
                            fmax=fmax,
                            htk=htk)

    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (freqs[2:n_mels+2] - freqs[:n_mels])

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = (fftfreqs - freqs[i]) / (freqs[i+1] - freqs[i])
        upper = (freqs[i+2] - fftfreqs) / (freqs[i+2] - freqs[i+1])

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper)) * enorm[i]

    return weights

#######################################################   
def mel_to_hz(mels, htk=False):
    """Convert mel bin numbers to frequencies
    """

    mels = np.atleast_1d(mels)

    if htk:
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region
    log_t = (mels >= min_log_mel)

    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

    return freqs
#######################################################   
def hz_to_mel(frequencies, htk=False):
    """Convert Hz to Mels
    """

    frequencies = np.atleast_1d(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    log_t = (frequencies >= min_log_hz)
    mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep

    return mels



#############################################################################################
def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    """Compute the center frequencies of mel bands.
    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk=htk)
    
#############################################################################################
def fft_frequencies(sr=22050, n_fft=2048):
    '''Alternative implementation of `np.fft.fftfreqs`
    '''

    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft//2),
                       endpoint=True)

#############################################################################################
def _spectrogram(y=None, S=None, n_fft=2048, hop_length=512, power=1,cached_spec=None,y_start=None):
    '''Helper function to retrieve a magnitude spectrogram.

    This is primarily used in feature extraction functions that can operate on
    either audio time-series or spectrogram input.

    '''        
    if S is not None:
        # Infer n_fft from spectrogram shape
        n_fft = 2 * (S.shape[0] - 1)
    else:
        # Otherwise, compute a magnitude spectrogram from input
        if cached_spec is not None:
            lagg=int(round(n_fft/(hop_length)))
            y_start2=int(y_start//hop_length)+1
            y_start3=int((y_start2-lagg*2)*hop_length)
            y2=y[y_start3:]
            S2 = np.abs(stft(y2, n_fft=n_fft, hop_length=hop_length))**power
            S=np.concatenate((cached_spec[:,:-1*lagg+1],S2[:,lagg+1:]),axis=1)
        else:
            S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length))**power

    return S, n_fft
########################################################################################
def stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann',
         center=True, dtype=np.complex64):
    """Short-time Fourier transform (STFT)

    Returns a complex-valued matrix D such that
        `np.abs(D[f, t])` is the magnitude of frequency bin `f`
        at frame `t`

        `np.angle(D[f, t])` is the phase of frequency bin `f`
        at frame `t`
    """

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Pad the time series so that frames are centered
    if center:
        valid_audio(y)
        y = np.pad(y, int(n_fft // 2), mode='reflect')

    # Window the time series.
    y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype,
                           order='F')
# Constrain STFT block sizes to 256 KB
    MAX_MEM_BLOCK = 2**8 * 2**10
    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                          stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        # RFFT and Conjugate here to match phase from DPWE code
        stft_matrix[:, bl_s:bl_t] = fft.fft(fft_window *
                                            y_frames[:, bl_s:bl_t],
                                            axis=0)[:stft_matrix.shape[0]].conj()

    return stft_matrix
#######################################################################################
def pad_center(data, size, axis=-1, **kwargs):
    '''Wrapper for np.pad to automatically center an array prior to padding.
    This is analogous to `str.center()`
    '''

    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ParameterError(('Target size ({:d}) must be '
                              'at least input size ({:d})').format(size, n))

    return np.pad(data, lengths, **kwargs)


############################################
def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0, ref_power=Deprecated()):
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units

    This computes the scaling ``10 * log10(S / ref)`` in a numerically
    stable way.
    """

    if amin <= 0:
        raise ParameterError('amin must be strictly positive')

    magnitude = np.abs(S)

    ref = rename_kw('ref_power', ref_power,
                    'ref', ref,
                    '0.5', '0.6')

    if six.callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError('top_db must be non-negative')
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


logamplitude = power_to_db
###########################################################################
def normalize(S, norm=np.inf, axis=0, threshold=None, fill=None):
    '''Normalize an array along a chosen axis.
    '''

    # Avoid div-by-zero
    if threshold is None:
        threshold = tiny(S)

    elif threshold <= 0:
        raise ParameterError('threshold={} must be strictly '
                             'positive'.format(threshold))

    if fill not in [None, False, True]:
        raise ParameterError('fill={} must be None or boolean'.format(fill))

    if not np.all(np.isfinite(S)):
        raise ParameterError('Input must be finite')

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S).astype(np.float)

    # For max/min norms, filling with 1 works
    fill_norm = 1

    if norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        if fill is True:
            raise ParameterError('Cannot normalize with norm=0 and fill=True')

        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag**norm, axis=axis, keepdims=True)**(1./norm)

        if axis is None:
            fill_norm = mag.size**(-1./norm)
        else:
            fill_norm = mag.shape[axis]**(-1./norm)

    elif norm is None:
        return S

    else:
        raise ParameterError('Unsupported norm: {}'.format(repr(norm)))

    # indices where norm is below the threshold
    small_idx = length < threshold

    if fill is None:
        # Leave small indices un-normalized
        length[small_idx] = 1.0
        Snorm = S / length

    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length[small_idx] = np.nan
        Snorm = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length[small_idx] = np.inf
        Snorm = S / length

    return Snorm


########################################################################
def frame(y, frame_length=2048, hop_length=512):
    '''Slice a time series into overlapping frames.
    '''

    if len(y) < frame_length:
        raise ParameterError('Buffer is too short (n={:d})'
                             ' for frame_length={:d}'.format(len(y), frame_length))

    if hop_length < 1:
        raise ParameterError('Invalid hop_length: {:d}'.format(hop_length))

    if not y.flags['C_CONTIGUOUS']:
        raise ParameterError('Input buffer must be contiguous.')

    valid_audio(y)

    # Compute the number of frames that will fit. The end may get truncated.
    n_frames = 1 + int((len(y) - frame_length) / hop_length)

    # Vertical stride is one sample
    # Horizontal stride is `hop_length` samples
    y_frames = as_strided(y, shape=(frame_length, n_frames),
                          strides=(y.itemsize, hop_length * y.itemsize))
    return y_frames
####################################################################
def valid_audio(y, mono=True):
    '''Validate whether a variable contains valid, mono audio data.
    '''

    if not isinstance(y, np.ndarray):
        raise ParameterError('data must be of type numpy.ndarray')

    if mono and y.ndim != 1:
        raise ParameterError('Invalid shape for monophonic audio: '
                             'ndim={:d}, shape={}'.format(y.ndim, y.shape))
    elif y.ndim > 2:
        raise ParameterError('Invalid shape for audio: '
                             'ndim={:d}, shape={}'.format(y.ndim, y.shape))

    if not np.isfinite(y).all():
        raise ParameterError('Audio buffer is not finite everywhere')

    return True
################################################################################
def sync(data, idx, aggregate=None, pad=True, axis=-1):
    """Synchronous aggregation of a multi-dimensional array between boundaries
    """

    if aggregate is None:
        aggregate = np.mean

    shape = list(data.shape)

    if np.all([isinstance(_, slice) for _ in idx]):
        slices = idx
    elif np.all([np.issubdtype(type(_), np.int) for _ in idx]):
        slices = index_to_slice(np.asarray(idx), 0, shape[axis], pad=pad)
    else:
        raise ParameterError('Invalid index set: {}'.format(idx))

    agg_shape = list(shape)
    agg_shape[axis] = len(slices)

    data_agg = np.empty(agg_shape, order='F' if np.isfortran(data) else 'C', dtype=data.dtype)

    idx_in = [slice(None)] * data.ndim
    idx_agg = [slice(None)] * data_agg.ndim

    for (i, segment) in enumerate(slices):
        idx_in[axis] = segment
        idx_agg[axis] = i
        data_agg[idx_agg] = aggregate(data[idx_in], axis=axis)

    return data_agg
###############################################################    
def index_to_slice(idx, idx_min=None, idx_max=None, step=None, pad=True):
    '''Generate a slice array from an index array.
    '''

    # First, normalize the index set
    idx_fixed = fix_frames(idx, idx_min, idx_max, pad=pad)

    # Now convert the indices to slices
    return [slice(start, end, step) for (start, end) in zip(idx_fixed, idx_fixed[1:])]
###############################################################################################
def fix_frames(frames, x_min=0, x_max=None, pad=True):
    '''Fix a list of frames to lie within [x_min, x_max]
    '''

    frames = np.asarray(frames)

    if np.any(frames < 0):
        raise ParameterError('Negative frame index detected')

    if pad and (x_min is not None or x_max is not None):
        frames = np.clip(frames, x_min, x_max)

    if pad:
        pad_data = []
        if x_min is not None:
            pad_data.append(x_min)
        if x_max is not None:
            pad_data.append(x_max)
        frames = np.concatenate((pad_data, frames))

    if x_min is not None:
        frames = frames[frames >= x_min]

    if x_max is not None:
        frames = frames[frames <= x_max]

    return np.unique(frames).astype(int)
##############################################################################
def tiny(x):
    '''Compute the tiny-value corresponding to an input's data type.
    '''

    # Make sure we have an array view
    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, float) or np.issubdtype(x.dtype, complex):
        dtype = x.dtype
    else:
        dtype = np.float32

    return np.finfo(dtype).tiny

###############################################################################

def beat_track(y=None, sr=22050, onset_envelope=None, hop_length=512,
               start_bpm=120.0, tightness=100, trim=True, bpm=None, cached_spec=None, y_start=None,
               units='frames'):
    r'''Dynamic programming beat tracker.

    Beats are detected in three stages, following the method of [1]_:
      1. Measure onset strength
      2. Estimate tempo from onset correlation
      3. Pick peaks in onset strength approximately consistent with estimated
         tempo

    .. [1] Ellis, Daniel PW. "Beat tracking by dynamic programming."
           Journal of New Music Research 36.1 (2007): 51-60.
           http://labrosa.ee.columbia.edu/projects/beattrack/
    '''

    # First, get the frame->beat strength profile if we don't already have one
    if onset_envelope is None:
        if y is None:
            raise ParameterError('y or onset_envelope must be provided')

        onset_envelope,SS = onset_strength(y=y,
                                              sr=sr,
                                              hop_length=hop_length,
                                              aggregate=np.mean,center=False)
    else:
        SS=np.arange(1,2)
    # Do we have any onsets to grab?
    if not onset_envelope.any():
        return (0, np.array([], dtype=int))

    # Estimate BPM if one was not provided
    if bpm is None:
        bpm = estimate_tempo(onset_envelope,
                             sr=sr,
                             hop_length=hop_length,
                             start_bpm=start_bpm)

    # Then, run the tracker
    beats = __beat_tracker(onset_envelope,
                           bpm,
                           float(sr) / hop_length,
                           tightness,
                           trim)

    if units == 'frames':
        pass
    elif units == 'samples':
        beats = frames_to_samples(beats, hop_length=hop_length)
    elif units == 'time':
        beats = frames_to_time(beats, hop_length=hop_length, sr=sr)
    else:
        raise ParameterError('Invalid unit type: {}'.format(units))

    return bpm, beats,SS

###############################################################################
def frames_to_samples(frames, hop_length=512, n_fft=None):
    """Converts frame indices to audio sample indices
    """

    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    return (np.atleast_1d(frames) * hop_length + offset).astype(int)

###############################################################################
def frames_to_time(frames, sr=22050, hop_length=512, n_fft=None):
    """Converts frame counts to time (seconds)
    """

    samples = frames_to_samples(frames,
                                hop_length=hop_length,
                                n_fft=n_fft)

    return samples_to_time(samples, sr=sr)
###############################################################################
def samples_to_time(samples, sr=22050):
    '''Convert sample indices to time (in seconds).
    '''

    return np.atleast_1d(samples) / float(sr)


###############################################################################
def localmax(x, axis=0):
    """Find local maxima in an array `x`.
    """

    paddings = [(0, 0)] * x.ndim
    paddings[axis] = (1, 1)

    x_pad = np.pad(x, paddings, mode='edge')

    inds1 = [slice(None)] * x.ndim
    inds1[axis] = slice(0, -2)

    inds2 = [slice(None)] * x.ndim
    inds2[axis] = slice(2, x_pad.shape[axis])

    return (x > x_pad[inds1]) & (x >= x_pad[inds2])
###############################################################################
def tempo_frequencies(n_bins, hop_length=512, sr=22050):
    '''Compute the frequencies (in beats-per-minute) corresponding
    to an onset auto-correlation or tempogram matrix.
    '''

    bin_frequencies = np.zeros(int(n_bins), dtype=np.float)

    bin_frequencies[0] = np.inf
    bin_frequencies[1:] = 60.0 * sr / (hop_length * np.arange(1.0, n_bins))

    return bin_frequencies
###############################################################################

def autocorrelate(y, max_size=None, axis=-1):
    """Bounded auto-correlation
    """

    if max_size is None:
        max_size = y.shape[axis]

    max_size = int(min(max_size, y.shape[axis]))

    # Compute the power spectrum along the chosen axis
    # Pad out the signal to support full-length auto-correlation.
    powspec = np.abs(fft.fft(y, n=2 * y.shape[axis] + 1, axis=axis))**2

    # Convert back to time domain
    autocorr = fft.ifft(powspec, axis=axis, overwrite_x=True)

    # Slice down to max_size
    subslice = [slice(None)] * autocorr.ndim
    subslice[axis] = slice(max_size)

    autocorr = autocorr[subslice]

    if not np.iscomplexobj(y):
        autocorr = autocorr.real

    return autocorr

###############################################################################


def estimate_tempo(onset_envelope, sr=22050, hop_length=512, start_bpm=120,
                   std_bpm=1.0, ac_size=2.0, duration=17.0, offset=0.0):
    """Estimate the tempo (beats per minute) from an onset envelope
    """

    if start_bpm <= 0:
        raise ParameterError('start_bpm must be strictly positive')

    fft_res = float(sr) / hop_length

    # Chop onsets to X[(upper_limit - duration):upper_limit]
    # or as much as will fit
    maxcol = int(min(len(onset_envelope)-1,
                     np.round((offset + duration) * fft_res)))

    mincol = int(max(0, maxcol - np.round(duration * fft_res)))

    # Use auto-correlation out of 4 seconds (empirically set??)
    ac_window = min(maxcol, np.round(ac_size * fft_res))

    # Compute the autocorrelation
    x_corr = autocorrelate(onset_envelope[mincol:maxcol], ac_window)[1:]

    # Get the BPM values for each bin, skipping the 0-lag bin
    bpms = tempo_frequencies(ac_window, hop_length=hop_length, sr=sr)[1:]

    # Weight the autocorrelation by a log-normal distribution centered start_bpm
    x_corr *= np.exp(-0.5 * ((np.log2(bpms) - np.log2(start_bpm)) / std_bpm)**2)

    # Get the local maximum of weighted correlation
    x_peaks = localmax(x_corr)

    # Zero out all peaks before the first negative
    x_peaks[:np.argmax(x_corr < 0)] = False

    # Choose the best peak out of .33, .5, 2, 3 * start_period
    candidates = np.argmax(x_peaks * x_corr) * np.asarray([1./3, 0.5, 1, 2, 3])

    candidates = candidates[candidates < ac_window].astype(int)

    best_period = np.argmax(x_corr[candidates])

    if candidates[best_period] > 0:
        return bpms[candidates[best_period]]

    return start_bpm
###############################################################################

def __beat_tracker(onset_envelope, bpm, fft_res, tightness, trim):
    """Internal function that tracks beats in an onset strength envelope.
    """

    if bpm <= 0:
        raise ParameterError('bpm must be strictly positive')
	
    # convert bpm to a sample period for searching
    period = round(60.0 * fft_res / bpm)

    # localscore is a smoothed version of AGC'd onset envelope
    localscore = __beat_local_score(onset_envelope, period)

    # run the DP
    backlink, cumscore = __beat_track_dp(localscore, period, tightness)

    # get the position of the last beat
    beats = [__last_beat(cumscore)]

    # Reconstruct the beat path from backlinks
    while backlink[beats[-1]] >= 0:
        beats.append(backlink[beats[-1]])

    # Put the beats in ascending order
    # Convert into an array of frame numbers
    beats = np.array(beats[::-1], dtype=int)

    # Discard spurious trailing beats
    beats = __trim_beats(localscore, beats, trim)

    return beats
###############################################################################

# -- Helper functions for beat tracking
def __normalize_onsets(onsets):
    '''Maps onset strength function into the range [0, 1]'''

    norm = onsets.std(ddof=1)
    if norm > 0:
        onsets = onsets / norm
    return onsets
###############################################################################

def __beat_local_score(onset_envelope, period):
    '''Construct the local score for an onset envlope and given period'''

    window = np.exp(-0.5 * (np.arange(-period, period+1)*32.0/period)**2)
    return scipy.signal.convolve(__normalize_onsets(onset_envelope),
                                 window,
                                 'same')
###############################################################################

def __beat_track_dp(localscore, period, tightness):
    """Core dynamic program for beat tracking"""

    backlink = np.zeros_like(localscore, dtype=int)
    cumscore = np.zeros_like(localscore)

    # Search range for previous beat
    window = np.arange(-2 * period, -np.round(period / 2) + 1, dtype=int)

    # Make a score window, which begins biased toward start_bpm and skewed
    if tightness <= 0:
        raise ParameterError('tightness must be strictly positive')

    txwt = -tightness * (np.log(-window / period) ** 2)

    # Are we on the first beat?
    first_beat = True
    for i, score_i in enumerate(localscore):

        # Are we reaching back before time 0?
        z_pad = np.maximum(0, min(- window[0], len(window)))

        # Search over all possible predecessors
        candidates = txwt.copy()
        candidates[z_pad:] = candidates[z_pad:] + cumscore[window[z_pad:]]

        # Find the best preceding beat
        beat_location = np.argmax(candidates)

        # Add the local score
        cumscore[i] = score_i + candidates[beat_location]

        # Special case the first onset.  Stop if the localscore is small
        if first_beat and score_i < 0.01 * localscore.max():
            backlink[i] = -1
        else:
            backlink[i] = window[beat_location]
            first_beat = False

        # Update the time range
        window = window + 1

    return backlink, cumscore
###############################################################################

def __last_beat(cumscore):
    """Get the last beat from the cumulative score array"""

    maxes = localmax(cumscore)
    med_score = np.median(cumscore[np.argwhere(maxes)])

    # The last of these is the last beat (since score generally increases)
    return np.argwhere((cumscore * maxes * 2 > med_score)).max()
###############################################################################

def __trim_beats(localscore, beats, trim):
    """Final post-processing: throw out spurious leading/trailing beats"""

    smooth_boe = scipy.signal.convolve(localscore[beats],
                                       scipy.signal.hann(5),
                                       'same')

    if trim:
        threshold = 0.5 * ((smooth_boe**2).mean()**0.5)
    else:
        threshold = 0.0

    valid = np.argwhere(smooth_boe > threshold)

    return beats[valid.min():valid.max()]
###############################################################################
