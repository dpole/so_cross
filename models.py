import numpy as np

def power_law(freq, freq0, beta, lmin, lmax, ell0, alpha, amplitude):
    """ Cross spectrum of a power law in frequency and multipoles

    Parameters
    ----------
    freq: array
        Frequencies at which the cross is evaluated. Shape (freq)
    freq0: float
        Reference frequency of the SED
    beta: float or array
        Spectral index of the SED. If array, the shape is (TEB)
    lmin: float
        Minimum multipole
    lmax: float
        Maximum multipole
    ell0: float
        Reference multipole
    alpha: float or array
        Power of the scaling in multipoles. If array, shape is `(TEB)`.
        Note that the scaling of the e.g., `TE` power spectrum will be
        `(alpha_T + alpha_E) / 2`
    amplitude: float or array
        Amplitude of the power spectra at `ell0` and `freq0`. 
        Shape is `([TEB,] TEB)`. If only one dimension is provided it is assumed
        to be the diagonal

    Returns
    -------
    cross: ndarray
        Shape is `([TEB, TEB,] freq, freq, ell)`, `TEB` dimensions are present
        only if they were present in any of the inputs. If some of the inputs
        didn't provide the dimension that input is considered constant across
        the TEB dimension.

    """

    ells = np.arange(lmin, lmax+1, 1.) / ell0

    try:
        sed = (freq / freq0)**beta[:, np.newaxis]  # (TEB, freq)
        sed_t = sed[:, np.newaxis, :, np.newaxis, np.newaxis]  # (TEB, 1, freq, 1, 1)
        sed = sed[:, np.newaxis, :, np.newaxis]  #  (TEB, 1, freq, 1)
    except TypeError:
        sed = (freq / freq0)**beta  # (freq)
        sed = sed[:, np.newaxis]  # (freq, 1)
        sed_t = sed[:, np.newaxis]  # (freq, 1, 1)

    try:
        alpha = (alpha + alpha[:, np.newaxis]) / 2.  # (TEB, TEB)
        alpha = alpha[..., np.newaxis, np.newaxis, np.newaxis]
    except TypeError:
        pass

    try:
        if amplitude.ndim == 1:
            amplitude = np.diag(amplitude)
        amplitude = amplitude[:, :, np.newaxis, np.newaxis, np.newaxis]
    except AttributeError:
        pass

    return amplitude * ells**alpha * sed * sed_t
