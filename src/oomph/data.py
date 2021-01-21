from . import PACKAGEDIR
import pandas as pd
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
from tqdm import tqdm

from skimage.transform import hough_line, hough_line_peaks

from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.convolution import convolve, Box1DKernel
from astropy.constants import h, c, k_B, sigma_sb
import astropy.units as u

import warnings

import pymc3 as pm
import theano.tensor as tt
import exoplanet as xo
import corner


_bandpass = pd.read_csv(PACKAGEDIR + '/data/bandpass.dat', header=None, delimiter=' ')
_wav = np.arange(300, 1100, 0.1)*u.nm
_bandpass = np.interp(_wav.to(u.angstrom).value, np.asarray(_bandpass[0]), np.asarray(_bandpass[1]))

def _bb(T, wav=_wav):
    """Calculate blackbody"""
    a = (2 * h * c**2/wav.to(u.m)**5)
    b = np.expm1((h * c)/(wav.to(u.m) * k_B * T))
    return a/b


def _flare_time_series(t, te=0.01, tg=0.01):
    """Model flare"""
    tmask = t < 0
    return np.exp(-(t**2)/(2*tg**2)) * tmask + np.exp(-(t**2)/(te)) * ~tmask


class CoRoTMultiPhotometry(object):
    def __init__(self, id, teff, hdu):
        """ Class to handle CoRoT multi band photometry

        Use `from_archive` to download a source

        Parameters
        ----------
        id : int
            CoRoT ID
        teff : float
            Effective temperature of target (e.g. use Gaia)
        hdu : astropy.io.fits
            Fits object with data
        """
        self.id = id
        self.teff = teff
        self.hdu = hdu
        self._open()
        self.r_g_mean = -2.5 * np.log10(np.median(self.rlc.flux.value/self.glc.flux.value))
        self.b_g_mean = -2.5 * np.log10(np.median(self.blc.flux.value/self.glc.flux.value))

    @staticmethod
    def from_archive(id, teff):
        """Downloads a CoRoT source from the archive.

        Parameters
        ----------
        id : int
            CoRoT ID
        teff : float
            Effective temperature of target (e.g. use Gaia)
        """
        def _get_url(id):
            df = pd.read_csv(PACKAGEDIR + '/data/catalog.csv')
            loc = np.where(df.CoRoT == id)[0]
            if len(loc) == 0:
                raise ValueError(f'No target {id} found with color photometry')
            id, year, month, day, FileName_a, FileName_b, FileName_c = np.asarray(df.loc[loc[0]])
            url = f'http://idoc-corot.ias.u-psud.fr/sitools/datastorage/user/corotstorage/N2-4.4/{year}/{month:02}/{day:02}/EN2_STAR_CHR_{id:010}_{year}{month:02}{day:02}T{FileName_a:06}_{FileName_b:06}T{FileName_c:06}.fits'
            return url
        hdu = fits.open(_get_url(id))
        return CoRoTMultiPhotometry(id=id, teff=teff, hdu=hdu)

    def __repr__(self):
        return f'CoRoT {self.id}'

    def _open(self):
        t = self.hdu[1].data['DATETT']
        b = self.hdu[1].data['BLUEFLUX']
        g = self.hdu[1].data['GREENFLUX']
        r = self.hdu[1].data['REDFLUX']
        w = self.hdu[1].data['WHITEFLUX']
        be = self.hdu[1].data['BLUEFLUXDEV']
        ge = self.hdu[1].data['GREENFLUXDEV']
        re = self.hdu[1].data['REDFLUXDEV']

        # Removes any significant outliers
        k = re/r > 0.5
        k |= be/b > 0.5
        k |= ge/g > 0.5
        k |= sigma_clip(np.ma.masked_array(np.gradient(w, t), k), sigma=5).mask

        k = ~k
        k = (convolve(k, Box1DKernel(5)) == 1)

        self.rlc = lk.LightCurve(time=t[k], flux=r[k], flux_err=r[k]**0.5)
        self.glc = lk.LightCurve(time=t[k], flux=g[k], flux_err=g[k]**0.5)
        self.blc = lk.LightCurve(time=t[k], flux=b[k], flux_err=b[k]**0.5)
        self.lc = self.rlc + self.glc + self.blc


    def calibrate(self, p1_lim=(495, 505), p2_lim=(550, 560), p1_num=200, p2_num=201, plot=True):
        """Calibrates the CoRoT multiband photometry, based on the input effective temperature value.

        Parameters
        ----------
        p1_lim: tuple of ints
            Limits to search for the blue edge of the green band bandpass
        p2_lim: tuple of ints
            Limits to search for the red edge of the green band bandpass
        p1_num: int
            Number of points to search for the blue edge of the green band pass
        p2_num: int
            Number of points to search for the red edge of the green band pass
        plot: bool
            Whether or not to plot the data

        Returns
        -------
        fig: matplotlib.pyplot.figure
            If plot is `True`, will return a figure, else will return None
        """
        spec = _bb(self.teff*u.K) * _bandpass
        def func(p1=507, p2=585, show=False):
            bmask = (_wav.to(u.nm).value > 300) & (_wav.to(u.nm).value < p1)
            gmask = (_wav.to(u.nm).value > p1) & (_wav.to(u.nm).value < p2)
            rmask = (_wav.to(u.nm).value > p2) & (_wav.to(u.nm).value < 11000)

            b = np.trapz(spec[bmask].value, _wav[bmask].value)
            g = np.trapz(spec[gmask].value, _wav[gmask].value)
            r = np.trapz(spec[rmask].value, _wav[rmask].value)
            r_g = -2.5 * np.log10(r/g)
            b_g = -2.5 * np.log10(b/g)

            chi = (self.r_g_mean - r_g)**2 * (self.b_g_mean - b_g)**2
            return chi

        p1s = np.linspace(p1_lim[0], p1_lim[1], p1_num)
        p2s = np.linspace(p2_lim[0], p2_lim[1], p2_num)
        chi = np.zeros((p1_num, p2_num))
        for idx, p1 in tqdm(enumerate(p1s), total=p1_num, desc='Calibrating'):
            for jdx, p2 in enumerate(p2s):
                chi[idx, jdx] = func(p1, p2)


        if plot:
            with plt.style.context('seaborn-white'):
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                im = ax[0].pcolormesh(p1s, p2s, np.log10(chi).T, cmap='viridis')
                cbar = plt.colorbar(im, ax=ax[0])
                ax[0].set_xlabel('Point 1 [nm]')
                ax[0].set_ylabel('Point 2 [nm]')
                cbar.set_label('log$_{10}$ Chi')


        # Hough Transform finds two lines
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        h, theta, d = hough_line((np.log10(chi).T < -6).astype(float), theta=tested_angles)
        origin = np.array((0, chi.shape[1]))
        ls = []
        tupes = []
        idx = 0
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
            if idx >= 2:
                continue
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            tupes.append(np.vstack([np.asarray([origin[0], y0]), np.asarray([origin[1], y1])]))
            ls.append((np.polyfit(origin, (y0, y1), 1)))
            idx += 1

        if idx != 2:
            warnings.warn('Could not identify correct calibration, try altering the `p1_lim` and `p2_lim` arguments.')
            if plot:
                return fig
            return

        # Find intersection of two lines
        s = np.vstack(tupes)        # s for stacked
        h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
        l1 = np.cross(h[0], h[1])           # get first line
        l2 = np.cross(h[2], h[3])           # get second line
        x, y, z = np.cross(l1, l2)          # point of intersection
        p1_loc, p2_loc =  x/z, y/z
        p1 = np.interp(p1_loc, np.arange(len(p1s)), p1s)
        p2 = np.interp(p2_loc, np.arange(len(p2s)), p2s)
        self.p1 = p1
        self.p2 = p2
        self._build_masks()
        if plot:
            with plt.style.context('seaborn-white'):
                ax[0].scatter(p1, p2, c='r', label=f'{p1.astype(int)}, {p2.astype(int)}')
                ax[0].legend(frameon=True)

                bmask = (_wav.to(u.nm).value > 300) & (_wav.to(u.nm).value < p1)
                gmask = (_wav.to(u.nm).value > p1) & (_wav.to(u.nm).value < p2)
                rmask = (_wav.to(u.nm).value > p2) & (_wav.to(u.nm).value < 11000)
                ax[1].fill_between(_wav/10, _bandpass * bmask, color='b', alpha=0.4)
                ax[1].fill_between(_wav/10, _bandpass * gmask, color='g', alpha=0.4, label=f'{np.round(p1).astype(int)} - {np.round(p2).astype(int)}nm')
                ax[1].fill_between(_wav/10, _bandpass * rmask, color='r', alpha=0.4)
                ax[1].legend()
                ax[1].set_xlabel('Wavelength [nm]')
                ax[1].set_ylabel("Throughput")
            return fig
        return

    def _build_masks(self):
        self.bmask = (_wav.to(u.nm).value > 300) & (_wav.to(u.nm).value < self.p1)
        self.gmask = (_wav.to(u.nm).value > self.p1) & (_wav.to(u.nm).value < self.p2)
        self.rmask = (_wav.to(u.nm).value > self.p2) & (_wav.to(u.nm).value < 11000)


    def _estimate_stellar_temperature(self, cadence_mask, plot=False, remove_outliers=True):
        r_g_perc = -2.5 * np.log10(np.median(self.rlc.flux.value[cadence_mask]/self.glc.flux.value[cadence_mask]))
        b_g_perc = -2.5 * np.log10(np.median(self.blc.flux.value[cadence_mask]/self.glc.flux.value[cadence_mask]))

        teffs = np.linspace(self.teff - 50, self.teff + 50, 2)
        r_gs = np.zeros(len(teffs))
        b_gs = np.zeros(len(teffs))
        for idx, teff in enumerate(teffs):
            spec = _bb(teff*u.K)  * _bandpass
            b = np.trapz(spec[self.bmask].value, _wav[self.bmask].value)
            g = np.trapz(spec[self.gmask].value, _wav[self.gmask].value)
            r = np.trapz(spec[self.rmask].value, _wav[self.rmask].value)
            r_gs[idx] = (-2.5 * np.log10(r/g))
            b_gs[idx] = (-2.5 * np.log10(b/g))

        s = np.vstack([np.hstack(list(teffs)*2), np.hstack([r_gs - r_g_perc, b_gs - b_g_perc])]).T
        h1 = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
        l1 = np.cross(h1[0], h1[1])           # get first line
        l2 = np.cross(h1[2], h1[3])           # get second line
        x, y, z = np.cross(l1, l2)          # point of intersection
        teff = x/z

        spec = _bb(teff*u.K)  * _bandpass
        b = np.trapz(spec[self.bmask].value, _wav[self.bmask].value)
        g = np.trapz(spec[self.gmask].value, _wav[self.gmask].value)
        r = np.trapz(spec[self.rmask].value, _wav[self.rmask].value)
        if plot:
            with plt.style.context('seaborn-white'):
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                rlc = (self.rlc/self.glc)[cadence_mask]
                blc = (self.blc/self.glc)[cadence_mask]
                rlc.errorbar(ax=ax[0], c='r')
                blc.errorbar(ax=ax[1], c='b')

                ax[0].axhline((r/g), c='r', label='Best Fit r/g')
                ax[1].axhline((b/g), c='b', label='Best Fit b/g')
            return teff, fig
        return teff

    def fit_flare(self, cadence_mask, teff=None, r_star=0.8*u.solRad, bin=0.0025, draws=500, tune=100, chains=2, cores=2):
        """
        Fits the flare energy in a given region of the CoRoT data

        Parameters
        ----------
        cadence_mask : np.array of booleans
            Cadences to fit.
        teff : float
            The assumed temperature of the photosphere. If none, will use the
            effective temperature in `self`.
        r_star: float * astropy.unit.solRad
            Stellar radius, used to set the estimated flare energy in Joules.
        bin : float
            The time resolution to bin to
        draws : int
            Number of draws in pymc3 sampling
        tune : int
            Tune parameter pymc3 sampling
        chains : int
            Chains in pymc3 sampling
        cores : int
            Cores in pymc3 sampling

        Returns
        -------
        fig1 : matplotlib.pyplot.figure
            Figure containing the corner plot of fit parameters
        fig2 : matplotlib.pyplot.figure
            Figure containing the color light curves, and best fit flare function
        samples : pandas.DataFrame
            DataFrame containing the samples
        """
        bandpass = pd.read_csv(PACKAGEDIR + '/data/bandpass.dat', header=None, delimiter=' ')
        wav = np.arange(300, 1100, 5)*u.nm
        bandpass = np.interp(wav.to(u.angstrom).value, np.asarray(bandpass[0]), np.asarray(bandpass[1]))

        bmask = (wav.to(u.nm).value > 300) & (wav.to(u.nm).value < self.p1)
        gmask = (wav.to(u.nm).value > self.p1) & (wav.to(u.nm).value < self.p2)
        rmask = (wav.to(u.nm).value > self.p2) & (wav.to(u.nm).value < 11000)

        unit = u.joule * u.day * u.nm/(u.s * u.m**3)
        const = ((np.pi * (r_star)**2) * unit).to(u.joule).value

        if teff is None:
            teff = self.teff

        def _bb_tt(T):
            a = (2 * h.value * c.value**2/(wav.value*1e-9)**5)
            b = tt.expm1((h.value * c.value)/(wav.value*1e-9 * k_B.value * T))
            return a/b


        wb = np.diff(wav[bmask].value)[:, None]
        wg = np.diff(wav[gmask].value)[:, None]
        wr = np.diff(wav[rmask].value)[:, None]

        star_spec = _bb(teff*u.K, wav)
        def _flare_time_series_tt(amp=0.1, te=0.01, tg=0.01):
            return (tt.exp(-(t**2)/(2*tg**2)) * tmask + tt.exp(-(t**2)/(te)) * ~tmask) * amp

        def _flare_tt(teff_flare, flare_time_series):
            flare_spec = _bb_tt(teff_flare)
            spec = (flare_spec[:, None] * flare_time_series) + star_spec[:, None]
            spec *= bandpass[:, None]
            b = tt.sum(wb * (spec[bmask][:-1] + spec[bmask][1:])/2, axis=0)
            g = tt.sum(wg * (spec[gmask][:-1] + spec[gmask][1:])/2, axis=0)
            r = tt.sum(wr * (spec[rmask][:-1] + spec[rmask][1:])/2, axis=0)
            return r/g, b/g

        rlc = (self.rlc/self.glc)[cadence_mask].bin(bin, aggregate_func=np.nanmedian).remove_nans()
        blc = (self.blc/self.glc)[cadence_mask].bin(bin, aggregate_func=np.nanmedian).remove_nans()

        rlc.time -= blc.time.jd[np.argmax(blc.flux.value)]
        blc.time -= blc.time.jd[np.argmax(blc.flux.value)]
        t = np.copy(blc.time.jd)
        tmask = t < 0

        with pm.Model() as model:
            teff_flare = pm.Uniform('teff_flare', lower=4000, upper=30000, testval=10000)
            amp = pm.Uniform('amp', lower=0, upper=1, testval=0.1)
            te = pm.Uniform('te', lower=0, upper=0.03, testval=0.01)
            tg = pm.Uniform('tg', lower=0, upper=0.03, testval=0.01)
            flare_time_series = _flare_time_series_tt(amp, te, tg)
            r_corr = pm.Normal('r_corr', mu=0, sd=rlc.flux*0.05)
            b_corr = pm.Normal('b_corr', mu=0, sd=blc.flux*0.05)

            r, b = _flare_tt(teff_flare=teff_flare, flare_time_series=flare_time_series)
            r -= r_corr
            b -= b_corr
            pm.Normal("obs", mu=tt.concatenate([r, b]),
               observed=np.hstack([rlc.flux, blc.flux]), sd=np.hstack([rlc.flux_err, blc.flux_err]))
            map_soln = xo.optimize()

        tg = map_soln['tg']
        r_corr = map_soln['r_corr']
        b_corr = map_soln['b_corr']

        with pm.Model() as model:
            teff_flare = pm.Uniform('teff_flare', lower=4000, upper=30000, testval=map_soln['teff_flare'])
            amp = pm.Uniform('amp', lower=0, upper=1, testval=map_soln['amp'])
            te = pm.Uniform('te', lower=0, upper=0.03, testval=map_soln['te'])
            tg = pm.Uniform('tg', lower=0, upper=0.03, testval=map_soln['tg'])
            flare_time_series = _flare_time_series_tt(amp, te, tg)

            r, b = _flare_tt(teff_flare=teff_flare, flare_time_series=flare_time_series)
            r -= r_corr
            b -= b_corr
            pm.Normal("obs", mu=tt.concatenate([r, b]),
               observed=np.hstack([rlc.flux, blc.flux]), sd=np.hstack([rlc.flux_err, blc.flux_err]))
            map_soln = xo.optimize()

            L = tt.sum(np.diff(t) * (flare_time_series[:-1] + flare_time_series[1:])/2)
            flare_spec = _bb_tt(teff_flare)
            E = pm.Deterministic('energy', (tt.sum(np.diff(wav.value)  * (flare_spec[:-1] + flare_spec[1:])/2)) * const)

            trace = pm.sample(
                draws=draws,
                tune=tune,
                start=map_soln,
                chains=chains,
                cores=cores,
                target_accept=0.95,
            )


        samples = pm.trace_to_dataframe(trace)


        fig1 = corner.corner(samples)

        fig2, ax = plt.subplots(1, 2, facecolor='w', figsize=(10, 5))
        rlc.errorbar(c='r', ls='', ax=ax[0])
        ax[0].plot(rlc.time.jd, xo.eval_in_model(r, model=model, point=map_soln), c='r', label='Best Fit')
        ax[0].set(xlabel='Time from Flare Peak')

        blc.errorbar(ax=ax[1], c='b', ls='')
        ax[1].plot(blc.time.jd, xo.eval_in_model(b, model=model, point=map_soln), c='b', label='Best Fit')
        ax[1].set(ylabel='', xlabel='Time from Flare Peak')

        return fig1, fig2, samples
