import pytest
import oomph
import lightkurve as lk
import numpy as np

# Mark remote data
def test_data():
    url = 'http://idoc-corot.ias.u-psud.fr/sitools/datastorage/user/corotstorage/N2-4.4/2007/02/03/EN2_STAR_CHR_0102706314_20070203T130553_20070402T070158.fits'
    c = oomph.CoRoTMultiPhotometry.from_archive(102706314, 4405)
    for attr in ['lc', 'rlc', 'glc', 'blc']:
        assert hasattr(c, attr)
        assert isinstance(getattr(c, attr), lk.lightcurve.LightCurve)

def test_calibrate():
    url = 'http://idoc-corot.ias.u-psud.fr/sitools/datastorage/user/corotstorage/N2-4.4/2007/02/03/EN2_STAR_CHR_0102706314_20070203T130553_20070402T070158.fits'
    c = oomph.CoRoTMultiPhotometry.from_archive(102706314, 4405)
    fig = c.calibrate(p1_lim=(450, 550), p2_lim=(500, 600), p1_num=100, p2_num=101, plot=True)
    fig.savefig('demo.png', bbox_inches='tight', dpi=150)
    fig = c.calibrate(p1_lim=(c.p1-30, c.p1+30), p2_lim=(c.p2-30, c.p2+30), p1_num=100, p2_num=101, plot=True)
    fig.savefig('demo.png', bbox_inches='tight', dpi=150)
    assert np.isclose(c.p1, 484, atol=2)
    assert np.isclose(c.p2, 541, atol=2)

def test_estimate_teff():
    url = 'http://idoc-corot.ias.u-psud.fr/sitools/datastorage/user/corotstorage/N2-4.4/2007/02/03/EN2_STAR_CHR_0102706314_20070203T130553_20070402T070158.fits'
    c = oomph.CoRoTMultiPhotometry.from_archive(102706314, 4405)
    _ = c.calibrate(p1_lim=(480, 490), p2_lim=(537, 547), p1_num=30, p2_num=31, plot=False)
    cadence_mask = (c.lc.time.jd > 54135.07256172 + 1) & (c.lc.time.jd < 54135.07256172+ 2)
    teff, fig = c._estimate_stellar_temperature(cadence_mask, plot=True)

def test_fit_flare():
    url = 'http://idoc-corot.ias.u-psud.fr/sitools/datastorage/user/corotstorage/N2-4.4/2007/02/03/EN2_STAR_CHR_0102706314_20070203T130553_20070402T070158.fits'
    c = oomph.CoRoTMultiPhotometry.from_archive(101562259, 4084)
    c.p1 = 498
    c.p2 = 555
    c._build_masks()
    cadence_mask = (c.lc.time.jd > 54294.05) & (c.lc.time.jd < 54295.5)
    c.fit_flare(cadence_mask)
