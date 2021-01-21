import oomph
import lightkurve as lk
import numpy as np
import pickle

def build_paper():
    url = 'http://idoc-corot.ias.u-psud.fr/sitools/datastorage/user/corotstorage/N2-4.4/2007/02/03/EN2_STAR_CHR_0102706314_20070203T130553_20070402T070158.fits'
    c = oomph.CoRoTMultiPhotometry.from_archive(101562259, 4084)
    fig = c.calibrate(p1_lim=(450, 550), p2_lim=(500, 600), p1_num=100, p2_num=101, plot=True)
    fig.savefig('paper/demo.png', bbox_inches='tight', dpi=150)
    fig = c.calibrate(p1_lim=(c.p1-30, c.p1+30), p2_lim=(c.p2-30, c.p2+30), p1_num=100, p2_num=101, plot=True)
    fig.savefig('paper/demo.png', bbox_inches='tight', dpi=150)

    c._build_masks()
    cadence_masks = (c.lc.time.jd > 54294.05) & (c.lc.time.jd < 54295.5)
    for cadence_mask in cadence_masks:
        fig1, fig2, samples = c.fit_flare(cadence_mask)
        fig1.savefig('paper/flare1_corner.png', bbox_inches='tight', dpi=150)
        fig2.savefig('paper/flare1.png', bbox_inches='tight', dpi=150)
        pickle.dump(samples, open('paper/flare1_samples.p', 'wb'))
