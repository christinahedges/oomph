import pytest
import oomph
import lightkurve as lk
import numpy as np
import pickle
from tqdm import tqdm


def test_paper():
    c = oomph.CoRoTMultiPhotometry.from_archive(101562259, 4084)
    fig = c.calibrate(p1_lim=(450, 550), p2_lim=(500, 600), p1_num=100, p2_num=101, plot=True)
    fig.savefig('paper/demo.png', bbox_inches='tight', dpi=150)
    fig = c.calibrate(p1_lim=(c.p1-30, c.p1+30), p2_lim=(c.p2-30, c.p2+30), p1_num=100, p2_num=101, plot=True)
    fig.savefig('paper/demo.png', bbox_inches='tight', dpi=150)


    cadence_masks = [(c.lc.time.jd > 54294.9) & (c.lc.time.jd < 54295.6),
                     (c.lc.time.jd > 54370.2) & (c.lc.time.jd < 54370.9),
                     (c.lc.time.jd > 54315.5) & (c.lc.time.jd < 54315.9),
                     (c.lc.time.jd > 54309.8) & (c.lc.time.jd < 54310.2)]

    for idx, cadence_mask in enumerate(tqdm(cadence_masks)):
        fig1, fig2, samples = c.fit_flare(cadence_mask)
        fig1.savefig(f'paper/flare{idx}_corner.pdf', bbox_inches='tight')
        fig2.savefig(f'paper/flare{idx}.pdf', bbox_inches='tight')
        pickle.dump(samples, open(f'paper/flare{idx}_samples.p', 'wb'))
