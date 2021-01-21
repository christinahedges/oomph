import pytest
import oomph
import lightkurve as lk
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt


def test_paper_lightcurves():
    c = oomph.CoRoTMultiPhotometry.from_archive(101562259, 4084)
    ax = c.lc.plot(c='k')
    c.rlc.plot(c='r', ax=ax, lw=0.1)
    c.blc.plot(c='b', ax=ax, lw=0.1)
    c.glc.plot(c='g', ax=ax, lw=0.1)
    plt.savefig('paper/lightcurve1.pdf', bbox_inches='tight')

    cadence_masks = [(c.lc.time.jd > 54294.9) & (c.lc.time.jd < 54295.6),
                     (c.lc.time.jd > 54370.2) & (c.lc.time.jd < 54370.9),
                     (c.lc.time.jd > 54315.5) & (c.lc.time.jd < 54315.9),
                     (c.lc.time.jd > 54309.8) & (c.lc.time.jd < 54310.2)]

    with plt.style.context('seaborn-white'):
        fig, ax = plt.subplots(2, 1, figsize=(10, 5), facecolor='w')
        (c.rlc/c.glc).bin(0.04, aggregate_func=np.nanmedian).plot(c='r', ax=ax[0], lw=0.3)
        (c.blc/c.glc).bin(0.04, aggregate_func=np.nanmedian).plot(c='b', ax=ax[1], lw=0.3)

        ylims0 = ax[0].get_ylim()
        ylims1 = ax[1].get_ylim()
        for cadence_mask in cadence_masks:
            ts = [np.min(c.lc.time.jd[cadence_mask]), np.max(c.lc.time.jd[cadence_mask])]
            ax[0].fill_between(ts, *ylims0, color='k', alpha=0.2)
            ax[1].fill_between(ts, *ylims1, color='k', alpha=0.2)
        ax[0].set_ylim(ylims0)
        ax[1].set_ylim(ylims1)
        ax[0].set(ylabel=('Flux$_R$/Flux$_G$'))
        ax[1].set(ylabel=('Flux$_B$/Flux$_G$'))
        plt.savefig('paper/lightcurve2.pdf', bbox_inches='tight')


def test_run_fit_flares():
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
