from . import PACKAGEDIR
import pandas as pd


def _get_url(id):
    df = pd.read_csv(PACKAGEDIR + 'src/oomph/data/database.csv')
    loc = np.where(df.CoRoT == id)[0]
    if len(loc) == 0:
        raise ValueError(f'No target {id} found with color photometry')
    id, year, month, day, FileName_a, FileName_b, FileName_c = df.loc[loc[0]]
    url = f'http://idoc-corot.ias.u-psud.fr/sitools/datastorage/user/corotstorage/N2-4.4/{year}/{month:02}/{day:02}/EN2_STAR_CHR_{id:010}_{year}{month:02}{day:02}T{FileName_a:06}_{FileName_b:06}T{FileName_c:06}.fits'
    return url
