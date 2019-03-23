import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_colors():
    """ utility function for getting colors for variables

    Returns:
    -------
    : (list)
        [holaps, modis, gleam, chirps, runoff]
    """
    h_col = sns.color_palette()[0]
    m_col = sns.color_palette()[1]
    g_col = sns.color_palette()[2]
    c_col = sns.color_palette()[3]
    r_col = sns.color_palette()[4]

    return [h_col, m_col, g_col, c_col]
