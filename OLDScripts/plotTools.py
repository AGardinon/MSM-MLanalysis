import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt

# -------------------------------------------------- #
# --- general fig,ax env

def get_axes(L, max_col=2, fig_frame=(3.3,3.), res=200):
    cols = L if L <= max_col else max_col
    rows = int(L / max_col) + int(L % max_col != 0)
    fig, axes = plt.subplots(rows,
                             cols,
                             figsize=(cols * fig_frame[0], rows * fig_frame[1]),
                             dpi=res)
    if L > 1:
        axes = axes.flatten()
        for s in range(L, max_col*rows):
            for side in ['bottom', 'right', 'top', 'left']:
                axes[s].spines[side].set_visible(False)
            axes[s].set_yticks([])
            axes[s].set_xticks([])
            axes[s].xaxis.set_ticks_position('none')
            axes[s].yaxis.set_ticks_position('none')
    return fig, axes


def remove_frame(axes):
    for side in ['bottom', 'right', 'top', 'left']:
        axes.spines[side].set_visible(False)
    axes.set_yticks([])
    axes.set_xticks([])
    axes.xaxis.set_ticks_position('none')
    axes.yaxis.set_ticks_position('none')
    

def make_colors(clust, mode='tab10'):
    if np.min(clust) == -1:
        N = len(np.unique(clust)) - 1
        colors = sns.color_palette(mode, N) + [(0,0,0)]
    else:
        N = len(np.unique(clust))
        colors = sns.color_palette(mode, N)
    return colors
    
# -------------------------------------------------- #
# --- FES

def setFES1D(X, bins, kb='kcal', temp=300,
             interval=None, fill_empty=True):
    kb_mode = {
        'kJ': 0.00831446261815324,
        'kcal': 0.00198720425864083,
        'unit': 1.0 / temp
    }
    KBT = kb_mode[kb] * temp
    hist, edges = np.histogram(X,
                               bins=bins, range=interval,
                               density=True)
    FES = -1 * KBT * np.log(hist)
    FES = FES - np.min(FES)
    if fill_empty:
        max_ = np.max(FES[FES != np.inf])
        FES[FES == np.inf] = max_
    # get min value
    edges_ = edges[:-1]
    min_value = edges_[FES == np.min(FES)]
    return FES, edges, min_value


def convtoFES2D(X, Y, bins, kb='kcal', temp=300,
                fill_empty=True, interval=None):
    kb_mode = {
        'kJ': 0.00831446261815324,
        'kcal': 0.00198720425864083,
        'unit': 1.0 / temp
    }
    KBT = kb_mode[kb] * temp
    H, xedges, yedges = np.histogram2d(X, Y, bins=bins,
                                       range=interval, density=True)
    FES = -1 * KBT * np.log(H)
    FES = FES - np.min(FES)
    if fill_empty:
        max_ = np.max(FES[FES != np.inf])
        FES[FES == np.inf] = max_
    return FES, xedges, yedges


def setFES2D(x, y, bins, kb='kcal', temp=300,
             fill_empty=True, range_fes=None):
    X = np.asarray(x)
    Y = np.asarray(y)

    FES, xed, yed = convtoFES2D(X.flatten(),
                                Y.flatten(),
                                bins,
                                kb=kb, temp=temp,
                                fill_empty=fill_empty,
                                interval=range_fes)

    Xmin = xed.min()
    Xmax = xed.max()
    Ymin = yed.min()
    Ymax = yed.max()
    xx = np.arange(Xmin, Xmax, ((Xmax - Xmin) / bins))
    yy = np.arange(Ymin, Ymax, ((Ymax - Ymin) / bins))
    XX, YY = np.meshgrid(xx, yy)
    return XX, YY, FES.T


def plotFES2D(X, Y, Z, levels, figure, axes,
              colorbar=True, cbar_label=None,
              contlabels=True, 
              ghost=False):
    cont = axes.contour(X, Y, Z, levels,
                        colors='k', linewidths=0.5, zorder=2)
    if not ghost:
        surf = axes.contourf(X, Y, Z, levels,
                             cmap='coolwarm_r', zorder=1)
        cbar = figure.colorbar(surf,ax=axes)
        if cbar_label:
            cbar.set_label(cbar_label)
    if contlabels:
        axes.clabel(cont, inline=True, colors='k',
                    fontsize=8, fmt='%1.1f', zorder=3)
        
        
# -------------------------------------------------- #
# --- CHARTS


def plot_donut(data, labels, axes, pieargs, size=0.3, radius=1, annotate=1):
	circle = plt.Circle((0,0),0.40,fc='white')
	wedges, texts = axes.pie(data, startangle=90, radius=radius,
		                     wedgeprops=dict(width=size, edgecolor='k'), **pieargs)
	axes.add_patch(circle)

	if annotate == 1:
		bbox_props = dict(boxstyle="square,pad=0.1", fc="w", ec="k", lw=0.72)
		kw = dict(arrowprops=dict(arrowstyle="-"), 
			      bbox=bbox_props, zorder=0, va="center")

		for i, p in enumerate(wedges):
			ang = (p.theta2 - p.theta1)/2. + p.theta1
			y = np.sin(np.deg2rad(ang))
			x = np.cos(np.deg2rad(ang))
			horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
			connectionstyle = "angle,angleA=0,angleB={}".format(ang)
			kw["arrowprops"].update({"connectionstyle": connectionstyle})
			axes.annotate(labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                		horizontalalignment=horizontalalignment, **kw)
	elif annotate == 0:
		axes.legend(wedges, labels, loc="center left",
					bbox_to_anchor=(1, 0, 0.5, 1), prop=dict(size=15))

	return axes


# --------

def plot_feature_histograms(xyzall,
                            feature_labels=None,
                            ax=None,
                            ylog=False,
                            outfile=None,
                            n_bins=50,
                            ignore_dim_warning=False,
                            **kwargs):
    import numpy as _np
    r"""Feature histogram plot
    Parameters
    ----------
    xyzall : np.ndarray(T, d)
        (Concatenated list of) input features; containing time series data to be plotted.
        Array of T data points in d dimensions (features).
    feature_labels : iterable of str or pyemma.Featurizer, optional, default=None
        Labels of histogramed features, defaults to feature index.
    ax : matplotlib.Axes object, optional, default=None.
        The ax to plot to; if ax=None, a new ax (and fig) is created.
    ylog : boolean, default=False
        If True, plot logarithm of histogram values.
    n_bins : int, default=50
        Number of bins the histogram uses.
    outfile : str, default=None
        If not None, saves plot to this file.
    ignore_dim_warning : boolean, default=False
        Enable plotting for more than 50 dimensions (on your own risk).
    **kwargs: kwargs passed to pyplot.fill_between. See the doc of pyplot for options.
    Returns
    -------
    fig : matplotlib.Figure object
        The figure in which the used ax resides.
    ax : matplotlib.Axes object
        The ax in which the historams were plotted.
    """

    if not isinstance(xyzall, _np.ndarray):
        raise ValueError('Input data hast to be a numpy array. Did you concatenate your data?')

    if xyzall.shape[1] > 50 and not ignore_dim_warning:
        raise RuntimeError('This function is only useful for less than 50 dimensions. Turn-off this warning '
                           'at your own risk with ignore_dim_warning=True.')

    if feature_labels is not None:
        if not isinstance(feature_labels, list):
            from pyemma.coordinates.data.featurization.featurizer import MDFeaturizer as _MDFeaturizer
            if isinstance(feature_labels, _MDFeaturizer):
                feature_labels = feature_labels.describe()
            else:
                raise ValueError('feature_labels must be a list of feature labels, '
                                 'a pyemma featurizer object or None.')
        if not xyzall.shape[1] == len(feature_labels):
            raise ValueError('feature_labels must have the same dimension as the input data xyzall.')

    # make nice plots if user does not decide on color and transparency
    if 'color' not in kwargs.keys():
        kwargs['color'] = 'b'
    if 'alpha' not in kwargs.keys():
        kwargs['alpha'] = .25

    import matplotlib.pyplot as _plt
    # check input
    if ax is None:
        fig, ax = _plt.subplots()
    else:
        fig = ax.get_figure()

    hist_offset = -.2
    for h, coordinate in enumerate(reversed(xyzall.T)):
        hist, edges = _np.histogram(coordinate, bins=n_bins)
        if not ylog:
            y = hist / hist.max()
        else:
            y = _np.zeros_like(hist) + _np.NaN
            pos_idx = hist > 0
            y[pos_idx] = _np.log(hist[pos_idx]) / _np.log(hist[pos_idx]).max()
        ax.fill_between(edges[:-1], y + h + hist_offset, y2=h + hist_offset, **kwargs)
        ax.axhline(y=h + hist_offset, xmin=0, xmax=1, color='k', linewidth=.2)
    ax.set_ylim(hist_offset, h + hist_offset + 1)

    # formatting
    if feature_labels is None:
        feature_labels = [str(n) for n in range(xyzall.shape[1])]
        ax.set_ylabel('Feature histograms')

    ax.set_yticks(_np.array(range(len(feature_labels))) + .3)
    ax.set_yticklabels(feature_labels[::-1])
    ax.set_xlabel('Feature values')

    # save
    if outfile is not None:
        fig.savefig(outfile)
    return fig, ax