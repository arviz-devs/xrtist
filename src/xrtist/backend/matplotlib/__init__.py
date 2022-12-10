"""Matplotlib interface layer."""

from typing import Dict, Any

from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.pyplot import subplots

__all__ = ["create_plotting_grid", "line", "scatter"]

class UnsetDefault:
    pass


unset = UnsetDefault()


def create_plotting_grid(
    number,
    rows=1,
    cols=1,
    squeeze=True,
    sharex=False,
    sharey=False,
    polar=False,
    subplot_kws=None,
    **kwargs
):
    """Create a chart with a grid of plotting targets in it.

    Parameters
    ----------
    number : int
        Number of axes required
    rows, cols : int
        Number of rows and columns.
    squeeze : bool
    sharex, sharey : bool
    polar : bool
    subplot_kws : bool
        Passed to `~matplotlib.pyplot.subplots` as ``subplot_kw``
    **kwargs: dict, optional
        Passed to `~matplotlib.pyplot.subplots`

    Returns
    -------
    `~matplotlib.figure.Figure`
    `~matplotlib.axes.Axes` or ndarray of `~matplotlib.axes.Axes`
    """
    if subplot_kws is None:
        subplot_kws = {}
    subplot_kws = subplot_kws.copy()
    if polar:
        subplot_kws["projection"] = "polar"
    fig, axes = subplots(
        rows, cols, sharex=sharex, sharey=sharey, squeeze=squeeze, subplot_kw=subplot_kws, **kwargs
    )
    extra = (rows * cols) - number
    if extra > 0:
        for i, ax in enumerate(axes.ravel("C")):
            if i >= number:
                ax.set_axis_off()
    return fig, axes


def _filter_kwargs(kwargs, artist, artist_kws):
    kwargs = {key: value for key, value in kwargs.items() if value is not unset}
    if artist is not None:
        artist_kws = normalize_kwargs(artist_kws.copy(), artist)
    return {**artist_kws, **kwargs}


def line(x, y, target, *, color=unset, alpha=unset, linewidth=unset, linestyle=unset, **artist_kws):
    kwargs = dict(color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
    return target.plot(x, y, **_filter_kwargs(kwargs, Line2D, artist_kws))


def scatter(
    x,
    y,
    target,
    *,
    size=unset,
    marker=unset,
    alpha=unset,
    facecolor=unset,
    edgecolor=unset,
    edgewidth=unset,
    **artist_kws
):
    kwargs = dict(
        s=size, marker=marker, alpha=alpha, c=facecolor, edgecolors=edgecolor, linewidths=edgewidth
    )
    return target.scatter(x, y, **_filter_kwargs(kwargs, None, artist_kws))
