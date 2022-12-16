"""Bokeh interface layer."""

import numpy as np
from bokeh.layouts import gridplot
from bokeh.plotting import figure


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
    **kwargs,
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
        Passed to `~bokeh.plotting.figure`
    **kwargs: dict, optional
        Passed to `~bokeh.layouts.gridplot`

    Returns
    -------
    `~bokeh.layouts.gridplot` or None
    `~bokeh.plotting.figure` or ndarray of `~bokeh.plotting.figure`
    """
    if subplot_kws is None:
        subplot_kws = {}
    subplot_kws = subplot_kws.copy()

    figures = np.empty((rows, cols), dtype=object)

    if polar:
        subplot_kws.setdefault("x_axis_type", None)
        subplot_kws.setdefault("y_axis_type", None)

    for row in range(rows):
        for col in range(cols):
            if (row == 0) and (col == 0) and (sharex or sharey):
                p = figure(**subplot_kws)  # pylint: disable=invalid-name
                figures[row, col] = p
                if sharex:
                    subplot_kws["x_range"] = p.x_range
                if sharey:
                    subplot_kws["y_range"] = p.y_range
            elif row * cols + (col + 1) > number:
                figures[row, col] = None
            else:
                figures[row, col] = figure(**subplot_kws)
    if squeeze and figures.size == 1:
        return None, figures[0, 0]
    layout = gridplot(figures.tolist(), **kwargs)
    return layout, figures.squeeze() if squeeze else figures


def _filter_kwargs(kwargs, artist_kws):
    kwargs = {key: value for key, value in kwargs.items() if value is not unset}
    return {**artist_kws, **kwargs}


def line(x, y, target, *, color=unset, alpha=unset, linewidth=unset, linestyle=unset, **artist_kws):
    kwargs = dict(color=color, alpha=alpha, line_width=linewidth, line_dash=linestyle)
    return target.line(x, y, **_filter_kwargs(kwargs, artist_kws))


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
    **artist_kws,
):
    kwargs = dict(
        size=size,
        marker=marker,
        line_alpha=alpha,
        fill_alpha=alpha,
        fill_color=facecolor,
        line_color=edgecolor,
        line_width=edgewidth,
    )
    return target.scatter(x, y, **_filter_kwargs(kwargs, artist_kws))


def text(x, y, string, target, *, size=unset, alpha=unset, color=unset, **artist_kws):
    kwargs = dict(text_font_size=size, alpha=alpha, color=color)
    return target.text(x, y, string, **_filter_kwargs(kwargs, artist_kws))
