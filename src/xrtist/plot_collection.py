"""Plot collection classes."""
from importlib import import_module
import numpy as np
import xarray as xr


from arviz.sel_utils import xarray_sel_iter


def sel_subset(sel, present_dims):
    return {key: value for key, value in sel.items() if key in present_dims}


def subset_ds(ds, var_name, sel):
    out = ds[var_name].sel(sel_subset(sel, ds[var_name].dims))
    return out.item()


class PlotCollection:
    def __init__(self, data, viz_ds, aes=None, **kwargs):

        self.data = data
        self.preprocessed_data = None
        self.viz = viz_ds
        self.ds = xr.Dataset()

        if aes is None:
            aes = {}

        for aes_key, dims in aes.items():
            aes_raw_shape = [len(data[dim]) for dim in dims]
            n_aes = np.prod(aes_raw_shape)
            aes_vals = kwargs.get(aes_key, [None])
            n_aes_vals = len(aes_vals)
            if n_aes_vals > n_aes:
                aes_vals = aes_vals[:n_aes]
            elif n_aes_vals < n_aes:
                aes_vals = np.tile(aes_vals, (n_aes // n_aes_vals) + 1)[:n_aes]
            self.ds[aes_key] = xr.DataArray(
                np.array(aes_vals).reshape(aes_raw_shape),
                dims=dims,
                coords={dim: data.coords[dim] for dim in dims if dim in data.coords},
            )
        self.aes = aes

    @property
    def base_loop_dims(self):
        return set(self.viz["plot"].dims)

    @classmethod
    def wrap(
        cls,
        data,
        cols=None,
        col_wrap=4,
        backend="matplotlib",
        plot_grid_kws=None,
        **kwargs,
    ):
        if plot_grid_kws is None:
            plot_grid_kws = {}
        plots_raw_shape = [len(data[col]) for col in cols]
        n_plots = np.prod(plots_raw_shape)
        if n_plots <= col_wrap:
            n_rows, n_cols = 1, n_plots
        else:
            n_rows = n_plots // col_wrap + 1
            n_cols = col_wrap
        plot_bknd = import_module(f".backend.{backend}", package="xrtist")
        fig, ax_ary = plot_bknd.create_plotting_grid(n_plots, n_rows, n_cols, **plot_grid_kws)
        col_id, row_id = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
        viz_ds = xr.Dataset(
            {
                "chart": fig,
                "plot": (cols, ax_ary.flatten()[:n_plots].reshape(plots_raw_shape)),
                "row": (cols, row_id.flatten()[:n_plots].reshape(plots_raw_shape)),
                "col": (cols, col_id.flatten()[:n_plots].reshape(plots_raw_shape)),
            },
            coords={col: data[col] for col in cols},
        )
        return cls(data, viz_ds, **kwargs)

    @classmethod
    def grid(
        cls,
        data,
        cols,
        rows,
        backend="matplotlib",
        plot_grid_kws=None,
        **kwargs,
    ):
        if plot_grid_kws is None:
            plot_grid_kws = {}
        n_cols = np.prod([len(data[col]) for col in cols])
        n_rows = np.prod([len(data[row]) for row in rows])
        n_plots = n_cols * n_rows
        plot_bknd = import_module(f".backend.{backend}", package="xrtist")
        fig, ax_ary = plot_bknd.create_plotting_grid(n_plots, n_rows, n_cols, **plot_grid_kws)
        dims = tuple((*rows, *cols))  # use provided dim orders, not existing ones
        col_id, row_id = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
        plots_raw_shape = [len(data[dim]) for dim in dims]
        viz_ds = xr.Dataset(
            {
                "chart": fig,
                "plot": (dims, ax_ary.flatten().reshape(plots_raw_shape)),
                "row": (dims, row_id.flatten().reshape(plots_raw_shape)),
                "col": (dims, col_id.flatten().reshape(plots_raw_shape)),
            },
            coords={dim: data[dim] for dim in dims},
        )
        return cls(data, viz_ds, **kwargs)

    def _update_aes(self, ignore_aes=frozenset()):
        aes = [aes_key for aes_key in self.aes.keys() if aes_key not in ignore_aes]
        aes_dims = [dim for sublist in list(self.aes.values()) for dim in sublist]
        all_loop_dims = self.base_loop_dims.union(aes_dims)
        return aes, all_loop_dims

    def plot_iterator(self, ignore_aes=frozenset()):
        _, all_loop_dims = self._update_aes(ignore_aes)
        plotters = xarray_sel_iter(
            self.data, skip_dims={dim for dim in self.data.dims if dim not in all_loop_dims}
        )
        for var_name, sel, isel in plotters:
            yield var_name, sel, isel

    def map(
        self,
        fun,
        fun_label=None,
        *,
        ignore_aes=frozenset(),
        preprocessed=False,
        subset_info=False,
        store_artist=True,
        **kwargs,
    ):
        aes, all_loop_dims = self._update_aes(ignore_aes)
        plotters = xarray_sel_iter(
            self.data, skip_dims={dim for dim in self.data.dims if dim not in all_loop_dims}
        )
        artist_dims = [dim for dim in self.data.dims if dim in all_loop_dims]
        artist_shape = [len(self.data[dim]) for dim in artist_dims]

        if fun_label is None:
            fun_label = fun.__name__

        if store_artist:
            self.viz[fun_label] = xr.DataArray(
                np.empty(artist_shape, dtype=object),
                dims=artist_dims,
                coords={dim: self.data[dim] for dim in artist_dims},
            )
        for var_name, sel, isel in plotters:
            da = self.data.sel(sel)
            ax = subset_ds(self.viz, "plot", sel)

            aes_kwargs = {}
            for aes_key in aes:
                aes_kwargs[aes_key] = subset_ds(self.ds, aes_key, sel)

            fun_kwargs = {**kwargs, **aes_kwargs}
            if preprocessed:
                if self.preprocessed_data is None:
                    raise ValueError(
                        "You must manually set the `preprocessed_data` to use preprocessed=True"
                    )
                pre_da = self.preprocessed_data.sel(sel_subset(sel, self.preprocessed_data.dims))
                fun_kwargs["preprocessed_data"] = pre_da
            if subset_info:
                fun_kwargs = {**fun_kwargs, "var_name": var_name, "sel": sel, "isel": isel}
            aux_artist = fun(da, ax=ax, **fun_kwargs)
            if store_artist:
                self.viz[fun_label].loc[sel] = aux_artist

    def add_legend(self, aes, artist, **kwargs):
        pass
