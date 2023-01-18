"""Plot collection classes."""
from importlib import import_module

import numpy as np
import xarray as xr
from arviz.sel_utils import xarray_sel_iter
from datatree import DataTree


def sel_subset(sel, present_dims):
    return {key: value for key, value in sel.items() if key in present_dims}


def subset_ds(ds, var_name, sel):
    subset_dict = sel_subset(sel, ds[var_name].dims)
    if subset_dict:
        out = ds[var_name].sel(subset_dict)
    else:
        out = ds[var_name]
    if out.size == 1:
        return out.item()
    return out.values


def _process_facet_dims(data, facet_dims):
    if not facet_dims:
        return 1, {}
    facets_per_var = {}
    if "__variable__" in facet_dims:
        for var_name, da in data.items():
            lenghts = [len(da[dim]) for dim in facet_dims if dim in da.dims]
            facets_per_var[var_name] = np.prod(lenghts) if lenghts else 1
        n_facets = np.sum(list(facets_per_var.values()))
    else:
        missing_dims = {
            var_name: [dim for dim in facet_dims if dim not in da.dims]
            for var_name, da in data.items()
        }
        missing_dims = {k: v for k, v in missing_dims.items() if v}
        if any(missing_dims.values()):
            raise ValueError(
                "All variables must have all facetting dimensions, but found the following "
                f"dims to be missing in these variables: {missing_dims}"
            )
        n_facets = np.prod([data.sizes[dim] for dim in facet_dims])
    return n_facets, facets_per_var


class PlotCollection:
    def __init__(self, data, viz_ds, aes=None, backend=None, **kwargs):

        self.data = data
        self.preprocessed_data = None
        self.viz = viz_ds
        self.ds = xr.Dataset()
        if backend is not None:
            self.backend = backend

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
        if cols is None:
            plots_raw_shape = ()
            n_plots = 1
        else:
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
        if n_plots > 1:
            viz_ds = xr.Dataset(
                {
                    "chart": fig,
                    "plot": (cols, ax_ary.flatten()[:n_plots].reshape(plots_raw_shape)),
                    "row": (cols, row_id.flatten()[:n_plots].reshape(plots_raw_shape)),
                    "col": (cols, col_id.flatten()[:n_plots].reshape(plots_raw_shape)),
                },
                coords={col: data[col] for col in cols},
            )
        else:
            viz_ds = xr.Dataset(
                {
                    "chart": fig,
                    "plot": ax_ary,
                    "row": 0,
                    "col": 0,
                },
            )
        return cls(data, viz_ds, backend=backend, **kwargs)

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
        n_cols = np.prod([data.sizes[col] for col in cols])
        n_rows = np.prod([data.sizes[row] for row in rows])
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
        return cls(data, viz_ds, backend=backend, **kwargs)

    def _update_aes(self, ignore_aes=frozenset()):
        aes = [aes_key for aes_key in self.aes.keys() if aes_key not in ignore_aes]
        aes_dims = [dim for aes_key in aes for dim in self.aes[aes_key]]
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
            target = subset_ds(self.viz, "plot", sel)

            aes_kwargs = {}
            for aes_key in aes:
                aes_kwargs[aes_key] = subset_ds(self.ds, aes_key, sel)

            fun_kwargs = {**aes_kwargs, **kwargs}
            fun_kwargs["backend"] = self.backend
            if preprocessed:
                if self.preprocessed_data is None:
                    raise ValueError(
                        "You must manually set the `preprocessed_data` to use preprocessed=True"
                    )
                pre_da = self.preprocessed_data.sel(sel_subset(sel, self.preprocessed_data.dims))
                fun_kwargs["preprocessed_data"] = pre_da
            if subset_info:
                fun_kwargs = {**fun_kwargs, "var_name": var_name, "sel": sel, "isel": isel}
            aux_artist = fun(da, target=target, **fun_kwargs)
            if store_artist:
                self.viz[fun_label].loc[sel] = aux_artist

    def add_legend(self, aes, artist, **kwargs):
        pass


class PlotMuseum:
    def __init__(self, data, viz_dt, aes_dt=None, aes=None, backend=None, **kwargs):

        self.data = data
        self.preprocessed_data = None
        self.viz = viz_dt
        self.dt = aes_dt

        if backend is not None:
            self.backend = backend

        if aes is None:
            aes = {}

        self._aes = aes
        self._kwargs = kwargs

    def generate_aes_dt(self, aes, **kwargs):
        if aes is None:
            aes = {}
        self._aes = aes
        self._kwargs = kwargs
        self.dt = DataTree()
        for var_name, da in self.data.items():
            ds = xr.Dataset()
            for aes_key, dims in aes.items():
                aes_vals = kwargs.get(aes_key, [None])
                aes_raw_shape = [da.sizes[dim] for dim in dims if dim in da.dims]
                if not aes_raw_shape:
                    ds[aes_key] = aes_vals[0]
                    continue
                n_aes = np.prod(aes_raw_shape)
                n_aes_vals = len(aes_vals)
                if n_aes_vals > n_aes:
                    aes_vals = aes_vals[:n_aes]
                elif n_aes_vals < n_aes:
                    aes_vals = np.tile(aes_vals, (n_aes // n_aes_vals) + 1)[:n_aes]
                ds[aes_key] = xr.DataArray(
                    np.array(aes_vals).reshape(aes_raw_shape),
                    dims=dims,
                    coords={dim: da.coords[dim] for dim in dims if dim in da.coords},
                )
            DataTree(name=var_name, parent=self.dt, data=ds)

    @property
    def base_loop_dims(self):
        if "plot" in self.viz.data_vars:
            return set(self.viz["plot"].dims)
        return set(dim for da in self.viz.children.values() for dim in da["plot"].dims)

    def get_viz(self, var_name):
        return self.viz if "plot" in self.viz.data_vars else self.viz[var_name]

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
        if cols is None:
            cols = []
        if plot_grid_kws is None:
            plot_grid_kws = {}

        n_plots, plots_per_var = _process_facet_dims(data, cols)
        if n_plots <= col_wrap:
            n_rows, n_cols = 1, n_plots
        else:
            div_mod = divmod(n_plots, col_wrap)
            n_rows = div_mod[0] + (div_mod[1] != 0)
            n_cols = col_wrap

        plot_bknd = import_module(f".backend.{backend}", package="xrtist")
        fig, ax_ary = plot_bknd.create_plotting_grid(
            n_plots, n_rows, n_cols, squeeze=False, **plot_grid_kws
        )
        col_id, row_id = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
        viz_dict = {}
        flat_ax_ary = ax_ary.flatten()[:n_plots]
        flat_row_id = row_id.flatten()[:n_plots]
        flat_col_id = col_id.flatten()[:n_plots]
        if "__variable__" not in cols:
            dims = cols  # use provided dim orders, not existing ones
            plots_raw_shape = [data.sizes[dim] for dim in dims]
            viz_dict["/"] = xr.Dataset(
                {
                    "chart": fig,
                    "plot": (dims, flat_ax_ary.reshape(plots_raw_shape)),
                    "row": (dims, flat_row_id.reshape(plots_raw_shape)),
                    "col": (dims, flat_col_id.reshape(plots_raw_shape)),
                },
                coords={dim: data[dim] for dim in dims},
            )
        else:
            viz_dict["/"] = xr.Dataset({"chart": xr.DataArray(fig)})
            all_dims = cols
            facet_cumulative = 0
            for var_name, da in data.items():
                dims = [dim for dim in all_dims if dim in da.dims]
                plots_raw_shape = [data.sizes[dim] for dim in dims]
                col_slice = (
                    slice(None, None)
                    if var_name not in plots_per_var
                    else slice(facet_cumulative, facet_cumulative + plots_per_var[var_name])
                )
                facet_cumulative += plots_per_var[var_name]
                viz_dict[var_name] = xr.Dataset(
                    {
                        "plot": (
                            dims,
                            flat_ax_ary[col_slice].reshape(plots_raw_shape),
                        ),
                        "row": (
                            dims,
                            flat_row_id[col_slice].reshape(plots_raw_shape),
                        ),
                        "col": (
                            dims,
                            flat_col_id[col_slice].reshape(plots_raw_shape),
                        ),
                    }
                )
        viz_dt = DataTree.from_dict(viz_dict)
        return cls(data, viz_dt, backend=backend, **kwargs)

    @classmethod
    def grid(
        cls,
        data,
        cols=None,
        rows=None,
        backend="matplotlib",
        plot_grid_kws=None,
        **kwargs,
    ):
        if cols is None:
            cols = []
        if rows is None:
            rows = []
        if plot_grid_kws is None:
            plot_grid_kws = {}
        repeated_dims = [col for col in cols if col in rows]
        if repeated_dims:
            raise ValueError("The same dimension can't be used for both cols and rows.")

        n_cols, cols_per_var = _process_facet_dims(data, cols)
        n_rows, rows_per_var = _process_facet_dims(data, rows)

        n_plots = n_cols * n_rows
        plot_bknd = import_module(f".backend.{backend}", package="xrtist")
        fig, ax_ary = plot_bknd.create_plotting_grid(
            n_plots, n_rows, n_cols, squeeze=False, **plot_grid_kws
        )
        col_id, row_id = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
        viz_dict = {}
        if "__variable__" not in cols and "__variable__" not in rows:
            dims = tuple((*rows, *cols))  # use provided dim orders, not existing ones
            plots_raw_shape = [data.sizes[dim] for dim in dims]
            viz_dict["/"] = xr.Dataset(
                {
                    "chart": fig,
                    "plot": (dims, ax_ary.flatten().reshape(plots_raw_shape)),
                    "row": (dims, row_id.flatten().reshape(plots_raw_shape)),
                    "col": (dims, col_id.flatten().reshape(plots_raw_shape)),
                },
                coords={dim: data[dim] for dim in dims},
            )
        else:
            viz_dict["/"] = xr.Dataset({"chart": xr.DataArray(fig)})
            all_dims = tuple((*rows, *cols))  # use provided dim orders, not existing ones
            facet_cumulative = 0
            for var_name, da in data.items():
                dims = [dim for dim in all_dims if dim in da.dims]
                plots_raw_shape = [data.sizes[dim] for dim in dims]
                row_slice = (
                    slice(None, None)
                    if var_name not in rows_per_var
                    else slice(facet_cumulative, facet_cumulative + rows_per_var[var_name])
                )
                col_slice = (
                    slice(None, None)
                    if var_name not in cols_per_var
                    else slice(facet_cumulative, facet_cumulative + cols_per_var[var_name])
                )
                if rows_per_var:
                    facet_cumulative += rows_per_var[var_name]
                else:
                    facet_cumulative += cols_per_var[var_name]
                viz_dict[var_name] = xr.Dataset(
                    {
                        "plot": (
                            dims,
                            ax_ary[row_slice, col_slice].flatten().reshape(plots_raw_shape),
                        ),
                        "row": (
                            dims,
                            row_id[row_slice, col_slice].flatten().reshape(plots_raw_shape),
                        ),
                        "col": (
                            dims,
                            col_id[row_slice, col_slice].flatten().reshape(plots_raw_shape),
                        ),
                    }
                )
        viz_dt = DataTree.from_dict(viz_dict)
        return cls(data, viz_dt, backend=backend, **kwargs)

    def _update_aes(self, ignore_aes=frozenset()):
        aes = [aes_key for aes_key in self._aes.keys() if aes_key not in ignore_aes]
        aes_dims = [dim for aes_key in aes for dim in self._aes[aes_key]]
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
        artist_dims=None,
        **kwargs,
    ):
        if self.dt is None:
            self.generate_aes_dt(self._aes, **self._kwargs)
        if artist_dims is None:
            artist_dims = {}
        if fun_label is None:
            fun_label = fun.__name__

        aes, all_loop_dims = self._update_aes(ignore_aes)
        plotters = xarray_sel_iter(
            self.data, skip_dims={dim for dim in self.data.dims if dim not in all_loop_dims}
        )
        if store_artist:
            for var_name, da in self.data.items():
                if var_name not in self.viz.children:
                    DataTree(name=var_name, parent=self.viz)
                inherited_dims = [dim for dim in da.dims if dim in all_loop_dims]
                artist_shape = [da.sizes[dim] for dim in inherited_dims] + list(
                    artist_dims.values()
                )
                all_artist_dims = inherited_dims + list(artist_dims.keys())

                self.viz[var_name][fun_label] = xr.DataArray(
                    np.empty(artist_shape, dtype=object),
                    dims=all_artist_dims,
                    coords={dim: self.data[dim] for dim in inherited_dims},
                )

        for var_name, sel, isel in plotters:
            da = self.data[var_name].sel(sel)
            target = subset_ds(self.get_viz(var_name), "plot", sel)

            aes_kwargs = {}
            for aes_key in aes:
                aes_kwargs[aes_key] = subset_ds(self.dt[var_name], aes_key, sel)

            fun_kwargs = {**aes_kwargs, **kwargs}
            fun_kwargs["backend"] = self.backend
            if preprocessed:
                if self.preprocessed_data is None:
                    raise ValueError(
                        "You must manually set the `preprocessed_data` to use preprocessed=True"
                    )
                pre_da = self.preprocessed_data.sel(sel_subset(sel, self.preprocessed_data.dims))
                fun_kwargs["preprocessed_data"] = pre_da
            if subset_info:
                fun_kwargs = {**fun_kwargs, "var_name": var_name, "sel": sel, "isel": isel}
            aux_artist = fun(da, target=target, **fun_kwargs)
            if store_artist:
                self.viz[var_name][fun_label].loc[sel] = aux_artist

    def add_legend(self, aes, artist, **kwargs):
        pass
