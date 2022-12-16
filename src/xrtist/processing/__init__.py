import arviz as az

__all__ = ("kde",)


def kde(da, dims=None, grid_len=512, **kwargs):
    if dims is None:
        dims = ["chain", "draw"]
    return az.wrap_xarray_ufunc(
        az.kde,
        da,
        ufunc_kwargs={"n_output": 2, "n_input": 1, "n_dims": len(dims)},
        func_kwargs={**kwargs, "out_shape": [(grid_len,), (grid_len,)], "grid_len": grid_len},
        output_core_dims=[["kde_dim"], ["kde_dim"]],
        input_core_dims=[dims],
    )
