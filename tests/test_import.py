# pylint: disable=no-self-use, redefined-outer-name
import pytest

from xarray_einstats import tutorial
from xrtist import PlotCollection


@pytest.fixture(scope="module")
def dataarray():
    return tutorial.generate_mcmc_like_dataset(3)["mu"]


class TestPlotCollectionInit:
    def test_wrap(self, dataarray):
        pc = PlotCollection.wrap(
            dataarray, cols=["team"], aes={"color": ["chain"]}, color=[f"C{i}" for i in range(4)]
        )
        assert "plot" in pc.viz
        assert "chart" in pc.viz
        assert "chain" in pc.ds

    def test_grid(self, dataarray):
        pc = PlotCollection.grid(
            dataarray,
            cols=["team"],
            rows=["chain"],
            aes={"color": ["chain"]},
            color=[f"C{i}" for i in range(4)],
        )
        assert "plot" in pc.viz
        assert "team" in pc.viz["plot"].dims
        assert "chain" in pc.viz["plot"].dims
        assert "chart" in pc.viz
        assert "chain" in pc.ds
