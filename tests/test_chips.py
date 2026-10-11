import pytest

from burnscars.data.chips import parse_name


def test_parse_observed_name():
    r = parse_name("subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif")
    assert r["chip_id"] == "T10SEH.2018190.v1"          # IBM split-list form
    assert (r["tile"], r["utm_zone"], r["lat_band"]) == ("T10SEH", 10, "S")
    assert (r["year"], r["doy"], r["date"]) == (2018, 190, "2018-07-09")
    assert r["kind"] == "scene" and r["sensor"] == "S30" and r["hls_version"] == "v1.4"
    assert parse_name("HLS.S30.T10SEH.2018190.v1.4.mask.tif")["kind"] == "mask"


@pytest.mark.parametrize("bad", ["HLS.S30.T10SEH.2018400.v1.4_merged.tif",
                                 "HLS.S30.T10SEH.2018190.v1.4.tif", "foo.tif"])
def test_parse_rejects(bad):
    with pytest.raises(ValueError):
        parse_name(bad)
