"""Tests for dataset loaders."""

import goldeneye.datasets as datasets


def test_dataset_loaders_available() -> None:
    """Test that all dataset loaders are exported correctly."""
    # Captioning datasets
    assert hasattr(datasets, "load_rsicd")
    assert hasattr(datasets, "load_ucm_captions")
    assert hasattr(datasets, "load_sydney_captions")
    assert hasattr(datasets, "load_nwpu_captions")
    assert hasattr(datasets, "load_fair1m_caption")
    assert hasattr(datasets, "load_rs5m")
    assert hasattr(datasets, "load_landsat_captions")
    assert hasattr(datasets, "load_rsteller")

    # VQA datasets
    assert hasattr(datasets, "load_rs_vqa")
    assert hasattr(datasets, "load_rsvqa_hr")
    assert hasattr(datasets, "load_geochat_instruct")
    assert hasattr(datasets, "load_geochat_bench")
    assert hasattr(datasets, "load_vrsbench")
    assert hasattr(datasets, "load_lrs_vqa")
    assert hasattr(datasets, "load_xhrbench")
    assert hasattr(datasets, "load_lrs_gro")

    # Instruction datasets
    assert hasattr(datasets, "load_rs_visual_instructions")
    assert hasattr(datasets, "load_geozero_eval")
    assert hasattr(datasets, "load_uavit_1m")
    assert hasattr(datasets, "load_uavbench")
    assert hasattr(datasets, "load_geopix_instruct")
    assert hasattr(datasets, "load_earthdial_dataset")

    # Benchmarks
    assert hasattr(datasets, "load_de_dataset")
    assert hasattr(datasets, "load_xlrs_bench")
    assert hasattr(datasets, "load_urbench")

    # Chain-of-Thought datasets
    assert hasattr(datasets, "load_rs_eot")

    # Change captioning datasets
    assert hasattr(datasets, "load_rscc")
    assert hasattr(datasets, "load_changechat")

    # Geo-localization datasets
    assert hasattr(datasets, "load_geotext1652")

    # Referring expression datasets
    assert hasattr(datasets, "load_refgeo")
    assert hasattr(datasets, "load_pregres")
    assert hasattr(datasets, "load_gres")

    # Attribute datasets
    assert hasattr(datasets, "load_evattrs")

    # SAR datasets
    assert hasattr(datasets, "load_sarlang")

    # Disaster datasets
    assert hasattr(datasets, "load_disaster_m3")

    # Pre-training datasets
    assert hasattr(datasets, "load_vhm_versad")


def test_dataset_exports_in_all() -> None:
    """Test that all loaders are in __all__."""
    expected_exports = [
        # Benchmarks
        "load_de_dataset",
        "load_xlrs_bench",
        "load_urbench",
        # Captioning datasets
        "load_rsicd",
        "load_ucm_captions",
        "load_sydney_captions",
        "load_nwpu_captions",
        "load_fair1m_caption",
        "load_rs5m",
        "load_rscid_captions",
        "load_landsat_captions",
        "load_rsteller",
        # VQA datasets
        "load_rs_vqa",
        "load_rsvqa_hr",
        "load_geochat_instruct",
        "load_geochat_bench",
        "load_vrsbench",
        "load_lrs_vqa",
        "load_xhrbench",
        "load_lrs_gro",
        # Instruction datasets
        "load_rs_visual_instructions",
        "load_geozero_eval",
        "load_uavit_1m",
        "load_uavbench",
        "load_geopix_instruct",
        "load_earthdial_dataset",
        # Chain-of-Thought datasets
        "load_rs_eot",
        # Change captioning datasets
        "load_rscc",
        "load_changechat",
        # Geo-localization datasets
        "load_geotext1652",
        # Referring expression datasets
        "load_refgeo",
        "load_pregres",
        "load_gres",
        # Attribute datasets
        "load_evattrs",
        # SAR datasets
        "load_sarlang",
        # Disaster datasets
        "load_disaster_m3",
        # Pre-training datasets
        "load_vhm_versad",
    ]
    for export in expected_exports:
        assert export in datasets.__all__, f"{export} not in __all__"
