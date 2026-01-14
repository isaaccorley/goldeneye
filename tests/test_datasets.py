"""Tests for dataset loaders."""

import goldeneye.datasets as datasets


def test_dataset_loaders_available() -> None:
    """Test that all dataset loaders are exported correctly."""
    # Captioning datasets
    assert hasattr(datasets, "load_rsicd")
    assert hasattr(datasets, "stream_rsicd")
    assert hasattr(datasets, "load_ucm_captions")
    assert hasattr(datasets, "stream_ucm_captions")
    assert hasattr(datasets, "load_sydney_captions")
    assert hasattr(datasets, "stream_sydney_captions")
    assert hasattr(datasets, "load_nwpu_captions")
    assert hasattr(datasets, "stream_nwpu_captions")
    assert hasattr(datasets, "load_fair1m_caption")
    assert hasattr(datasets, "stream_fair1m_caption")
    assert hasattr(datasets, "load_rs5m")
    assert hasattr(datasets, "stream_rs5m")

    # VQA datasets
    assert hasattr(datasets, "load_rs_vqa")
    assert hasattr(datasets, "stream_rs_vqa")
    assert hasattr(datasets, "load_rsvqa_hr")
    assert hasattr(datasets, "stream_rsvqa_hr")
    assert hasattr(datasets, "load_geochat_instruct")
    assert hasattr(datasets, "stream_geochat_instruct")
    assert hasattr(datasets, "load_geochat_bench")
    assert hasattr(datasets, "stream_geochat_bench")

    # Instruction datasets
    assert hasattr(datasets, "load_rs_visual_instructions")
    assert hasattr(datasets, "stream_rs_visual_instructions")

    # Existing datasets
    assert hasattr(datasets, "load_de_dataset")
    assert hasattr(datasets, "stream_de_dataset")
    assert hasattr(datasets, "load_xlrs_bench")
    assert hasattr(datasets, "stream_xlrs_bench")


def test_dataset_exports_in_all() -> None:
    """Test that all loaders are in __all__."""
    expected_exports = [
        # DE-Dataset
        "load_de_dataset",
        "stream_de_dataset",
        # XLRS-Bench
        "load_xlrs_bench",
        "stream_xlrs_bench",
        # RS Captioning
        "load_rsicd",
        "stream_rsicd",
        "load_ucm_captions",
        "stream_ucm_captions",
        "load_sydney_captions",
        "stream_sydney_captions",
        "load_nwpu_captions",
        "stream_nwpu_captions",
        "load_fair1m_caption",
        "stream_fair1m_caption",
        "load_rs5m",
        "stream_rs5m",
        # RS VQA
        "load_rs_vqa",
        "stream_rs_vqa",
        "load_rsvqa_hr",
        "stream_rsvqa_hr",
        "load_geochat_instruct",
        "stream_geochat_instruct",
        "load_geochat_bench",
        "stream_geochat_bench",
        # RS Instructions
        "load_rs_visual_instructions",
        "stream_rs_visual_instructions",
    ]
    for export in expected_exports:
        assert export in datasets.__all__, f"{export} not in __all__"
