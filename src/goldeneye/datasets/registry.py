"""Dataset registry for goldeneye.

Provides a unified interface for loading geospatial VLM datasets.
"""

from __future__ import annotations

from collections.abc import Callable

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

from goldeneye.datasets.changechat import load_changechat
from goldeneye.datasets.de_dataset import load_de_dataset
from goldeneye.datasets.disaster_m3 import load_disaster_m3
from goldeneye.datasets.earthdial_dataset import load_earthdial_dataset
from goldeneye.datasets.evattrs import load_evattrs
from goldeneye.datasets.fair1m_caption import load_fair1m_caption
from goldeneye.datasets.geochat_bench import load_geochat_bench
from goldeneye.datasets.geochat_instruct import load_geochat_instruct
from goldeneye.datasets.geopix_instruct import load_geopix_instruct
from goldeneye.datasets.geotext1652 import load_geotext1652
from goldeneye.datasets.geozero_eval import load_geozero_eval
from goldeneye.datasets.gres import load_gres
from goldeneye.datasets.landsat30_au import load_landsat_captions
from goldeneye.datasets.lrs_gro import load_lrs_gro
from goldeneye.datasets.lrs_vqa import load_lrs_vqa
from goldeneye.datasets.nwpu_captions import load_nwpu_captions
from goldeneye.datasets.pregres import load_pregres
from goldeneye.datasets.refgeo import load_refgeo
from goldeneye.datasets.rs5m import load_rs5m
from goldeneye.datasets.rs_eot import load_rs_eot
from goldeneye.datasets.rs_visual_instructions import load_rs_visual_instructions
from goldeneye.datasets.rs_vqa_waltonfuture import load_rs_vqa
from goldeneye.datasets.rscc import load_rscc
from goldeneye.datasets.rscid_captions import load_rscid_captions
from goldeneye.datasets.rsicd import load_rsicd
from goldeneye.datasets.rsteller import load_rsteller
from goldeneye.datasets.rsvqa_hr import load_rsvqa_hr
from goldeneye.datasets.sarlang import load_sarlang
from goldeneye.datasets.sydney_captions import load_sydney_captions
from goldeneye.datasets.uavbench import load_uavbench
from goldeneye.datasets.uavit_1m import load_uavit_1m
from goldeneye.datasets.ucm_captions import load_ucm_captions
from goldeneye.datasets.urbench import load_urbench
from goldeneye.datasets.vhm_versad import load_vhm_versad
from goldeneye.datasets.vrsbench import load_vrsbench
from goldeneye.datasets.xhrbench import load_xhrbench
from goldeneye.datasets.xlrs_bench import load_xlrs_bench

DatasetType = Dataset | DatasetDict | IterableDataset | IterableDatasetDict

# Maps codename -> HuggingFace dataset ID
_DATASET_IDS: dict[str, str] = {
    # Benchmarks
    "xlrs-bench": "initiacms/XLRS-Bench-lite",
    "de-dataset": "earth-insights/DE-Dataset",
    "urbench": "Yiyuan/URBench",
    "geochat-bench": "MBZUAI/GeoChat_Bench",
    "vrsbench": "lhrs/VRSBench",
    "xhrbench": "XHRBench/XHRBench",
    "geozero-eval": "Peng-YM/RS-Eval-VQA-Dataset",
    # Captioning datasets
    "rsicd": "isaaccorley/rsicd",
    "ucm-captions": "isaaccorley/ucm-captions",
    "sydney-captions": "isaaccorley/sydney-captions",
    "nwpu-captions": "isaaccorley/nwpu-captions",
    "fair1m-caption": "wangzhecheng/fair1m_caption",
    "rs5m": "zilongzhong/RS5M",
    "rscid-captions": "isaaccorley/rscid-captions",
    "landsat-captions": "wangzhecheng/Landsat_30_AU",
    "rsteller": "BigData-KSU/RSTeller",
    # VQA datasets
    "rs-vqa": "waltonfuture/LHRS-VQA",
    "rsvqa-hr": "jonathan-roberts1/RSVQA-HR",
    "geochat-instruct": "MBZUAI/GeoChat_Instruct",
    "lrs-vqa": "jonathan-roberts1/LRS-VQA",
    "lrs-gro": "jonathan-roberts1/LRS-GRO",
    # Instruction datasets
    "rs-visual-instructions": "shizukanao/RS-visual-instructions",
    "uavit-1m": "HIT-UAV/UAVIT-1M",
    "uavbench": "ZhanYang-nwpu/UAVBench",
    "geopix-instruct": "MBZUAI/GeoPixInstruct",
    "earthdial": "akshaydudhane/EarthDial-Dataset",
    # Chain-of-Thought datasets
    "rs-eot": "jirvin16/RS-EoT",
    # Change captioning datasets
    "rscc": "HIT-UAV/RSCC",
    "changechat": "HIT-UAV/ChangeChat",
    # Geo-localization datasets
    "geotext1652": "layneins/GeoText-1652",
    # Referring expression datasets
    "refgeo": "VisionXLab/RefGeo",
    "pregres": "Pregres/PreGRes",
    "gres": "Geo-R1/GRES",
    # Attribute datasets
    "evattrs": "liarzone/EVAttrs-95K",
    # SAR datasets
    "sarlang": "XAI4SAR/SARLang-1M",
    # Disaster datasets
    "disaster-m3": "Kingdrone-Junjue/DisasterM3",
    # Pre-training datasets
    "vhm-versad": "FitzPC/VHM_VersaD",
}

# Maps codename -> loader function
_DATASET_LOADERS: dict[str, Callable[..., DatasetType]] = {
    # Benchmarks
    "xlrs-bench": load_xlrs_bench,
    "de-dataset": load_de_dataset,
    "urbench": load_urbench,
    "geochat-bench": load_geochat_bench,
    "vrsbench": load_vrsbench,
    "xhrbench": load_xhrbench,
    "geozero-eval": load_geozero_eval,
    # Captioning datasets
    "rsicd": load_rsicd,
    "ucm-captions": load_ucm_captions,
    "sydney-captions": load_sydney_captions,
    "nwpu-captions": load_nwpu_captions,
    "fair1m-caption": load_fair1m_caption,
    "rs5m": load_rs5m,
    "rscid-captions": load_rscid_captions,
    "landsat-captions": load_landsat_captions,
    "rsteller": load_rsteller,
    # VQA datasets
    "rs-vqa": load_rs_vqa,
    "rsvqa-hr": load_rsvqa_hr,
    "geochat-instruct": load_geochat_instruct,
    "lrs-vqa": load_lrs_vqa,
    "lrs-gro": load_lrs_gro,
    # Instruction datasets
    "rs-visual-instructions": load_rs_visual_instructions,
    "uavit-1m": load_uavit_1m,
    "uavbench": load_uavbench,
    "geopix-instruct": load_geopix_instruct,
    "earthdial": load_earthdial_dataset,
    # Chain-of-Thought datasets
    "rs-eot": load_rs_eot,
    # Change captioning datasets
    "rscc": load_rscc,
    "changechat": load_changechat,
    # Geo-localization datasets
    "geotext1652": load_geotext1652,
    # Referring expression datasets
    "refgeo": load_refgeo,
    "pregres": load_pregres,
    "gres": load_gres,
    # Attribute datasets
    "evattrs": load_evattrs,
    # SAR datasets
    "sarlang": load_sarlang,
    # Disaster datasets
    "disaster-m3": load_disaster_m3,
    # Pre-training datasets
    "vhm-versad": load_vhm_versad,
}


def list_datasets() -> list[str]:
    """List all available dataset codenames.

    Returns
    -------
    list[str]
        List of dataset codenames that can be passed to load_dataset()

    Examples
    --------
    >>> import goldeneye
    >>> goldeneye.list_datasets()
    ['xlrs-bench', 'de-dataset', 'urbench', ...]
    """
    return list(_DATASET_IDS.keys())


def get_dataset_info(codename: str) -> dict[str, str]:
    """Get information about a dataset.

    Parameters
    ----------
    codename : str
        Dataset identifier from list_datasets()

    Returns
    -------
    dict[str, str]
        Dictionary with 'codename' and 'hf_id' keys

    Raises
    ------
    ValueError
        If the codename is not found in the registry
    """
    if codename not in _DATASET_IDS:
        msg = f"Dataset {codename} not found. Available datasets: {list_datasets()}"
        raise ValueError(msg)

    return {
        "codename": codename,
        "hf_id": _DATASET_IDS[codename],
    }


def load_dataset(
    codename: str,
    split: str | None = None,
    streaming: bool = False,
    cache_dir: str | None = None,
    **kwargs: object,
) -> DatasetType:
    """Load a geospatial VLM dataset by codename.

    Parameters
    ----------
    codename : str
        Dataset identifier from list_datasets()
    split : str | None, optional
        Dataset split to load (e.g., 'train', 'test'), by default None
    streaming : bool, optional
        If True, stream the dataset without downloading it entirely, by default False
    cache_dir : str | None, optional
        Directory to cache the dataset, by default None
    **kwargs : object
        Additional keyword arguments passed to the underlying loader function

    Returns
    -------
    Dataset | DatasetDict | IterableDataset | IterableDatasetDict
        The loaded dataset

    Raises
    ------
    ValueError
        If the codename is not found in the registry

    Examples
    --------
    >>> import goldeneye
    >>> # List available datasets
    >>> goldeneye.list_datasets()
    ['xlrs-bench', 'de-dataset', 'urbench', ...]
    >>> # Load a dataset
    >>> dataset = goldeneye.load_dataset("xlrs-bench", split="train")
    >>> # Stream a dataset
    >>> dataset = goldeneye.load_dataset("rsicd", streaming=True)
    """
    if codename not in _DATASET_LOADERS:
        msg = f"Dataset {codename} not found. Available datasets: {list_datasets()}"
        raise ValueError(msg)

    loader_func = _DATASET_LOADERS[codename]

    # Build kwargs for the loader function
    load_kwargs: dict[str, object] = {"streaming": streaming}
    if split is not None:
        load_kwargs["split"] = split
    if cache_dir is not None:
        load_kwargs["cache_dir"] = cache_dir
    load_kwargs.update(kwargs)

    return loader_func(**load_kwargs)
