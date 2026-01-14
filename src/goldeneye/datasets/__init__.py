"""Goldeneye Dataset Loaders.

This module provides loaders for remote sensing VLM datasets.
All loaders support both standard loading and streaming.
"""

from goldeneye.datasets.changechat import load_changechat, stream_changechat
from goldeneye.datasets.de_dataset import load_de_dataset
from goldeneye.datasets.disaster_m3 import load_disaster_m3
from goldeneye.datasets.earthdial_dataset import load_earthdial_dataset
from goldeneye.datasets.evattrs import load_evattrs, stream_evattrs
from goldeneye.datasets.fair1m_caption import load_fair1m_caption
from goldeneye.datasets.geochat_bench import load_geochat_bench
from goldeneye.datasets.geochat_instruct import load_geochat_instruct
from goldeneye.datasets.geopix_instruct import load_geopix_instruct
from goldeneye.datasets.geotext1652 import load_geotext1652, stream_geotext1652
from goldeneye.datasets.geozero_eval import load_geozero_eval
from goldeneye.datasets.gres import load_gres, stream_gres
from goldeneye.datasets.landsat30_au import (
    load_landsat_captions,
    stream_landsat_captions,
)
from goldeneye.datasets.lrs_gro import load_lrs_gro
from goldeneye.datasets.lrs_vqa import load_lrs_vqa
from goldeneye.datasets.nwpu_captions import load_nwpu_captions
from goldeneye.datasets.pregres import load_pregres, stream_pregres
from goldeneye.datasets.refgeo import load_refgeo, stream_refgeo
from goldeneye.datasets.rs5m import load_rs5m
from goldeneye.datasets.rs_eot import load_rs_eot, stream_rs_eot
from goldeneye.datasets.rs_visual_instructions import (
    load_rs_visual_instructions,
)
from goldeneye.datasets.rs_vqa_waltonfuture import load_rs_vqa
from goldeneye.datasets.rscc import load_rscc, stream_rscc
from goldeneye.datasets.rscid_captions import load_rscid_captions
from goldeneye.datasets.rsicd import load_rsicd
from goldeneye.datasets.rsteller import load_rsteller, stream_rsteller
from goldeneye.datasets.rsvqa_hr import load_rsvqa_hr
from goldeneye.datasets.sarlang import load_sarlang, stream_sarlang
from goldeneye.datasets.sydney_captions import load_sydney_captions
from goldeneye.datasets.uavbench import load_uavbench
from goldeneye.datasets.uavit_1m import load_uavit_1m
from goldeneye.datasets.ucm_captions import load_ucm_captions
from goldeneye.datasets.urbench import load_urbench, stream_urbench
from goldeneye.datasets.vhm_versad import load_vhm_versad
from goldeneye.datasets.vrsbench import load_vrsbench
from goldeneye.datasets.xhrbench import load_xhrbench
from goldeneye.datasets.xlrs_bench import load_xlrs_bench

__all__ = [
    # Benchmarks
    "load_de_dataset",
    "load_xlrs_bench",
    "load_urbench",
    "stream_urbench",
    # Captioning datasets
    "load_rsicd",
    "load_ucm_captions",
    "load_sydney_captions",
    "load_nwpu_captions",
    "load_fair1m_caption",
    "load_rs5m",
    "load_rscid_captions",
    "load_landsat_captions",
    "stream_landsat_captions",
    "load_rsteller",
    "stream_rsteller",
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
    "stream_rs_eot",
    # Change captioning datasets
    "load_rscc",
    "stream_rscc",
    "load_changechat",
    "stream_changechat",
    # Geo-localization datasets
    "load_geotext1652",
    "stream_geotext1652",
    # Referring expression datasets
    "load_refgeo",
    "stream_refgeo",
    "load_pregres",
    "stream_pregres",
    "load_gres",
    "stream_gres",
    # Attribute datasets
    "load_evattrs",
    "stream_evattrs",
    # SAR datasets
    "load_sarlang",
    "stream_sarlang",
    # Disaster datasets
    "load_disaster_m3",
    # Pre-training datasets
    "load_vhm_versad",
]
