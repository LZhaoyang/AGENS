"""Build a multimodal CSI-text dataset with Sionna RT.

This script:
1. Loads several built-in Sionna ray-tracing scenes.
2. Samples UE positions from a radio map under multiple channel-condition presets.
3. Generates CIRs and frequency-domain CSI samples.
4. Derives simple channel descriptors from CIRs.
5. Creates English user-intent text aligned with each CSI sample.
6. Saves the dataset to an HDF5 file and provides a PyTorch Dataset wrapper.

The implementation is organized for readability and publication:
- English-only comments and docstrings
- Clear configuration objects
- Consistent naming
- Safer receiver management
- Removal of duplicated code blocks
- Explicit bug fixes for label handling and path-solver placement
"""

from __future__ import annotations

import gc
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import drjit as dr
import h5py
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader, Dataset

# Sionna imports
try:
    import sionna.phy
    import sionna.rt
except ImportError as exc:
    import sys

    if "google.colab" in sys.modules:
        print("Installing Sionna and restarting the runtime. Please run the cell again.")
        os.system("pip install sionna")
        os.kill(os.getpid(), 5)
    raise exc

from sionna.phy.channel import CIRDataset, GenerateOFDMChannel
from sionna.phy.ofdm import ResourceGrid
from sionna.rt import PathSolver, PlanarArray, RadioMapSolver, Receiver, Transmitter, load_scene


# =============================================================================
# Runtime setup
# =============================================================================

def configure_runtime(cuda_visible_devices: Optional[str] = None) -> None:
    """Configure TensorFlow/Sionna runtime behavior."""
    if os.getenv("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "" if cuda_visible_devices is None else cuda_visible_devices

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as exc:
            print(exc)

    tf.get_logger().setLevel("ERROR")

    gc.collect()
    dr.flush_malloc_cache()


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class DistanceConfig:
    name: str
    min_dist: float
    max_dist: float


@dataclass(frozen=True)
class GainConfig:
    name: str
    min_gain_db: float
    max_gain_db: float


@dataclass(frozen=True)
class MultipathConfig:
    name: str
    max_depth: int
    max_num_paths_per_src: int


@dataclass(frozen=True)
class SpeedConfig:
    name: str
    velocity: Tuple[float, float, float]


@dataclass(frozen=True)
class SystemConfig:
    subcarrier_spacing: float = 30e3
    num_tx: int = 1
    num_rx: int = 1
    num_tx_ant: int = 4
    num_rx_ant: int = 2
    batch_size_cir: int = 10
    target_num_cirs: int = 50
    fft_size: int = 128
    cp_len: int = 10
    num_time_steps: int = 14
    pilot_symbol_indices: Tuple[int, int] = (2, 9)
    radio_map_max_depth: int = 5
    samples_per_tx: int = 10**7
    output_h5: str = "/home/ubuntu/MC-LLM/csi_text_dataset.h5"


SCENES: Dict[str, object] = {
    "munich": sionna.rt.scene.munich,
    "etoile": sionna.rt.scene.etoile,
    "florence": sionna.rt.scene.florence,
    "san_francisco": sionna.rt.scene.san_francisco,
}

TX_POSITIONS: Dict[str, List[float]] = {
    "munich": [8.5, 21.0, 27.0],
    "etoile": [8.5, 21.0, 27.0],
    "florence": [8.5, 21.0, 27.0],
    "san_francisco": [8.5, 21.0, 27.0],
}

TX_LOOK_AT: Dict[str, List[float]] = {
    "munich": [45.0, 90.0, 1.5],
    "etoile": [45.0, 90.0, 1.5],
    "florence": [45.0, 90.0, 1.5],
    "san_francisco": [45.0, 90.0, 1.5],
}

DISTANCE_CONFIGS = [
    DistanceConfig("near", 10.0, 400.0),
    DistanceConfig("mid", 20.0, 400.0),
    DistanceConfig("far", 30.0, 450.0),
]

GAIN_CONFIGS = [
    GainConfig("all_gain", -130.0, 0.0),
    GainConfig("weak_only", -140.0, -90.0),
    GainConfig("medium", -110.0, -60.0),
]

MULTIPATH_CONFIGS = [
    MultipathConfig("few_paths", 5, 10000),
    MultipathConfig("normal", 8, 8000),
    MultipathConfig("rich", 6, 4000),
]

SPEED_CONFIGS = [
    SpeedConfig("static", (0.5, 0.5, 0.0)),
    SpeedConfig("pedestrian", (1.0, 1.0, 0.0)),
    SpeedConfig("vehicular", (3.0, 3.0, 0.0)),
]

REQUEST_CLASSES = [
    "high_data_rate",
    "urllc",
    "energy_saving",
]

USER_NEED_TEMPLATES = {
    "high_data_rate": [
        "I would like the network to offer as high a data rate as possible for HD video streaming and large file downloads.",
        "Please prioritize throughput. I need to transfer a large amount of data and require a high data rate.",
        "This session needs high bandwidth support. Please maximize the downlink data rate.",
        "Configure the link for a high-throughput scenario, such as video streaming or cloud gaming.",
    ],
    "urllc": [
        "I care more about reliability and latency. The bit error rate should be extremely low with minimal delay.",
        "This is similar to an industrial control scenario and requires ultra-reliable, low-latency communication.",
        "Please prioritize link stability and delay, even at the cost of reduced data rate.",
        "This link must avoid packet loss and retransmissions as much as possible, with very strict latency requirements.",
    ],
    "energy_saving": [
        "I want to minimize the transmit power and prioritize energy-efficient communication.",
        "The current service does not need a high data rate, so reducing power consumption and extending battery life is preferred.",
        "Please reduce power consumption as much as possible while maintaining basic connectivity.",
        "I prefer a low-power configuration, even if that means sacrificing some throughput.",
    ],
}


# =============================================================================
# Resource grid and channel helpers
# =============================================================================

def build_resource_grid(cfg: SystemConfig) -> ResourceGrid:
    """Create the OFDM resource grid."""
    return ResourceGrid(
        num_ofdm_symbols=cfg.num_time_steps,
        fft_size=cfg.fft_size,
        subcarrier_spacing=cfg.subcarrier_spacing,
        num_tx=cfg.num_tx,
        num_guard_carriers=[9, 10],
        dc_null=True,
        num_streams_per_tx=cfg.num_tx_ant,
        cyclic_prefix_length=cfg.cp_len,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=list(cfg.pilot_symbol_indices),
    )


class CIRGenerator:
    """Yield one CIR sample at a time for Sionna's CIRDataset wrapper."""

    def __init__(self, a: np.ndarray, tau: np.ndarray):
        self._a = tf.constant(a, tf.complex64)
        self._tau = tf.constant(tau, tf.float32)
        self._num_samples = int(self._a.shape[0])

    def __call__(self):
        index = 0
        while True:
            yield self._a[index], self._tau[index]
            index = (index + 1) % self._num_samples


# =============================================================================
# Text-generation helpers
# =============================================================================

def extract_channel_features_from_cir(
    a_i: np.ndarray,
    tau_i: np.ndarray,
    ue_pos_i: np.ndarray,
    tx_pos: np.ndarray,
) -> Dict[str, float]:
    """Extract simple physical descriptors from a single CIR sample."""
    a_i = np.asarray(a_i)
    tau_i = np.asarray(tau_i)
    ue_pos_i = np.asarray(ue_pos_i)
    tx_pos = np.asarray(tx_pos)

    assert a_i.ndim >= 2, "CIR tensor must have at least [paths, time_steps] dimensions."
    *antenna_dims, _, _ = a_i.shape

    if antenna_dims:
        power_paths_time = np.mean(np.abs(a_i) ** 2, axis=tuple(range(len(antenna_dims))))
    else:
        power_paths_time = np.abs(a_i) ** 2

    power_per_path = np.mean(power_paths_time, axis=-1)
    total_power = float(np.sum(power_per_path) + 1e-12)
    total_power_db = 10.0 * np.log10(total_power)

    strongest_path_power = float(np.max(power_per_path) + 1e-12)
    effective_path_mask = power_per_path > (strongest_path_power * 1e-2)
    num_effective_paths = int(np.sum(effective_path_mask))

    tau_paths = tau_i.reshape(-1, tau_i.shape[-1])[0]
    tau_max = float(np.max(tau_paths))

    weights = power_per_path / (np.sum(power_per_path) + 1e-12)
    tau_mean = float(np.sum(weights * tau_paths))
    tau_rms = float(np.sqrt(np.sum(weights * (tau_paths - tau_mean) ** 2)))

    distance = float(np.linalg.norm(ue_pos_i - tx_pos))
    los_like = bool(power_per_path[0] / (total_power + 1e-12) > 0.8)

    return {
        "total_power_db": total_power_db,
        "num_eff_paths": num_effective_paths,
        "tau_rms": tau_rms,
        "tau_max": tau_max,
        "distance": distance,
        "los_like": los_like,
    }


def describe_environment(features: Dict[str, float], rng: random.Random) -> str:
    """Render an optional environment sentence from physical channel features."""
    dist = features["distance"]
    total_power_db = features["total_power_db"]
    num_paths = features["num_eff_paths"]
    tau_rms_us = features["tau_rms"] * 1e6
    los_like = features["los_like"]

    if dist < 80:
        position_desc = "near the cell center close to the base station"
    elif dist < 200:
        position_desc = "somewhere in the middle of the cell"
    else:
        position_desc = "near the cell edge"

    fading_desc = (
        "dominated by a line-of-sight component with relatively mild fading"
        if los_like
        else "strongly affected by non-line-of-sight multipath fading"
    )

    if num_paths <= 2:
        multipath_desc = "only a few multipath components, and the channel is close to flat fading"
    elif num_paths <= 6:
        multipath_desc = "a moderate number of multipath components"
    else:
        multipath_desc = "many multipath components and strong frequency selectivity"

    draw = rng.random()
    if draw < 0.3:
        return ""

    if draw < 0.7:
        candidates = [
            f"The user is located {position_desc}, and the channel is {fading_desc}.",
            f"The current position is {position_desc}, where {multipath_desc} is observed.",
            f"The user is outdoors in an urban environment, roughly {position_desc}, and the channel is {fading_desc}.",
        ]
        return rng.choice(candidates)

    detailed = [
        (
            f"The user is about {dist:.1f} meters away from the base station, located {position_desc}, "
            f"with an overall path gain of about {total_power_db:.1f} dB. The channel exhibits {multipath_desc}, "
            f"with an RMS delay spread of around {tau_rms_us:.2f} microseconds and is {fading_desc}."
        ),
        (
            f"According to the ray-tracing results, the UE-BS distance is approximately {dist:.0f} meters, "
            f"with an aggregated path gain of about {total_power_db:.1f} dB. There are roughly {num_paths} effective "
            f"multipath components and an RMS delay spread of {tau_rms_us:.2f} microseconds, indicating a channel that is {fading_desc}."
        ),
        (
            f"In this location, the channel has about {num_paths} multipath components and an RMS delay spread on the order of "
            f"{tau_rms_us:.2f} microseconds. The user is {position_desc}, and the overall path loss is "
            f"{'moderate' if total_power_db > -90 else 'quite significant'}."
        ),
    ]
    return rng.choice(detailed)


def sample_label_from_features(features: Dict[str, float], rng: random.Random) -> int:
    """Sample a request class with a weak bias from channel conditions.

    Index mapping is consistent with REQUEST_CLASSES:
        0 -> high_data_rate
        1 -> urllc
        2 -> energy_saving
    """
    dist = features["distance"]
    power_db = features["total_power_db"]

    if dist < 80 and power_db > -90:
        draw = rng.random()
        if draw < 0.55:
            return 0  # high_data_rate
        if draw < 0.80:
            return 2  # energy_saving
        return 1      # urllc

    if dist > 200 or power_db < -100:
        draw = rng.random()
        if draw < 0.50:
            return 1  # urllc
        if draw < 0.80:
            return 2  # energy_saving
        return 0      # high_data_rate

    draw = rng.random()
    if draw < 0.34:
        return 0
    if draw < 0.67:
        return 1
    return 2


def render_text_for_sample(features: Dict[str, float], label_id: int, rng: random.Random) -> Tuple[str, str]:
    """Create the final English text description for one sample."""
    label_name = REQUEST_CLASSES[label_id]
    need_sentence = rng.choice(USER_NEED_TEMPLATES[label_name])
    environment_sentence = describe_environment(features, rng)

    if not environment_sentence:
        return need_sentence, label_name

    if rng.random() < 0.5:
        return f"{environment_sentence} {need_sentence}", label_name
    return f"{need_sentence} {environment_sentence}", label_name


def generate_text_dataset_from_csi(
    a: np.ndarray,
    tau: np.ndarray,
    ue_pos_all: np.ndarray,
    tx_pos: np.ndarray,
    seed: int = 0,
    use_feature_based_label: bool = True,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Generate intent text, numeric labels, and label names for CSI samples."""
    rng = random.Random(seed)
    a = np.asarray(a)
    tau = np.asarray(tau)
    ue_pos_all = np.asarray(ue_pos_all)

    texts: List[str] = []
    label_ids: List[int] = []
    label_names: List[str] = []

    for i in range(a.shape[0]):
        features = extract_channel_features_from_cir(a[i], tau[i], ue_pos_all[i], tx_pos)
        if use_feature_based_label:
            label_id = sample_label_from_features(features, rng)
        else:
            label_id = rng.randint(0, len(REQUEST_CLASSES) - 1)

        text_i, label_name_i = render_text_for_sample(features, label_id, rng)
        texts.append(text_i)
        label_ids.append(label_id)
        label_names.append(label_name_i)

    return texts, np.asarray(label_ids, dtype=np.int32), np.asarray(label_names, dtype=object)


# =============================================================================
# Sionna scene helpers
# =============================================================================

def safe_remove_receiver(scene, name: str) -> None:
    """Remove a receiver if it already exists."""
    try:
        scene.remove(name)
    except Exception:
        pass


def setup_scene(scene_name: str, cfg: SystemConfig):
    """Load a Sionna scene and configure BS/UE arrays."""
    scene = load_scene(SCENES[scene_name])

    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=cfg.num_tx_ant // 2,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="cross",
    )

    tx = Transmitter(
        name="tx",
        position=TX_POSITIONS[scene_name],
        look_at=TX_LOOK_AT[scene_name],
        display_radius=3.0,
    )
    scene.add(tx)

    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=cfg.num_rx_ant // 2,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="cross",
    )

    for i in range(cfg.batch_size_cir):
        receiver_name = f"rx-{i}"
        safe_remove_receiver(scene, receiver_name)
        scene.add(
            Receiver(
                name=receiver_name,
                position=[0.0, 0.0, 1.5],
                velocity=(0.0, 0.0, 0.0),
                display_radius=1.0,
                color=(1.0, 0.0, 0.0),
            )
        )

    return scene


def sample_cirs_for_configuration(
    scene,
    cfg: SystemConfig,
    resource_grid: ResourceGrid,
    scene_name: str,
    dist_cfg: DistanceConfig,
    gain_cfg: GainConfig,
    mp_cfg: MultipathConfig,
    speed_cfg: SpeedConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate CIR, delay, UE position, and frequency-domain CSI arrays for one config."""
    print(
        f"\n--> Scene [{scene_name}], "
        f"Config [dist-{dist_cfg.name}_gain-{gain_cfg.name}_mp-{mp_cfg.name}_spd-{speed_cfg.name}]"
    )

    rm_solver = RadioMapSolver()
    print("Computing radio map...")
    radio_map = rm_solver(
        scene,
        max_depth=cfg.radio_map_max_depth,
        cell_size=(1.0, 1.0),
        samples_per_tx=cfg.samples_per_tx,
    )

    path_solver = PathSolver()
    num_runs = int(np.ceil(cfg.target_num_cirs / cfg.batch_size_cir))

    a_batches: List[np.ndarray] = []
    tau_batches: List[np.ndarray] = []
    ue_position_batches: List[np.ndarray] = []
    max_num_paths = 0

    for run_idx in range(num_runs):
        print(f"   Run {run_idx + 1}/{num_runs}", end="\r")

        ue_pos, _ = radio_map.sample_positions(
            num_pos=cfg.batch_size_cir,
            metric="path_gain",
            min_val_db=gain_cfg.min_gain_db,
            max_val_db=gain_cfg.max_gain_db,
            min_dist=dist_cfg.min_dist,
            max_dist=dist_cfg.max_dist,
            seed=run_idx,
        )

        ue_pos_np = np.asarray(dr.detach(ue_pos))[0]

        for rx_idx in range(cfg.batch_size_cir):
            receiver = scene.receivers[f"rx-{rx_idx}"]
            receiver.position = ue_pos_np[rx_idx]
            receiver.velocity = speed_cfg.velocity

        # Important bug fix:
        # Path solving must happen only once after all receivers are updated.
        paths = path_solver(
            scene,
            max_depth=mp_cfg.max_depth,
            max_num_paths_per_src=mp_cfg.max_num_paths_per_src,
        )

        a_batch, tau_batch = paths.cir(
            sampling_frequency=cfg.subcarrier_spacing,
            num_time_steps=cfg.num_time_steps,
            out_type="numpy",
        )

        a_batches.append(a_batch)
        tau_batches.append(tau_batch)
        ue_position_batches.append(ue_pos_np)
        max_num_paths = max(max_num_paths, a_batch.shape[-2])

    print()

    a_padded = []
    tau_padded = []
    for a_batch, tau_batch in zip(a_batches, tau_batches):
        num_paths = a_batch.shape[-2]
        a_padded.append(
            np.pad(
                a_batch,
                [[0, 0], [0, 0], [0, 0], [0, 0], [0, max_num_paths - num_paths], [0, 0]],
                constant_values=0,
            )
        )
        tau_padded.append(
            np.pad(
                tau_batch,
                [[0, 0], [0, 0], [0, max_num_paths - num_paths]],
                constant_values=0,
            )
        )

    a_cat = np.concatenate(a_padded, axis=0)
    tau_cat = np.concatenate(tau_padded, axis=0)
    ue_pos_all = np.concatenate(ue_position_batches, axis=0)

    # Convert to uplink-style organization.
    a_ul = np.expand_dims(a_cat, axis=0)
    tau_ul = np.expand_dims(tau_cat, axis=0)
    a_ul = np.transpose(a_ul, [1, 0, 2, 3, 4, 5, 6])
    tau_ul = np.transpose(tau_ul, [1, 0, 2, 3])

    link_power = np.sum(np.abs(a_ul) ** 2, axis=(1, 2, 3, 4, 5, 6))
    valid_mask = link_power > 0.0

    a_final = a_ul[valid_mask]
    tau_final = tau_ul[valid_mask]
    ue_pos_final = ue_pos_all[valid_mask]

    cir_generator = CIRGenerator(a_final, tau_final)
    channel_model = CIRDataset(
        cir_generator,
        a_final.shape[0],
        cfg.num_rx,
        cfg.num_rx_ant,
        cfg.num_tx,
        cfg.num_tx_ant,
        a_final.shape[-2],
        cfg.num_time_steps,
    )
    ofdm_channel = GenerateOFDMChannel(channel_model, resource_grid, normalize_channel=True)
    h_freq = ofdm_channel().numpy()

    print(
        "   Final shapes:",
        "a", a_final.shape,
        "tau", tau_final.shape,
        "ue_pos", ue_pos_final.shape,
        "h_freq", h_freq.shape,
    )

    return a_final, tau_final, ue_pos_final, h_freq


# =============================================================================
# Dataset storage and loading
# =============================================================================

def get_h5_string_dtype():
    """Return an HDF5-compatible UTF-8 string dtype."""
    try:
        return h5py.string_dtype(encoding="utf-8")
    except AttributeError:
        return h5py.special_dtype(vlen=str)


def save_dataset_to_h5(
    output_path: str,
    h_freq: np.ndarray,
    label_ids: np.ndarray,
    texts: np.ndarray,
    label_names: np.ndarray,
) -> None:
    """Save the generated dataset to an HDF5 file."""
    string_dtype = get_h5_string_dtype()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, "w") as hf:
        hf.create_dataset("h", data=h_freq)
        hf.create_dataset("label_id", data=label_ids.astype(np.int32))
        hf.create_dataset("text", data=np.asarray(texts, dtype=object), dtype=string_dtype)
        hf.create_dataset("label_name", data=np.asarray(label_names, dtype=object), dtype=string_dtype)


class CSIMultiModalDataset(Dataset):
    """PyTorch dataset for CSI-text multimodal learning.

    Supported HDF5 layouts:
    1. Time-domain CIR version
       - "a": complex CIR
       - optional "tau": path delays
    2. Frequency-domain version
       - "h": complex frequency-domain channel

    Required fields:
       - "label_id"
       - "label_name"
       - "text"
    """

    def __init__(
        self,
        h5_path: str,
        tokenizer=None,
        max_text_len: int = 128,
        in_memory: bool = True,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.in_memory = in_memory

        with h5py.File(h5_path, "r") as hf:
            if "a" in hf:
                self.csi_key = "a"
            elif "h" in hf:
                self.csi_key = "h"
            else:
                raise KeyError(f"Neither 'a' nor 'h' exists in the HDF5 file. Keys: {list(hf.keys())}")

            self.has_tau = "tau" in hf
            lengths = {
                self.csi_key: hf[self.csi_key].shape[0],
                "label_id": hf["label_id"].shape[0],
                "label_name": hf["label_name"].shape[0],
                "text": hf["text"].shape[0],
            }
            if self.has_tau:
                lengths["tau"] = hf["tau"].shape[0]

            self.num_samples = min(lengths.values())

            if in_memory:
                self.csi = hf[self.csi_key][: self.num_samples]
                self.tau = hf["tau"][: self.num_samples] if self.has_tau else None
                self.label_id = hf["label_id"][: self.num_samples]
                self.label_name = np.asarray([
                    x.decode("utf-8") if isinstance(x, bytes) else x
                    for x in hf["label_name"][: self.num_samples]
                ])
                self.text = np.asarray([
                    x.decode("utf-8") if isinstance(x, bytes) else x
                    for x in hf["text"][: self.num_samples]
                ])
            else:
                self.csi = None
                self.tau = None
                self.label_id = None
                self.label_name = None
                self.text = None

    def __len__(self) -> int:
        return self.num_samples

    def _read_single_item(self, idx: int):
        with h5py.File(self.h5_path, "r") as hf:
            csi_i = hf[self.csi_key][idx]
            tau_i = hf["tau"][idx] if self.has_tau else None
            label_id_i = hf["label_id"][idx]
            label_name_i = hf["label_name"][idx]
            text_i = hf["text"][idx]

        if isinstance(label_name_i, bytes):
            label_name_i = label_name_i.decode("utf-8")
        if isinstance(text_i, bytes):
            text_i = text_i.decode("utf-8")

        return csi_i, tau_i, label_id_i, label_name_i, text_i

    def __getitem__(self, idx: int):
        if self.in_memory:
            csi_i = self.csi[idx]
            tau_i = self.tau[idx] if self.has_tau else None
            label_id_i = self.label_id[idx]
            label_name_i = self.label_name[idx]
            text_i = self.text[idx]
        else:
            csi_i, tau_i, label_id_i, label_name_i, text_i = self._read_single_item(idx)

        csi_i = np.asarray(csi_i)
        csi_feat = np.stack([np.real(csi_i), np.imag(csi_i)], axis=-1)

        sample = {
            "csi_a": torch.from_numpy(csi_feat).to(torch.float32),
            "label_id": torch.tensor(label_id_i, dtype=torch.long),
            "label_name": label_name_i,
            "text": text_i,
        }

        if tau_i is not None:
            sample["tau"] = torch.from_numpy(np.asarray(tau_i)).to(torch.float32)

        if self.tokenizer is not None:
            encoded = self.tokenizer(
                text_i,
                truncation=True,
                padding="max_length",
                max_length=self.max_text_len,
                return_tensors="pt",
            )
            sample["input_ids"] = encoded["input_ids"].squeeze(0)
            sample["attention_mask"] = encoded["attention_mask"].squeeze(0)

        return sample


# =============================================================================
# Tensor reshaping helper
# =============================================================================

def csi_ri_to_bsd(csi_ri: torch.Tensor) -> torch.Tensor:
    """Convert CSI from [B,R,Ra,Tx,Ta,S,F,2] to [B,S,D]."""
    assert csi_ri.dim() == 8, f"Expected 8D tensor [B,R,Ra,Tx,Ta,S,F,2], got {csi_ri.shape}."
    bsz, num_rx, num_rx_ant, num_tx, num_tx_ant, num_sym, num_fft, complex_dim = csi_ri.shape
    assert complex_dim == 2, f"The last dimension must be 2 for real/imag, got {complex_dim}."

    csi_perm = csi_ri.permute(0, 5, 1, 2, 3, 4, 6, 7)
    flattened_dim = num_rx * num_rx_ant * num_tx * num_tx_ant * num_fft * complex_dim
    return csi_perm.reshape(bsz, num_sym, flattened_dim)


# =============================================================================
# Main pipeline
# =============================================================================

def build_dataset(cfg: SystemConfig) -> None:
    """Build and save the full CSI-text dataset."""
    configure_runtime(cuda_visible_devices="6")
    resource_grid = build_resource_grid(cfg)

    all_h_freq: List[np.ndarray] = []
    all_ue_pos: List[np.ndarray] = []
    all_texts: List[np.ndarray] = []
    all_label_ids: List[np.ndarray] = []
    all_label_names: List[np.ndarray] = []

    for scene_name in SCENES:
        print(f"\n=== Processing scene: {scene_name} ===")
        scene = setup_scene(scene_name, cfg)
        tx_pos = np.asarray(TX_POSITIONS[scene_name], dtype=np.float32)

        for dist_cfg in DISTANCE_CONFIGS:
            for gain_cfg in GAIN_CONFIGS:
                for mp_cfg in MULTIPATH_CONFIGS:
                    for speed_cfg in SPEED_CONFIGS:
                        a_final, tau_final, ue_pos_final, h_freq = sample_cirs_for_configuration(
                            scene=scene,
                            cfg=cfg,
                            resource_grid=resource_grid,
                            scene_name=scene_name,
                            dist_cfg=dist_cfg,
                            gain_cfg=gain_cfg,
                            mp_cfg=mp_cfg,
                            speed_cfg=speed_cfg,
                        )

                        texts, label_ids, label_names = generate_text_dataset_from_csi(
                            a=a_final,
                            tau=tau_final,
                            ue_pos_all=ue_pos_final,
                            tx_pos=tx_pos,
                            seed=42,
                            use_feature_based_label=True,
                        )

                        all_h_freq.append(h_freq)
                        all_ue_pos.append(ue_pos_final)
                        all_texts.append(np.asarray(texts, dtype=object))
                        all_label_ids.append(label_ids)
                        all_label_names.append(label_names)

    h_freq_all = np.concatenate(all_h_freq, axis=0)
    ue_pos_all = np.concatenate(all_ue_pos, axis=0)
    texts_all = np.concatenate(all_texts, axis=0)
    label_ids_all = np.concatenate(all_label_ids, axis=0)
    label_names_all = np.concatenate(all_label_names, axis=0)

    print("Final dataset summary:")
    print("  h_freq:", h_freq_all.shape)
    print("  ue_pos:", ue_pos_all.shape)
    print("  texts:", len(texts_all))
    print("  label_ids:", len(label_ids_all))
    print("  label_names:", len(label_names_all))

    save_dataset_to_h5(
        output_path=cfg.output_h5,
        h_freq=h_freq_all,
        label_ids=label_ids_all,
        texts=texts_all,
        label_names=label_names_all,
    )

    dataset = CSIMultiModalDataset(cfg.output_h5)
    print("Saved dataset length:", len(dataset))

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in loader:
        csi_bsd = csi_ri_to_bsd(batch["csi_a"])
        print("csi_bsd shape:", csi_bsd.shape)
        print("raw csi_a shape:", batch["csi_a"].shape)
        print("label_id shape:", batch["label_id"].shape)
        print("label_id:", batch["label_id"])
        print("example text:", batch["text"][0])
        print("example label:", batch["label_name"][0])
        break


if __name__ == "__main__":
    build_dataset(SystemConfig())
