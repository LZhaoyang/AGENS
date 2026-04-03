"""Link-level strategy evaluation and exhaustive search for Sionna-based PHY adaptation.

This module evaluates one-step link adaptation strategies over a fixed CSI sample,
computes a multi-objective reward, and optionally searches for the best strategy
by exhaustive enumeration.

Main capabilities:
1. Prepare source bits for different modulation and coding settings.
2. Run a full PHY chain for a single sample.
3. Compute a reward balancing BER, rate, latency, and power.
4. Evaluate a batch of strategies.
5. Exhaustively search the best strategy per sample.
6. Plot normalization curves used by the reward function.

Notes:
- This refactor keeps the original algorithmic intent while improving readability.
- Comments and docstrings are fully in English.
- Several logic bugs in the original script are fixed and documented.
"""

from __future__ import annotations


import math
import os
import random
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
import sionna.phy
from sionna.phy.channel import ApplyOFDMChannel
from sionna.phy.mapping import BinarySource
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper
from sionna.phy.utils import compute_ber, ebnodb2no

from channel_estimation import *
from data import *
from encoder import *
from equalization import *
from modulation import *
from proceeder import *
from spreading import *


# =============================================================================
# Runtime configuration
# =============================================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.random.set_seed(1234)
tf.config.set_visible_devices([], "GPU")

warnings.filterwarnings("ignore", category=UserWarning, module="sionna")
warnings.filterwarnings("ignore", category=FutureWarning, module="sionna")


# =============================================================================
# Strategy space definitions
# =============================================================================

# Canonical strategy layout used throughout this refactored module:
# strategy = [modulation, code_rate_index, power_control, encoder, channel_estimator, equalizer, precoder]
STRATEGY_DIM = 7
IDX_MOD = 0
IDX_RATE = 1
IDX_POWER = 2
IDX_ENCODER = 3
IDX_CE = 4
IDX_EQ = 5
IDX_PRECODER = 6

VALID_RATES: Dict[int, List[float]] = {
    0: [1 / 3, 1 / 2, 2 / 3, 3 / 4, 0.8],  # LDPC
    1: [1 / 3, 1 / 2, 2 / 3, 3 / 4, 0.8],  # Polar
    2: [1 / 3, 1 / 2],                     # Convolutional
}


# =============================================================================
# Utility helpers
# =============================================================================

def set_all_seeds(seed: int) -> None:
    """Set seeds for reproducibility across NumPy, TensorFlow, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def to_scalar_float(x) -> float:
    """Convert a TensorFlow/PyTorch/numpy scalar-like object to a Python float."""
    try:
        if hasattr(x, "numpy"):
            x = x.numpy()
        if hasattr(x, "item"):
            x = x.item()
    except Exception:
        pass
    return float(x)



def modulation_order_from_strategy(method_idx: int) -> int:
    """Map modulation strategy index to constellation order."""
    mapping = {
        0: 4,    # QPSK
        1: 16,   # 16-QAM
        2: 64,   # 64-QAM
        3: 256,  # 256-QAM
    }
    if method_idx not in mapping:
        raise ValueError(f"Invalid modulation strategy index: {method_idx}")
    return mapping[method_idx]



def prepare_bits_for_modulation(
    bits_global_i: tf.Tensor,
    mod_strategy: int,
    rate: float,
    rg: ResourceGrid,
) -> Tuple[tf.Tensor, int, int]:
    """Pad or truncate payload bits to match the selected coding and modulation setup."""
    modulation_order = modulation_order_from_strategy(mod_strategy)
    bits_per_symbol = int(np.log2(modulation_order))

    num_coded_bits = int(rg.num_data_symbols) * bits_per_symbol
    num_info_bits = int(num_coded_bits * rate)

    payload_bits = bits_global_i.shape[-1]
    if payload_bits < num_info_bits:
        pad_len = num_info_bits - payload_bits
        pad = tf.zeros([1, 1, rg.num_streams_per_tx, pad_len], dtype=bits_global_i.dtype)
        bits_i = tf.concat([bits_global_i, pad], axis=3)
    else:
        bits_i = bits_global_i[..., :num_info_bits]

    return bits_i, num_info_bits, num_coded_bits



def power_control(pc_strategy: int, rg_i: ResourceGrid, num_info_bits: int) -> Tuple[float, float, float, float]:
    """Apply a discrete transmit-power strategy and derive its energy/power descriptors."""
    power_levels_db = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    tx_power_offset_db = power_levels_db[int(pc_strategy) % len(power_levels_db)]

    p_ref = 1.0
    p_tx = p_ref * (10.0 ** (tx_power_offset_db / 10.0))

    subcarrier_spacing = rg_i.subcarrier_spacing
    fft_size = rg_i.fft_size
    t_fft = 1.0 / subcarrier_spacing
    ts = 1.0 / (fft_size * subcarrier_spacing)
    t_cp = rg_i.cyclic_prefix_length * ts
    t_sym = t_fft + t_cp
    t_block = float(rg_i.num_ofdm_symbols) * t_sym

    if num_info_bits <= 0:
        e_b = np.inf
    else:
        e_b = p_tx * t_block / float(num_info_bits)

    # Logarithmic excess-power measure.
    p_extra = float(np.log(p_tx / p_ref + 1e-12))

    return tx_power_offset_db, p_tx, e_b, p_extra


# =============================================================================
# Reward normalization
# =============================================================================

def norm_ber(ber: float, ber_ref: float = 1e-2) -> float:
    """Normalize BER into a reward-friendly range."""
    ber = float(ber)

    if ber < 0.1 + (ber_ref / 5):
        ber_n = -np.log10(ber + 1e-12) / (-np.log10(ber_ref))
    elif ber < 0.25 + (ber_ref / 5):
        ber_n = 0.5 * (0.25 - ber) / 0.25
    elif ber < 0.3 + (ber_ref / 5):
        ber_n = -15.0 * (ber - 0.25)
    else:
        ber_n = -200.0 * (ber - 0.25)

    return float(np.clip(ber_n, -8.0, 1.0))



def norm_rate(rate_eff: float, rate_ref: float = 0.7) -> float:
    """Normalize effective rate."""
    rate_eff = float(rate_eff)

    if rate_eff <= 0.0:
        return -2.0

    if rate_eff < rate_ref:
        rate_n = rate_eff / rate_ref
    else:
        extra = (rate_eff - rate_ref) / max(1e-6, 4.0 - rate_ref)
        rate_n = 1.0 + 0.2 * extra

    return float(np.clip(rate_n, -2.0, 1.2))



def norm_time(time_exec: float, time_ref: float = 5.0) -> float:
    """Normalize execution time."""
    time_exec = float(time_exec)

    if time_exec <= 0.0:
        return 1.0
    if time_exec <= time_ref:
        time_n = 1.0 - time_exec / time_ref
    elif time_exec <= 3 * time_ref:
        time_n = -(time_exec - time_ref) / (2 * time_ref)
    else:
        time_n = -1.0 - (time_exec - 3 * time_ref) / time_ref

    return float(np.clip(time_n, -5.0, 1.0))



def norm_power(p_extra: float, p_extra_ref: float = 0.921) -> float:
    """Normalize excess-power usage."""
    p_extra = float(p_extra)

    if p_extra <= 0.0:
        return 1.0
    if p_extra <= p_extra_ref:
        power_n = 1.0 - p_extra / p_extra_ref
    elif p_extra <= 2 * p_extra_ref:
        power_n = -(p_extra - p_extra_ref) / p_extra_ref
    else:
        power_n = -1.0 - (p_extra - 2 * p_extra_ref) / p_extra_ref

    return float(np.clip(power_n, -5.0, 1.0))



def ber_ref_from_snr(ebno_db: float) -> float:
    """Choose a BER reference threshold based on SNR.

    Important bug fix:
    The original threshold ordering was reversed, which made later conditions unreachable.
    This version checks from the lowest SNR upward.
    """
    eb = float(ebno_db)
    if eb < -12:
        return 0.20
    if eb < -10:
        return 0.15
    if eb < -9:
        return 0.10
    return 1e-2



def rl_reward(
    ber,
    rate_eff,
    time_exec,
    p_extra,
    pref_label: Optional[int] = None,
    ebno_db: Optional[float] = None,
    ber_ref: float = 1e-1,
    rate_ref: float = 3.0,
    time_ref: float = 0.6,
    p_extra_ref: float = 0.921,
    w_ber: float = 0.4,
    w_rate: float = 0.3,
    w_time: float = 0.15,
    w_power: float = 0.15,
) -> float:
    """Compute the scalar RL reward for a single link sample.

    Preference labels:
        0 -> high_data_rate
        1 -> urllc
        2 -> energy_saving
        others -> balanced fallback
    """
    ber = to_scalar_float(ber)
    rate_eff = to_scalar_float(rate_eff)
    time_exec = to_scalar_float(time_exec)
    p_extra = to_scalar_float(p_extra)

    if pref_label is not None:
        if hasattr(pref_label, "item"):
            pref_label = int(pref_label.item())
        else:
            pref_label = int(pref_label)

        if pref_label == 0:
            w_ber, w_rate, w_time, w_power = 0.05, 0.80, 0.10, 0.05
        elif pref_label == 1:
            w_ber, w_rate, w_time, w_power = 0.80, 0.05, 0.10, 0.05
        elif pref_label == 2:
            w_ber, w_rate, w_time, w_power = 0.10, 0.05, 0.05, 0.80
        else:
            w_ber, w_rate, w_time, w_power = 0.40, 0.30, 0.15, 0.15

    if ebno_db is not None:
        ber_ref = ber_ref_from_snr(to_scalar_float(ebno_db))

    ber_n = norm_ber(ber, ber_ref=ber_ref)
    rate_n = norm_rate(rate_eff, rate_ref=rate_ref)
    time_n = norm_time(time_exec, time_ref=time_ref)
    power_n = norm_power(p_extra, p_extra_ref=p_extra_ref)

    weight_sum = w_ber + w_rate + w_time + w_power
    if weight_sum <= 0:
        w_ber_n = w_rate_n = w_time_n = w_power_n = 0.25
    else:
        w_ber_n = w_ber / weight_sum
        w_rate_n = w_rate / weight_sum
        w_time_n = w_time / weight_sum
        w_power_n = w_power / weight_sum

    reward = (
        w_ber_n * ber_n
        + w_rate_n * rate_n
        + w_time_n * time_n
        + w_power_n * power_n
    )
    return float(reward)


# =============================================================================
# PHY execution pipeline
# =============================================================================

def decode_strategy(strategy_i: Sequence[int]) -> Dict[str, int]:
    """Decode a 7-dimensional action vector into named strategy components."""
    if len(strategy_i) != STRATEGY_DIM:
        raise ValueError(f"Expected strategy length {STRATEGY_DIM}, got {len(strategy_i)}")

    return {
        "modulation": int(strategy_i[IDX_MOD]),
        "rate_index": int(strategy_i[IDX_RATE]),
        "power": int(strategy_i[IDX_POWER]),
        "encoder": int(strategy_i[IDX_ENCODER]),
        "channel_estimator": int(strategy_i[IDX_CE]),
        "equalizer": int(strategy_i[IDX_EQ]),
        "precoder": int(strategy_i[IDX_PRECODER]),
    }



def run_single_sample(
    bits_global_i: tf.Tensor,
    strategy_i: Sequence[int],
    ebno_db,
    h_freq_i,
    rg: ResourceGrid,
    stream_management: StreamManagement,
    pref_label,
):
    """Run the full PHY chain and reward evaluation for one sample."""
    payload_bits = bits_global_i.shape[3]
    decoded_strategy = decode_strategy(strategy_i)

    enc_strategy = decoded_strategy["encoder"]
    mod_strategy = decoded_strategy["modulation"]
    ce_strategy = decoded_strategy["channel_estimator"]
    eq_strategy = decoded_strategy["equalizer"]
    dec_strategy = enc_strategy
    rate_strategy = decoded_strategy["rate_index"]
    precoder_strategy = decoded_strategy["precoder"]
    pc_strategy = decoded_strategy["power"]

    rate_list = VALID_RATES[enc_strategy]
    rate = rate_list[rate_strategy % len(rate_list)]

    bits_i, num_info_bits, num_coded_bits = prepare_bits_for_modulation(bits_global_i, mod_strategy, rate, rg)

    start_time = time.time()

    coded_i, encoder_i = channel_coding(bits_i, enc_strategy, int(num_info_bits), int(num_coded_bits), rate)
    coded_i = coded_i[:, :, :, :num_coded_bits]

    syms_i = modulation(coded_i, mod_strategy)

    rg_i = ResourceGrid(
        num_ofdm_symbols=rg.num_ofdm_symbols,
        fft_size=rg.fft_size,
        subcarrier_spacing=rg.subcarrier_spacing,
        num_tx=rg.num_tx,
        num_guard_carriers=rg.num_guard_carriers,
        dc_null=rg.dc_null,
        num_streams_per_tx=rg.num_streams_per_tx,
        cyclic_prefix_length=rg.cyclic_prefix_length,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=[2, 9],
    )
    rg_mapper_i = ResourceGridMapper(rg_i)
    x_rg_i = rg_mapper_i(syms_i)

    tx_power_offset_db, p_tx, e_b, p_extra = power_control(pc_strategy, rg_i, int(num_info_bits))
    effective_ebno_db = ebno_db + tx_power_offset_db

    n0_i = ebnodb2no(
        effective_ebno_db,
        math.log2(modulation_order_from_strategy(mod_strategy)),
        rate,
        rg_i,
    )

    x_precoded_i, g_i = precoder(x_rg_i, h_freq_i, rg_i, stream_management, precoder_strategy)

    ofdm_apply = ApplyOFDMChannel(add_awgn=True, normalize_channel=True)
    y_rg_i = ofdm_apply(x_precoded_i, h_freq_i, n0_i)

    h_hat_i, err_var_i = channel_estimation(y_rg_i, ce_strategy, rg_i, n0_i, g_i)
    x_hat_i, n0_eff_i = equalization(
        y_rg_i,
        h_hat_i,
        eq_strategy,
        rg_i,
        stream_management,
        err_var_i,
        n0_i,
    )

    llr_i = demodulation(x_hat_i, mod_strategy, n0_eff_i)
    bits_hat_full = channel_decoding(llr_i, dec_strategy, encoder_i, int(num_info_bits), rate)

    bits_hat_i = bits_hat_full[:, :, :, :payload_bits]
    bits_i_truth = bits_i[:, :, :, :payload_bits]

    ber_i = compute_ber(bits_i_truth, bits_hat_i)
    elapsed = time.time() - start_time

    total_info_bits = tf.cast(tf.size(bits_i[0]), tf.float32)
    total_re = tf.cast(rg_i.fft_size * rg_i.num_ofdm_symbols, tf.float32) * bits_i.shape[2]
    rate_eff = total_info_bits / total_re

    reward = rl_reward(ber_i, rate_eff, elapsed, p_extra, pref_label, ebno_db=ebno_db)

    return ber_i, bits_hat_i, elapsed, p_extra, rate_eff, reward



def run_link(
    batch_size: int,
    ebno_db,
    strategy: np.ndarray,
    bits_global: tf.Tensor,
    h_freq,
    rg: ResourceGrid,
    stream_management: StreamManagement,
    pref_label,
    max_harq_rounds: int = 1,
):
    """Evaluate a batch of samples independently."""
    ber_list = []
    time_list = []
    rate_list = []
    pextra_list = []
    rewards = []

    for i in range(batch_size):
        bits_global_i = bits_global[i : i + 1]
        strategy_i = strategy[i]
        h_freq_i = h_freq[i : i + 1]
        ebno_db_i = ebno_db[i : i + 1]
        pref_label_i = pref_label[i : i + 1]

        ber_i, bits_hat_i, elapsed, p_extra, rate_eff, reward = run_single_sample(
            bits_global_i,
            strategy_i,
            ebno_db_i,
            h_freq_i,
            rg,
            stream_management,
            pref_label_i,
        )

        ber_list.append(to_scalar_float(ber_i))
        time_list.append(float(elapsed))
        rate_list.append(to_scalar_float(rate_eff))
        pextra_list.append(float(p_extra))
        rewards.append(float(reward))

    return (
        np.asarray(ber_list, dtype=np.float32),
        np.asarray(time_list, dtype=np.float32),
        np.asarray(rate_list, dtype=np.float32),
        np.asarray(pextra_list, dtype=np.float32),
        np.asarray(rewards, dtype=np.float32),
    )



def generate_strategy(batch_size: int) -> np.ndarray:
    """Randomly generate strategy vectors.

    Important bug fix:
    The original comments and ranges did not match the actual strategy layout.
    This version follows the canonical strategy order defined at the top.
    """
    modulation_range = 4
    rate_range = 5
    power_range = 7
    encoder_range = 3
    ce_range = 3
    eq_range = 3
    precoder_range = 2

    strategy = np.zeros((batch_size, STRATEGY_DIM), dtype=np.int32)
    for i in range(batch_size):
        strategy[i, IDX_MOD] = np.random.randint(0, modulation_range)
        strategy[i, IDX_RATE] = np.random.randint(0, rate_range)
        strategy[i, IDX_POWER] = np.random.randint(0, power_range)
        strategy[i, IDX_ENCODER] = np.random.randint(0, encoder_range)
        strategy[i, IDX_CE] = np.random.randint(0, ce_range)
        strategy[i, IDX_EQ] = np.random.randint(0, eq_range)
        strategy[i, IDX_PRECODER] = np.random.randint(0, precoder_range)
    return strategy








# =============================================================================
# Demonstration entry point
# =============================================================================

def main() -> None:
    """Example entry point for local testing."""

    fft_size = 128
    scs = 30e3
    cp_len = 10
    pilot_symbol_indices = [2, 9]

    num_tx = 1
    num_rx = 1
    num_ut_ant = 2
    num_bs_ant = 4
    num_streams_per_tx = num_ut_ant
    num_time_steps = 14

    rg = ResourceGrid(
        num_ofdm_symbols=num_time_steps,
        fft_size=fft_size,
        subcarrier_spacing=scs,
        num_tx=num_tx,
        num_guard_carriers=[9, 10],
        dc_null=True,
        num_streams_per_tx=num_streams_per_tx,
        cyclic_prefix_length=cp_len,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=pilot_symbol_indices,
    )

    batch_size = 8
    rx_tx_association = np.array([[1]])
    stream_management = StreamManagement(rx_tx_association, num_streams_per_tx)

    strategy = np.array(
        [
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 1],
            [0, 0, 3, 0, 1, 0, 0],
            [2, 1, 0, 2, 0, 0, 0],
            [2, 0, 0, 2, 1, 0, 0],
            [2, 0, 0, 2, 2, 0, 0],
            [2, 4, 0, 2, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    bits_global = BinarySource()([batch_size, 1, rg.num_streams_per_tx, 512])
    h5_path = "/home/ubuntu/MC-LLM/csi_text_dataset.h5"

    set_all_seeds(1024)

    dataset = CSIMultiModalDataset(h5_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    for batch in dataloader:
        csi = batch["csi_a"]
        labels = batch["label_id"]
        print("Loaded labels:", labels)
        break

    # Reuse the first CSI sample across the batch for a controlled sanity test.
    csi_0 = csi[:1]
    csi = csi_0.expand_as(csi)

    comm_data_np = csi.detach().cpu().numpy()
    h_real = comm_data_np[..., 0]
    h_imag = comm_data_np[..., 1]
    csi_complex = h_real + 1j * h_imag

    snr = -5 * np.ones((batch_size, 1), dtype=np.float32)

    # Important bug fix:
    # The original code overwrote labels with a tensor of length 4 while batch_size was 8.
    # Here we keep the labels aligned with the batch size.
    labels = labels[:batch_size]

    ber, elapsed, rate_eff, pextra, rewards = run_link(
        batch_size,
        snr,
        strategy,
        bits_global,
        csi_complex,
        rg,
        stream_management,
        labels,
    )
    print("Batch rewards:", rewards)


    



if __name__ == "__main__":
    main()
