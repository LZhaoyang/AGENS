from sionna.phy.ofdm import RZFPrecoder
import tensorflow as tf


def zf_precoder_simple(s, H):
    """
    s: [B, num_streams, ...]
    H: [B, num_rx, num_tx, ...]
    """
    H_H = tf.math.conj(tf.transpose(H, perm=[0,2,1,...]))        # [B, num_tx, num_rx, ...]
    HH_H = tf.matmul(H, H_H)                                     # [B, num_rx, num_rx, ...]
    HH_H_inv = tf.linalg.inv(HH_H + 1e-9*tf.eye(tf.shape(HH_H)[-1]))
    W = tf.matmul(H_H, HH_H_inv)                                 # [B, num_tx, num_rx, ...]
    # 单用户 num_rx=1 时，简化为 [B, num_tx, ...]
    w = W[..., 0, :]                                             # [B, num_tx, ...]
    # 归一化
    power = tf.reduce_mean(tf.abs(w)**2, axis=1, keepdims=True) + 1e-9
    w_norm = w / tf.sqrt(power)
    x = w_norm * s
    return x


def mrt_precoder(x_rg, H_est):
    """
    x_rg  : [B, N_tx, N_streams, N_sym, N_sc]
    H_est : [B, N_rx, N_rx_ant, N_tx, N_tx_ant, N_sym, N_sc]
    返回:
      x_precoded : [B, N_tx, N_streams, N_sym, N_sc]
      H_eff      : [B, N_rx_tot, 1, N_sym, N_sc]  # 单 stream 的等效信道
    """

    # === 1) 把 7 维信道压成 5 维 [B, N_rx_tot, N_tx_tot, N_sym, N_sc] ===
    shape = tf.shape(H_est)
    B      = shape[0]
    N_rx   = shape[1] * shape[2]   # 总接收天线数
    N_tx   = shape[3] * shape[4]   # 总发射天线数
    N_sym  = shape[5]
    N_sc   = shape[6]

    # [B, N_rx_tot, N_tx_tot, N_sym, N_sc]
    H_flat = tf.reshape(H_est, [B, N_rx, N_tx, N_sym, N_sc])

    # === 2) 做 H^H 得到发射端方向向量 ===
    # H_flat: [B, N_rx, N_tx, S, F]
    # H^H   : [B, N_tx, N_rx, S, F]
    H_herm = tf.math.conj(tf.transpose(H_flat, perm=[0, 2, 1, 3, 4]))

    # 简单处理：对所有接收天线求平均，得到一个 beamforming 向量
    # w: [B, N_tx, S, F]
    H_herm = tf.math.conj(tf.transpose(H_flat, perm=[0, 2, 1, 3, 4]))
    w = tf.reduce_mean(H_herm, axis=2)   # [B, N_tx_tot, S, F]，复数

    # === 3) 归一化发射功率 ===
    power = tf.reduce_mean(tf.abs(w)**2, axis=1, keepdims=True) + 1e-9
    scale = tf.cast(tf.sqrt(power), w.dtype)  # 保证和 w 同 dtype (complex64)
    w = w / scale

    # === 4) 应用到 x_rg 上 ===
    # x_rg: [B, N_tx, N_streams, S, F]
    # 扩展 w 到 stream 维度
    w_exp = tf.expand_dims(w, axis=2)                 # [B, N_tx, 1, S, F]
    w_exp = tf.broadcast_to(w_exp, tf.shape(x_rg))    # [B, N_tx, N_streams, S, F]

    x_precoded = x_rg * w_exp

    # === 5) 计算等效信道 H_eff = H_flat * w（单 stream 情况）===
    # H_flat: [B, N_rx_tot, N_tx_tot, S, F]
    # w     : [B, N_tx_tot, S, F]
    # einsum over tx 维度 → [B, N_rx_tot, S, F]
    H_eff = tf.einsum('brtks,btks->brks', H_flat, w)

    # 加一个 stream 维统一接口: [B, N_rx_tot, 1, S, F]
    H_eff = tf.expand_dims(H_eff, axis=2)

    return x_precoded, H_eff

def precoder(x_rg, H_est, rg, sm, method_idx):


    if method_idx == 0:
        # ZF
        rzfp = RZFPrecoder(rg, sm, return_effective_channel=True)
        x_precoded, g = rzfp(x_rg, H_est)
    elif method_idx == 1:
        # RZF (你现在的实现)
        rzfp = RZFPrecoder(rg, sm, return_effective_channel=True)
        x_precoded, g = rzfp(x_rg, H_est, 2)
    else:
        raise ValueError("Unknown precoding method_idx")

    return x_precoded, g

