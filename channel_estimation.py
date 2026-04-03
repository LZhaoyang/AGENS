from sionna.phy.ofdm import LSChannelEstimator, LMMSEInterpolator
import tensorflow as tf
# def sample_channel(batch_size):
#     # Sample random topologies
#     topology = gen_single_sector_topology(batch_size, 1, 'umi', min_ut_velocity=SPEED, max_ut_velocity=SPEED)
#     channel_sampler = GenerateOFDMChannel(CHANNEL_MODEL, rg)
#     CHANNEL_MODEL.set_topology(*topology)

#     # Sample channel frequency responses
#     # [batch size, 1, num_rx_ant, 1, 1, num_ofdm_symbols, fft_size]
#     h_freq = channel_sampler(batch_size)
#     # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
#     h_freq = h_freq[:,0,:,0,0]

#     return h_freq

# @tf.function(jit_compile=True) # Use XLA for speed-up
# def estimate_covariance_matrices(num_it, batch_size, rg):
#     freq_cov_mat = tf.zeros([rg.fft_size, rg.fft_size], tf.complex64)
#     time_cov_mat = tf.zeros([rg.num_ofdm_symbols, rg.num_ofdm_symbols], tf.complex64)
#     space_cov_mat = tf.zeros([rg.num_tx, rg.num_tx], tf.complex64)
#     for _ in tf.range(num_it):
#         # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
#         h_samples = sample_channel(batch_size)

#         #################################
#         # Estimate frequency covariance
#         #################################
#         # [batch size, num_rx_ant, fft_size, num_ofdm_symbols]
#         h_samples_ = tf.transpose(h_samples, [0,1,3,2])
#         # [batch size, num_rx_ant, fft_size, fft_size]
#         freq_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
#         # [fft_size, fft_size]
#         freq_cov_mat_ = tf.reduce_mean(freq_cov_mat_, axis=(0,1))
#         # [fft_size, fft_size]
#         freq_cov_mat += freq_cov_mat_

#         ################################
#         # Estimate time covariance
#         ################################
#         # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
#         time_cov_mat_ = tf.matmul(h_samples, h_samples, adjoint_b=True)
#         # [num_ofdm_symbols, num_ofdm_symbols]
#         time_cov_mat_ = tf.reduce_mean(time_cov_mat_, axis=(0,1))
#         # [num_ofdm_symbols, num_ofdm_symbols]
#         time_cov_mat += time_cov_mat_

#         ###############################
#         # Estimate spatial covariance
#         ###############################
#         # [batch size, num_ofdm_symbols, num_rx_ant, fft_size]
#         h_samples_ = tf.transpose(h_samples, [0,2,1,3])
#         # [batch size, num_ofdm_symbols, num_rx_ant, num_rx_ant]
#         space_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
#         # [num_rx_ant, num_rx_ant]
#         space_cov_mat_ = tf.reduce_mean(space_cov_mat_, axis=(0,1))
#         # [num_rx_ant, num_rx_ant]
#         space_cov_mat += space_cov_mat_

#     freq_cov_mat /= tf.complex(tf.cast(rg.num_ofdm_symbols*num_it, tf.float32), 0.0)
#     time_cov_mat /= tf.complex(tf.cast(rg.fft_size*num_it, tf.float32), 0.0)
#     space_cov_mat /= tf.complex(tf.cast(rg.fft_size*num_it, tf.float32), 0.0)

#     return freq_cov_mat, time_cov_mat, space_cov_mat

def estimate_batch_covariance(h_samples):
    s = tf.shape(h_samples)
    B, RX = s[0], s[1]
    UT = s[2]
    TX = s[3]
    BS = s[4]
    P  = s[5]
    T  = s[6]    # time domain channel length
    B_flat = B * RX * TX * BS 
    h_samples = tf.transpose(h_samples, [0, 2, 5, 6, 1, 3, 4])
    h_samples = tf.reshape(h_samples, [B_flat, UT, P, T])
    # 频域
    h_freq = tf.transpose(h_samples, [0, 1, 3, 2])
    freq_cov = tf.reduce_mean(tf.matmul(h_freq, h_freq, adjoint_b=True), axis=(0,1))
    # 时间域
    time_cov = tf.reduce_mean(tf.matmul(h_samples, h_samples, adjoint_b=True), axis=(0,1))
    # 空间域
    h_space = tf.transpose(h_samples, [0, 2, 1, 3])
    space_cov = tf.reduce_mean(tf.matmul(h_space, h_space, adjoint_b=True), axis=(0,1))
    return freq_cov, time_cov, space_cov

def select_effective_subcarriers_h(h, rg):
    # h: [B, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
    K = rg.fft_size
    left_guard, right_guard = rg.num_guard_carriers  # [5,6]
    idx = tf.range(left_guard, K - right_guard)      # [5, ..., 25]

    if rg.dc_null:
        dc = K // 2                                  # 16
        idx = tf.boolean_mask(idx, tf.not_equal(idx, dc))

    # 取最后一维（子载波维）
    h_eff = tf.gather(h, idx, axis=-1)
    return h_eff, idx


def channel_estimation(rx_freq, method_idx, rg, n0, h):


    if method_idx == 0:  # LS
        ls_est = LSChannelEstimator(rg, interpolation_type="nn")
        out, err_var = ls_est(rx_freq, n0)
    elif method_idx == 1:  # linear
        ls_est = LSChannelEstimator(rg, interpolation_type="lin")
        out, err_var = ls_est(rx_freq, n0)
    elif method_idx == 2:  # MMSE
        h, idx = select_effective_subcarriers_h(h, rg)
        freq_cov_mat, time_cov_mat, space_cov_mat = estimate_batch_covariance(h)

        lmmse_int_freq_first = LMMSEInterpolator(rg.pilot_pattern, time_cov_mat, freq_cov_mat, space_cov_mat, order='t-f-s')
        ls_est = LSChannelEstimator(rg, interpolator=lmmse_int_freq_first)
        out, err_var = ls_est(rx_freq, n0)

    return out, err_var