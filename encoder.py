import tensorflow as tf
from sionna.phy.mapping import BinarySource
from sionna.phy.fec.polar.utils import generate_5g_ranking, generate_rm_code
from sionna.phy.fec.conv import ConvEncoder, ViterbiDecoder
from sionna.phy.fec.turbo import TurboEncoder, TurboDecoder
from sionna.phy.fec.linear import LinearEncoder, OSDecoder
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.fec.polar import PolarEncoder, Polar5GEncoder, PolarSCLDecoder, Polar5GDecoder
from sionna.phy.fec.utils import load_parity_check_examples
import sionna.phy
import numpy as np
from sionna.phy.utils import compute_ber
from modulation import *
import time
import warnings

warnings.filterwarnings(
    "ignore",
    message="5G Polar codes use an integrated CRC that cannot be materialized with SC decoding and, thus, causes a degraded performance. Please consider SCL decoding instead.",
    module="sionna"
)

class SegmentedPolarEncoder:
    def __init__(self, k_seg=256, n_seg=512):
        self.k_seg = k_seg
        self.n_seg = n_seg

    
        self.seg_encoder = Polar5GEncoder(k=k_seg, n=n_seg)

    def encode(self, bits, k_total, n_total):
        """
        bits : shape [..., k_total]
        return: coded_bits shape [..., n_total]
        """

        # ---- Sanity Check ----
        assert bits.shape[-1] == k_total, \
            f"input bits last dim={bits.shape[-1]}, but k_total={k_total}"

        # ---- flat batch to [..., k_total] → [B_flat, k_total] ----
        orig_shape = tf.shape(bits)
        B_flat = tf.reduce_prod(orig_shape[:-1])
        bits_flat = tf.reshape(bits, [B_flat, k_total])


        num_segments = int(np.ceil(k_total / self.k_seg))

        coded_segments = []

        start = 0
        for _ in range(num_segments):
            end = min(start + self.k_seg, k_total)
            seg = bits_flat[:, start:end]      # shape [B_flat, seg_len]

            seg_len = end - start
            pad_len = self.k_seg - seg_len

            # ---- 补 0 ----
            if pad_len > 0:
                pad = tf.zeros([B_flat, pad_len], dtype=bits.dtype)
                seg = tf.concat([seg, pad], axis=-1)  # [B_flat, k_seg]

            # ---- Polar 编码 ----
            seg_coded = self.seg_encoder(seg)  # [B_flat, n_seg]
            coded_segments.append(seg_coded)

            start = end

        # ---- 拼接所有段 ----
        coded_concat = tf.concat(coded_segments, axis=-1)  # [B_flat, num_segments*n_seg]

        # ---- 截断到 n_total ----
        coded_final = coded_concat[:, :n_total]            # shape [B_flat, n_total]

        # ---- reshape 回原批次维 ----
        out_shape = tf.concat([orig_shape[:-1], [n_total]], axis=0)
        coded_final = tf.reshape(coded_final, out_shape)
       
        return coded_final, self.seg_encoder

class SegmentedPolarDecoder:
    def __init__(self, k_seg=256, n_seg=512, seg_encoder=None):
        self.k_seg = k_seg
        self.n_seg = n_seg

        # 单段解码器
        if seg_encoder == None:
            self.seg_encoder = Polar5GEncoder(k=k_seg, n=n_seg)
        else:
            self.seg_encoder = seg_encoder
        self.seg_decoder = Polar5GDecoder(self.seg_encoder, dec_type="SC")
       
    def decode(self, llr, k_total):
        """
        llr: Tensor, shape [..., n_total]
        return: decoded bits, shape [..., k_total]
        """


        # 需要的段数
        num_segments = int(np.ceil(k_total / self.k_seg))

        decoded_segments = []
        start = 0

        for i in range(num_segments):
            end = start + self.n_seg   # 每段取 n_seg 长度

            # ---- 取段 (保持前面所有维度不变) ----
            seg_llr = llr[..., start:end]   # shape [..., n_seg]

            # 如果最后段不够 n_seg，需要补零
            actual_len = seg_llr.shape[-1]
            if actual_len < self.n_seg:
                pad_len = self.n_seg - actual_len
                pad = tf.zeros(seg_llr.shape[:-1] + (pad_len,), dtype=seg_llr.dtype)
                seg_llr = tf.concat([seg_llr, pad], axis=-1)

            # ---- 解码（shape [..., k_seg]）----
            seg_decoded = self.seg_decoder(seg_llr)

            decoded_segments.append(seg_decoded)
            start = end

        # ---- 拼接所有段，沿最后一维 ----
        decoded_concat = tf.concat(decoded_segments, axis=-1)  # shape [..., num_segments*k_seg]

        # ---- 截断到 k_total ----
        decoded_final = decoded_concat[..., :k_total]

        return decoded_final




def channel_coding(bits, method_idx, k, n, rate):
   

    if method_idx == 0:
        encoder = LDPC5GEncoder(k=k, n=n)

        coded_bits = encoder(bits)
        return coded_bits, encoder

    elif method_idx == 1:
        
        enc = SegmentedPolarEncoder(int(512*rate), int(512))
        # Polar with segmentation
        coded_bits, encoder = enc.encode(bits, k, n)
        return coded_bits, encoder

    elif method_idx == 2:
        encoder = ConvEncoder(rate=rate, constraint_length=8)
        coded_bits = encoder(bits)
        return coded_bits, encoder

    elif method_idx == 3:
        encoder = TurboEncoder(rate=rate, constraint_length=4, terminate=False)
        coded_bits = encoder(bits)
        return coded_bits, encoder

    
    
def channel_decoding(codeword, method_idx, encoder_or_encoders, k_total, rate):
    
    if method_idx == 0:
        decoder = LDPC5GDecoder(encoder_or_encoders, num_iter=20)
        return decoder(codeword)

    elif method_idx == 1:
        # Polar segmented decode
        decoder = SegmentedPolarDecoder(int(512*rate), int(512), seg_encoder=encoder_or_encoders)
        return decoder.decode(codeword, k_total)

    elif method_idx == 2:
        decoder = ViterbiDecoder(gen_poly=encoder_or_encoders.gen_poly, method="soft_llr")
        return decoder(codeword)

    elif method_idx == 3:
        decoder = TurboDecoder(encoder_or_encoders, num_iter=8)
        return decoder(codeword)



if __name__ == "__main__":
    encoder = SegmentedPolarEncoder()

    awgn_channel = sionna.phy.channel.AWGN()
    no = sionna.phy.utils.ebnodb2no(5, num_bits_per_symbol=2, coderate=0.5)

    batch_size = 64
    num_tx = 1
    num_tx_ant = 4
    k = 2048
    n = 4096
    rate = k/n
    method_idx = 0
    bits = BinarySource()([batch_size, num_tx, num_tx_ant, k])  
    t0 = time.time()
    coded_bits, encoder = channel_coding(bits, method_idx, k, n, rate)
    coded_bits = modulation(coded_bits, 0)
    coded_bits = awgn_channel(coded_bits,no)
    llr_i = demodulation(coded_bits, 0, no)
    bits_reg = channel_decoding(llr_i, method_idx, encoder, k, rate)
    ber_i = compute_ber(bits, bits_reg)
    t1 = time.time()
    print(t1-t0)
    print(ber_i)
    print(bits[0])
    print(bits_reg[0])