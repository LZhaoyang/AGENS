from sionna.phy.ofdm import LMMSEEqualizer, MFEqualizer, ZFEqualizer
def equalization(rx_freq, H_est, method_idx, rg, sm, err_var, no):


    if method_idx == 0:
        lmmse_equ = LMMSEEqualizer(rg, sm)
    elif method_idx == 1:
        lmmse_equ = MFEqualizer(rg, sm)
    elif method_idx == 2:
        lmmse_equ = ZFEqualizer(rg, sm)

    
    x_hat, no_eff = lmmse_equ(rx_freq, H_est, err_var, no)
    return x_hat, no_eff
