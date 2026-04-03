from sionna.phy.mapping import Constellation, Mapper, Demapper


def modulation(coded_bits, method_idx):
    if method_idx == 0:
        m_order = 2
    elif method_idx == 1:
        m_order = 4     
    elif method_idx == 2:
        m_order = 6
    elif method_idx == 3:
        m_order = 8
    # elif method_idx == 4:
    #     m_order = 10
    mapper = Mapper("qam", m_order)
    out = mapper(coded_bits)
    return out

    
    
def demodulation(rx_symbols, method_idx, no_eff):
    if method_idx == 0:
        num_bits_per_symbol = 2
    elif method_idx == 1:
        num_bits_per_symbol = 4
    elif method_idx == 2:
        num_bits_per_symbol = 6
    elif method_idx == 3:
        num_bits_per_symbol = 8
    # elif method_idx == 4:
    #     num_bits_per_symbol = 10
        
    mapper = Demapper("app", "qam", num_bits_per_symbol)
    out = mapper(rx_symbols, no_eff)
    return out