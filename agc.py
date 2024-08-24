def compute_gain_var(in_data, size_in):
    mean = np.complex64(0.0 + 0.0j)
    var1 = np.complex64(0.0 + 0.0j)
    var2 = np.complex64(0.0 + 0.0j)
    factor = 0x7fff

    # Calculate mean
    for sample in range(size_in):
        delta = in_data[sample] - mean
        i = sample + 1
        q = delta / i
        mean += q

    print(f"********** Mean:  {mean.real:.10f} + {mean.imag:.10f}j **********")

    # Compute variance
    for sample in range(size_in):
        diff = in_data[sample] - mean
        i = (sample // 2) + 1
        if sample % 2 == 0:
            delta = np.complex64(diff.real * diff.real + diff.imag * diff.imag) - var1
            q = delta / i
            var1 += q
        else:
            delta = np.complex64(diff.real * diff.real + diff.imag * diff.imag) - var2
            q = delta / i
            var2 += q

    print(f"********** Vars:  {var1.real:.10f} + {var1.imag:.10f}j, {var2.real:.10f} + {var2.imag:.10f}j **********")

    # Merge standard deviations
    tmpvar = (var1 + var2) * 0.5
    var = np.complex64(np.sqrt(tmpvar.real) + np.sqrt(tmpvar.imag) * 1j)
    print(f"********** Var:   {var.real:.10f} + {var.imag:.10f}j **********")

    var = var * np.complex64(var_variance)  # Assuming var_variance is defined elsewhere
    print(f"********** 4*Var: {var.real:.10f} + {var.imag:.10f}j **********")

    # Compute gain
    gain = var.real
    if var.imag > gain:
        gain = var.imag

    # Ignore zero variance samples and apply no gain
    if int(gain) == 0:
        gain = 1.0
    else:
        gain = factor / gain

    return gain
