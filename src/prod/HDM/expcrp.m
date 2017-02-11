function N = expcrp(alpha, n)
    N = alpha * (psi(alpha + n) - psi(alpha));
end