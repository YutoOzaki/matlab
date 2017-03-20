function N = varcrp(alpha, n)
    N = alpha * (psi(alpha + n) - psi(alpha)) + (alpha^2)*(psi(1, alpha + n) - psi(1, alpha));
end