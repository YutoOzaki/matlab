function p = diagmvnpdf(X, MU, SIG)
    SIG = diag(SIG)';
    
    d = size(X, 2); 
    A = bsxfun(@minus, X, MU).^2;
    B = bsxfun(@rdivide, A, SIG);
    p = exp(-0.5*d*log(2*pi) - 0.5*sum(log(SIG)) - 0.5*sum(B, 2));
end