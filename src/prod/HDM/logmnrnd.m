function idx = logmnrnd(logp)
    p = exp(logp - max(logp));
    p = p./sum(p);
    idx = mnrnd(1, p);
end