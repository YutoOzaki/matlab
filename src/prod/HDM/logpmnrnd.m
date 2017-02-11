function idx = logpmnrnd(logp)
    p = exp(logp - max(logp));
    p = p./sum(p);
    idx = mnrnd(1, p);
    idx = find(idx == 1);
end