function loss = fprop(x, encnet, decnet, priornet, lossnode)
    za = encnet.connect.forwardprop(x);
    zh = encnet.activate.forwardprop(za);
    zmu = encnet.mu.forwardprop(zh);
    lnzsig = encnet.sig.forwardprop(zh);
    zsig = encnet.exp.forwardprop(lnzsig);
    z = encnet.reparam.forwardprop(struct('mu', zmu, 'sig', zsig));

    L = size(z,3);
    z = reshape(z, [size(z,1), size(z,2)*L]);

    xa = decnet.connect.forwardprop(z);
    xh = decnet.activate.forwardprop(xa);
    xmu = decnet.mu.forwardprop(xh);
    lnxsig = decnet.sig.forwardprop(xh);
    xsig = decnet.exp.forwardprop(lnxsig);
    xmu = reshape(xmu, [size(xmu,1), size(xmu,2)/L, L]);
    xsig = reshape(xsig, [size(xsig,1), size(xsig,2)/L, L]);

    z = priornet.reparam.forwardprop(struct('mu', zmu, 'sig', zsig));
    gam = priornet.weight.forwardprop(z);

    loss = lossnode.forwardprop(struct(...
        'x', x,...
        'xmu', xmu,...
        'xsig', xsig,...
        'zmu', zmu,...
        'zsig', zsig,...
        'gam', gam,...
        'eta_mu', priornet.weight.prms.eta_mu,...
        'eta_sig', exp(priornet.weight.prms.eta_r),...
        'PI', priornet.weight.prms.PI...
        ));
end