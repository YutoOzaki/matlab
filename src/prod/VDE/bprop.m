function bprop(x, encnet, decnet, priornet, lossnode)
    dL = lossnode.backwardprop([]);
    %lossnode.localgradcheck();
    
    delta = priornet.weight.backwardprop(dL);
    
    L = size(dL.xmu,3);
    dxmu = decnet.mu.backwardprop(L.*reshape(dL.xmu, [size(dL.xmu,1), size(dL.xmu,2)*L]));
    dxsig = decnet.exp.backwardprop(L.*reshape(dL.xsig, [size(dL.xsig,1), size(dL.xsig,2)*L]));
    dlnxsig = decnet.sig.backwardprop(dxsig);
    dxh = decnet.activate.backwardprop(dxmu + dlnxsig);
    dxa = decnet.connect.backwardprop(dxh);
end