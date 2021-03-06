function [loss, x_recon] = fprop(x, nets, lossnode)
    encnet = nets.encnet;
    names = fieldnames(encnet);
    input = x;
    for i=1:length(names)
        input = encnet.(names{i}).forwardprop(input);
    end
    
    encrpm = nets.encrpm;
    zmu = encrpm.mu.forwardprop(input);
    zsig = encrpm.exp.forwardprop(encrpm.lnsigsq.forwardprop(input));
    z = encrpm.reparam.forwardprop(struct('mu', zmu, 'sig', zsig));
    
    L = size(z, 3);
    z = reshape(z, [size(z,1), size(z,2)*L]);
    
    decnet = nets.decnet;
    names = fieldnames(decnet);
    input = z;
    for i=1:length(names)
        input = decnet.(names{i}).forwardprop(input);
    end
    
    decrpm = nets.decrpm;
    xmu = decrpm.mu.forwardprop(input);
    xsig = decrpm.exp.forwardprop(decrpm.lnsigsq.forwardprop(input));
    
    xmu = reshape(xmu, [size(xmu,1), size(xmu,2)/L, L]);
    xsig = reshape(xsig, [size(xsig,1), size(xsig,2)/L, L]);
    
    xmu = mean(xmu, 3);
    xsig = mean(xsig, 3);
    x_recon = decrpm.reparam.forwardprop(struct('mu', xmu, 'sig', xsig));
    
    priornet = nets.priornet;
    %*try reusing z
    %*z = priornet.reparam.forwardprop(struct('mu', zmu, 'sig', zsig));
    gam = priornet.weight.forwardprop(z);
    
    loss = lossnode.forwardprop(struct(...
        'x', x,...
        'x_recon', x_recon,...
        'xmu', xmu,...
        'xsig', xsig,...
        'zmu', zmu,...
        'zsig', zsig,...
        'gam', gam,...
        'eta_mu', priornet.weight.prms.eta_mu,...
        'eta_sig', exp(priornet.weight.prms.eta_lnsig),...
        'PI', priornet.weight.getPI()...
        ));
end