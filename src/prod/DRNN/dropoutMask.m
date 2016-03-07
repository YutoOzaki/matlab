function x = dropoutMask(x,p)
    [r,c,t] = size(x);
    %x = x.*repmat(binornd(1,p,[r c]),[1 1 t]);
    x = x.*binornd(1,p,[r c t]);
end