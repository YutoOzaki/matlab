function PI = softmax(p)
    C = max(p);
    buf = exp(p - C);
    PI = buf./sum(buf);
end