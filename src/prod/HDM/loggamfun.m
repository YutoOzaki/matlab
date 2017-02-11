%% Approximation of log-gamma function
% For example, compare result of the calculation below.
% x = (0:2:200)';
% A = log(gamma(x));
% B = loggamfun(x);
% C = gammaln(x);
% disp([A B C]);
function y = loggamfun(x)
    y1 = log(gamma(x));
    idx = isinf(y1);
    
    x2 = x(idx);
    y2 = 0.5.*(log(2*pi) - log(x2)) + x2.*(log(x2 + 1./(12*x2 - 1./(10*x2))) - 1);
    
    y = y1;
    y(idx) = y2;
end