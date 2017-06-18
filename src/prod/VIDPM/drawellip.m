function drawellip(MU, SIG)
    [cu, cv, phi] = ellipprm(SIG); 
    p = trajectory(MU(1), MU(2), cu, cv, phi);
    plot(p(:,1), p(:,2));
end

function [X,Y] = trajectory(x, y, a, b, angle, steps)
    %# This functions returns points to draw an ellipse
    %#
    %#  @param x     X coordinate
    %#  @param y     Y coordinate
    %#  @param a     Semimajor axis
    %#  @param b     Semiminor axis
    %#  @param angle Angle of the ellipse (in degrees)
    %#
 
    narginchk(5, 6);
    if nargin<6, steps = 36; end
 
    beta = -angle * (pi / 180);
    sinbeta = sin(beta);
    cosbeta = cos(beta);
 
    alpha = linspace(0, 360, steps)' .* (pi / 180);
    sinalpha = sin(alpha);
    cosalpha = cos(alpha);
 
    X = x + (a * cosalpha * cosbeta - b * sinalpha * sinbeta);
    Y = y + (a * cosalpha * sinbeta + b * sinalpha * cosbeta);
 
    if nargout==1, X = [X Y]; end
end

function [cu, cv, phi] = ellipprm(covar)
    %# This functions returns parameters to draw an ellipse
    %#
    %#  @param covar 2-D covariance matrix
    %#
 
    narginchk(1, 1);
 
    sigu = 0.5*(covar(1,1) + covar(2,2) + sqrt((covar(1,1)-covar(2,2))^2 + 4*covar(1,2)^2));
    sigv = 0.5*(covar(1,1) + covar(2,2) - sqrt((covar(1,1)-covar(2,2))^2 + 4*covar(1,2)^2));
    
    if sigv > sigu
     tmp = sigu;
     sigu = sigv;
     sigv = tmp;
    end
    
    cu = 3.035*sqrt(sigu);
    cv = 3.035*sqrt(sigv);
    phi = -(sigu-covar(1,1))/covar(1,2) *180/pi;
 
    if nargout==1, cu = [cu cv phi]; end
end