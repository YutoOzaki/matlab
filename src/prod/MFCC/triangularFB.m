function [trifil,cbin] = triangularFB(coefnum, f_start, f_end, FFTL, unitpow, f_s)
    if f_s/2 < f_end
        f_end = f_s/2;
    end

    cbin = zeros(coefnum+2,1);
    
    Mel1 = 2595*log10(1+f_start/700);
    Mel2 = 2595*log10(1+(f_end)/700);
    
    f_c = zeros(coefnum+2,1);
    f_c(1) = f_start;
    f_c(end) = f_end;
    
    for i=1:coefnum
        f_c(i+1) = Mel1 + (Mel2-Mel1)/(coefnum+1)*i;
        f_c(i+1) = (10^(f_c(i+1)/2595) - 1)*700;
        cbin(i+1) = round(f_c(i+1)/f_s * FFTL) + 1;
    end
    
    cbin(1) = round(f_start/f_s*FFTL) + 1;
    cbin(coefnum+2) = round(f_end/f_s*FFTL) + 1;

    trifil = zeros(coefnum,1+FFTL/2);
    for k=2:coefnum+1
        for i=cbin(k-1):cbin(k)
            trifil(k-1,i) = (i-cbin(k-1)+1)/(cbin(k)-cbin(k-1)+1);
        end
        for i=cbin(k)+1:cbin(k+1)
            trifil(k-1,i) = (1 - (i-cbin(k))/(cbin(k+1)-cbin(k)+1));
        end
    end

    if unitpow
        filterArea = sum(trifil,2);
        for i=1:coefnum
            trifil(i,:) = trifil(i,:)./filterArea(i);
        end
    end
end