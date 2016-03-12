function [blockWiseMFCC,patchInfo] = MFCC(x,hprms)
    %% get hyperparameters
    coef_range  = hprms.coef_range;
    mfbc        = hprms.mfbc;
    d           = hprms.d;
    dd          = hprms.dd;
    N           = hprms.N;
    M           = hprms.M;
    preemp      = hprms.preemp;
    w_z         = hprms.w_z;
    blocks      = hprms.blocks;
    patch       = hprms.patch;
    normalize   = hprms.normalize;
    zcaeps      = hprms.zcaeps;
    
    melfilbank  = hprms.melfilbank;
    ceplifter   = hprms.ceplifter;
    FFTL        = hprms.FFTL;
    FFTL_half   = hprms.FFTL_half;
    ACF         = hprms.ACF;
    CF          = hprms.CF;
    dim         = hprms.dim;
    
    blockWiseMFCC = zeros(dim,patch,blocks);
    patchInfo = zeros(2,patch);
    
    %% main loop
    L = length(x);
    rnd_start = randperm(L - (FFTL+(blocks-1)*M) + 1);

    i = 1;
    patchCounter = 1;
    while patchCounter <= patch        
        N_start = rnd_start(i);
        N_end = N_start + N - 1;
        vec2frame = zeros(FFTL,blocks);
        patchInfo(1,patchCounter) = N_start;
        
        k = 1;
        while k <= blocks
            frame = x(N_start:N_end);

            if isinf(log(sum(frame.^2)/N)) == 0
                vec2frame(1:N,k) = frame;
                N_start = N_start + M;
                N_end = N_start + N - 1;

                k = k + 1;
            else
                k = 1;
                i = i + 1;              
                N_start = rnd_start(i);
                N_end = N_start + N - 1;
                patchInfo(1,patchCounter) = N_start;
            end
        end
        i = i + 1;
        patchInfo(2,patchCounter) = N_end;

        %% Energy
        logE = log(sum(vec2frame.^2)/N);

        %% Pre-emphasis
        s_pe = filter(preemp,1,vec2frame);
        s_pe = s_pe ./ ACF;

        %% FFT
        s_w = diag(w_z) * s_pe;
        bin = fft(s_w,FFTL);
        bin = abs(bin).*CF;

        %% Mel-filtering
        fbank = melfilbank * bin(FFTL_half,:);
        f = log(fbank);

        %% Get cepstrum coefficients
        if mfbc
            C = f;
        else
            C = dct(f);
            C(1,:) = logE;
            %C(1,:) = C(1,:) * sqrt(coefnum);
            %C(2:coefnum,:) = C(2:coefnum,:) .* sqrt(coefnum/2);

            C = diag(ceplifter) * C;
        end
        C = C(coef_range,:);
        
        %% store result
        blockWiseMFCC(:,patchCounter,:) = C;
        patchCounter = patchCounter + 1;
    end
end