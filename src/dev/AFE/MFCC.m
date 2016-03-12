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
    
    melfilbank  = hprms.melfilbank;
    ceplifter   = hprms.ceplifter;
    FFTL        = hprms.FFTL;
    FFTL_half   = hprms.FFTL_half;
    ACF         = hprms.ACF;
    CF          = hprms.CF;
    dim         = hprms.dim;
    
    blockWiseMFCC = zeros(dim,patch,blocks);
    patchInfo   = zeros(2,patch);
    
    %% delta MFCC
    if d > 0
        dnmt_d = sum((1:d).^2);
        dmfcc = @deltaMFCC./dnmt_d;
    else
        dmfcc = @(a,b,c) [];
    end
    
    if dd > 0
        dnmt_dd = sum((1:dd).^2);
        ddmfcc = @deltaMFCC./dnmt_dd;
    else
        ddmfcc = @(a,b,c) [];
    end
    
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
        
        %% delta coefficients
        delta = dmfcc(C,d,blocks);
        deltadelta = ddmfcc(delta,dd,blocks);
        bufbwm = [C;delta;deltadelta];
        
        %% store result
        blockWiseMFCC(:,patchCounter,:) = bufbwm;
        patchCounter = patchCounter + 1;
    end
end

%% debugging
%{
i = randi(blocks);
%% time domain
figure(1)
subplot(211); plot(vec2frame(:,i));
subplot(212); plot(s_w(:,i));

%% frequency domain
figure(2)
subplot(411); plot(bin(FFTL_half,i));
subplot(412); plot(fbank(:,i));
subplot(413); plot(f(:,i));
subplot(414); plot(C(:,i));

%% final output
figure(3)
plot(bufbwm(:,i));
%}