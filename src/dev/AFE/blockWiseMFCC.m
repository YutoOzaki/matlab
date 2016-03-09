function [blockWise, labels, sourceInfo] = blockWiseMFCC(url,mapObj,coef_range,blocks,patch,N,M,preemp,w,trifil,ceplifter,isdct)
    %% Correction factors
    FFTL = 2^nextpow2(N);
    
    ACF = sum(w)/FFTL;
    ENBWCF = sum(w.*w)/(sum(w)^2) * FFTL;
    CF = 1/sqrt(ENBWCF*FFTL^2);
    
    %% internal variables
    samples = length(url);
    FFTL_half = 1:(floor(FFTL/2) + 1);
    coef_num = length(coef_range);
    dim = coef_num;
    
    %% Variable for output
    blockWise = zeros(samples*patch,dim,blocks);
    labels = zeros(samples*patch,1);
    sourceInfo = cell(samples*patch,1);
    rndidx = 1:samples;
    %rndidx = randperm(samples);

    w_z = [w; zeros(FFTL-N,1)];

    %% Compute MFCC
    for l=1:samples
        %% read audio file
        audioFile = url{rndidx(l),1};
        y = audioread(audioFile);

        %% attach genre label to data
        label_str = strsplit(audioFile,'\');
        label_num = mapObj(label_str{end-1});

        fprintf('loop %d %s\n', l, label_str{end});

        %% Loop start
        L = length(y);
        rnd_start = randperm(L - FFTL*blocks +  1);

        i = 1;
        patchCounter = 1;
        while patchCounter <= patch        
            N_start = rnd_start(i);
            N_end = N_start + N - 1;
            vec2frame = zeros(FFTL,blocks);

            k = 1;
            while k <= blocks
                frame = y(N_start:N_end);

                if 0 == isinf(log(sum(frame.^2)/N))
                    vec2frame(1:N,k) = frame;
                    N_start = N_start + M;
                    N_end = N_start + N - 1;

                    k = k + 1;
                else
                    k = 1;
                    i = i + 1;              
                    N_start = rnd_start(i);
                    N_end = N_start + N - 1;               
                end
            end
            i = i + 1;

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
            fbank = trifil * bin(FFTL_half,:);

            f = log(fbank);

            %% Get cepstrum coefficients
            if isdct == 1
                C = dct(f);
                C(1,:) = logE;
                %C(1) = C(1) * sqrt(coef);
                %C(2:coef) = C(2:coef) .* sqrt(coef/2);

                C = diag(ceplifter) * C;
            else 
                C = f;
            end
            C = C(coef_range,:);

            %{
            figure(1);
            plot(y(N_start - M*blocks:N_start-1));
            figure(2);
            surf(C,'LineStyle','None','EdgeColor','None');
            view(0,90); axis tight; colormap gray; drawnow
            pause
            %}

            blockWise((l-1)*patch+patchCounter,:,:) = C;
            patchCounter = patchCounter + 1;
        end

        labels((l-1)*patch+1:l*patch,1) = label_num;
        sourceInfo((l-1)*patch+1:l*patch,1) = {audioFile};
    end
end