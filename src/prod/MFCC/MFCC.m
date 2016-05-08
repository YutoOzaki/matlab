classdef MFCC < handle
    properties
        coefnum
        MAX_DURATION
        N, M
        preemp
        w, ACF, CF
        useMelSpec, dctHelper
        coef_range
        ceplifter
        mfccMap, mfbHelper
    end
    
    methods
        function obj = MFCC()
            hprms = MFCCModeler();
            
            obj.coefnum     = hprms.coefnum;
            obj.MAX_DURATION = hprms.MAX_DURATION;
            obj.N           = hprms.N;
            obj.M           = hprms.M;
            obj.preemp      = hprms.preemp;
            obj.w           = hprms.w(obj.N);
            obj.ACF         = sum(obj.w)/obj.N;
            
            ENBWCF          = sum(obj.w.^2)/(sum(obj.w)^2) * obj.N;
            obj.CF          = sum(obj.w.^2)./(ENBWCF*obj.N^2);
            
            obj.useMelSpec  = hprms.useMelSpec;
            
            if obj.useMelSpec
                obj.dctHelper = @(obj, x, a) x;
            else
                obj.dctHelper = @myDCT;
            end
            
            obj.coef_range  = hprms.coef_range;
            initLifteringFun(obj, hprms.liftering);
        
            obj.mfccMap = containers.Map('KeyType', 'int32', 'ValueType', 'any');
            obj.mfbHelper = @(obj, fs) createMelFilterBank(obj, fs, hprms.mfb_type, hprms.f_start,...
                hprms.f_end, hprms.unitpow, 2^nextpow2(obj.N));
            %{
            dnmt_d      = hprms.dnmt_d;
            dnmt_dd     = hprms.dnmt_dd;
            dmfcc       = hprms.dmfcc;
            ddmfcc      = hprms.ddmfcc;
            d           = hprms.d;
            dd          = hprms.dd;
            %}
        end
        
        function C = myDCT(obj, F, logE)
            C = dct(F);
            
            C(1,:) = logE;
            %C(1,:) = C(1,:) * sqrt(obj.coefnum);
            %C(2:end,:) = C(2:end,:) .* sqrt(obj.coefnum/2);

            C = obj.ceplifter * C;
        end
        
        function initLifteringFun(obj, liftering)
            if strcmp(liftering{1}, 'sinusoidal')
                obj.ceplifter = 1 + (liftering{2}/2)*sin(pi*(1:obj.coefnum)'/liftering{2});
            elseif strcmp(liftering{1}, 'exponential')
                a = liftering{2}(1);
                tau = liftering{2}(2);
                obj.ceplifter = ((1:obj.coefnum).^a).*exp(-((1:obj.coefnum).^2)./(2*tau^2));
            else
                obj.ceplifter = ones(obj.coefnum,1);
            end
            
            obj.ceplifter = diag(obj.ceplifter);
        end
        
        function melfilbank = createMelFilterBank(obj, fs, mfb_type, f_start, f_end, unitpow, FFTL)
            if mfb_type == 1
                melfilbank = triangularFB(obj.coefnum, f_start, f_end, FFTL, unitpow, fs);
            else
                melfilbank = mfccFB40(FFTL);
                FFTL_half = 1:(floor(FFTL/2) + 1);
                melfilbank = melfilbank(FFTL_half,:)';
            end
        end
        
        function cellFormat = extract(obj, filePath)
            batchSize = length(filePath);
            cellFormat = cell(batchSize,1);
            
            for i=1:batchSize
                [x,fs] = audioread(filePath{i});
                cellFormat{i} = takeMFCC(obj, x, fs);
            end
        end
        
        function C = takeMFCC(obj, x, fs)
            if isKey(obj.mfccMap, fs)
                melfilbank = obj.mfccMap(fs);
            else
                melfilbank = obj.mfbHelper(obj, fs);
                obj.mfccMap(fs) = melfilbank;
            end
            
            L = length(x);
            L_MAX = obj.MAX_DURATION * fs;
            if(L > L_MAX)
                x = x(1:L_MAX);
                L = L_MAX;
            end

            %% frame matrix
            T = floor((L - obj.N)/obj.M) + 1;
            adjustedL = L - (T-1)*obj.M - obj.N;
            x = x(1:L - adjustedL);
            
            %% add slight DC to avoid Inf in log computation
            x = x + eps;
            
            %% Pre-emphasis
            s_pe = filter(obj.preemp, 1, x);
            s_pe = s_pe ./ obj.ACF;
            
            %% FFT
            [~,~,~,frame2vec] = spectrogram(s_pe, obj.w, obj.N-obj.M, obj.N, fs);
            frame2vec = frame2vec.*fs.*obj.CF;
           
            %% Energy
            logE = log(sum(frame2vec.^2, 1));

            %% Mel-filtering
            fbank = melfilbank * frame2vec;
            f = log(fbank);

            %% Get cepstrum coefficients
            C = obj.dctHelper(obj, f, logE);
            C = C(obj.coef_range,:);

            %% delta coefficients
            %{
            delta = dmfcc(C,d,blocks)./dnmt_d;
            deltadelta = ddmfcc(delta,dd,blocks)./dnmt_dd;
            bufbwm = [C;delta;deltadelta];
            %}
        end
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