function [mel_filters, freqs] = mfccFB40(nfft)
    nlinfilt = 13; %linear scaling bands
    nlogfilt = 27; %log scaling bands
    nfilt = nlinfilt + nlogfilt;
    N = nfilt + 2;
    freqs = zeros(1, N);

    if edit == 1
        lowfreq = 0;
        linsc = 933.3333/(nlinfilt-1);
        f40 = 7300;
    else
        lowfreq = 400/3.0;
        linsc = 200/3.0;
        f40 = 6400;
    end

    logsc = exp(log(f40/1000)/nlogfilt);
    fs = 16000;
    half = round(nfft/2);

    for i=1:nlinfilt
        freqs(i) = lowfreq + (i-1)*linsc;
    end
    for i=(nlinfilt+1):N
        freqs(i) = freqs(nlinfilt) * logsc^(i-nlinfilt);
    end

    nfreqs = zeros(1,nfft);
    for i=1:nfft
        nfreqs(i) = fs*(i-1)/nfft; %frequency resolution is fs/nfft
    end
    mel_filters = zeros(nfilt,nfft);

    for i=1:nfilt
        triangle = zeros(1,nfft);

        low = freqs(i);
        cen = freqs(i+1);
        hi = freqs(i+2);

        [~,low_i] = min(abs(nfreqs-low));
        [~,cen_i] = min(abs(nfreqs-cen));
        [~,hi_i] = min(abs(nfreqs-hi));

        for k=1:half
            if k<low_i
                triangle(k) = 0;
            elseif low_i<=k && k<cen_i
                triangle(k) = 2*(k-low_i)/((cen_i-low_i)*(hi_i-low_i));
            elseif cen_i<=k && k<=hi_i
                triangle(k) = 2*(hi_i-k)/((hi_i-cen_i)*(hi_i-low_i));
            else
                triangle(k) = 0;
            end
        end

        mel_filters(i,:) = triangle;
    end

    mel_filters = mel_filters';
end

%% Reference
% 1. Ganchev, T., Fakotakis, N., Kokkinakis, G.: Comparative Evaluation of
%    Various MFCC Implementations on the Speaker Verification Task. In
%    Proceedings of the 10th International Conference on Speach and
%    Computer, volume 1, pp.191-194 (2005)
%
% 2. Davis, S.B., Mermelstein, P.: Comparison of Parametric Representations
%    for Monosyllabic Word Recognition in Continuously Spoken Sentences.
%    IEEE Transactions on Acoustic, Speech, and Signal Processing,
%    Vol.28(4), pp.357-366 (1980)
%
% 3. Slaney, M.: Auditory Toolbox Version 2: Technial Report #1998-010, 
%    Interval Research Corporation (1998)