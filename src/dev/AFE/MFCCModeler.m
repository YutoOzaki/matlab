function hprms = MFCCModeler()
    hprms = struct(....
        'method',       'MFCC',...
        'mfb_type',     1,...
        'f_start',      10,...
        'f_end',        10000,...
        'coefnum',      40,....
        'coef_range',   1:40,...
        'unitpow',      true,...
        'liftering',    0,...
        'liftcoef',     32,...
        'mfbc',         false,...
        'd',            0,...
        'dd',           0,...
        'N',            1024,...        
        'M',            1024,...
        'preemp',       [1 0],...
        'w',            @hamming,...
        'blocks',       80,...
        'patch',        80,...
        'checkfs',      22050 ...
    );
end

%% notes
% method        : MFCC
% mfb_type      : The type of mel-filter banks.  1 -> parametric, 2 -> FB-40[1].
% f_start       : lowest center frequency of mel-fiter banks.
% f_end         : highest center frequency of mel-fiter banks.
% coefnum       : this defines the number of mel-filter banks.
% coef_range    : typically 2:14. Note the first index is DC component.
% unitpow       : this defines whether to normalize each filter bank to have unit power.
% liftering     : this defines whether to apply liftering. 0 -> false, 1 -> sinusoidal lifter[2], 2 -> exponential lifter[2]
% liftcoef      : coefficients of liftering. This is ignored if liftering is false.
% mfbc          : MFCC returns mel-filter bank coefficients when this is true.
% d             : delta MFCC.
% dd            : delta delta MFCC. This is ignored when d is false.
% N             : length of the FFT segment on audio data.
% M             : length of the non-overlap samples of Fourier transform.
% preemp        : coefficients for pre-emphasis filter. Typically [1 -0.97] for speech recognition.
% w             : window function.
% blocks        : the number of blocks consisting block-wise MFCC.
% patch         : the number of block-wise MFCC extracting from single audio data. (local "audio" patch, as like local image patch).
% checkfs       : required sampling frequency of audio data
%
% [1] Todor Ganchev, Nikos Fakotakis, George Kokkinakis: "Comparative Evaluation of
%     Various MFCC Implementation on the Speaker Verification Task, In
%     Proceedings of the 10th International Conference on Speech and
%     Computer (SPECOM 2005), volume 1, pages 191-194, 2005
% [2] K. K. Paliwal: "Decorrelated and Liftered Filter-Bank Energies for Robust Speech
%     Recognition", In fifth international symposium on siganl processing
%     and applications, Aug 1999.