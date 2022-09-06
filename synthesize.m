% code with the Matlab R2022a version
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% create Rayleigh fading channel
% In Rayleigh model, only Non Line of Sight(NLOS) components are simulated 
% between transmitter and receiver. It is assumed that there is no LOS path 
% exists between transmitter and receiver.

rayleigh_chan = comm.RayleighChannel( ...
    'SampleRate',1.92e6, ...
    'PathDelays',0, ...
    'AveragePathGains',0, ...
    'MaximumDopplerShift',2000, ...
    'ChannelFiltering',false, ...
    'NumSamples',1e5, ...
    'FadingTechnique','Sum of sinusoids');

% synthesize Rayleigh channel gain data (both real and imaginary part)
rayleigh_gain = rayleigh_chan();


% create Rician fading channel
% In rician model, both Line of Sight (LOS) and non Line of Sight(NLOS) 
% components are simulated between transmitter and receiver.

rician_chan = comm.RicianChannel( ...
    'SampleRate',1e6, ...
    'NumSamples',1e5, ...
    'KFactor',2.8, ...
    'DirectPathDopplerShift',5.0, ...
    'DirectPathInitialPhase',0.5, ...
    'MaximumDopplerShift',50, ...
    'DopplerSpectrum',doppler('Bell', 8), ...
    'ChannelFiltering',false);

% synthesize Rician channel gain data (both real and imaginary part)
rician_gain = rician_chan();


% create Nakagami fading channel
% It was originally developed empirically based on measurements. The Matlab did
% not provide an Object for the Nakagami fading channel yet.

nakagami_dist = makedist('Nakagami','mu',1,'omega',1);

% synthesize Nakagami channel gain data (both real and imaginary part)
nakagami_gain = random(nakagami_dist,[1e5,1]) + i*random(nakagami_dist,[1e5,1]);


% create Weibull fading channel
% Weibull distribution represents another generalization of the Rayleigh 
% distribution. The Matlab did not provide an Object for the Weibull fading 
% channel yet.

weibull_dist = makedist('Weibull','A',1,'B',1);

% synthesize Weibull channel gain data (both real and imaginary part)
weibull_gain = random(weibull_dist,[1e5,1]) + i*random(weibull_dist,[1e5,1]);


% save channel gain data into .mat file for later reuse by Python
save('channel_gain.mat', ...
     'rayleigh_gain','rician_gain','nakagami_gain','weibull_gain');
load('channel_gain.mat')


% The power received at the receiver from the transmitter is given by 
% G_{ij}|A_{ij}|^2p_j, where G_{ij} stands for the path gain (real-valued; for
% power, not amplitude) adopted from the well-known path loss model G_{ij} 
% = 10^{-12.8}d_{ij}^{-3.76}, where d_{ij} is the Euclidean distance between 
% the i-th transmitter and the j-th receiver and 10^{-12.8} is the attenuation 
% factor that represents power variations due to path loss, and A_{ij} models 
% the signal amplitude (complex-valued; for amplitude, not power) under the 
% specific fading types as shown above.

% According to this formula, we can generate power amplification factors using 
% G_{ij}|A_{ij}|^2 = G_{ij}(Re{A_{ij}}^2 + Im{A_{ij}}^2) computed under
% specific channel fading models. The actual power received at the receivers is 
% found by multiplying with the actual power transmitted at the transmitters
% G_{ij}|A_{ij}|^2p_j. The distance should be prescribed beforehand to compute 
% G_{ij}, and we frequently omit the attenuation factor 10^{-12.8} for 
% simplicity.

% However, this Matlab script generate channel gains only, and save them into 
% a .mat file for later reuse by Python.
