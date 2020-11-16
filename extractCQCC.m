function coeff = extractCQCC(x, fs)
addpath(genpath('Baselines/CQCC_v1.0'));
coeff = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
