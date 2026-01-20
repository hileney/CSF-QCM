clear; close all; clc;

if ~exist('coeffs', 'var')
    if exist('butterfly_fornberg.m', 'file')
        fprintf('Loading mapping coefficients...\n');
        run('butterfly_fornberg.m'); 
    else
        error('Error: butterfly_fornberg.m not found, cannot load mapping coefficients.');
    end
end

fprintf('1. Generating ground truth wavefront (Target RMS = 0.05λ)...\n');

N_eval = 256; 
x_vec = linspace(-1.6, 1.6, N_eval);
[X_eval, Y_eval] = meshgrid(x_vec, x_vec);
[Theta_eval, R_eval] = cart2pol(X_eval, Y_eval);

theta_ctrl = deg2rad([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]);
r_ctrl = [1.0, 0.9, 0.6, 0.25, 0.6, 0.9, 1.0, 0.9, 0.6, 0.25, 0.6, 0.9, 1.0]; 
theta_dense = linspace(0, 2*pi, 400);
r_dense = makima(theta_ctrl, r_ctrl, theta_dense);
[xb1, yb1] = pol2cart(theta_dense, r_dense);
mask_eval = inpolygon(X_eval, Y_eval, xb1, yb1);

z_modes = 36;
rng(2025); 
c_true = (rand(z_modes, 1) - 0.5);
c_true(1) = 0; 
for k=1:z_modes
    [n_val, ~] = get_noll_nm(k);
    c_true(k) = c_true(k) / (n_val + 1); 
end

get_z = @(c, r, t) sum(reshape(cell2mat(arrayfun(@(j) c(j)*zernike_func(j, r, t), ...
    1:z_modes, 'UniformOutput', false)), [size(r), z_modes]), 3);

W_raw = get_z(c_true, R_eval, Theta_eval);
[W_true, ~] = force_rms(W_raw, mask_eval, 0.058);

fprintf('2. Precomputing sampling matrix...\n');

N_r_samp = 24; N_th_samp = 64;   
r_vec = linspace(0.01, 0.99, N_r_samp);
t_vec = linspace(0, 2*pi, N_th_samp+1); t_vec(end) = [];
[R_comp, T_comp] = meshgrid(r_vec, t_vec);
W_comp = R_comp .* exp(1i * T_comp);
Z_phys_samp = polyval_func(coeffs, W_comp);

x_samp = real(Z_phys_samp(:));
y_samp = imag(Z_phys_samp(:));
r_c_flat = R_comp(:); t_c_flat = T_comp(:);

in_mask = inpolygon(x_samp, y_samp, xb1, yb1);
x_samp = x_samp(in_mask); y_samp = y_samp(in_mask);
r_c_flat = r_c_flat(in_mask); t_c_flat = t_c_flat(in_mask);

W_samples_clean = interp2(X_eval, Y_eval, W_true, x_samp, y_samp, 'cubic');
valid_s = ~isnan(W_samples_clean);
x_samp = x_samp(valid_s); y_samp = y_samp(valid_s);
r_c_flat = r_c_flat(valid_s); t_c_flat = t_c_flat(valid_s);
W_samples_clean = W_samples_clean(valid_s);

A = zeros(length(x_samp), z_modes);
for j = 1:z_modes
    A(:, j) = zernike_func(j, r_c_flat, t_c_flat);
end

R_dense = linspace(0, 1, 100); T_dense = linspace(0, 2*pi, 200);
[RR, TT] = meshgrid(R_dense, T_dense);
WW_dense = RR .* exp(1i * TT);
ZZ_dense = polyval_func(coeffs, WW_dense);
XX_dense = real(ZZ_dense); YY_dense = imag(ZZ_dense);

fprintf('3. Starting Monte Carlo speckle noise scan (Speckle Monte Carlo)...\n');

noise_levels = linspace(0.05, 0.20, 16); 
n_trials = 50; 
rms_mean = zeros(length(noise_levels), 1);
rms_std  = zeros(length(noise_levels), 1);

U_signal = 1.0 .* exp(1i * W_samples_clean);

fprintf('   Calculating: ');
for k = 1:length(noise_levels)
    sigma = noise_levels(k);
    trial_errors = zeros(n_trials, 1);
    
    if mod(k, 4) == 0, fprintf(' %.0f%% ', (k/length(noise_levels))*100); else, fprintf('.'); end
    
    for t = 1:n_trials
        n_real = sigma * randn(size(W_samples_clean));
        n_imag = sigma * randn(size(W_samples_clean));
        U_noise = (n_real + 1i * n_imag) / sqrt(2);
        
        U_total = U_signal + U_noise;
        
        phase_noise = angle(U_total ./ U_signal);
        
        W_input = W_samples_clean + phase_noise;
        
        c_fit = A \ W_input;
        
        W_recon_vals = get_z(c_fit, RR, TT);
        F = scatteredInterpolant(XX_dense(:), YY_dense(:), W_recon_vals(:), 'natural', 'none');
        W_recon = F(X_eval, Y_eval);
        
        diff = W_true - W_recon;
        trial_errors(t) = sqrt(nanmean(diff(:).^2));
    end
    
    rms_mean(k) = mean(trial_errors);
    rms_std(k)  = std(trial_errors);
end
fprintf('\nCalculation complete.\n');

figure('Color', 'w', 'Name', 'Speckle Robustness Analysis');
errorbar(noise_levels, rms_mean, rms_std, 'o-', 'LineWidth', 1.5, ...
    'MarkerFaceColor', 'r', 'Color', 'r', 'CapSize', 10);
grid on;

xlabel('Speckle Contrast / Scatter Amplitude (\sigma)', 'FontSize', 12);
ylabel('Reconstruction RMS Error (\lambda)', 'FontSize', 12);
title({'Robustness to Speckle Noise', 'Phasor Addition Model (Monte Carlo N=50)'}, 'FontSize', 13);
xlim([0.04, 0.21]);

function [W_out, final_rms] = force_rms(W_in, mask, target_rms)
    vals = W_in(mask);
    vals = vals - mean(vals); 
    current_rms = sqrt(mean(vals.^2));
    if current_rms == 0, scale = 0; else, scale = target_rms / current_rms; end
    vals_scaled = vals * scale;
    W_out = nan(size(W_in)); W_out(mask) = vals_scaled;
    final_rms = sqrt(mean(vals_scaled.^2));
end

function [n, m] = get_noll_nm(j)
    persistent n_table m_table;
    if isempty(n_table)
        n_table = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7];
        m_table = [0, 1,-1, 0,-2, 2,-1, 1,-3, 3, 0,-2, 2,-4, 4,-1, 1,-3, 3,-5, 5, 0,-2, 2,-4, 4,-6, 6,-1, 1,-3, 3,-5, 5,-7, 7];
    end
    if j > length(n_table), n=0; m=0; return; end
    n = n_table(j); m = m_table(j);
end

function Z = zernike_func(j, r, t)
    [n, m] = get_noll_nm(j);
    R_nl = zeros(size(r));
    for k = 0:(n-abs(m))/2
        num = (-1)^k * factorial(n-k);
        den = factorial(k) * factorial((n+m)/2 - k) * factorial((n-m)/2 - k);
        R_nl = R_nl + (num/den) * r.^(n-2*k);
    end
    if m >= 0, Z = R_nl .* cos(m*t); else, Z = R_nl .* sin(abs(m)*t); end
end

function z = polyval_func(coeffs, w)
    z = zeros(size(w)); ww = ones(size(w));
    for k = 1:length(coeffs), z = z + coeffs(k) * ww; ww = ww .* w; end
end
