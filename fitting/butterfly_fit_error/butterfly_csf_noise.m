clear; close all; clc;

data_file = 'mapping_result.mat';
if ~exist(data_file, 'file')
    error('File %s not found. Run CSF_Beltrami_Solver.m first.', data_file);
end
load(data_file, 'XY_Phys', 'UV_Calc');
fprintf('1. CSF data loaded successfully...\n');

u_samp = UV_Calc(:, 1); v_samp = UV_Calc(:, 2);
[theta_samp, r_samp] = cart2pol(u_samp, v_samp);

max_r = max(r_samp);
if max_r > 1.000001, r_samp = r_samp / max_r; end

x_samp = XY_Phys(:, 1); y_samp = XY_Phys(:, 2);

fprintf('2. Generating ground truth wavefront (Target RMS = 0.05λ)...\n');

N_eval = 256; 
x_vec = linspace(-1.6, 1.6, N_eval);
[X_eval, Y_eval] = meshgrid(x_vec, x_vec);

theta_c = deg2rad([0,30,60,90,120,150,180,210,240,270,300,330,360]);
r_c = [1.0,0.9,0.6,0.25,0.6,0.9,1.0,0.9,0.6,0.25,0.6,0.9,1.0];
[xb, yb] = pol2cart(linspace(0,2*pi,2000), makima(theta_c, r_c, linspace(0,2*pi,2000)));
mask_eval = inpolygon(X_eval, Y_eval, xb, yb);

z_modes = 36;
rng(2025); 
c_true = (rand(z_modes, 1) - 0.5);
c_true(1) = 0; 
for k=1:z_modes
    [n_val, ~] = get_noll_nm(k);
    c_true(k) = c_true(k) / (n_val + 1); 
end

[Theta_eval, R_eval] = cart2pol(X_eval, Y_eval);
get_z = @(c, r, t) sum(reshape(cell2mat(arrayfun(@(j) c(j)*zernike_func(j, r, t), ...
    1:z_modes, 'UniformOutput', false)), [size(r), z_modes]), 3);

W_raw = get_z(c_true, R_eval, Theta_eval);
[W_true, ~] = force_rms(W_raw, mask_eval, 0.06);

fprintf('3. Building fitting matrix...\n');

W_clean_all = interp2(X_eval, Y_eval, W_true, x_samp, y_samp, 'cubic');

valid_idx = ~isnan(W_clean_all) & inpolygon(x_samp, y_samp, xb, yb);

x_fit = x_samp(valid_idx); y_fit = y_samp(valid_idx);
r_fit = r_samp(valid_idx); t_fit = theta_samp(valid_idx);
W_clean_samp = W_clean_all(valid_idx);

A = zeros(length(x_fit), z_modes);
for j = 1:z_modes
    A(:, j) = zernike_func(j, r_fit, t_fit);
end

fprintf('4. Starting CSF Monte Carlo speckle noise scan...\n');

noise_levels = linspace(0.05, 0.20, 16);
n_trials = 50; 
rms_mean = zeros(length(noise_levels), 1);
rms_std  = zeros(length(noise_levels), 1);

U_signal = 1.0 .* exp(1i * W_clean_samp);

fprintf('   Progress: ');
for k = 1:length(noise_levels)
    sigma = noise_levels(k);
    trial_errors = zeros(n_trials, 1);
    
    if mod(k, 4) == 0, fprintf(' %.0f%% ', (k/length(noise_levels))*100); else, fprintf('.'); end
    
    for t = 1:n_trials
        n_real = sigma * randn(size(W_clean_samp));
        n_imag = sigma * randn(size(W_clean_samp));
        U_noise = (n_real + 1i * n_imag) / sqrt(2);
        
        U_total = U_signal + U_noise;
        phase_noise = angle(U_total ./ U_signal);
        
        W_input = W_clean_samp + phase_noise;
        
        c_fit = A \ W_input;
        
        W_fit_on_nodes = A * c_fit;
        
        F = scatteredInterpolant(x_fit, y_fit, W_fit_on_nodes, 'natural', 'nearest');
        W_recon = F(X_eval, Y_eval);
        
        diff = W_true - W_recon;
        trial_errors(t) = sqrt(nanmean(diff(:).^2));
    end
    
    rms_mean(k) = mean(trial_errors);
    rms_std(k)  = std(trial_errors);
end
fprintf('\nCalculation complete.\n');

figure('Color', 'w', 'Name', 'CSF Speckle Robustness');
errorbar(noise_levels, rms_mean, rms_std, 's-', 'LineWidth', 1.5, ...
    'Color', [0.85, 0.33, 0.1], 'MarkerFaceColor', [0.85, 0.33, 0.1], 'CapSize', 10);
grid on;
xlabel('Speckle Contrast / Scatter Amplitude (\sigma)', 'FontSize', 12);
ylabel('Reconstruction RMS Error (\lambda)', 'FontSize', 12);
title({'CSF-Beltrami Robustness Analysis', 'Speckle Noise Model (Monte Carlo N=50)'}, 'FontSize', 13);
xlim([0.04, 0.21]);

save('csf_noise_data.mat', 'noise_levels', 'rms_mean', 'rms_std');
fprintf('Data saved to csf_noise_data.mat\n');

function [W_out, final_rms] = force_rms(W_in, mask, target_rms)
    vals = W_in(mask); vals = vals - mean(vals);
    scale = target_rms / sqrt(mean(vals.^2));
    W_out = nan(size(W_in)); W_out(mask) = vals * scale;
    final_rms = target_rms;
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
