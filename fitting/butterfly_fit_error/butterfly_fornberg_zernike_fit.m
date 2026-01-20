clear; close all; clc;

if ~exist('coeffs', 'var')
    fprintf('Coeffs not detected, running butterfly_fornberg.m...\n');
    run('butterfly_fornberg.m'); 
end

fprintf('1. Generating ground truth wavefront (Target RMS = 0.05λ)...\n');

N_eval = 512;
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
target_rms = 0.058;
[W_true, true_rms] = force_rms(W_raw, mask_eval, target_rms);

fprintf('2. Generating conformal sampling points...\n');

N_r_samp = 24;    
N_th_samp = 64;   
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

W_samples = interp2(X_eval, Y_eval, W_true, x_samp, y_samp, 'cubic');
valid_s = ~isnan(W_samples);
x_samp = x_samp(valid_s); y_samp = y_samp(valid_s);
r_c_flat = r_c_flat(valid_s); t_c_flat = t_c_flat(valid_s);
W_samples = W_samples(valid_s);

fprintf('   Valid sampling points: %d\n', length(x_samp));

fprintf('3. Performing Zernike fitting...\n');
A = zeros(length(W_samples), z_modes);
for j = 1:z_modes
    A(:, j) = zernike_func(j, r_c_flat, t_c_flat);
end
c_fit = A \ W_samples;

fprintf('4. Reconstruction and error evaluation...\n');
R_dense = linspace(0, 1, 150);
T_dense = linspace(0, 2*pi, 300);
[RR, TT] = meshgrid(R_dense, T_dense);
WW_dense = RR .* exp(1i * TT);
ZZ_dense = polyval_func(coeffs, WW_dense);
XX_dense = real(ZZ_dense); YY_dense = imag(ZZ_dense);

W_fit_dense_vals = get_z(c_fit, RR, TT);
F = scatteredInterpolant(XX_dense(:), YY_dense(:), W_fit_dense_vals(:), 'natural', 'none');
W_fit_recon = F(X_eval, Y_eval);
W_fit_recon(~mask_eval) = NaN;

Residual = W_true - W_fit_recon;
rms_fit_error = sqrt(nanmean(Residual(:).^2));
fprintf('   Fitting RMS Error: %.5f λ\n', rms_fit_error);

pv_true=max(W_true(:))-min(W_true(:));
fprintf('   Ture PV: %.5f λ \n', pv_true);

pv_conf=max(Residual(:))-min(Residual(:));
fprintf('   Fornberg Conformal Fit PV: %.5f λ \n', pv_conf);

climit = [-0.16, 0.16]; 

figure('Name', 'Ground Truth', 'NumberTitle', 'off', 'Color', 'w', 'Position', [620, 500, 500, 400]);
plot_wf(X_eval, Y_eval, W_true, mask_eval, 'Ground Truth Wavefront', climit, [0.5, 0.6]);

figure('Name', 'Reconstructed', 'NumberTitle', 'off', 'Color', 'w', 'Position', [100, 50, 500, 400]);
plot_wf(X_eval, Y_eval, W_fit_recon, mask_eval, 'Fitted Wavefront (Reconstructed)', climit, [0.5, 0.6]);

figure('Name', 'Residual Error', 'NumberTitle', 'off', 'Color', 'w', 'Position', [620, 50, 500, 400]);
plot_wf(X_eval, Y_eval, Residual, mask_eval, ...
    sprintf('Full Aperture Residual\n(RMS = %.4f\\lambda)', rms_fit_error), climit/2,[0.5, 0.6]);

function [W_out, final_rms] = force_rms(W_in, mask, target_rms)
    vals = W_in(mask);
    vals = vals - mean(vals);
    current_rms = sqrt(mean(vals.^2));
    scale = target_rms / current_rms;
    vals_scaled = vals * scale;
    W_out = nan(size(W_in));
    W_out(mask) = vals_scaled;
    final_rms = sqrt(mean(vals_scaled.^2));
end

function plot_wf(X, Y, W, mask, txt, clim_range, cb_scale)
    W_vis = W; W_vis(~mask) = NaN;
    s = pcolor(X, Y, W_vis);
    set(s, 'EdgeColor', 'none'); shading interp;
    colormap(gca, jet(256)); 
    axis equal tight; axis off;
    title(txt, 'FontSize', 12, 'Interpreter', 'tex');
    clim(clim_range);
    
    cb = colorbar;
    cb.Label.String = 'Amplitude (\lambda)';
    
    if nargin > 6 && ~isempty(cb_scale)
        pos = cb.Position; 
        pos(3) = pos(3) * cb_scale(1);
        if length(cb_scale) > 1
            original_height = pos(4);
            new_height = original_height * cb_scale(2);
            diff_h = original_height - new_height;
            
            pos(4) = new_height;
            pos(2) = pos(2) + diff_h / 2;
        end
        cb.Position = pos;
    end
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
