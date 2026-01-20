clear; close all; clc;

data_file = 'mapping_result.mat';
if ~exist(data_file, 'file')
    error('File %s not found. Run CSF_Beltrami_Solver.m first.', data_file);
end
load(data_file, 'XY_Phys', 'UV_Calc');

fprintf('CSF data loaded successfully. Mesh nodes: %d\n', size(XY_Phys, 1));

x_samp = XY_Phys(:, 1);
y_samp = XY_Phys(:, 2);
u_samp = UV_Calc(:, 1);
v_samp = UV_Calc(:, 2);

[theta_samp, r_samp] = cart2pol(u_samp, v_samp);

max_r = max(r_samp);
if max_r > 1.0001
    fprintf('Warning: Computation domain radius slightly out of bounds (Max r = %.6f), normalizing...\n', max_r);
    r_samp = r_samp / max_r;
end

fprintf('Generating ground truth wavefront (Target RMS = 0.05λ)...\n');

N_eval = 512;
x_vec = linspace(-1.6, 1.6, N_eval);
[X_eval, Y_eval] = meshgrid(x_vec, x_vec);

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

[Theta_eval, R_eval] = cart2pol(X_eval, Y_eval);
get_z = @(c, r, t) sum(reshape(cell2mat(arrayfun(@(j) c(j)*zernike_func(j, r, t), ...
    1:z_modes, 'UniformOutput', false)), [size(r), z_modes]), 3);

W_raw = get_z(c_true, R_eval, Theta_eval);
target_rms = 0.05;
[W_true, true_rms] = force_rms(W_raw, mask_eval, target_rms);

fprintf('Sampling wavefront at CSF nodes...\n');

W_meas = interp2(X_eval, Y_eval, W_true, x_samp, y_samp, 'cubic');

valid_idx = ~isnan(W_meas) & inpolygon(x_samp, y_samp, xb1, yb1);
x_fit = x_samp(valid_idx);
y_fit = y_samp(valid_idx);
r_fit = r_samp(valid_idx);
t_fit = theta_samp(valid_idx);
W_meas = W_meas(valid_idx);

fprintf('   Valid fitting points: %d\n', length(x_fit));

fprintf('Performing CSF-Zernike fitting...\n');

A = zeros(length(x_fit), z_modes);
for j = 1:z_modes
    A(:, j) = zernike_func(j, r_fit, t_fit);
end

c_fit = A \ W_meas;

fprintf('Reconstructing full aperture wavefront...\n');

W_fit_at_samples = A * c_fit;

F = scatteredInterpolant(x_fit, y_fit, W_fit_at_samples, 'natural', 'linear');
W_fit_recon = F(X_eval, Y_eval);
W_fit_recon(~mask_eval) = NaN;

Residual = W_true - W_fit_recon;
rms_fit_error = sqrt(nanmean(Residual(:).^2));

fprintf('------------------------------------------------\n');
fprintf('CSF fitting RMS Error: %.5f λ\n', rms_fit_error);
fprintf('Comparison: Target RMS = %.5f λ\n', true_rms);
fprintf('------------------------------------------------\n');

pv_true=max(W_true(:))-min(W_true(:));
fprintf('   Ture PV: %.5f λ \n', pv_true);

pv_conf=max(Residual(:))-min(Residual(:));
fprintf('   CSF Fit PV: %.5f λ \n', pv_conf);

climit = [-0.16, 0.16]; 

figure('Name', 'Ground Truth', 'NumberTitle', 'off', 'Color', 'w', 'Position', [620, 500, 500, 400]);
plot_wf(X_eval, Y_eval, W_true, mask_eval, 'Ground Truth Wavefront', climit, [0.5, 0.5]);

figure('Name', 'CSF Reconstructed', 'NumberTitle', 'off', 'Color', 'w', 'Position', [100, 50, 500, 400]);
plot_wf(X_eval, Y_eval, W_fit_recon, mask_eval, 'CSF-Fitted Wavefront', climit, [0.5, 0.5]);

figure('Name', 'CSF Residual', 'NumberTitle', 'off', 'Color', 'w', 'Position', [620, 50, 500, 400]);
plot_wf(X_eval, Y_eval, Residual, mask_eval, ...
    sprintf('CSF Residual Error\n(RMS = %.4f\\lambda)', rms_fit_error), climit/2, [0.5, 0.5]);

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
            orig_h = pos(4);
            new_h = orig_h * cb_scale(2);
            pos(4) = new_h;
            pos(2) = pos(2) + (orig_h - new_h)/2;
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
