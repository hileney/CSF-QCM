clear; close all; clc;

fprintf('Generating butterfly boundary...\n');

angles_deg = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360];
theta_ctrl = deg2rad(angles_deg);
r_ctrl = [1.0, 0.9, 0.6, 0.25, 0.6, 0.9, 1.0, 0.9, 0.6, 0.25, 0.6, 0.9, 1.0]; 

N_dense = 1024; 
theta_dense = linspace(0, 2*pi, N_dense+1); 
theta_dense(end) = [];

r_dense = makima(theta_ctrl, r_ctrl, theta_dense);
[x_dense, y_dense] = pol2cart(theta_dense, r_dense);
z_boundary = x_dense + 1i * y_dense;

scale_factor = max(abs(z_boundary));
z_boundary = z_boundary / scale_factor;

fprintf('Solving conformal mapping coefficients (may take a few seconds)...\n');

N_coeffs = 512; 
[coeffs, err_history] = solve_polygon_map(z_boundary, N_coeffs);

fprintf('Iteration complete. Final boundary error: %.2e\n', err_history(end));

figure('Position', [100, 100, 1200, 550], 'Color', [1 1 1]);

subplot(1, 2, 1);
[x_ctrl, y_ctrl] = pol2cart(theta_ctrl, r_ctrl/scale_factor);
plot(x_ctrl, y_ctrl, 'ro', 'MarkerSize', 6, 'LineWidth', 1.5); hold on;

z_recon = polyval_func(coeffs, exp(1i * linspace(0, 2*pi, 500)));
plot(real(z_recon), imag(z_recon), 'b-', 'LineWidth', 2);

legend('Original control points', 'Conformal map reconstruction', 'Location', 'best');
title({'Boundary Fitting Check'}, 'FontSize', 12);
axis equal; grid on;
xlim([-1.2 1.2]); ylim([-1.2 1.2]);

subplot(1, 2, 2);
hold on;

plot(real(z_recon), imag(z_recon), 'k-', 'LineWidth', 2);

r_vals = linspace(0.2, 0.98, 16);
theta_plot = linspace(0, 2*pi, 600);
for r = r_vals
    w_circle = r * exp(1i * theta_plot);
    z_circle = polyval_func(coeffs, w_circle);
    plot(real(z_circle), imag(z_circle), 'b-', 'LineWidth', 0.8, 'Color', [0 0.4 0.8 0.5]);
end

theta_vals = linspace(0, 2*pi, 64); 
r_line = linspace(0, 1, 100);
for th = theta_vals
    w_line = r_line * exp(1i * th);
    z_line = polyval_func(coeffs, w_line);
    plot(real(z_line), imag(z_line), 'r-', 'LineWidth', 0.8, 'Color', [0.8 0.2 0.2 0.6]);
end

axis equal; 
title({'Direct Conformal Map', 'Note the Severe Crowding at the Waist'}, 'FontSize', 12);
xlim([-1.2 1.2]); ylim([-1.2 1.2]);
axis off;

fprintf('Performing grid uniformity comparison analysis...\n');

N_r_stats = 100; N_th_stats = 360;   
r_vec = linspace(0.01, 0.99, N_r_stats);
th_vec = linspace(0, 2*pi, N_th_stats+1);

Z_grid = zeros(length(r_vec), length(th_vec));
for i = 1:length(r_vec)
    Z_grid(i, :) = polyval_func(coeffs, r_vec(i) * exp(1i * th_vec));
end

grid_areas = [];
for i = 1:N_r_stats-1
    for j = 1:N_th_stats
        p1 = Z_grid(i, j); p2 = Z_grid(i, j+1);
        p3 = Z_grid(i+1, j+1); p4 = Z_grid(i+1, j);
        nodes = [p1; p2; p3; p4];
        grid_areas = [grid_areas; polyarea(real(nodes), imag(nodes))];
    end
end
ratio_forn = grid_areas / mean(grid_areas);

x_forn = sort(ratio_forn);
y_forn = (1:length(x_forn))' / length(x_forn);
window_size_f = round(length(x_forn) * 0.02);
x_forn_smooth = smoothdata(x_forn, 'gaussian', window_size_f);

qcm_data_file = 'qcm_area_stats.mat';
has_qcm_data = false;
if exist(qcm_data_file, 'file')
    loaded_data = load(qcm_data_file);
    ratio_qcm = loaded_data.ratio_qcm;
    has_qcm_data = true;
    
    x_qcm = sort(ratio_qcm);
    y_qcm = (1:length(x_qcm))' / length(x_qcm);
    window_size_q = round(length(x_qcm) * 0.02);
    x_qcm_smooth = smoothdata(x_qcm, 'gaussian', window_size_q);
else
    fprintf('Note: %s not found. Showing Fornberg results only.\n', qcm_data_file);
end

figure('Position', [150, 150, 800, 600], 'Color', 'w');

h1 = semilogx(x_forn_smooth, y_forn, 'b-', 'LineWidth', 3);
hold on;

legend_str = {'Fornberg / Wegmann (Conformal)'};
if has_qcm_data
    h2 = semilogx(x_qcm_smooth, y_qcm, 'r-', 'LineWidth', 3);
    legend_str{end+1} = 'CSF-QCM (Proposed)';
end

xline(1.0, 'k--', 'LineWidth', 1.5);

grid on;
set(gca, 'FontSize', 12, 'LineWidth', 1.2, 'TickDir', 'out');

set(gca, 'XScale', 'log'); 
set(gca, 'XMinorGrid', 'on', 'YMinorGrid', 'on');

xlabel('Normalized Area Ratio ($A_i / \bar{A}$) [Log Scale]', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Cumulative Probability (ECDF)', 'FontSize', 14);
title({'Grid Uniformity Comparison', 'Fornberg vs. CSF-QCM'}, 'FontSize', 14);

legend(legend_str, 'Location', 'southeast', 'FontSize', 12, 'Box', 'on');

stats_msg = {'\bf Statistics Comparison \rm'};
stats_msg{end+1} = sprintf('\\color{blue}Fornberg Max: %.1f', max(ratio_forn));
if has_qcm_data
    stats_msg{end+1} = sprintf('\\color{red}CSF-QCM Max:  %.2f', max(ratio_qcm));
    improvement = max(ratio_forn) / max(ratio_qcm);
    stats_msg{end+1} = sprintf('\\color{black}Improvement:  %.1fx', improvement);
end
text(0.05, 0.75, stats_msg, 'Units', 'normalized', ...
    'BackgroundColor', 'w', 'EdgeColor', 'k', ...
    'Margin', 6, 'FontSize', 11, 'Interpreter', 'tex');

max_val = max(ratio_forn);
if has_qcm_data
    max_val = max(max_val, max(ratio_qcm));
end
xlim([0.01, max_val * 2]); 

fprintf('Comparison plot completed (log scale corrected).\n');

function z = polyval_func(coeffs, w)
    z = zeros(size(w));
    ww = ones(size(w));
    for k = 1:length(coeffs)
        z = z + coeffs(k) * ww;
        ww = ww .* w;
    end
end

function [coeffs, err_history] = solve_polygon_map(z_target, N)
    coeffs = zeros(N, 1);
    coeffs(2) = 1.0; 
    
    max_iter = 100;
    err_history = zeros(max_iter, 1);
    
    ang_target = angle(z_target);
    ang_target = unwrap(ang_target);
    
    for iter = 1:max_iter
        w_grid = exp(1i * linspace(0, 2*pi, length(z_target)+1));
        w_grid(end) = [];
        z_curr = polyval_func(coeffs, w_grid).'; 
        
        ang_curr = angle(z_curr);
        ang_curr = unwrap(ang_curr);
        
        phase_shift = ang_target(1) - ang_curr(1);
        ang_curr_shifted = ang_curr + phase_shift;
        
        z_projected = interp1(ang_target, z_target, ang_curr_shifted, 'linear', 'extrap');
        
        diff_vec = z_projected - z_curr;
        err_history(iter) = norm(diff_vec);
        
        if err_history(iter) < 1e-3
            err_history = err_history(1:iter);
            break;
        end
        
        coeffs_new = fft(z_projected) / length(z_projected);
        
        coeffs_update = zeros(N, 1);
        n_keep = min(N, length(coeffs_new));
        coeffs_update(1:n_keep) = coeffs_new(1:n_keep);
        
        damping = 0.3;
        coeffs = coeffs + damping * (coeffs_update - coeffs);
    end
end
