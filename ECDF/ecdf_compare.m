clear; close all; clc;

aperture_type = 1;  
N_pts = 600;        

fprintf('Generating boundary (Type %d)...\n', aperture_type);

switch aperture_type
    case 1 
        str_title = 'Type I: Butterfly (Deep Waist)';
        angles_deg = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360];
        theta_ctrl = deg2rad(angles_deg);
        r_ctrl = [1.0, 0.9, 0.6, 0.25, 0.6, 0.9, 1.0, 0.9, 0.6, 0.25, 0.6, 0.9, 1.0]; 
        
        t_dense = linspace(0, 2*pi, N_pts+1); 
        t_dense(end) = []; 
        r_dense = makima(theta_ctrl, r_ctrl, t_dense);
        x = r_dense .* cos(t_dense);
        y = r_dense .* sin(t_dense);
        
        [x_ctrl_plot, y_ctrl_plot] = pol2cart(theta_ctrl, r_ctrl);
        
    case 2
        str_title = 'Type II: Rounded Rectangle (High Aspect Ratio)';
        L = 3.0; H = 1.0; R = 0.35; 
        
        pts_per_unit = N_pts / (2*(L+H)); 
        n_line_L = round((L-2*R) * pts_per_unit);
        n_line_H = round((H-2*R) * pts_per_unit);
        n_arc = round(0.5*pi*R * pts_per_unit);
        n_line_L = max(n_line_L, 2); n_line_H = max(n_line_H, 2); n_arc = max(n_arc, 5);
        
        y_R = linspace(-H/2+R, H/2-R, n_line_H); x_R = (L/2) * ones(size(y_R));
        th_tr = linspace(0, pi/2, n_arc); x_TR = (L/2-R) + R*cos(th_tr); y_TR = (H/2-R) + R*sin(th_tr);
        x_T = linspace(L/2-R, -L/2+R, n_line_L); y_T = (H/2) * ones(size(x_T));
        th_tl = linspace(pi/2, pi, n_arc); x_TL = (-L/2+R) + R*cos(th_tl); y_TL = (H/2-R) + R*sin(th_tl);
        y_L = linspace(H/2-R, -H/2+R, n_line_H); x_L = (-L/2) * ones(size(y_L));
        th_bl = linspace(pi, 3*pi/2, n_arc); x_BL = (-L/2+R) + R*cos(th_bl); y_BL = (-H/2+R) + R*sin(th_bl);
        x_B = linspace(-L/2+R, L/2-R, n_line_L); y_B = (-H/2) * ones(size(x_B));
        th_br = linspace(3*pi/2, 2*pi, n_arc); x_BR = (L/2-R) + R*cos(th_br); y_BR = (-H/2+R) + R*sin(th_br);
        
        x_raw = [x_R, x_TR, x_T, x_TL, x_L, x_BL, x_B, x_BR];
        y_raw = [y_R, y_TR, y_T, y_TL, y_L, y_BL, y_B, y_BR];
        
        [x, y] = reparameterize_curve(x_raw, y_raw, N_pts);
        
    otherwise
        error('Unknown aperture type');
end

z_boundary = x + 1i * y;
scale_factor = max(abs(z_boundary));
z_boundary = z_boundary / scale_factor;

fprintf('Solving conformal map coefficients (Fornberg/Wegmann)...\n');
N_coeffs = 512; 
[coeffs, err_history] = solve_polygon_map(z_boundary, N_coeffs);
fprintf('Completed. Final boundary error: %.2e\n', err_history(end));

fprintf('Performing grid uniformity comparison...\n');

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
    fprintf('Loading comparison data from %s...\n', qcm_data_file);
    loaded_data = load(qcm_data_file);
    
    if isfield(loaded_data, 'ratio_qcm')
        ratio_qcm = loaded_data.ratio_qcm;
    elseif isfield(loaded_data, 'area_ratios')
        ratio_qcm = loaded_data.area_ratios;
    else
        warning('ratio_qcm variable not found.');
        ratio_qcm = [];
    end
    
    if ~isempty(ratio_qcm)
        has_qcm_data = true;
        x_qcm = sort(ratio_qcm);
        y_qcm = (1:length(x_qcm))' / length(x_qcm);
        window_size_q = round(length(x_qcm) * 0.02);
        x_qcm_smooth = smoothdata(x_qcm, 'gaussian', window_size_q);
    end
else
    fprintf('Note: %s not found. Showing Fornberg only.\n', qcm_data_file);
end

figure('Position', [150, 150, 800, 600], 'Color', 'w');

h1 = semilogx(x_forn_smooth, y_forn, 'b-', 'LineWidth', 3);
hold on;

legend_str = {'Schwarz-Christoffel Mapping'};
if has_qcm_data
    h2 = semilogx(x_qcm_smooth, y_qcm, 'r-', 'LineWidth', 3);
    legend_str{end+1} = 'CSF-QCM (Proposed)';
end

grid on; set(gca, 'FontSize', 15, 'LineWidth', 1.5, 'TickDir', 'out');
set(gca, 'XScale', 'log'); set(gca, 'XMinorGrid', 'on', 'YMinorGrid', 'on');
xlabel('Normalized Area Ratio ($A_i / \bar{A}$) [Log Scale]', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('Cumulative Probability (ECDF)', 'FontSize', 16);
title('SC Mapping vs. CSF-QCM ', 'FontSize', 18);
legend(legend_str, 'Location', 'southeast', 'FontSize', 14, 'Box', 'on');

stats_msg = {'\bf Statistics Comparison \rm'};
stats_msg{end+1} = sprintf('\\color{blue}SC Mapping Max Ratio: %.1f', max(ratio_forn));
if has_qcm_data
    stats_msg{end+1} = sprintf('\\color{red}CSF-QCM Max Ratio:  %.2f', max(ratio_qcm));
    improvement = max(ratio_forn) / max(ratio_qcm);
    stats_msg{end+1} = sprintf('\\color{black}Improvement:  %.1fx', improvement);
end
text(0.5, 0.5, stats_msg, 'Units', 'normalized', ...
    'BackgroundColor', 'w', 'EdgeColor', 'k', ...
    'Margin', 6, 'FontSize', 16, 'Interpreter', 'tex');

max_val = max(ratio_forn);
if has_qcm_data, max_val = max(max_val, max(ratio_qcm)); end
xlim([0.01, max_val * 2]); 

fprintf('Comparison analysis completed.\n');

function z = polyval_func(coeffs, w)
    z = zeros(size(w)); ww = ones(size(w));
    for k = 1:length(coeffs)
        z = z + coeffs(k) * ww; ww = ww .* w;
    end
end

function [coeffs, err_history] = solve_polygon_map(z_target, N)
    coeffs = zeros(N, 1); coeffs(2) = 1.0; 
    max_iter = 100; err_history = zeros(max_iter, 1);
    
    ang_target = angle(z_target); ang_target = unwrap(ang_target);
    
    for iter = 1:max_iter
        w_grid = exp(1i * linspace(0, 2*pi, length(z_target)+1)); w_grid(end) = [];
        z_curr = polyval_func(coeffs, w_grid).'; 
        
        ang_curr = angle(z_curr); ang_curr = unwrap(ang_curr);
        phase_shift = ang_target(1) - ang_curr(1);
        ang_curr_shifted = ang_curr + phase_shift;
        
        z_projected = interp1(ang_target, z_target, ang_curr_shifted, 'linear', 'extrap');
        
        diff_vec = z_projected - z_curr;
        err_history(iter) = norm(diff_vec);
        if err_history(iter) < 1e-3, err_history = err_history(1:iter); break; end
        
        coeffs_new = fft(z_projected) / length(z_projected);
        coeffs_update = zeros(N, 1);
        n_keep = min(N, length(coeffs_new));
        coeffs_update(1:n_keep) = coeffs_new(1:n_keep);
        
        damping = 0.2; 
        coeffs = coeffs + damping * (coeffs_update - coeffs);
    end
end

function [x_new, y_new] = reparameterize_curve(x, y, N_out)
    dx = diff(x); dy = diff(y);
    dist = sqrt(dx.^2 + dy.^2);
    cum_dist = [0, cumsum(dist)];
    total_len = cum_dist(end);
    target_dist = linspace(0, total_len, N_out + 1); target_dist(end) = []; 
    [cum_dist_unique, unique_idx] = unique(cum_dist);
    x_new = interp1(cum_dist_unique, x(unique_idx), target_dist, 'pchip');
    y_new = interp1(cum_dist_unique, y(unique_idx), target_dist, 'pchip');
end
