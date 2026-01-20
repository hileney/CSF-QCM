clear; clc; close all;

if ~exist('mapping_result.mat', 'file')
    error('Run CSF_QCM_Solver.m first!');
end
load('mapping_result.mat');

F_x = scatteredInterpolant(UV_Calc(:,1), UV_Calc(:,2), XY_Phys(:,1), 'natural', 'linear');
F_y = scatteredInterpolant(UV_Calc(:,1), UV_Calc(:,2), XY_Phys(:,2), 'natural', 'linear');

figure('Position', [100, 100, 800, 600], 'Color', 'w');
ax = gca; hold on;

TR = triangulation(tri, XY_Phys);
bdy_edges = freeBoundary(TR);

num_bdy = size(bdy_edges, 1);
ordered_idx = zeros(num_bdy, 1);
current_node = bdy_edges(1, 1);
for i = 1:num_bdy
    ordered_idx(i) = current_node;
    row_idx = find(bdy_edges(:,1) == current_node);
    current_node = bdy_edges(row_idx, 2);
end
ordered_idx(end+1) = ordered_idx(1); 

bdy_x_raw = XY_Phys(ordered_idx, 1);
bdy_y_raw = XY_Phys(ordered_idx, 2);

dx = diff(bdy_x_raw); dy = diff(bdy_y_raw);
ds = sqrt(dx.^2 + dy.^2);
t_raw = [0; cumsum(ds)];
t_raw = t_raw / t_raw(end);

t_fine = linspace(0, 1, 500); 
bdy_x_smooth = interp1(t_raw, bdy_x_raw, t_fine, 'spline');
bdy_y_smooth = interp1(t_raw, bdy_y_raw, t_fine, 'spline');

plot(bdy_x_smooth, bdy_y_smooth, 'k-', 'LineWidth', 2.5);

theta_vals = linspace(0, 2*pi, 48); 
r_line = linspace(0, 1, 100);
for th = theta_vals
    u_l = r_line * cos(th);
    v_l = r_line * sin(th);
    plot(F_x(u_l, v_l), F_y(u_l, v_l), 'r-', 'LineWidth', 0.8, 'Color', [0.8 0.2 0.2 0.6]);
end

r_vals = linspace(0.1, 0.98, 15);
th_fine = linspace(0, 2*pi, 400);
for r = r_vals
    u_c = r * cos(th_fine);
    v_c = r * sin(th_fine);
    plot(F_x(u_c, v_c), F_y(u_c, v_c), 'b-', 'LineWidth', 1.0, 'Color', [0 0.4 0.8 0.8]);
end

axis equal; axis off;
title('CSF-QCM Result (Smooth Boundary)', 'FontSize', 14, 'Interpreter', 'latex');

fprintf('Plot completed with smoothed boundary.\n');

fprintf('Computing CSF-QCM grid area statistics...\n');

N_r_stats = 100;    
N_th_stats = 360;   

r_vec = linspace(0.01, 0.99, N_r_stats);
th_vec = linspace(0, 2*pi, N_th_stats+1);

X_grid = zeros(length(r_vec), length(th_vec));
Y_grid = zeros(length(r_vec), length(th_vec));

for i = 1:length(r_vec)
    u_ring = r_vec(i) * cos(th_vec);
    v_ring = r_vec(i) * sin(th_vec);
    X_grid(i, :) = F_x(u_ring, v_ring);
    Y_grid(i, :) = F_y(u_ring, v_ring);
end

qcm_areas = [];
for i = 1:N_r_stats-1
    for j = 1:N_th_stats
        x_poly = [X_grid(i,j), X_grid(i,j+1), X_grid(i+1,j+1), X_grid(i+1,j)];
        y_poly = [Y_grid(i,j), Y_grid(i,j+1), Y_grid(i+1,j+1), Y_grid(i+1,j)];
        area_val = polyarea(x_poly, y_poly);
        qcm_areas = [qcm_areas; area_val];
    end
end

mean_area_qcm = mean(qcm_areas);
ratio_qcm = qcm_areas / mean_area_qcm;

x_raw = sort(ratio_qcm);
n_samples = length(x_raw);
y_ecdf = (1:n_samples)' / n_samples;

window_size = round(n_samples * 0.02); 
if window_size < 5, window_size = 5; end

try
    x_smooth = smoothdata(x_raw, 'gaussian', window_size);
catch
    x_smooth = filter(ones(1,window_size)/window_size, 1, x_raw);
end

figure('Position', [150, 150, 700, 500], 'Color', 'w');

h_line = semilogx(x_smooth, y_ecdf, 'r-', 'LineWidth', 3); 
hold on;
xline(1.0, 'k--', 'LineWidth', 1.5, 'Label', 'Ideal Uniformity');

grid on;
set(gca, 'XMinorGrid', 'on', 'YMinorGrid', 'on');
set(gca, 'FontSize', 12, 'LineWidth', 1.2);
set(gca, 'TickDir', 'out');

xlabel('Normalized Area Ratio ($A_i / \bar{A}$) [Log Scale]', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Cumulative Probability (ECDF)', 'FontSize', 14);
title({'CSF-QCM Grid Uniformity Analysis', 'Smoothed ECDF Curve'}, 'FontSize', 14);

msg = {
    ['\bf Statistics (CSF-QCM) \rm'];
    sprintf('Max Ratio: %.2f', max(ratio_qcm));
    sprintf('Min Ratio: %.4f', min(ratio_qcm));
    sprintf('Median:   %.2f', median(ratio_qcm))
};

text(0.05, 0.75, msg, 'Units', 'normalized', ...
    'BackgroundColor', 'w', 'EdgeColor', 'r', ...
    'Margin', 6, 'FontSize', 10, 'Interpreter', 'tex');

legend(h_line, 'CSF-QCM (Proposed)', 'Location', 'southeast');

xlim([0.01, 50]); 

fprintf('CSF-QCM ECDF plotted.\n');
fprintf('Max Ratio: %.2f\n', max(ratio_qcm));

fprintf('Saving CSF-QCM statistics...\n');
save_filename = 'qcm_area_stats.mat';
save(save_filename, 'ratio_qcm');
fprintf('Data saved to: %s\n', save_filename);
