clear; clc; close all;

fprintf('[Step 1] Generating Type III Mesh...\n');

N_bdy = 400;
t = linspace(0, 2*pi, N_bdy+1); t(end) = []; 

r_out = 1.0 + 0.05*sin(3*t) + 0.04*cos(5*t);
[x_out, y_out] = pol2cart(t, r_out);

r_in = 0.40 + 0.02*sin(4*t); 
[x_in_local, y_in_local] = pol2cart(t, r_in);

x_in = x_in_local + 0.4; 
y_in = y_in_local - 0.1;

density = 45; 
bbox = [min(x_out), max(x_out); min(y_out), max(y_out)];
[xx, yy] = meshgrid(linspace(bbox(1,1), bbox(1,2), density), ...
                    linspace(bbox(2,1), bbox(2,2), density));

in_outer = inpolygon(xx, yy, x_out, y_out);
in_inner = inpolygon(xx, yy, x_in, y_in);
mask = in_outer & (~in_inner); 
nodes_internal = [xx(mask), yy(mask)];

nodes = [[x_out(:), y_out(:)]; [x_in(:), y_in(:)]; nodes_internal];
n_out = length(x_out);
n_in = length(x_in);
n_total = size(nodes, 1);

idx_out = 1:n_out;
idx_in = (n_out+1):(n_out+n_in);
idx_boundary = [idx_out, idx_in];

DT = delaunayTriangulation(nodes);
tri = DT.ConnectivityList;
centers = incenter(DT);
c_in_outer = inpolygon(centers(:,1), centers(:,2), x_out, y_out);
c_in_inner = inpolygon(centers(:,1), centers(:,2), x_in, y_in);
tri = tri(c_in_outer & (~c_in_inner), :);

fprintf('   Mesh Generated: %d Vertices, %d Triangles.\n', n_total, size(tri,1));

K_target = zeros(n_total, 1);
K_target(idx_out) = 2 * pi / n_out;
K_target(idx_in) = -2 * pi / n_in;

fprintf('[Step 2] Running Ricci Flow Optimization...\n');

u = zeros(n_total, 1); 
L0 = compute_edge_lengths(nodes, tri); 

max_iter = 50;
for iter = 1:max_iter
    [K_curr, L_cot, L_metric] = compute_geom(tri, L0, u, n_total, idx_boundary);
    
    delta_K = K_target - K_curr;
    err = norm(delta_K, inf);
    
    if mod(iter, 10) == 0 || iter == 1
        fprintf('   Iter %2d: Curvature Error = %.2e\n', iter, err);
    end
    
    if err < 1e-6, break; end
    
    H = L_cot + 1e-9 * speye(n_total); 
    du = H \ delta_K;
    u = u + 1.0 * du; 
end
fprintf('   Ricci Flow Converged.\n');

fprintf('[Step 3] Visualizing Inverse Grid (Blue Rings + Red Spokes)...\n');

[~, L_final, ~] = compute_geom(tri, L_metric, zeros(n_total,1), n_total, idx_boundary);

bdy_U = zeros(n_total, 1);
is_fixed = false(n_total, 1);
is_fixed(idx_out) = true; bdy_U(idx_out) = 1; 
is_fixed(idx_in)  = true; bdy_U(idx_in)  = 0; 

free_ids = find(~is_fixed);
U_field = bdy_U;
U_field(free_ids) = -L_final(free_ids, free_ids) \ (L_final(free_ids, is_fixed) * bdy_U(is_fixed));

grid_res = 800; 
x_range = linspace(min(nodes(:,1)), max(nodes(:,1)), grid_res);
y_range = linspace(min(nodes(:,2)), max(nodes(:,2)), grid_res);
[XG, YG] = meshgrid(x_range, y_range);

F = scatteredInterpolant(nodes(:,1), nodes(:,2), U_field, 'natural', 'none');
UG = F(XG, YG);

figure('Name', 'Ricci Flow Grid Crowding', 'Color', 'w', 'Position', [200, 200, 800, 600]);
axis equal; axis off; hold on;

xlim([min(x_out)-0.1, max(x_out)+0.1]);
ylim([min(y_out)-0.1, max(y_out)+0.1]);

x_out=[x_out,x_out(1)];y_out=[y_out,y_out(1)];
x_in=[x_in,x_in(1)];y_in=[y_in,y_in(1)];
fill(x_out, y_out, [0.96 0.98 1.0], 'EdgeColor', 'none'); 
fill(x_in, y_in, [1 1 1], 'EdgeColor', 'k', 'LineWidth', 1.5);
plot(x_out, y_out, 'k', 'LineWidth', 1.5);

num_rings = 15;
levels = linspace(0, 1, num_rings+2); 
levels = levels(2:end-1); 
contour(XG, YG, UG, levels, 'LineColor', 'b', 'LineWidth', 1.0);

[DX, DY] = gradient(UG); 

num_spokes = 64;
seed_indices = round(linspace(1, n_in, num_spokes+1)); 
seed_indices(end) = []; 
sx = x_in(seed_indices);
sy = y_in(seed_indices);

h_lines = streamline(XG, YG, DX, DY, sx, sy);
set(h_lines, 'Color', 'r', 'LineWidth', 1.0);

title('Baseline (Ricci Flow): Grid Crowding Visualization', 'FontSize', 14);

fprintf('[Step 4] Calculating ACCURATE Grid Statistics based on Gradient Field...\n');

num_tri = size(tri, 1);
grad_sq_vals = zeros(num_tri, 1);

for k = 1:num_tri
    idx = tri(k, :);
    nodes_tri = nodes(idx, :); 
    x1=nodes_tri(1,1); y1=nodes_tri(1,2);
    x2=nodes_tri(2,1); y2=nodes_tri(2,2);
    x3=nodes_tri(3,1); y3=nodes_tri(3,2);
    
    u_vals = U_field(idx);
    
    M = [1, x1, y1; 1, x2, y2; 1, x3, y3];
    coeffs = M \ u_vals; 
    a = coeffs(2); 
    b = coeffs(3);
    
    grad_sq_vals(k) = a^2 + b^2;
end

grid_cell_areas = 1 ./ (grad_sq_vals + 1e-12);
ratio_ricci_real = grid_cell_areas / mean(grid_cell_areas);

x_ricci = sort(ratio_ricci_real);
y_ricci = (1:length(x_ricci))' / length(x_ricci);

window_size = round(length(x_ricci) * 0.02);
if window_size < 3, window_size = 3; end
x_ricci_smooth = smoothdata(x_ricci, 'gaussian', window_size);

figure('Position', [150, 150, 700, 500], 'Color', 'w');

h = semilogx(x_ricci_smooth, y_ricci, 'r-', 'LineWidth', 3); hold on; 
xline(1.0, 'k--', 'LineWidth', 1.5); 

grid on; set(gca, 'XScale', 'log'); 
set(gca, 'FontSize', 14, 'LineWidth', 1.5, 'TickDir', 'out');
xlabel('Normalized Grid Area Ratio ($A_{grid} / \bar{A}$) [Log Scale]', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('Cumulative Probability (ECDF)', 'FontSize', 16);
title('Corrected Statistics: Ricci Flow Grid Distortion', 'FontSize', 16);

max_val = max(ratio_ricci_real);
min_val = min(ratio_ricci_real);
stats_msg = {
    '\bf Ricci Flow Stats (Gradient Based) \rm';
    sprintf('Max Ratio: %.2f', max_val);  
    sprintf('Min Ratio: %.4f', min_val);
};
text(0.05, 0.7, stats_msg, 'Units', 'normalized', ...
    'BackgroundColor', 'w', 'EdgeColor', 'r', 'FontSize', 14);

xlim([0.01, max(10, max_val*1.2)]); 

ratio_ricci = ratio_ricci_real; 
save('ricci_area_stats.mat', 'ratio_ricci', 'nodes', 'tri'); 

function [K, L, L_new] = compute_geom(tri, L0, u, n, bdy_idx)
    i1=tri(:,1); i2=tri(:,2); i3=tri(:,3);
    l1 = L0.l1 .* exp((u(i2)+u(i3))/2);
    l2 = L0.l2 .* exp((u(i1)+u(i3))/2);
    l3 = L0.l3 .* exp((u(i1)+u(i2))/2);
    L_new.l1=l1; L_new.l2=l2; L_new.l3=l3;
    cos1 = (l2.^2+l3.^2-l1.^2)./(2.*l2.*l3);
    cos2 = (l1.^2+l3.^2-l2.^2)./(2.*l1.*l3);
    cos3 = (l1.^2+l2.^2-l3.^2)./(2.*l1.*l2);
    cos1=max(-1,min(1,cos1)); cos2=max(-1,min(1,cos2)); cos3=max(-1,min(1,cos3));
    ang1=acos(cos1); ang2=acos(cos2); ang3=acos(cos3);
    cot1=1./tan(ang1); cot2=1./tan(ang2); cot3=1./tan(ang3);
    I = [i2; i3; i3; i1; i1; i2];
    J = [i3; i2; i1; i3; i2; i1];
    V = 0.5 * [cot1; cot1; cot2; cot2; cot3; cot3];
    L = sparse(I, J, -V, n, n);
    L = L + sparse(1:n, 1:n, -sum(L,2), n, n);
    angle_sum = sparse(tri(:), 1, [ang1; ang2; ang3], n, 1);
    K = 2*pi - angle_sum;
    K(bdy_idx) = pi - angle_sum(bdy_idx); 
end

function L0 = compute_edge_lengths(XY, tri)
    p1=XY(tri(:,1),:); p2=XY(tri(:,2),:); p3=XY(tri(:,3),:);
    L0.l1 = sqrt(sum((p2-p3).^2,2));
    L0.l2 = sqrt(sum((p1-p3).^2,2));
    L0.l3 = sqrt(sum((p1-p2).^2,2));
end
