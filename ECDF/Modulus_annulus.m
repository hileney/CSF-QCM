clear; clc; close all;

N_bdy = 800;
t = linspace(0, 2*pi, N_bdy+1)'; t(end) = []; 

r_out = 1.0 + 0.05*sin(3*t) + 0.04*cos(5*t);
[x_out, y_out] = pol2cart(t, r_out);

r_in_local = 0.40 + 0.02*sin(4*t); 
[x_in_local, y_in_local] = pol2cart(t, r_in_local);

shift_x = 0.40;
shift_y = -0.10;

x_in = x_in_local + shift_x; 
y_in = y_in_local + shift_y;

dists = [];
for i = 1:length(x_in)
    d = sqrt((x_out - x_in(i)).^2 + (y_out - y_in(i)).^2);
    dists(end+1) = min(d);
end
min_gap = min(dists);

fprintf('   Geometry generated.\n');
fprintf('   Min Gap: %.4f\n', min_gap);

if min_gap < 0.001
    error('Geometry penetration or gap too small!');
end

fprintf('2. Generating adaptive mesh...\n');

x_range = linspace(-1.2, 1.2, 150);
[X, Y] = meshgrid(x_range, x_range);

mask = inpolygon(X, Y, x_out, y_out) & ~inpolygon(X, Y, x_in, y_in);
P_inner = [X(mask), Y(mask)];

P_all = [x_out, y_out; x_in, y_in; P_inner];

N_out = length(x_out);
N_in  = length(x_in);

C_out = [(1:N_out-1)', (2:N_out)'; N_out, 1];
idx_in_start = N_out + 1;
idx_in_end   = N_out + N_in;
C_in = [(idx_in_start:idx_in_end-1)', (idx_in_start+1:idx_in_end)'; idx_in_end, idx_in_start];

DT = delaunayTriangulation(P_all, [C_out; C_in]);

pts = DT.Points;
connect = DT.ConnectivityList;
centers = (pts(connect(:,1),:) + pts(connect(:,2),:) + pts(connect(:,3),:)) / 3;

is_inside_hole = inpolygon(centers(:,1), centers(:,2), x_in, y_in);
is_outside_outer = ~inpolygon(centers(:,1), centers(:,2), x_out, y_out);
valid_tri = ~(is_inside_hole | is_outside_outer);

T = connect(valid_tri, :);
Nodes = pts;
N_nodes = size(Nodes, 1);

fprintf('   Mesh ready: %d nodes, %d elements\n', N_nodes, size(T,1));

fprintf('3. Solving PDE for potential field...\n');

I = zeros(size(T,1)*9, 1);
J = zeros(size(T,1)*9, 1);
V = zeros(size(T,1)*9, 1);
idx_k = 0;

for k = 1:size(T, 1)
    tri = T(k, :);
    p = Nodes(tri, :);
    
    b = [p(2,2)-p(3,2); p(3,2)-p(1,2); p(1,2)-p(2,2)];
    c = [p(3,1)-p(2,1); p(1,1)-p(3,1); p(2,1)-p(1,1)];
    Area = 0.5 * det([1 1 1; p']);
    
    if Area < 1e-12, continue; end
    
    Ke = (b*b' + c*c') / (4*Area);
    
    [Ti, Tj] = meshgrid(tri, tri);
    I(idx_k+1:idx_k+9) = Ti(:);
    J(idx_k+1:idx_k+9) = Tj(:);
    V(idx_k+1:idx_k+9) = Ke(:);
    idx_k = idx_k + 9;
end

I = I(1:idx_k); J = J(1:idx_k); V = V(1:idx_k);
K_mat = sparse(I, J, V, N_nodes, N_nodes);

bdy_in_idx = (N_out+1):(N_out+N_in);
bdy_out_idx = 1:N_out;

u = zeros(N_nodes, 1);
is_fixed = false(N_nodes, 1);
u(bdy_in_idx) = 1;  is_fixed(bdy_in_idx) = true;
u(bdy_out_idx) = 0; is_fixed(bdy_out_idx) = true;

free_nodes = ~is_fixed;
rhs = -K_mat(free_nodes, is_fixed) * u(is_fixed);
u(free_nodes) = K_mat(free_nodes, free_nodes) \ rhs;

if any(isnan(u))
    error('FEM solution failed with NaN.');
end

energy = u' * K_mat * u;
R_val = exp(-2*pi / energy);

fprintf('\n================ RESULTS ================\n');
fprintf('Dirichlet Energy (E) : %.6f\n', energy);
fprintf('-----------------------------------------\n');
fprintf('Optimal Conformal Modulus R : %.8f\n', R_val);
fprintf('-----------------------------------------\n');

figure('Color', 'w', 'Position', [100, 100, 1000, 500]);

subplot(1, 2, 1);
trisurf(T, Nodes(:,1), Nodes(:,2), u, 'EdgeColor', 'none', 'FaceColor', 'interp');
view(2); axis equal; 
title(sprintf('Harmonic Potential u\n(Energy = %.4f)', energy));
colorbar; colormap jet;
xlabel('X'); ylabel('Y');

subplot(1, 2, 2);
plot(x_out, y_out, 'k-', 'LineWidth', 2); hold on;
plot(x_in, y_in, 'b-', 'LineWidth', 2);
[XX, YY] = meshgrid(linspace(-1.2, 1.2, 100));
ZZ = griddata(Nodes(:,1), Nodes(:,2), u, XX, YY);
contour(XX, YY, ZZ, 20, 'LineWidth', 0.5);
axis equal; 
title(sprintf('Geometry Check & Equipotential Lines\nMin Gap = %.4f', min_gap));
legend('Outer Bdy', 'Inner Bdy', 'Equipotentials');
xlim([-1.2 1.2]); ylim([-1.2 1.2]);

fprintf('Record R_optimal value (%.8f) for CSF algorithm.\n', R_val);
