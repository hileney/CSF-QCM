clear; clc; close all;

aperture_type = 1;       
safety_threshold = 0.98; 
target_circularity = 1e-3; 
stagnation_tol = 1e-7;     

fprintf('Initializing Fast Solver for Type %d...\n', aperture_type);

N_bdy = 300; 

switch aperture_type
    case 1 
        fprintf('Generating Butterfly Aperture...\n');
        angles_deg = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360];
        theta_ctrl = deg2rad(angles_deg);
        r_ctrl = [1.0, 0.9, 0.6, 0.25, 0.6, 0.9, 1.0, 0.9, 0.6, 0.25, 0.6, 0.9, 1.0]; 
        
        t_dense = linspace(0, 2*pi, N_bdy*2 + 1); 
        t_dense(end) = []; 
        r_dense = makima(theta_ctrl, r_ctrl, t_dense);
        x_init = r_dense .* cos(t_dense);
        y_init = r_dense .* sin(t_dense);
        
        [x, y] = reparameterize_curve(x_init, y_init, N_bdy);
        
    case 2 
        L = 3.0; H = 1.0; R = 0.35; 
        pts_per_unit = N_bdy / (2*(L+H)); 
        n_line_L = max(2, round((L-2*R) * pts_per_unit));
        n_line_H = max(2, round((H-2*R) * pts_per_unit));
        n_arc = max(5, round(0.5*pi*R * pts_per_unit));
        
        y_R = linspace(-H/2+R, H/2-R, n_line_H); x_R = (L/2) * ones(size(y_R));
        th_tr = linspace(0, pi/2, n_arc); 
        x_TR = (L/2-R) + R*cos(th_tr); y_TR = (H/2-R) + R*sin(th_tr);
        x_T = linspace(L/2-R, -L/2+R, n_line_L); y_T = (H/2) * ones(size(x_T));
        th_tl = linspace(pi/2, pi, n_arc); 
        x_TL = (-L/2+R) + R*cos(th_tl); y_TL = (H/2-R) + R*sin(th_tl);
        y_L = linspace(H/2-R, -H/2+R, n_line_H); x_L = (-L/2) * ones(size(y_L));
        th_bl = linspace(pi, 3*pi/2, n_arc);
        x_BL = (-L/2+R) + R*cos(th_bl); y_BL = (-H/2+R) + R*sin(th_bl);
        x_B = linspace(-L/2+R, L/2-R, n_line_L); y_B = (-H/2) * ones(size(x_B));
        th_br = linspace(3*pi/2, 2*pi, n_arc);
        x_BR = (L/2-R) + R*cos(th_br); y_BR = (-H/2+R) + R*sin(th_br);
        
        x_raw = [x_R, x_TR, x_T, x_TL, x_L, x_BL, x_B, x_BR];
        y_raw = [y_R, y_TR, y_T, y_TL, y_L, y_BL, y_B, y_BR];
        [x, y] = reparameterize_curve(x_raw, y_raw, N_bdy);
end

if polyarea(x, y) < 1e-6, x = fliplr(x); y = fliplr(y); end
x = x - mean(x); y = y - mean(y);
initial_area = polyarea(x, y);
scale0 = sqrt(pi / initial_area); 
x = x * scale0; y = y * scale0;

bdy_nodes_init = [x(:), y(:)];
N_bdy = size(bdy_nodes_init, 1);

fprintf('Generating Mesh (Balanced Aspect Ratio)...\n');
density = 40; 
bbox = [min(x), max(x); min(y), max(y)];
[grid_x, grid_y] = meshgrid(linspace(bbox(1,1), bbox(1,2), density), ...
                            linspace(bbox(2,1), bbox(2,2), density));
in = inpolygon(grid_x, grid_y, x, y);
inner_nodes = [grid_x(in), grid_y(in)];

nodes_init = [bdy_nodes_init; inner_nodes];
DT = delaunayTriangulation(nodes_init);
tri = DT.ConnectivityList;
centers = incenter(DT);
in_tri = inpolygon(centers(:,1), centers(:,2), x, y);
tri = tri(in_tri, :);

num_total = size(nodes_init, 1);
bdy_indices = 1:N_bdy;
inner_indices = (N_bdy+1):num_total;

L_cot = compute_cotangent_laplacian(nodes_init, tri);
L_ii = L_cot(inner_indices, inner_indices);
L_ib = L_cot(inner_indices, bdy_indices);
decomp_L = decomposition(L_ii + 1e-9*speye(size(L_ii)), 'chol');

fprintf('System Ready: %d nodes, %d triangles.\n', num_total, size(tri,1));

max_iter = 50000; 
dt = 1e-3;       
dt_max = 0.05;   
dt_min = 1e-9;   

current_bdy_x = bdy_nodes_init(:,1);
current_bdy_y = bdy_nodes_init(:,2);
current_nodes = nodes_init;

figure('Name', 'Robust CSF-QCM Monitor', 'Color', 'w', 'Position', [50, 100, 1000, 400]);
subplot(1,2,1); h_mesh = patch('Faces', tri, 'Vertices', current_nodes, 'FaceColor', 'none', 'EdgeColor', [0.6 0.6 0.6]); 
hold on; h_bdy = plot(current_bdy_x, current_bdy_y, 'b-', 'LineWidth', 2); axis equal; 
title('Evolution'); xlim([-2 2]); ylim([-2 2]); grid on;
subplot(1,2,2); h_mu = histogram(0,0:0.05:1); title('Beltrami |\mu|'); xlim([0 1.2]);

iter = 0;
converged = false;
prev_circularity = inf;
stagnation_counter = 0;

while iter < max_iter && ~converged
    iter = iter + 1;
    
    x_c = current_bdy_x; y_c = current_bdy_y;
    x_prev = circshift(x_c, 1); x_next = circshift(x_c, -1);
    y_prev = circshift(y_c, 1); y_next = circshift(y_c, -1);
    
    dx = 0.5*(x_next - x_prev); dy = 0.5*(y_next - y_prev);
    ddx = x_next - 2*x_c + x_prev; ddy = y_next - 2*y_c + y_prev;
    ds_sq = dx.^2 + dy.^2 + 1e-12;
    curvature = (dx .* ddy - dy .* ddx) ./ (ds_sq.^(1.5));
    
    max_kappa = 50.0; 
    curvature(curvature > max_kappa) = max_kappa;
    curvature(curvature < -max_kappa) = -max_kappa;
    
    nx = -dy ./ sqrt(ds_sq); ny = dx ./ sqrt(ds_sq);
    
    step_accepted = false;
    
    while ~step_accepted
        prop_x = current_bdy_x + dt * curvature .* nx;
        prop_y = current_bdy_y + dt * curvature .* ny;
        
        [prop_x, prop_y] = tangential_smoothing(prop_x, prop_y);
        
        current_area = polyarea(prop_x, prop_y);
        scale_factor = sqrt(pi / current_area);
        prop_x = prop_x * scale_factor;
        prop_y = prop_y * scale_factor;
        
        cx_drift = mean(prop_x); cy_drift = mean(prop_y);
        prop_x = prop_x - cx_drift;
        prop_y = prop_y - cy_drift;
        
        U_in = decomp_L \ (-L_ib * prop_x);
        V_in = decomp_L \ (-L_ib * prop_y);
        
        prop_nodes = current_nodes;
        prop_nodes(bdy_indices, :) = [prop_x, prop_y];
        prop_nodes(inner_indices, :) = [U_in, V_in];
        
        mu = compute_beltrami(nodes_init, prop_nodes, tri);
        max_mu = max(abs(mu));
        

        if iter < 100
            current_threshold = 1.2; 
        else
            current_threshold = safety_threshold;
        end
        
        if max_mu < current_threshold
            step_accepted = true;
            
            if max_mu < 0.6
                dt = min(dt * 1.1, dt_max); 
            elseif max_mu < 0.85
                dt = min(dt * 1.02, dt_max);
            end
        else
            dt = dt * 0.5;
            if dt < dt_min
                warning('Step size underflow at iter %d. Force stopping.', iter);
                converged = true; break; 
            end
        end
    end
    
    if converged, break; end

    disp_vec = sqrt((prop_x - current_bdy_x).^2 + (prop_y - current_bdy_y).^2);
    max_disp = max(disp_vec);

    current_bdy_x = prop_x;
    current_bdy_y = prop_y;
    current_nodes = prop_nodes;
    
    radii = sqrt(current_bdy_x.^2 + current_bdy_y.^2);
    circularity_err = std(radii) / mean(radii);
    
    if abs(circularity_err - prev_circularity) < stagnation_tol
        stagnation_counter = stagnation_counter + 1;
    else
        stagnation_counter = 0;
    end
    prev_circularity = circularity_err;
    
    if mod(iter, 50) == 0
        set(h_mesh, 'Vertices', current_nodes);
        set(h_bdy, 'XData', [current_bdy_x; current_bdy_x(1)], 'YData', [current_bdy_y; current_bdy_y(1)]);
        histogram(subplot(1,2,2), abs(mu), 50);
        title(subplot(1,2,2), sprintf('Iter %d | Max \\mu = %.3f | dt = %.2e', iter, max_mu, dt));
        drawnow limitrate;
        fprintf('Iter %4d | CircleErr %.2e | Disp %.2e | MaxMu %.4f | dt %.2e\n', ...
            iter, circularity_err, max_disp, max_mu, dt);
    end
    
    is_circular = (circularity_err < target_circularity);
    is_conformal = (max_mu < 0.02) && (circularity_err < 2e-3); 
    is_static = (max_disp < 1e-5) && (iter > 500); 

    if is_circular
        fprintf('Converged: Target Circularity Reached (%.2e) at iter %d!\n', circularity_err, iter);
        converged = true;
    elseif is_conformal
        fprintf('Converged: Conformal Limit Reached (Max Mu < 0.02) at iter %d.\n', iter);
        converged = true;
    elseif is_static
        fprintf('Converged: Geometry Stagnated (Disp < 1e-6) at iter %d.\n', iter);
        converged = true;
    elseif stagnation_counter > 1000
        fprintf('Converged: Error Stagnation Detected at iter %d.\n', iter);
        converged = true;
    end
end

fprintf('Finalizing Mapping Result...\n');

UV_Final = current_nodes;

center_uv = mean(UV_Final(bdy_indices, :));
UV_Final = UV_Final - center_uv;

r_final = sqrt(sum(UV_Final(bdy_indices,:).^2, 2));
UV_Final = UV_Final / mean(r_final);

XY_Phys = nodes_init;
UV_Calc = UV_Final;
save('mapping_result.mat', 'XY_Phys', 'UV_Calc', 'tri');

fprintf('Done. Saved mapping_result.mat\n');

function [xn, yn] = tangential_smoothing(x, y)
    x_next = circshift(x, -1); y_next = circshift(y, -1);
    x_prev = circshift(x, 1);  y_prev = circshift(y, 1);
    lx = 0.5*(x_next + x_prev) - x;
    ly = 0.5*(y_next + y_prev) - y;
    tx = x_next - x_prev;
    ty = y_next - y_prev;
    len = sqrt(tx.^2 + ty.^2);
    tx = tx ./ len; ty = ty ./ len;
    proj = lx .* tx + ly .* ty;
    shift_amount = 0.3; 
    xn = x + shift_amount * proj .* tx;
    yn = y + shift_amount * proj .* ty;
end

function [x_new, y_new] = reparameterize_curve(x, y, N)
    x = x(:)'; y = y(:)';
    x_c = [x, x(1)]; y_c = [y, y(1)];
    dx = diff(x_c); dy = diff(y_c);
    ds = sqrt(dx.^2 + dy.^2); ds(ds<1e-12) = 1e-12;
    s = [0, cumsum(ds)];
    t = s / s(end);
    [t_u, idx] = unique(t, 'stable');
    t_new = linspace(0, 1, N+1); t_new(end) = [];
    x_new = interp1(t_u, x_c(idx), t_new, 'pchip');
    y_new = interp1(t_u, y_c(idx), t_new, 'pchip');
end

function mu = compute_beltrami(Nodes_Source, Nodes_Target, Tri)
    u = Nodes_Source(:,1); v = Nodes_Source(:,2);
    x = Nodes_Target(:,1); y = Nodes_Target(:,2);
    w = u + 1i*v; z = x + 1i*y;
    idx1=Tri(:,1); idx2=Tri(:,2); idx3=Tri(:,3);
    w1=w(idx1); w2=w(idx2); w3=w(idx3);
    z1=z(idx1); z2=z(idx2); z3=z(idx3);
    a = conj(w2-w3); b = conj(w3-w1); c = conj(w1-w2);
    denom = w1.*a + w2.*b + w3.*c;
    dz_dw = (z1.*a + z2.*b + z3.*c) ./ denom;
    dz_dw_bar = (z1.*conj(a) + z2.*conj(b) + z3.*conj(c)) ./ conj(denom);
    mu = zeros(size(Tri,1), 1);
    mask = abs(dz_dw) > 1e-12;
    mu(mask) = dz_dw_bar(mask) ./ dz_dw(mask);
end

function L = compute_cotangent_laplacian(nodes, tri)
    n = size(nodes, 1);
    p1 = nodes(tri(:,1), :); p2 = nodes(tri(:,2), :); p3 = nodes(tri(:,3), :);
    v1 = p2-p3; v2 = p3-p1; v3 = p1-p2;
    l1 = sum(v1.^2,2); l2 = sum(v2.^2,2); l3 = sum(v3.^2,2);
    area = 0.5 * abs((p2(:,1)-p1(:,1)).*(p3(:,2)-p1(:,2)) - (p3(:,1)-p1(:,1)).*(p2(:,2)-p1(:,2)));
    area(area<1e-12) = 1e-12;
    cot1 = (l2+l3-l1)./(4*area); 
    cot2 = (l1+l3-l2)./(4*area); 
    cot3 = (l1+l2-l3)./(4*area);
    I = [tri(:,2); tri(:,3); tri(:,3); tri(:,1); tri(:,1); tri(:,2); tri(:,1); tri(:,2); tri(:,3)];
    J = [tri(:,3); tri(:,2); tri(:,1); tri(:,3); tri(:,2); tri(:,1); tri(:,1); tri(:,2); tri(:,3)];
    V = [-cot1; -cot1; -cot2; -cot2; -cot3; -cot3; cot2+cot3; cot1+cot3; cot1+cot2];
    L = sparse(I, J, V, n, n);
end
