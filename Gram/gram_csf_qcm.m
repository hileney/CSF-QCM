function gram_csf_qcm()
    clear; clc; close all;

    fprintf('Step 1: Generating physical mesh (Butterfly)...\n');
    [nodes, tri, bdy_indices] = generate_butterfly_mesh(100); 
    
    n_pts = size(nodes, 1);
    fprintf('    - Mesh vertices: %d\n', n_pts);
    fprintf('    - Triangles: %d\n', size(tri, 1));

    fprintf('Step 2: Executing CSF-QCM mapping simulation (Harmonic Map)...\n');
    uv_map = solve_harmonic_map(nodes, tri, bdy_indices);

    fprintf('Step 3: Calculating Gram matrix and condition number...\n');
    u = uv_map(:, 1);
    v = uv_map(:, 2);
    r_map = sqrt(u.^2 + v.^2);
    theta_map = atan2(v, u);
    
    n_modes = 36;
    Z = zernfun(n_modes, r_map, theta_map);
    
    G = (Z' * Z) / n_pts;
    
    cond_num = cond(G);
    
    fprintf('\n==========================================\n');
    fprintf('>>> Final Gram matrix condition number: %.4f <<<\n', cond_num);
    fprintf('==========================================\n');
    fprintf('(Ideal value is 1.0, traditional conformal mapping is ~900+)\n');

    figure('Name', 'CSF-QCM Validation', 'Position', [100, 100, 600, 600], 'Color', 'w');
    
    imagesc(abs(G));
    colorbar; axis square;
    title(['Cond = ' num2str(cond_num, '%.2f')],FontSize=18);
    xlabel('Zernike Mode Index',FontSize=14); ylabel('Zernike Mode Index',FontSize=14);
    clim([0 1]); colormap('parula');

    figure('Position', [100, 100, 600, 600], 'Color', 'w');
    scatter(u, v, 2, 'b', 'filled', 'MarkerFaceAlpha', 0.4); hold on;
    
    theta_circ = linspace(0, 2*pi, 200);
    plot(cos(theta_circ), sin(theta_circ), 'r--', 'LineWidth', 1.5);
    
    title({'Mapped Grid Distribution (Canonical Domain)', ['Quasi-Uniform Sampling (N=' num2str(n_pts) ')']});
    axis equal; axis([-1.1 1.1 -1.1 1.1]);
    box on;

    fprintf('Plotting complete.\n');
end

function [nodes, tri, bdy_indices] = generate_butterfly_mesh(density_factor)
    angles_deg = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360];
    theta_ctrl = deg2rad(angles_deg);
    r_ctrl = [1.0, 0.9, 0.6, 0.25, 0.6, 0.9, 1.0, 0.9, 0.6, 0.25, 0.6, 0.9, 1.0]; 
    
    N_bdy = 600;
    t_dense = linspace(0, 2*pi, N_bdy+1); t_dense(end) = [];
    r_dense = makima(theta_ctrl, r_ctrl, t_dense);
    
    bx = r_dense .* cos(t_dense);
    by = r_dense .* sin(t_dense);
    
    bx = bx - mean(bx); by = by - mean(by);
    scale = 1.0 / max(sqrt(bx.^2 + by.^2));
    bx = bx * scale; by = by * scale;
    
    bdy_nodes = [bx(:), by(:)];
    
    min_x = min(bx); max_x = max(bx);
    min_y = min(by); max_y = max(by);
    
    [xx, yy] = meshgrid(linspace(min_x, max_x, density_factor), ...
                        linspace(min_y, max_y, density_factor));
    
    in = inpolygon(xx, yy, bx, by);
    inner_nodes = [xx(in), yy(in)];
    
    nodes = [bdy_nodes; inner_nodes];
    num_bdy = size(bdy_nodes, 1);
    bdy_indices = 1:num_bdy; 
    
    DT = delaunayTriangulation(nodes);
    tri = DT.ConnectivityList;
    
    centers = incenter(DT);
    in_tri = inpolygon(centers(:,1), centers(:,2), bx, by);
    tri = tri(in_tri, :);
end

function L = compute_cotan_laplacian(nodes, tri)
    n = size(nodes, 1);
    
    p1 = nodes(tri(:,1), :);
    p2 = nodes(tri(:,2), :);
    p3 = nodes(tri(:,3), :);
    
    v1 = p2 - p3; 
    v2 = p3 - p1; 
    v3 = p1 - p2; 
    
    l1_sq = sum(v1.^2, 2);
    l2_sq = sum(v2.^2, 2);
    l3_sq = sum(v3.^2, 2);
    
    double_area = abs((p2(:,1)-p1(:,1)).*(p3(:,2)-p1(:,2)) - ...
                      (p3(:,1)-p1(:,1)).*(p2(:,2)-p1(:,2)));
    double_area(double_area < 1e-12) = 1e-12; 
    
    cot1 = (l2_sq + l3_sq - l1_sq) ./ (2 * double_area);
    cot2 = (l1_sq + l3_sq - l2_sq) ./ (2 * double_area);
    cot3 = (l1_sq + l2_sq - l3_sq) ./ (2 * double_area);
    
    I = [tri(:,2); tri(:,3); tri(:,3); tri(:,1); tri(:,1); tri(:,2)];
    J = [tri(:,3); tri(:,2); tri(:,1); tri(:,3); tri(:,2); tri(:,1)];
    V = [cot1;     cot1;     cot2;     cot2;     cot3;     cot3] * (-0.5);
    
    diag_I = [tri(:,1); tri(:,2); tri(:,3)];
    diag_J = [tri(:,1); tri(:,2); tri(:,3)];
    diag_V = [0.5*(cot2+cot3); 0.5*(cot1+cot3); 0.5*(cot1+cot2)];
    
    L = sparse([I; diag_I], [J; diag_J], [V; diag_V], n, n);
end

function uv_map = solve_harmonic_map(nodes, tri, bdy_indices)
    n = size(nodes, 1);
    
    L = compute_cotan_laplacian(nodes, tri);
    
    inner_indices = setdiff(1:n, bdy_indices);
    
    bdy_pts = nodes(bdy_indices, :);
    
    d = diff([bdy_pts; bdy_pts(1,:)]);
    seg_len = sqrt(sum(d.^2, 2));
    cum_len = [0; cumsum(seg_len)];
    total_len = cum_len(end);
    cum_len(end) = []; 
    
    thetas = (cum_len / total_len) * 2 * pi;
    u_bdy = cos(thetas);
    v_bdy = sin(thetas);
    
    L_ii = L(inner_indices, inner_indices);
    L_ib = L(inner_indices, bdy_indices);
    
    u_in = -L_ii \ (L_ib * u_bdy);
    v_in = -L_ii \ (L_ib * v_bdy);
    
    uv_map = zeros(n, 2);
    uv_map(bdy_indices, :) = [u_bdy, v_bdy];
    uv_map(inner_indices, :) = [u_in, v_in];
end

function Z = zernfun(n_modes, r, theta)
    noll_table = [
        0, 0; 1, 1; 1, -1; 2, 0; 2, -2; 2, 2; 3, -1; 3, 1; 3, -3; 3, 3;
        4, 0; 4, 2; 4, -2; 4, 4; 4, -4; 5, 1; 5, -1; 5, 3; 5, -3; 5, 5; 5, -5;
        6, 0; 6, -2; 6, 2; 6, -4; 6, 4; 6, -6; 6, 6;
        7, -1; 7, 1; 7, -3; 7, 3; 7, -5; 7, 5; 7, -7; 7, 7
    ];
    
    if n_modes > size(noll_table, 1)
        error('Only supports first 36 Zernike modes');
    end

    Z = zeros(length(r), n_modes);

    for j = 1:n_modes
        n = noll_table(j, 1);
        m = noll_table(j, 2);
        
        m_abs = abs(m);
        R = zeros(size(r));
        k_max = (n - m_abs) / 2;
        
        for k = 0:k_max
            num = (-1)^k * factorial(n - k);
            den = factorial(k) * factorial((n + m_abs)/2 - k) * factorial((n - m_abs)/2 - k);
            c = num / den;
            R = R + c * r.^(n - 2*k);
        end
        
        if m == 0
            norm_factor = sqrt(n + 1);
        else
            norm_factor = sqrt(2 * (n + 1));
        end
        
        if m == 0
            Z(:, j) = norm_factor * R;
        elseif m > 0
            Z(:, j) = norm_factor * R .* cos(m * theta);
        else
            Z(:, j) = norm_factor * R .* sin(-m * theta);
        end
        
        Z(r > 1.0001, j) = 0; 
    end
end
