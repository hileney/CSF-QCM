function gram_csf_Annulus()
    clear; clc; close all;

    fprintf('[Step 1] Initializing geometry (Type III: Eccentric Annulus)...\n');
    
    N_bdy = 500; 
    t = linspace(0, 2*pi, N_bdy+1)'; t(end) = []; 
    
    r_out_func = @(t) 1.0 + 0.05*sin(3*t) + 0.04*cos(5*t);
    [xo, yo] = pol2cart(t, r_out_func(t));
    
    r_in_func = @(t) 0.40 + 0.02*sin(4*t);
    [xi, yi] = pol2cart(t, r_in_func(t));
    xi = xi + 0.4; yi = yi - 0.1; 

    fprintf('[Step 2] Calculating conformal modulus...\n');
    [nodes_tmp, tri_tmp, idx_out, idx_in] = generate_unstructured_mesh(xo, yo, xi, yi);
    L = compute_cotan_laplacian(nodes_tmp, tri_tmp);
    
    n_nodes = size(nodes_tmp, 1);
    bdy_mask = false(n_nodes, 1);
    bdy_mask([idx_out; idx_in]) = true;
    u_bc = zeros(n_nodes, 1); u_bc(idx_out) = 1; u_bc(idx_in) = 0;
    
    free_nodes = find(~bdy_mask);
    U = u_bc;
    U(free_nodes) = -L(free_nodes, free_nodes) \ (L(free_nodes, bdy_mask) * u_bc(bdy_mask));
    
    E_dirichlet = U' * L * U;
    R_in = exp(-2 * pi / E_dirichlet);
    fprintf('    -> Exact conformal inner radius R_in: %.6f\n', R_in);

    fprintf('[Step 3] Generating structured physical mesh (Alignment Corrected)...\n');
    
    n_radial = 60;   
    n_angular = 180; 
    
    [xo_uni, yo_uni] = reparam_curve(xo, yo, n_angular);
    [xi_uni, yi_uni] = reparam_curve(xi, yi, n_angular);
    [xi_uni, yi_uni] = align_boundary_phase(xi_uni, yi_uni, xo_uni, yo_uni);
    
    rho = linspace(0, 1, n_radial)';
    X_mesh = zeros(n_radial, n_angular+1);
    Y_mesh = zeros(n_radial, n_angular+1);
    
    for j = 1:n_angular+1
        X_mesh(:, j) = (1-rho) * xi_uni(j) + rho * xo_uni(j);
        Y_mesh(:, j) = (1-rho) * yi_uni(j) + rho * yo_uni(j);
    end
    
    for k = 1:50
        X_new = X_mesh; Y_new = Y_mesh;
        for r = 2:n_radial-1
            for c = 1:n_angular+1
                c_pre = mod(c-2, n_angular) + 1;
                c_nxt = mod(c, n_angular) + 1;
                X_new(r,c) = 0.25*(X_mesh(r-1,c)+X_mesh(r+1,c)+X_mesh(r,c_pre)+X_mesh(r,c_nxt));
                Y_new(r,c) = 0.25*(Y_mesh(r-1,c)+Y_mesh(r+1,c)+Y_mesh(r,c_pre)+Y_mesh(r,c_nxt));
            end
        end
        X_mesh = X_new; Y_mesh = Y_new;
    end
    
    x_vec = X_mesh(1:end-1, 1:end-1); x_vec = x_vec(:);
    y_vec = Y_mesh(1:end-1, 1:end-1); y_vec = y_vec(:);

    fprintf('[Step 4] Calculating Gram matrix (Using Annular Zernikes)...\n');
    
    r_canon = linspace(R_in, 1, n_radial); 
    theta_canon = linspace(0, 2*pi, n_angular+1);
    r_c = 0.5 * (r_canon(1:end-1) + r_canon(2:end));
    th_c = 0.5 * (theta_canon(1:end-1) + theta_canon(2:end));
    [RR, TT] = meshgrid(r_c, th_c);
    rr_vec = RR(:); tt_vec = TT(:);
    
    Areas = zeros(n_radial-1, n_angular);
    for r = 1:n_radial-1
        for c = 1:n_angular
            p1 = [X_mesh(r,c), Y_mesh(r,c)];
            p2 = [X_mesh(r+1,c), Y_mesh(r+1,c)];
            p3 = [X_mesh(r+1,c+1), Y_mesh(r+1,c+1)];
            p4 = [X_mesh(r,c+1), Y_mesh(r,c+1)];
            nodes_poly = [p1; p2; p3; p4];
            Areas(r,c) = polyarea(nodes_poly(:,1), nodes_poly(:,2));
        end
    end
    weights_vec = Areas(:);
    Total_Area = sum(weights_vec);
    
    n_modes = 36;
    Z = zernfun_annular_ortho(n_modes, rr_vec, tt_vec, R_in);
    
    G = (Z' * (weights_vec .* Z));
    
    d = sqrt(diag(G));
    G_norm = G ./ (d * d');
    
    cond_num = cond(G_norm);
    
    fprintf('\n==========================================\n');
    fprintf('>>> Gram Condition: %.4f <<<\n', cond_num);
    fprintf('    (Should be < 10 for Conformal Maps)\n');
    fprintf('==========================================\n');

    figure('Position', [100, 100, 600, 600], 'Color', 'w');
    
    step_r = max(1, round(n_radial/30));
    step_a = max(1, round(n_angular/30));
    for i = 1:step_r:n_radial, plot(X_mesh(i,:), Y_mesh(i,:), 'b-', 'Color', [0 0.4 0.8 0.3]); hold on; end
    for j = 1:step_a:n_angular+1, plot(X_mesh(:,j), Y_mesh(:,j), 'r-', 'Color', [0.8 0.2 0.2 0.3]); end
    plot(X_mesh(1,:), Y_mesh(1,:), 'k-', 'LineWidth', 1.5);
    plot(X_mesh(end,:), Y_mesh(end,:), 'k-', 'LineWidth', 1.5);
    axis equal; axis off;
    title({'Structured Physical Mesh', ['R_{in} = ' num2str(R_in, '%.4f')]});
    
    figure('Position', [100, 300, 600, 600], 'Color', 'w');
    imagesc(abs(G_norm));
    colorbar; axis square;
    title(['Cond = ' num2str(cond_num, '%.2f')],FontSize=18);
    xlabel('Zernike Mode Index',FontSize=14); ylabel('Zernike Mode Index',FontSize=14);
end

function Z = zernfun_annular_ortho(n_modes, r, theta, R_in)
    noll_table = [0,0; 1,1; 1,-1; 2,0; 2,-2; 2,2; 3,-1; 3,1; 3,-3; 3,3; 
                  4,0; 4,2; 4,-2; 4,4; 4,-4; 5,1; 5,-1; 5,3; 5,-3; 5,5; 5,-5; 
                  6,0; 6,-2; 6,2; 6,-4; 6,4; 6,-6; 6,6; 
                  7,-1; 7,1; 7,-3; 7,3; 7,-5; 7,5; 7,-7; 7,7];
    if n_modes > size(noll_table, 1)
        error('n_modes exceeds current table size');
    end
    
    noll_table = noll_table(1:n_modes, :);
    
    n_ref = 5000;
    r_ref = linspace(R_in, 1, n_ref)';
    dr = r_ref(2) - r_ref(1);
    w_ref = r_ref * dr; 
    
    unique_m_abs = unique(abs(noll_table(:,2)));
    
    Z = zeros(length(r), n_modes);
    
    for m_abs = unique_m_abs'
        mode_indices = find(abs(noll_table(:,2)) == m_abs);
        radial_orders = noll_table(mode_indices, 1);
        
        unique_n = unique(radial_orders);
        num_radial_funcs = length(unique_n);
        
        Basis_std_ref = zeros(n_ref, num_radial_funcs);
        for k = 1:num_radial_funcs
            n = unique_n(k);
            Basis_std_ref(:, k) = get_std_radial(n, m_abs, r_ref);
        end
        
        Basis_std_in = zeros(length(r), num_radial_funcs);
        for k = 1:num_radial_funcs
            n = unique_n(k);
            Basis_std_in(:, k) = get_std_radial(n, m_abs, r);
        end
        
        Basis_ortho_in = zeros(size(Basis_std_in));
        Basis_ortho_ref = zeros(size(Basis_std_ref)); 
        
        for k = 1:num_radial_funcs
            vec_ref = Basis_std_ref(:, k);
            vec_in  = Basis_std_in(:, k);
            
            for j = 1:k-1
                prev_ref = Basis_ortho_ref(:, j);
                prev_in  = Basis_ortho_in(:, j);
                
                overlap = sum(vec_ref .* prev_ref .* w_ref);
                
                vec_ref = vec_ref - overlap * prev_ref;
                vec_in  = vec_in  - overlap * prev_in;
            end
            
            norm_val = sqrt(sum(vec_ref.^2 .* w_ref));
            Basis_ortho_ref(:, k) = vec_ref / norm_val;
            Basis_ortho_in(:, k)  = vec_in  / norm_val;
        end
        
        for k = 1:length(mode_indices)
            idx = mode_indices(k);
            n = noll_table(idx, 1);
            m = noll_table(idx, 2);
            
            n_idx = find(unique_n == n);
            R_vals = Basis_ortho_in(:, n_idx);
            
            if m == 0
                Z(:, idx) = R_vals;
            elseif m > 0
                Z(:, idx) = R_vals .* cos(m * theta);
            else
                Z(:, idx) = R_vals .* sin(-m * theta);
            end
            
            if m == 0
                norm_ang = sqrt(1 / (2*pi));
            else
                norm_ang = sqrt(1 / pi);
            end
            Z(:, idx) = Z(:, idx) / norm_ang; 
        end
    end
end

function R = get_std_radial(n, m_abs, r)
    R = zeros(size(r));
    k_max = (n - m_abs) / 2;
    for k = 0:k_max
        c = (-1)^k * factorial(n - k) / ...
            (factorial(k) * factorial((n + m_abs)/2 - k) * factorial((n - m_abs)/2 - k));
        R = R + c * r.^(n - 2*k);
    end
end

function [xi_aligned, yi_aligned] = align_boundary_phase(xi, yi, xo, yo)
    xi = xi(:); yi = yi(:); xo = xo(:); yo = yo(:);
    target_x = xo(1); target_y = yo(1);
    dists = (xi - target_x).^2 + (yi - target_y).^2;
    [~, idx] = min(dists);
    xi_temp = xi(1:end-1); yi_temp = yi(1:end-1);
    shift_amount = -(idx - 1);
    xi_shifted = circshift(xi_temp, shift_amount);
    yi_shifted = circshift(yi_temp, shift_amount);
    xi_aligned = [xi_shifted; xi_shifted(1)];
    yi_aligned = [yi_shifted; yi_shifted(1)];
end

function [nodes, tri, idx_out, idx_in] = generate_unstructured_mesh(xo, yo, xi, yi)
    bbox = [min(xo), max(xo); min(yo), max(yo)];
    density = 40;
    [xx, yy] = meshgrid(linspace(bbox(1,1), bbox(1,2), density), ...
                        linspace(bbox(2,1), bbox(2,2), density));
    in_o = inpolygon(xx, yy, xo, yo);
    in_i = inpolygon(xx, yy, xi, yi);
    mask = in_o & ~in_i;
    xn = xx(mask); yn = yy(mask);
    nodes = [[xo(:), yo(:)]; [xi(:), yi(:)]; [xn(:), yn(:)]];
    n_out = length(xo); idx_out = 1:n_out;
    idx_in = (n_out+1):(n_out+length(xi));
    DT = delaunayTriangulation(nodes);
    tri = DT.ConnectivityList;
    ct = incenter(DT);
    io = inpolygon(ct(:,1), ct(:,2), xo, yo);
    ii = inpolygon(ct(:,1), ct(:,2), xi, yi);
    tri = tri(io & ~ii, :);
end

function L = compute_cotan_laplacian(nodes, tri)
    n = size(nodes, 1);
    p1 = nodes(tri(:,1), :); p2 = nodes(tri(:,2), :); p3 = nodes(tri(:,3), :);
    v1 = p2-p3; v2 = p3-p1; v3 = p1-p2;
    area = 0.5 * abs((p2(:,1)-p1(:,1)).*(p3(:,2)-p1(:,2)) - (p3(:,1)-p1(:,1)).*(p2(:,2)-p1(:,2)));
    cot1 = (sum(v2.^2,2)+sum(v3.^2,2)-sum(v1.^2,2))./(4*area);
    cot2 = (sum(v1.^2,2)+sum(v3.^2,2)-sum(v2.^2,2))./(4*area);
    cot3 = (sum(v1.^2,2)+sum(v2.^2,2)-sum(v3.^2,2))./(4*area);
    I = [tri(:,2); tri(:,3); tri(:,3); tri(:,1); tri(:,1); tri(:,2); tri(:,1); tri(:,2); tri(:,3)];
    J = [tri(:,3); tri(:,2); tri(:,1); tri(:,3); tri(:,2); tri(:,1); tri(:,1); tri(:,2); tri(:,3)];
    V = [-cot1; -cot1; -cot2; -cot2; -cot3; -cot3; cot2+cot3; cot1+cot3; cot1+cot2] * 0.5;
    L = sparse(I, J, V, n, n);
end

function [xn, yn] = reparam_curve(x, y, N)
    x = x(:); y = y(:);
    if norm([x(1)-x(end), y(1)-y(end)]) > 1e-9, x(end+1)=x(1); y(end+1)=y(1); end
    dx = diff(x); dy = diff(y); dist = sqrt(dx.^2 + dy.^2); s = [0; cumsum(dist)];
    s_targets = linspace(0, s(end), N+1)'; 
    xn = interp1(s, x, s_targets, 'spline'); xn = xn(:);
    yn = interp1(s, y, s_targets, 'spline'); yn = yn(:);
end
