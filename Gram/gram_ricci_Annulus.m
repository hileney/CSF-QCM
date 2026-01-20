function gram_ricci_Annulus()
    clear; clc; close all;

    fprintf('[Step 1] Generating physical mesh and performing topological cut...\n');
    [nodes_raw, tri_raw, idx_out, idx_in] = generate_type_iii_mesh(50); 
    
    cut_path_indices = find_cut_path(nodes_raw, tri_raw, idx_in, idx_out);
    
    [nodes, tri, bdy_left, bdy_right, bdy_bottom, bdy_top] = ...
        slice_mesh_robust(nodes_raw, tri_raw, cut_path_indices, idx_in, idx_out);
    
    n_pts = size(nodes, 1);
    fprintf('    - Mesh cutting complete: %d vertices\n', n_pts);

    fprintf('[Step 2] Solving rectangular parameterization (u, v)...\n');
    
    L_cot = compute_cotan_laplacian(nodes, tri);
    
    v_bc = zeros(n_pts, 1);
    v_fixed = false(n_pts, 1);
    v_fixed(bdy_bottom) = true; v_bc(bdy_bottom) = 0;
    v_fixed(bdy_top)    = true; v_bc(bdy_top)    = 1;
    v_coord = solve_laplace(L_cot, v_fixed, v_bc);
    
    u_bc = zeros(n_pts, 1);
    u_fixed = false(n_pts, 1);
    u_fixed(bdy_left)  = true; u_bc(bdy_left)  = -1;
    u_fixed(bdy_right) = true; u_bc(bdy_right) =  1;
    u_coord = solve_laplace(L_cot, u_fixed, u_bc);

    fprintf('[Step 3] Building Zernike basis and performing Gram-Schmidt orthogonalization...\n');
    
    Areas = compute_vertex_areas(nodes, tri);
    W_diag = spdiags(Areas, 0, n_pts, n_pts);
    
    theta_map = (u_coord + 1) * pi; 
    rho_map   = v_coord;            
    
    n_modes = 36; 
    [Z_raw, mode_indices] = zernfun_standard(n_modes, rho_map, theta_map);
    
    fprintf('    - Raw Zernike basis matrix size: [%d x %d]\n', size(Z_raw));
    
    G_raw = Z_raw' * (Areas .* Z_raw);
    
    d_raw = sqrt(diag(G_raw));
    G_raw_norm = G_raw ./ (d_raw * d_raw'); 
    
    cond_raw = cond(G_raw_norm);
    fprintf('    -> Gram condition number before orthogonalization: %.2e (very large, unstable)\n', cond_raw);
    
    fprintf('    - Performing weighted QR decomposition...\n');
    M_weighted = sqrt(W_diag) * Z_raw;
    [Q_weighted, R_mat] = qr(M_weighted, 0);
    
    G_final = Q_weighted' * Q_weighted;
    cond_final = cond(G_final);
    
    fprintf('\n======================================================\n');
    fprintf('>>> Gram condition number after orthogonalization: %.4f (perfectly orthogonal) <<<\n', cond_final);
    fprintf('======================================================\n');

    figure('Position', [100, 100, 600, 600], 'Color', 'w');
    
    imagesc(abs(G_raw_norm));
    colorbar; axis square;
    title(['Cond = ' num2str(cond_raw, '%.2f')],FontSize=18);
    xlabel('Zernike Mode Index',FontSize=14); ylabel('Zernike Mode Index',FontSize=14);
    
end

function [Z, idx_list] = zernfun_standard(n_modes, r, theta)
    Z = zeros(length(r), n_modes);
    
    noll_table = [0,0; 1,1; 1,-1; 2,0; 2,-2; 2,2; 3,-1; 3,1; 3,-3; 3,3; 
                  4,0; 4,2; 4,-2; 4,4; 4,-4; 5,1; 5,-1; 5,3; 5,-3; 5,5; 5,-5;
                  6,0; 6,-2; 6,2; 6,-4; 6,4; 6,-6; 6,6;
                  7,-1; 7,1; 7,-3; 7,3; 7,-5; 7,5; 7,-7; 7,7];
    
    if n_modes > size(noll_table, 1), error('Modes > table size'); end
    idx_list = noll_table(1:n_modes, :);
    
    for j = 1:n_modes
        n = idx_list(j, 1);
        m = idx_list(j, 2);
        
        R = zernike_radial(n, abs(m), r);
        
        if m == 0
            Z(:, j) = R * sqrt(n+1);
        elseif m > 0
            Z(:, j) = R .* cos(m * theta) * sqrt(2*(n+1));
        else
            Z(:, j) = R .* sin(-m * theta) * sqrt(2*(n+1));
        end
    end
end

function R = zernike_radial(n, m, r)
    R = zeros(size(r));
    for s = 0 : (n-m)/2
        c = (-1)^s * factorial(n-s) / ...
            (factorial(s) * factorial((n+m)/2 - s) * factorial((n-m)/2 - s));
        R = R + c * r.^(n - 2*s);
    end
end

function [nodes_new, tri_new, left_bdy, right_bdy, bot_bdy, top_bdy] = ...
        slice_mesh_robust(nodes, tri, path, idx_in, idx_out)
    
    n_orig = size(nodes, 1);
    path = path(:); 
    n_path = length(path);
    
    nodes_new = [nodes; nodes(path, :)];
    map_dup = (n_orig + 1 : n_orig + n_path)'; 
    
    cut_edges = [path(1:end-1), path(2:end)];
    IsCutEdge = sparse(cut_edges(:,1), cut_edges(:,2), 1, n_orig, n_orig);
    IsCutEdge = IsCutEdge + IsCutEdge';
    
    TR = triangulation(tri, nodes);
    n_tri = size(tri, 1);
    adj_tri = sparse(n_tri, n_tri);
    
    edges_all = TR.edges;
    attachments = TR.edgeAttachments(edges_all);
    
    for i = 1:length(edges_all)
        ts = attachments{i};
        if length(ts) == 2
            u = edges_all(i, 1); v = edges_all(i, 2);
            if ~IsCutEdge(u, v) 
                adj_tri(ts(1), ts(2)) = 1;
                adj_tri(ts(2), ts(1)) = 1;
            end
        end
    end
    
    [tri_on_path_r, ~] = find(ismember(tri, path));
    tri_on_path_idx = unique(tri_on_path_r);
    seed = tri_on_path_idx(1);
    
    dists = distances(graph(adj_tri), seed);
    path_tris_dists = dists(tri_on_path_idx);
    
    threshold = max(path_tris_dists) * 0.4;
    group2_tris = tri_on_path_idx(path_tris_dists > threshold);
    
    tri_new = tri;
    for t_idx = group2_tris'
        for k = 1:3
            old_node = tri_new(t_idx, k);
            [is_p, loc] = ismember(old_node, path);
            if is_p
                tri_new(t_idx, k) = map_dup(loc);
            end
        end
    end
    
    left_bdy = path;          
    right_bdy = map_dup;      
    p_start_dup = map_dup(1);
    p_end_dup = map_dup(end);
    bot_bdy = unique([idx_in(:); p_start_dup]);
    top_bdy = unique([idx_out(:); p_end_dup]);
end

function x = solve_laplace(L, is_fixed, bc_vals)
    n = size(L, 1);
    free = find(~is_fixed);
    x = bc_vals;
    rhs = -L(free, is_fixed) * bc_vals(is_fixed);
    x(free) = L(free, free) \ rhs;
end

function areas = compute_vertex_areas(nodes, tri)
    n = size(nodes, 1);
    areas = zeros(n, 1);
    x = nodes(:,1); y = nodes(:,2);
    for i = 1:size(tri, 1)
        idx = tri(i, :);
        x1=x(idx(1)); y1=y(idx(1));
        x2=x(idx(2)); y2=y(idx(2));
        x3=x(idx(3)); y3=y(idx(3));
        curr_area = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1));
        areas(idx) = areas(idx) + curr_area / 3;
    end
end

function path = find_cut_path(nodes, tri, idx_in, idx_out)
    n = size(nodes, 1);
    adj = adjacency(graph(tri(:,1),tri(:,2),[],n)) | ...
          adjacency(graph(tri(:,2),tri(:,3),[],n)) | ...
          adjacency(graph(tri(:,3),tri(:,1),[],n));
    G = graph(adj);
    s = idx_in(1);
    d = distances(G, s, idx_out);
    [~, min_idx] = min(d);
    t = idx_out(min_idx);
    path = shortestpath(G, s, t);
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

function [nodes, tri, idx_out, idx_in] = generate_type_iii_mesh(density)
    N_bdy = 200;
    t = linspace(0, 2*pi, N_bdy+1); t(end) = [];
    r_out = 1.0 + 0.05*sin(3*t)+ 0.04*cos(5*t);
    [xo, yo] = pol2cart(t, r_out);
    r_in = 0.4 + 0.02*sin(4*t);
    [xi, yi] = pol2cart(t, r_in);
    xi = xi + 0.4; yi = yi - 0.1;
    bbox = [min(xo), max(xo); min(yo), max(yo)];
    [xx, yy] = meshgrid(linspace(bbox(1,1), bbox(1,2), density), ...
                        linspace(bbox(2,1), bbox(2,2), density));
    in_o = inpolygon(xx, yy, xo, yo);
    in_i = inpolygon(xx, yy, xi, yi);
    mask = in_o & ~in_i;
    xn = xx(mask); yn = yy(mask);
    nodes = [[xo(:), yo(:)]; [xi(:), yi(:)]; [xn(:), yn(:)]];
    n_out = length(xo); n_in = length(xi);
    idx_out = 1:n_out;
    idx_in = (n_out+1):(n_out+n_in);
    DT = delaunayTriangulation(nodes);
    tri = DT.ConnectivityList;
    ct = incenter(DT);
    io = inpolygon(ct(:,1), ct(:,2), xo, yo);
    ii = inpolygon(ct(:,1), ct(:,2), xi, yi);
    tri = tri(io & ~ii, :);
end
