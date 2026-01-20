clear; close all; clc;

fprintf('1. Initializing geometry and ground truth wavefront...\n');

N_eval = 200; 
x_vec = linspace(-1.6, 1.6, N_eval);
[X_eval, Y_eval] = meshgrid(x_vec, x_vec);
[Theta_eval, R_eval] = cart2pol(X_eval, Y_eval);

r_out_b = 1.0 + 0.05*sin(3*Theta_eval) + 0.04*cos(5*Theta_eval);
mask_out = R_eval <= r_out_b;
[Theta_in_eval, R_in_local] = cart2pol(X_eval - 0.4, Y_eval + 0.1); 
r_in_b = 0.40 + 0.02*sin(4*Theta_in_eval);
mask_in = R_in_local <= r_in_b;
mask_eval = mask_out & ~mask_in;

z_modes = 36;
rng(2025); 
c_true = (rand(z_modes, 1) - 0.5) ./ (1:z_modes)';
c_true(5:10) = c_true(5:10) * 2;

W_raw = zeros(size(X_eval));
for j = 1:z_modes
    W_raw = W_raw + c_true(j) * zernike_std(j, R_eval, Theta_eval);
end

vals = W_raw(mask_eval); vals = vals - mean(vals);
scale_factor = 0.05 / std(vals);
W_true = nan(size(W_raw)); 
W_true(mask_eval) = vals * scale_factor;

valid_idx = find(mask_eval);
x_samp = X_eval(valid_idx);
y_samp = Y_eval(valid_idx);
W_clean_samp = W_true(valid_idx);

fprintf('   Sample points: %d\n', length(x_samp));

fprintf('2. Precomputing Ricci Flow mapping (Method A)...\n');

[nodes, tri, bdy_out_idx, bdy_in_idx] = build_mesh_annulus(); 
n_nodes = size(nodes, 1);

K_target = zeros(n_nodes, 1); 
L = compute_edge_lengths(nodes, tri);
u = zeros(n_nodes, 1);
is_bdy = false(n_nodes, 1); is_bdy(bdy_out_idx)=true; is_bdy(bdy_in_idx)=true;

for iter = 1:60
    [K_curr, L_cot, ~] = compute_curv_metric(tri, L, u, n_nodes, is_bdy);
    delta_K = K_target - K_curr;
    if norm(delta_K, inf) < 1e-2, break; end 
    fixed_id = 1; H = L_cot; H(fixed_id,:)=0; H(:,fixed_id)=0; H(fixed_id,fixed_id)=1; delta_K(fixed_id)=0;
    du = H \ delta_K;
    if any(isnan(du)), break; end
    u = u + 0.5 * du; 
end
[~, ~, L_flat] = compute_curv_metric(tri, L, u, n_nodes, is_bdy);

pts_out = nodes(bdy_out_idx, :); pts_in = nodes(bdy_in_idx, :);
d2 = (pts_in(:,1)-pts_out(1,1)).^2 + (pts_in(:,2)-pts_out(1,2)).^2;
[~, min_idx] = min(d2);
start_node = bdy_out_idx(1); end_node = bdy_in_idx(min_idx);

E_all = [tri(:,1), tri(:,2); tri(:,2), tri(:,3); tri(:,3), tri(:,1)];
W_safe = [L_flat.l3; L_flat.l1; L_flat.l2];
G = graph(E_all(:,1), E_all(:,2), W_safe);
cut_path = shortestpath(G, start_node, end_node);
cut_edges = sort([cut_path(1:end-1)', cut_path(2:end)'], 2);

uv_tri = flatten_mesh(tri, L_flat, cut_edges);

uv_points = reshape(uv_tri, [], 2);
coeff_pca = pca(uv_points); 
uv_rot = uv_points * coeff_pca;
u_vals = uv_rot(:,1); v_vals = uv_rot(:,2);
range_u = range(u_vals); range_v = range(v_vals);

if range_v > range_u, temp=u_vals; u_vals=v_vals; v_vals=temp; temp=range_u; range_u=range_v; range_v=temp; end
epsilon_ricci = exp(-2*pi * (range_v/range_u));

TR = triangulation(tri, nodes);
ti = pointLocation(TR, [x_samp, y_samp]);
bc = cartesianToBarycentric(TR, ti, [x_samp, y_samp]);

uv_samp = zeros(length(x_samp), 2);
valid_ti_mask = ~isnan(ti);
idx_in_mesh = find(valid_ti_mask);

for k = 1:length(idx_in_mesh)
    pidx = idx_in_mesh(k);
    tid = ti(pidx);
    tri_uvs = squeeze(uv_tri(tid, :, :)); 
    tri_uvs_rot = tri_uvs * coeff_pca;
    if range(uv_rot(:,2)) > range(uv_rot(:,1))
         tri_uvs_rot = [tri_uvs_rot(:,2), tri_uvs_rot(:,1)];
    end
    uv_samp(pidx, :) = bc(pidx, :) * tri_uvs_rot;
end

theta_ricci = (uv_samp(:,1) - min(u_vals)) / range_u * 2 * pi;
phi_ricci   = (uv_samp(:,2) - min(v_vals)) / range_v;
rho_ricci   = epsilon_ricci .* (1/epsilon_ricci).^phi_ricci;
rho_ricci(~valid_ti_mask) = NaN;
A_ricci = zeros(length(x_samp), z_modes);
for j = 1:z_modes
    A_ricci(:, j) = zernike_std(j, rho_ricci, theta_ricci);
end

fprintf('3. Precomputing CSF mesh (Method B)...\n');

N_bdy = 360; t = linspace(0, 2*pi, N_bdy+1)'; t(end)=[];
r_out_geo = 1.0 + 0.05*sin(3*t) + 0.04*cos(5*t);
[x_out, y_out] = pol2cart(t, r_out_geo);
r_in_local = 0.40 + 0.02*sin(4*t); [x_in_l, y_in_l] = pol2cart(t, r_in_local);
x_in = x_in_l + 0.4; y_in = y_in_l - 0.1;

[x_out_u, y_out_u] = reparam_curve(x_out, y_out, 120);
[x_in_u, y_in_u] = reparam_curve(x_in, y_in, 120);

n_rad = 20; n_ang = 120;
r_vals = linspace(0, 1, n_rad)';
X_mesh = zeros(n_rad, n_ang+1); Y_mesh = zeros(n_rad, n_ang+1);
for j=1:n_ang+1
    X_mesh(:,j) = (1-r_vals)*x_in_u(j) + r_vals*x_out_u(j);
    Y_mesh(:,j) = (1-r_vals)*y_in_u(j) + r_vals*y_out_u(j);
end

for k=1:30
    X_new = X_mesh; Y_new = Y_mesh;
    for r=2:n_rad-1
        X_new(r,2:end-1) = 0.25*(X_mesh(r-1,2:end-1)+X_mesh(r+1,2:end-1)+X_mesh(r,1:end-2)+X_mesh(r,3:end));
        Y_new(r,2:end-1) = 0.25*(Y_mesh(r-1,2:end-1)+Y_mesh(r+1,2:end-1)+Y_mesh(r,1:end-2)+Y_mesh(r,3:end));
    end
    X_mesh = X_new; Y_mesh = Y_new;
end

len_out = sum(sqrt(diff(x_out_u).^2 + diff(y_out_u).^2));
len_in  = sum(sqrt(diff(x_in_u).^2  + diff(y_in_u).^2));
epsilon_csf = len_in / len_out;

theta_grid = repmat(linspace(0, 2*pi, n_ang+1), n_rad, 1);
rho_grid   = repmat(sqrt(linspace(epsilon_csf^2, 1, n_rad)'), 1, n_ang+1);

F_rho = scatteredInterpolant(X_mesh(:), Y_mesh(:), rho_grid(:), 'natural', 'nearest');
F_sin = scatteredInterpolant(X_mesh(:), Y_mesh(:), sin(theta_grid(:)), 'natural', 'nearest');
F_cos = scatteredInterpolant(X_mesh(:), Y_mesh(:), cos(theta_grid(:)), 'natural', 'nearest');

rho_csf = F_rho(x_samp, y_samp);
sin_csf = F_sin(x_samp, y_samp);
cos_csf = F_cos(x_samp, y_samp);
theta_csf = atan2(sin_csf, cos_csf);

A_csf = zeros(length(x_samp), z_modes);
for j = 1:z_modes
    A_csf(:, j) = zernike_std(j, rho_csf, theta_csf);
end

fprintf('4. Starting Monte Carlo comparison analysis (N=50)...\n');

noise_levels = linspace(0.05, 0.20, 10); 
noise_pct = noise_levels * 100;
n_trials = 50;

res_ricci_mean = zeros(length(noise_levels), 1);
res_ricci_std  = zeros(length(noise_levels), 1);
res_csf_mean   = zeros(length(noise_levels), 1);
res_csf_std    = zeros(length(noise_levels), 1);

U_ideal = exp(1i * W_clean_samp);

fprintf('   Progress: ');

for k = 1:length(noise_levels)
    sigma = noise_levels(k);
    
    errs_ricci = zeros(n_trials, 1);
    errs_csf   = zeros(n_trials, 1);
    
    if mod(k,2)==0, fprintf('%.0f%% ', k/length(noise_levels)*100); else, fprintf('.'); end
    
    for t = 1:n_trials
        n_real = sigma * randn(size(x_samp));
        n_imag = sigma * randn(size(x_samp));
        U_noise = (n_real + 1i * n_imag) / sqrt(2);
        
        U_total = U_ideal + U_noise;
        phase_noise = angle(U_total ./ U_ideal);
        
        W_noisy = W_clean_samp + phase_noise;
        
        valid_a = ~any(isnan(A_ricci), 2);
        c_ricci = A_ricci(valid_a, :) \ W_noisy(valid_a);
        w_rec_ricci = A_ricci(valid_a, :) * c_ricci;
        diff_ricci = W_clean_samp(valid_a) - w_rec_ricci;
        errs_ricci(t) = sqrt(mean(diff_ricci.^2));
        
        valid_b = ~any(isnan(A_csf), 2);
        c_csf = A_csf(valid_b, :) \ W_noisy(valid_b);
        w_rec_csf = A_csf(valid_b, :) * c_csf;
        diff_csf = W_clean_samp(valid_b) - w_rec_csf;
        errs_csf(t) = sqrt(mean(diff_csf.^2));
    end
    
    res_ricci_mean(k) = mean(errs_ricci);
    res_ricci_std(k)  = std(errs_ricci);
    res_csf_mean(k)   = mean(errs_csf);
    res_csf_std(k)    = std(errs_csf);
end
fprintf('\nCalculation complete.\n');

figure('Name', 'Robustness Comparison: Annulus', 'Color', 'w', 'Position', [100, 100, 680, 500]);
hold on; box on;

errorbar(noise_pct, res_ricci_mean, res_ricci_std, 'o-', ...
    'LineWidth', 1.5, 'Color', '#0072BD', ...
    'MarkerFaceColor', '#0072BD', 'CapSize', 8, ...
    'DisplayName', 'Ricci Flow Conformal');

errorbar(noise_pct, res_csf_mean, res_csf_std, 's-', ...
    'LineWidth', 1.5, 'Color', '#D95319', ...
    'MarkerFaceColor', '#D95319', 'CapSize', 8, ...
    'DisplayName', 'CSF-QCM (Proposed)');

grid on;
legend('Location', 'NorthWest', 'FontSize', 11);

xlabel('Input Noise Amplitude (%)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Reconstruction RMS Error (\lambda)', 'FontSize', 12, 'FontWeight', 'bold');
title({'Robustness Comparison (Type III Annulus)', 'Speckle Noise Model (Monte Carlo N=50)'}, ...
    'FontSize', 13);

xlim([4, 21]);
ylim([0, max([res_ricci_mean; res_csf_mean])*1.4]); 

save('annulus_robustness_data.mat', 'noise_levels', 'noise_pct', ...
     'res_ricci_mean', 'res_ricci_std', ...
     'res_csf_mean', 'res_csf_std');
fprintf('Data saved to annulus_robustness_data.mat\n');

function Z = zernike_std(j, r, t)
    [n, m] = get_noll_nm(j); R_nl = zeros(size(r));
    for k = 0:(n-abs(m))/2
        num = (-1)^k * factorial(n-k);
        den = factorial(k) * factorial((n+m)/2 - k) * factorial((n-m)/2 - k);
        R_nl = R_nl + (num/den) * r.^(n-2*k);
    end
    if m >= 0, Z = R_nl .* cos(m*t); else, Z = R_nl .* sin(abs(m)*t); end
end
function [n, m] = get_noll_nm(j)
    n_list = [0 1 1 2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7];
    m_list = [0 1 -1 0 -2 2 -1 1 -3 3 0 -2 2 -4 4 -1 1 -3 3 -5 5 0 -2 2 -4 4 -6 6 -1 1 -3 3 -5 5 -7 7];
    if j > length(n_list), n=0; m=0; else, n=n_list(j); m=m_list(j); end
end

function [xn, yn] = reparam_curve(x, y, N)
    x = x(:); y = y(:);
    if (x(1) ~= x(end)) || (y(1) ~= y(end)), x(end+1)=x(1); y(end+1)=y(1); end
    dx = diff(x); dy = diff(y); s = [0; cumsum(sqrt(dx.^2 + dy.^2))];
    s_targets = linspace(0, s(end), N+1);
    xn = interp1(s, x, s_targets, 'spline'); yn = interp1(s, y, s_targets, 'spline');
end

function [nodes, tri, idx_out, idx_in] = build_mesh_annulus()
    t=linspace(0,2*pi,120); t(end)=[];
    ro=1+.05*sin(3*t)+.04*cos(5*t); xo=ro.*cos(t); yo=ro.*sin(t);
    ri=.4+.02*sin(4*t); [xil, yil]=pol2cart(t,ri); xi=xil+0.4; yi=yil-0.1;
    
    [px,py]=meshgrid(linspace(-1.5,1.5,35)); px=px(:); py=py(:);
    valid = inpolygon(px,py,xo,yo) & ~inpolygon(px,py,xi,yi);
    nodes_inner=[xi' yi']; nodes_outer=[xo' yo'];
    nodes_internal=[px(valid) py(valid)];
    
    nodes = [nodes_outer; nodes_inner; nodes_internal];
    idx_out = 1:length(t); idx_in = (length(t)+1):(2*length(t));
    
    DT = delaunayTriangulation(nodes);
    c = incenter(DT);
    in = inpolygon(c(:,1),c(:,2),xo,yo) & ~inpolygon(c(:,1),c(:,2),xi,yi);
    tri = DT.ConnectivityList(in, :);
end

function L0 = compute_edge_lengths(XY, tri)
    p1=XY(tri(:,1),:); p2=XY(tri(:,2),:); p3=XY(tri(:,3),:);
    L0.l1=sqrt(sum((p2-p3).^2,2)); L0.l2=sqrt(sum((p1-p3).^2,2)); L0.l3=sqrt(sum((p1-p2).^2,2));
end

function [K, L_cot, L_new] = compute_curv_metric(tri, L0, u, n, is_bdy)
    i1=tri(:,1); i2=tri(:,2); i3=tri(:,3);
    l1=L0.l1.*exp((u(i2)+u(i3))/2); l2=L0.l2.*exp((u(i1)+u(i3))/2); l3=L0.l3.*exp((u(i1)+u(i2))/2);
    L_new=struct('l1',l1,'l2',l2,'l3',l3);
    f=@(a,b,c) max(-1,min(1,(b.^2+c.^2-a.^2)./(2.*b.*c)));
    ang1=acos(f(l1,l2,l3)); ang2=acos(f(l2,l1,l3)); ang3=acos(f(l3,l1,l2));
    cot1=1./tan(ang1); cot2=1./tan(ang2); cot3=1./tan(ang3);
    I=[i2;i3;i3;i1;i1;i2]; J=[i3;i2;i1;i3;i2;i1]; V=0.5*[cot1;cot1;cot2;cot2;cot3;cot3];
    L_cot=sparse(I,J,-V,n,n); L_cot=L_cot+sparse(1:n,1:n,-sum(L_cot,2),n,n);
    ang_sum = sparse(tri(:), 1, [ang1; ang2; ang3], n, 1);
    K = zeros(n, 1);
    K(~is_bdy) = 2*pi - ang_sum(~is_bdy); K(is_bdy) = pi - ang_sum(is_bdy); 
end

function uv_tri = flatten_mesh(tri, L, cut_edges)
    n_tri = size(tri,1); uv_tri = zeros(n_tri,3,2); visited = false(n_tri,1);
    l1=L.l1(1); l2=L.l2(1); l3=L.l3(1);
    p1=[0,0]; p2=[l3,0];
    cp1=(l3^2+l2^2-l1^2)/(2*l3*l2); p3=[l2*cp1, l2*sqrt(1-cp1^2)];
    uv_tri(1,:,:) = [p1; p2; p3]; visited(1)=true;
    queue = 1; head = 1;
    adj = neighbors(triangulation(tri, zeros(max(tri(:)),2)));
    
    while head <= length(queue)
        curr = queue(head); head=head+1;
        curr_uv = squeeze(uv_tri(curr,:,:));
        curr_nodes = tri(curr,:);
        for k=1:3
            next = adj(curr, k);
            if isnan(next) || visited(next), continue; end
            
            common = intersect(curr_nodes, tri(next,:));
            if length(common)~=2, continue; end
            if ismember(sort(common), cut_edges, 'rows'), continue; end
            
            [~,loc_curr] = ismember(common, curr_nodes);
            [~,loc_next] = ismember(common, tri(next,:));
            new_v_idx = setdiff(1:3, loc_next);
            
            A = curr_uv(loc_curr(1),:); B = curr_uv(loc_curr(2),:);
            l_next = [L.l1(next), L.l2(next), L.l3(next)];
            r_A = l_next(setdiff(1:3,[loc_next(1), new_v_idx]));
            r_B = l_next(setdiff(1:3,[loc_next(2), new_v_idx]));
            
            dAB = norm(A-B);
            visited(next) = true; queue(end+1) = next;
        end
    end
end
