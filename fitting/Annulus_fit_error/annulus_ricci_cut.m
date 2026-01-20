function annulus_ricci_cut()
clear; clc; close all;

fprintf('[Step 1] Generating Wavefront...\n');
N = 400; rng(2025);
x = linspace(-1.6, 1.6, N); y = linspace(-1.6, 1.6, N);
[X, Y] = meshgrid(x, y);
[Theta, R] = cart2pol(X, Y);

r_out_b = 1.0 + 0.05*sin(3*Theta) + 0.04*cos(5*Theta);
mask_out = R <= r_out_b;
[Theta_in, R_in_local] = cart2pol(X - 0.4, Y + 0.1); 
r_in_b = 0.40 + 0.02*sin(4*Theta_in);
mask_in = R_in_local <= r_in_b;
mask = mask_out & ~mask_in;

z_modes = 36;
coeffs_true = (rand(z_modes, 1) - 0.5) .* (1./(1:z_modes)');
coeffs_true(5:10) = coeffs_true(5:10) * 2; 
W_raw = zeros(size(X));
for j = 1:z_modes
    W_raw = W_raw + coeffs_true(j) * zernike_std(j, R, Theta);
end
vals = W_raw(mask); vals = vals - mean(vals);
W_true = nan(size(W_raw)); W_true(mask) = vals * (0.05 / std(vals));
fprintf('   Ground Truth RMS: %.4f lambda\n', std(W_true(mask)));

fprintf('[Step 2] Meshing & Ricci Flow...\n');
[nodes, tri, bdy_out_idx, bdy_in_idx] = build_mesh(); 
n_nodes = size(nodes, 1); n_tri = size(tri, 1);

K_target = zeros(n_nodes, 1); 
L = compute_edge_lengths(nodes, tri);
u = zeros(n_nodes, 1);
is_bdy = false(n_nodes, 1); 
is_bdy(bdy_out_idx) = true; is_bdy(bdy_in_idx) = true;

for iter = 1:60
    [K_curr, L_cot, ~] = compute_curv_metric_proper(tri, L, u, n_nodes, is_bdy);
    delta_K = K_target - K_curr;
    err = norm(delta_K, inf);
    if err < 1e-2, break; end 
    
    fixed_id = 1; 
    H = L_cot; H(fixed_id, :) = 0; H(:, fixed_id) = 0; H(fixed_id, fixed_id) = 1;
    delta_K(fixed_id) = 0;
    
    du = H \ delta_K;
    if any(isnan(du)), u = zeros(n_nodes, 1); break; end
    u = u + 0.5 * du; 
end
[~, ~, L_flat] = compute_curv_metric_proper(tri, L, u, n_nodes, is_bdy);

fprintf('[Step 3] Finding Shortest Geometric Cut...\n');
pts_out = nodes(bdy_out_idx, :);
pts_in  = nodes(bdy_in_idx, :);
dist_mat = zeros(length(bdy_out_idx), length(bdy_in_idx));
for i = 1:length(bdy_out_idx)
    d2 = (pts_in(:,1)-pts_out(i,1)).^2 + (pts_in(:,2)-pts_out(i,2)).^2;
    dist_mat(i, :) = d2';
end
[~, min_idx_linear] = min(dist_mat(:));
[r, c] = ind2sub(size(dist_mat), min_idx_linear);
start_node = bdy_out_idx(r); end_node = bdy_in_idx(c);

E_all = [tri(:,1), tri(:,2); tri(:,2), tri(:,3); tri(:,3), tri(:,1)];
W_safe = [L.l3; L.l1; L.l2]; 
E_sorted = sort(E_all, 2);
[E_unique, idx_unique, ~] = unique(E_sorted, 'rows');
W_unique = W_safe(idx_unique);
G_graph = graph(E_unique(:,1), E_unique(:,2), W_unique);

cut_path = shortestpath(G_graph, start_node, end_node);
cut_edges = sort([cut_path(1:end-1)', cut_path(2:end)'], 2);

uv_tri = zeros(n_tri, 3, 2); 
tri_visited = false(n_tri, 1);
neighbors_list = neighbors(triangulation(tri, nodes));

seed_id = 1;
l1=L_flat.l1(seed_id); l2=L_flat.l2(seed_id); l3=L_flat.l3(seed_id);
p1=[0,0]; p2=[l3,0];
cos_p1 = max(-1, min(1, (l3^2+l2^2-l1^2)/(2*l3*l2)));
p3=[l2*cos_p1, l2*sqrt(1-cos_p1^2)];
uv_tri(seed_id,:,:) = [p1; p2; p3];
tri_visited(seed_id) = true;

queue = seed_id; head = 1;
while head <= length(queue)
    curr_t = queue(head); head = head + 1;
    curr_uvs = squeeze(uv_tri(curr_t,:,:));
    curr_nodes = tri(curr_t,:);
    for k=1:3
        next_t = neighbors_list(curr_t, k);
        if isnan(next_t) || tri_visited(next_t), continue; end
        if k==1, e_nodes=sort([curr_nodes(2),curr_nodes(3)]);
        elseif k==2, e_nodes=sort([curr_nodes(1),curr_nodes(3)]);
        else, e_nodes=sort([curr_nodes(1),curr_nodes(2)]); end
        if ismember(e_nodes, cut_edges, 'rows'), continue; end 
        
        [~, loc_next] = ismember(e_nodes, tri(next_t,:));
        new_v_idx = setdiff(1:3, loc_next);
        [~, loc_curr] = ismember(e_nodes, curr_nodes);
        A_uv = curr_uvs(loc_curr(1),:); B_uv = curr_uvs(loc_curr(2),:);
        l_next = [L_flat.l1(next_t), L_flat.l2(next_t), L_flat.l3(next_t)];
        get_l = @(i,j) l_next(setdiff(1:3,[i,j]));
        r_A = get_l(loc_next(1), new_v_idx); r_B = get_l(loc_next(2), new_v_idx);
        d_AB = norm(A_uv - B_uv);
        ang = acos(max(-1, min(1, (d_AB^2+r_A^2-r_B^2)/(2*d_AB*r_A))));
        vc_A=complex(A_uv(1),A_uv(2)); vc_B=complex(B_uv(1),B_uv(2));
        is_ccw = (loc_next(2) == mod(loc_next(1),3)+1);
        offset = r_A * exp((1i*ang)*(2*is_ccw-1)) * ((vc_B-vc_A)/d_AB);
        new_uv = [real(vc_A+offset), imag(vc_A+offset)];
        uv_next=zeros(3,2); uv_next(loc_next(1),:)=A_uv; uv_next(loc_next(2),:)=B_uv; uv_next(new_v_idx,:)=new_uv;
        uv_tri(next_t,:,:) = uv_next;
        tri_visited(next_t) = true;
        queue(end+1) = next_t;
    end
end
uv_points = reshape(uv_tri,[],2);

fprintf('[Step 4] Mapping Coordinates...\n');
coeff_pca = pca(uv_points); 
uv_rot = uv_points * coeff_pca;
u_rot = uv_rot(:,1); v_rot = uv_rot(:,2);
range_u = max(u_rot) - min(u_rot); range_v = max(v_rot) - min(v_rot);
if range_v > range_u
    temp = u_rot; u_rot = v_rot; v_rot = temp;
    range_temp = range_u; range_u = range_v; range_v = range_temp;
end
epsilon_rec = exp(-2*pi * (range_v/range_u));

pix_idx = find(mask);
TR = triangulation(tri, nodes);
ti = pointLocation(TR, [X(pix_idx), Y(pix_idx)]);
valid_mask = ~isnan(ti);
valid_ti = ti(valid_mask);
valid_pix_idx = pix_idx(valid_mask);
bc = cartesianToBarycentric(TR, valid_ti, [X(valid_pix_idx), Y(valid_pix_idx)]);

pts_uv = zeros(length(valid_pix_idx), 2);
for k = 1:length(valid_pix_idx)
    pts_uv(k, :) = bc(k, :) * squeeze(uv_tri(valid_ti(k), :, :));
end
pts_uv_rot = pts_uv * coeff_pca;
if range(uv_rot(:,2)) > range(uv_rot(:,1))
     pts_uv_rot = [pts_uv_rot(:,2), pts_uv_rot(:,1)];
end

theta_fit = (pts_uv_rot(:,1) - min(u_rot)) / range_u * 2 * pi;
theta_fit = max(0, min(2*pi, theta_fit)); 
phi_fit = (pts_uv_rot(:,2) - min(v_rot)) / range_v;
phi_fit = max(0, min(1, phi_fit));       
rho_fit = epsilon_rec .* (1/epsilon_rec).^phi_fit;
rho_fit = max(epsilon_rec, min(1, rho_fit));

fprintf('[Step 5] Zernike Fitting (with Seam Stitching)...\n');

w_target = W_true(valid_pix_idx);

idx_left  = find(theta_fit < 0.2);
idx_right = find(theta_fit > 2*pi - 0.2);

rho_aug   = [rho_fit; rho_fit(idx_left); rho_fit(idx_right)];
theta_aug = [theta_fit; theta_fit(idx_left) + 2*pi; theta_fit(idx_right) - 2*pi];
w_aug     = [w_target; w_target(idx_left); w_target(idx_right)];

n_terms = 36; 
H_aug = zeros(length(rho_aug), n_terms);
for j=1:n_terms, H_aug(:,j) = zernike_std(j, rho_aug, theta_aug); end

[Q_aug, R_aug] = qr(H_aug, 0);
c_aug = Q_aug' * w_aug;

H_orig = zeros(length(rho_fit), n_terms);
for j=1:n_terms, H_orig(:,j) = zernike_std(j, rho_fit, theta_fit); end
c_std = R_aug \ c_aug; 

w_rec_vec = H_orig * c_std;

W_rec = nan(size(W_true)); W_rec(valid_pix_idx) = w_rec_vec;
Resid = nan(size(W_true)); Resid(valid_pix_idx) = w_target - w_rec_vec;

pv_true=max(W_true(:))-min(W_true(:));
fprintf('  Ture PV: %.5f λ \n', pv_true);

pv_rf=max(Resid(:))-min(Resid(:));
fprintf('  ricci flow PV error: %.5f λ \n', pv_rf);

figure('Color','w','Position',[50,50,1000,600]);
tiledlayout(2,2,'Padding','compact', 'TileSpacing', 'compact');
colormap(jet);

ax1 = nexttile;
pcolor(X,Y,W_true); shading interp; axis image off; hold on;
title('1. Ground Truth & Cut Path', 'FontSize', 12);
clim_range = [-0.15 0.15]; clim(clim_range);

ax2 = nexttile;
k_bdy = boundary(u_rot, v_rot, 0.8);
plot(u_rot(k_bdy), v_rot(k_bdy), 'k-', 'LineWidth', 1.5); hold on;
step = max(1, floor(length(pts_uv_rot)/3000));
scatter(pts_uv_rot(1:step:end,1), pts_uv_rot(1:step:end,2), 15, ...
        w_target(1:step:end), 'filled', 'MarkerFaceAlpha', 0.8);
title('2. Flattened Strip (UV Domain)', 'FontSize', 12);
axis equal off; 
clim(clim_range);
xlim([min(u_rot) max(u_rot)]); ylim([min(v_rot) max(v_rot)]);

ax3 = nexttile;
pcolor(X,Y,W_rec); shading interp; axis image off; 
title('3. Reconstruction (Seam Stitched)', 'FontSize', 12);
clim(clim_range);

ax4 = nexttile;
pcolor(X,Y,Resid); shading interp; axis image off; 
rms_val = std(Resid(:),'omitnan');
title(['4. Residual (RMS = ' num2str(rms_val,'%.4f') ' \lambda)'], 'FontSize', 12);
cb = colorbar; cb.Label.String = 'Wavefront Error (\lambda)';

sgtitle('Conformal Flattening & Zernike Fitting for Type III Aperture', 'FontSize', 14, 'FontWeight', 'bold');

end

function [nodes, tri, idx_out, idx_in] = build_mesh()
    t=linspace(0,2*pi,180); t(end)=[];
    ro=1+.05*sin(3*t)+.04*cos(5*t); xo=ro.*cos(t); yo=ro.*sin(t);
    ri=.4+.02*sin(4*t); xi=ri.*cos(t)+.4; yi=ri.*sin(t)-.1;
    [px,py]=meshgrid(linspace(-1.5,1.5,45)); px=px(:); py=py(:);
    valid = inpolygon(px,py,xo,yo) & ~inpolygon(px,py,xi,yi);
    nodes=[xo' yo'; xi' yi'; px(valid) py(valid)];
    idx_out=1:length(t); idx_in=(length(t)+1):(2*length(t));
    DT=delaunayTriangulation(nodes); c=incenter(DT);
    tri=DT.ConnectivityList(inpolygon(c(:,1),c(:,2),xo,yo) & ~inpolygon(c(:,1),c(:,2),xi,yi), :);
end

function [K, L_cot, L_new] = compute_curv_metric_proper(tri, L0, u, n, is_bdy)
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
    K(~is_bdy) = 2*pi - ang_sum(~is_bdy);
    K(is_bdy)  = pi - ang_sum(is_bdy); 
end

function L0 = compute_edge_lengths(XY, tri)
    p1=XY(tri(:,1),:); p2=XY(tri(:,2),:); p3=XY(tri(:,3),:);
    L0.l1=sqrt(sum((p2-p3).^2,2)); L0.l2=sqrt(sum((p1-p3).^2,2)); L0.l3=sqrt(sum((p1-p2).^2,2));
end

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
    n_list = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7];
    m_list = [0, 1, -1, 0, -2, 2, -1, 1, -3, 3, 0, -2, 2, -4, 4, -1, 1, -3, 3, -5, 5, 0, -2, 2, -4, 4, -6, 6, -1, 1, -3, 3, -5, 5, -7, 7];
    if j <= length(n_list), n=n_list(j); m=m_list(j);
    else, n=0; m=0; end
end
