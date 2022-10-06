clearvars
close all

gpud = 1;
reps = [1 2 3];
%%
for rep = reps
    tic
    clearvars -except rep gpud gf reps BX BY BZ
    %
    %     % GPU Activation
    %     if gpuDeviceCount>0 && rep>0
    %         gpud = 1;
    %     end
    % Diameter of jet
    const.D = 1;
    const.k_w = 0.0834;          % Wake spreading angle
    const.sigma_0 = 0.235;       % Wake width factor
    const.z_offset = const.D;    % Vertical offset
    const.U = 1;                 % Normalized Velocity
    %const.Bgeo = [0 0.44 1];     % Bgeo
    const.Bgeo = [0 1 0];     % Bgeo
    const.Bgeo = const.Bgeo./sum(const.Bgeo.^2);
    
    % Jensen wake width function - linear spreading
    % see Jensen, N. O. “A Note on Wind Generator Interaction.” Risø-M-2411., 1983.
    func.wake_width = @(D, z, k_w, z_offset) ...
        1 + k_w.*log(1 + exp((z-z_offset)./(D/2)));
    
    % Average Velocity Deficit Function - Streamwise cuton
    func.avg_vel_def     =   @(zs, U, D, k_w, z_offset) ...
        U./(2*(func.wake_width(D, zs, k_w, z_offset).^2)) ...
        .*(1+erf(zs./((D/2)*sqrt(2)*2)));
    
    % Gaussian Function
    func.gs = @(c, xs, ys, sigma) c*exp(-(xs.^2+ys.^2)./(2*sigma.^2));
    
    % Velocity field actuator disk
    func.W =   @(xs, ys, zs, sigma_0, U, D, k_w, z_offset) ...
        (D.^2./(8.*sigma_0.^2).*func.avg_vel_def(zs, U, D, k_w, z_offset))...
        .*func.gs(1,xs,ys, sigma_0.*func.wake_width(D, zs, k_w, z_offset));
    
    %% Show functions
    for idx = []
        % Trial gridding
        zs = linspace(-5,20,1000);
        ys = linspace(-1,1,100)';
        
        % Wake spreading function
        d_w = func.wake_width(const.D, zs, const.k_w, const.z_offset);
        figure('Name','Actuator Disk Models')
        subplot(1,3,1)
        plot(d_w, -zs,'LineWidth',2)
        ylabel('Vertical distance (z/D)')
        xlabel('Spreading Rate (r/D)')
        set(gca,'FontSize',24)
        
        % Wake velocity deficit
        deltaW = func.avg_vel_def(zs, const.U, const.D, const.k_w, const.z_offset);
        subplot(1,3,2)
        %title('Velocity Deficit')
        plot(deltaW, -zs,'LineWidth',2)
        ylabel('Vertical distance (z/D)')
        xlabel('Mean Velocity Deficit')
        set(gca,'FontSize',24)
        
        % Wave velocity contours
        colormap(bone)
        subplot(1,3,3)
        title('Actuator Disk Velocity Deficit')
        w_test = func.W(ys*0, ys, zs, const.sigma_0, const.U, const.D, const.k_w, const.z_offset);
        contourf(ys, -zs, -w_test'/2,15)
        ylabel('(z/D)')
        xlabel('(r/D)')
        set(gca,'FontSize',24)
        colorbar
        caxis([-1 0])
        shading interp
        %close all
        clearvars w_test zs ys d_w deltaW w_test
    end
    
    % Number of grid points
    %N = [1 1 8*4]*64*2^(rep-1);% [32 64 16]
    AR = 2;
    N = [1 1 AR]*128*2^(rep-2);
    
    % Limits of jet
    %lim = [1 1 8*4]*5*D; %[3 3 50];
    lim = [1 1 AR]*10*const.D; %[3 3 50];
    %% Fit the Ur to the Uz gradients
    zs = linspace(-lim(3), lim(3),1001);
    
    lin_r = 0;
    if lin_r
        rs = linspace(0, sqrt(lim(1).^2+lim(2).^2), 501)';
    else
        rs = logspace(-8, log10(sqrt(lim(1).^2+lim(2).^2)), 201)';
        rs = [0;rs];
    end
    
    [rs, zs] = meshgrid(rs, zs);    zs = [zs];   rs = [rs];
    dr = rs(1,:);
    dz = zs(:,1);
    [dW_dr, dW_dz] = gradient(  -func.W(rs, 0, zs, const.sigma_0, const.U, const.D, const.k_w, const.z_offset), dr , -dz);
    Ur = -cumtrapz(dr,[dW_dz].*rs,2)./rs;
    U_R = griddedInterpolant(rs',zs',Ur','linear','linear');
    %U_R = scatteredInterpolant(rs(:),zs(:),Ur(:),'linear','linear');
    
    clearvars Ur
    colormap(redbluecmap)
    for idx = []
        figure(1)
        subplot(1,4,1)
        Wdata = -func.W(rs, 0, zs, const.sigma_0, const.U, const.D, const.k_w, const.z_offset);
        [c h1] = contourf(rs, -zs,Wdata);
        %axis equal
        xlabel('r/D')
        ylabel('z/D')
        title('W')
        caxis([-max(abs(Wdata),[],'all') max(abs(Wdata),[],'all')]);
        shading interp
        colorbar('eastoutside')
        set(gca,'FontSize',16)
        axis([0 2 -10 2])
        
        subplot(1,4,2)
        Ur = -cumtrapz(dr,[dW_dz].*rs,2)./rs;
        h1 = contourf(rs, -zs,Ur);
        %axis equal
        xlabel('r/D')
        ylabel('z/D')
        title('Est. U_r')
        caxis([-max(abs(Ur),[],'all') max(abs(Ur),[],'all')]);
        colorbar('eastoutside')
        set(gca,'FontSize',16)
        axis([0 2 -10 2])
        
        
        subplot(1,4,3)
        data = dW_dz;
        contourf(rs,-zs,data);
        %axis equal
        xlabel('r/D')
        ylabel('z/D')
        title('\partial W/ \partial z')
        caxis([-max(abs(data),[],'all') max(abs(data),[],'all')]);
        colorbar('eastoutside')
        set(gca,'FontSize',16)
        axis([0 2 -10 2])
        
        
        subplot(1,4,4)
        data = dW_dr;
        contourf(rs,-zs,data);
        title('\partial W/ \partial r')
        %axis equal
        xlabel('r/D')
        ylabel('z/D')
        caxis([-max(abs(data),[],'all') max(abs(data),[],'all')]);
        colorbar('eastoutside')
        set(gca,'FontSize',16)
        axis([0 2 -10 2])
        
        
    end
    
    
    
    %%     Grid-mesh in the migration
    % CPU array for integration of volumes
    
    X = (linspace(-lim(1), lim(1), N(1)));
    Y = (linspace(-lim(2), lim(2), N(2)));
    Z = (linspace(-lim(3)*0-5, lim(3), N(3)));
    
    % Node spacing
    % dxp = 2*lim./(N-1);
    dxp = [mean(diff(X)) mean(diff(Y)) mean(diff(Z))];
    % end
    [xp, yp, zp] = ndgrid(  X, Y, Z);
    XP{1} = xp;   XP{2} = yp;   XP{3} = zp;
    colormap(redbluecmap)
    
    %%
    %   [FX,FY] = gradient( gs(c, x2d, y2d, sigma), dxp(1), dxp(2));
    
    clearvars gradUr gradUz
    Uz = -func.W(XP{1}, XP{2}, XP{3}, const.sigma_0, const.U, const.D, const.k_w, const.z_offset);
    Ur = U_R(XP{1}.^2+XP{2}.^2,XP{3});
    
    [dW_dx, dW_dy, dW_dz] = gradient(Uz  , dxp(1), dxp(2), dxp(3));
    gradUz{1} = dW_dx;   gradUz{2} = dW_dy;   gradUz{3} = dW_dz;
    
    Ux = Ur.*XP{1}./sqrt(XP{1}.^2+XP{2}.^2);
    [dU_dx, dU_dy, dU_dz] = gradient(Ux  , dxp(1), dxp(2), dxp(3));
    gradUx{1} = dU_dx;   gradUx{2} = dU_dy;   gradUx{3} = dU_dz;
    
    Uy = Ur.*XP{2}./sqrt(XP{1}.^2+XP{2}.^2);
    [dU_dx, dU_dy, dU_dz] = gradient(Uy  , dxp(1), dxp(2), dxp(3));
    gradUy{1} = dU_dx;   gradUy{2} = dU_dy;   gradUy{3} = dU_dz;
    clearvars dW_dx dW_dy dW_dz xp yp zp dU_dx dU_dy dU_dz
    
    %%
    %     Grid for r-points. Locations where we are solving for the magnetic
    %     field in 1-D Arrays
    y_r = [-fliplr(logspace(-2,3,51)*const.D) 0 logspace(-2,3,51)*const.D]';
    x_r = [-fliplr(logspace(-2,3,50)*const.D) 0 logspace(-2,3,50)*const.D]';
    
    %     Make the grid from the 1-D arrays
    [Xplot, Yplot] = ndgrid(x_r,y_r);
    r = zeros(numel(Xplot),3);
    r(:,1) = Xplot(:)*0; r(:,2) = Xplot(:); r(:,3) = Yplot(:);
    % r(:,1) = Xplot(:); r(:,2) = Yplot(:); r(:,3) = 0*Yplot(:);

    if rep == 1
        BX = zeros(length(reps),size(r,1));
        BY = zeros(length(reps),size(r,1));
        BZ = zeros(length(reps),size(r,1));
    end
    
    if gpud == 1
        X = gpuArray(X);
        Y = gpuArray(Y);
        Z = gpuArray(Z);
        x_gpu = gpuArray(XP{1});
        y_gpu = gpuArray(XP{2});
        z_gpu = gpuArray(XP{3});
        r = gpuArray(r);
        
        dUxdx_gpu = gpuArray(single(gradUx{2}));
        dUxdy_gpu = gpuArray(single(gradUx{1}));
        dUxdz_gpu = gpuArray(single(gradUx{3}));
        
        dUydx_gpu = gpuArray(single(gradUy{2}));
        dUydy_gpu = gpuArray(single(gradUy{1}));
        dUydz_gpu = gpuArray(single(gradUy{3}));
        
        dUzdx_gpu = gpuArray(single(gradUz{2}));
        dUzdy_gpu = gpuArray(single(gradUz{1}));
        dUzdz_gpu = gpuArray(single(gradUz{3}));
    end
    
    disp('Greens function integration')
    h = waitbar(0,'Please wait');
    
    tstart = tic;
    for idx = 1:size(r,1)
        if gpud == 1
            r_rp = sqrt(    (r(idx,1)-x_gpu).^2 +...
                (r(idx,2)-y_gpu).^2 +...
                (r(idx,3)-z_gpu).^2);
            % r_rp = gpuArray(r_rp);
            %% bx
            % bz = gpuArray(single(2*XP{2}.*func.gs(1,XP{1},XP{2},1)./r_rp./(4*pi)));
            bx = const.Bgeo(1).*dUxdx_gpu./r_rp;
            bx = bx + const.Bgeo(2).*dUxdy_gpu./r_rp;
            bx = bx + const.Bgeo(3).*dUxdz_gpu./r_rp;
            bx = trapz(X, trapz(Y, trapz(Z, bx, 3), 2),1)/(4*pi);
            % bx = trapz(Z, bx, 3); by = trapz(Y, bx, 2); by = trapz(X, bx, 1);
            
            %% by
            by = const.Bgeo(1).*dUydx_gpu./r_rp;
            by = by + const.Bgeo(2).*dUydy_gpu./r_rp;
            by = by + const.Bgeo(3).*dUydz_gpu./r_rp;
            by = trapz(X, trapz(Y, trapz(Z, by, 3), 2),1)/(4*pi);
            %by = trapz(Z, by, 3); by = trapz(Y, by, 2); by = trapz(X, by, 1);
            
            %% bz
            bz = const.Bgeo(1).*dUzdx_gpu./r_rp;
            bz = bz + const.Bgeo(2).*dUzdy_gpu./r_rp;
            bz = bz + const.Bgeo(3).*dUzdz_gpu./r_rp;
            bz = trapz(X, trapz(Y, trapz(Z, bz, 3), 2),1)/(4*pi);
            % bz = trapz(Z, bz, 3); bz = trapz(Y, bz, 2); bz = trapz(X, bz, 1);
            
        else
            r_rp = sqrt(    (r(idx,1)-XP{1}).^2 +...
                (r(idx,2)-XP{2}).^2 +...
                (r(idx,3)-XP{3}).^2);
            %temp = (2*XP{2}.*func.gs(1,XP{1},XP{2},1)./r_rp./(4*pi));
            %temp = (FY./r_rp);
            bx = const.Bgeo(1).*gradUx{2}./r_rp;
            bx = bx + const.Bgeo(2).*gradUx{1}./r_rp;
            bx = bx + const.Bgeo(3).*gradUx{3}./r_rp;
            bx = trapz(X, trapz(Y, trapz(Z, bx, 3), 2),1)/(4*pi);
            % bx = trapz(Z, bx, 3); by = trapz(Y, bx, 2); by = trapz(X, bx, 1);
            
            %% by
            by = const.Bgeo(1).*gradUy{2}./r_rp;
            by = by + const.Bgeo(2).*gradUy{1}./r_rp;
            by = by + const.Bgeo(3).*gradUy{3}./r_rp;
            by = trapz(X, trapz(Y, trapz(Z, by, 3), 2),1)/(4*pi);
            %by = trapz(Z, by, 3); by = trapz(Y, by, 2); by = trapz(X, by, 1);
            
            %% bz
            bz = const.Bgeo(1).*gradUz{2}./r_rp;
            bz = bz + const.Bgeo(2).*gradUz{1}./r_rp;
            bz = bz + const.Bgeo(3).*gradUz{3}./r_rp;
            bz = trapz(X, trapz(Y, trapz(Z, bz, 3), 2),1)/(4*pi);
            % bz = trapz(Z, bz, 3); bz = trapz(Y, bz, 2); bz = trapz(X, bz, 1);
            
        end
        
        if gpud == 1
            BZ(rep, idx) = gather(bz);
            BY(rep, idx) = gather(by);
            BX(rep, idx) = gather(bx);
        else
            BZ(rep, idx) = (bz);
            BX(rep, idx) = (bx);
            BY(rep, idx) = (by);
        end
        tleft = toc(tstart)/idx*size(r,1) -toc(tstart);
        if mod(idx,2) ==0
            waitbar(idx/size(r,1),h,sprintf('%0.1f sec elapsed + %0.1f sec left (%i/%i)',toc(tstart), tleft,idx,size(r,1)))
        end
    end
    gather(X); gather(Y); gather(Z);
    %         loglog(ray./sigma,gf(rep, :),'o-')
    %         set(gca, 'YScale','log');
    %         set(gca, 'XScale','log');
    grid on;
    box on;
    close(h)
    
    [rep  toc]
    
end
%%
Xplot   =   reshape(Xplot, size(x_r,1), size(y_r,1));
Yplot   =   reshape(Yplot, size(x_r,1), size(y_r,1));

mag_x    =    reshape(BX(end,:), size(x_r,1), size(y_r,1));
mag_y      =   reshape(BY(end,:), size(x_r,1), size(y_r,1));
mag_z     =    reshape(BZ(end,:), size(x_r,1), size(y_r,1));

%% Plotting

%plotConvergence(x_r,y_r, BZ)
%plotArea(Xplot,Yplot, mag_z)
plotArea(Xplot,Yplot, mag_z)

%plotArea(Xplot,Yplot, -func.W(Xplot, Yplot,0, const.sigma_0, const.U, const.D, const.k_w, const.z_offset))

%%
clf
plotDist(Xplot,Yplot, mag_z)
figure(1);hold on
vs = 4.225;
plot(unique(Xplot),(unique(Xplot)*50./((unique(Xplot)).^2.*sqrt((50)^2+(unique(Xplot)).^2)))'./vs.^2,'k-','LineWidth',2)
%plotArea(Xploplot(t,Yplot, U_R(Xplot,Yplot))
%plotArea(Xplot,Yplot, -func.gs(1,Xplot,Yplot,1))

%plotWandB(Xplot,Yplot, mag_z,const,func)
%%
%plotArea(t1.Xplot,t1.Yplot, t1.func.W(0, t1.Xplot, t1.Yplot, t1.const.sigma_0, t1.const.U, t1.const.D, t1.const.k_w, t1.const.z_offset))
%%


hold on
plotDist(t2.Xplot,t2.Yplot, t2.mag_z)
hold off
%%

F = scatteredInterpolant(Xplot(:),Yplot(:),double(mag_z(:)));
thetas = linspace(0,2*pi,100);
ff = F(cos(thetas), sin(thetas));
figure(1)
clf
plot(thetas,sin(thetas),'-','LineWidth',2)
hold on
plot(thetas',ff./max(ff),'o','LineWidth',1)
grid on
box on
xlabel('\theta')
ylabel('Amplitude')
axis([0 2*pi -1 1])
% F = scatteredInterpolant(Xplot(:),Yplot(:),Yplot(:).*gs(1,Xplot(:),Yplot(:),1));
% ff = F(cos(thetas), sin(thetas));
% plot(thetas',ff./max(ff),'x','LineWidth',1)
% legend('Sinusoid','Numerical','dw/dy')


matlab2tikz('Figure3_theta2.tex','width','3in','height','2in');

%%

function plotConvergence(x_r,y_r, data)
% Grid convergence Error

%% Error plot between the two grid spacings. Spacing differ by 2x
figure(1)
clf
gf = data;
colormap(redbluecmap)
% histogram(abs(diff(gf,1)'./(abs(gf(2,:)'))))
% % set(gca, 'YScale','log')
%plot(abs(diff(gf,1)'./(abs(gf(2,:)'))))
plot(diff(gf,2)'./mean(gf,2)')
% hold on
% plot(abs(gf(2,:)))
% hold on
% plot(abs(gf(1,:)))


GF_c    =   reshape(gf(end-1,:), size(x_r,1), size(y_r,1));

for ind = 1:size(gf,1)
    gf_c{ind} =  reshape(gf(ind,:), size(x_r,1), size(y_r,1));
end
%

% colormap(bone)

% p = surf(Xplot,-Yplot,data,'FaceAlpha','flat','AlphaDataMapping','scaled',...
%     'AlphaData',(abs(data)./max(abs(data),[],'all')).^.5);
% hold on
% [c, p] = contour3(Xplot,-Yplot,data,21,'LineColor','k');

%p.FaceAlpha = 0.5;

end
function plotArea(Xplot,Yplot, data)

%%
figure(2)
clf
rbcm = redblue(101);
rbcm = redbluecmap(11);
% [Xq,Yq] = meshgrid(1:3,linspace(1,11,11));
% rbcm = interp2(rbcm,Xq,Yq);
colormap(rbcm)

[c, p] = contourf(Xplot,-Yplot,data,linspace(-max(abs(data),[],'all'), max(abs(data),[],'all'),21),'LineColor','none','LineWidth',0.5);
diff(p.LevelList)
%p = surf(Xplot,Yplot,GF);
% p.AlphaData = abs(p.ZData)./max(p.ZData,[],'all');
% p.FaceAlpha= 'interp';
shading interp
%p.EdgeAlpha = 0.25;
xlabel('x/D');    ylabel('y/D');
cbar = colorbar; cbar.Label.String = 'b_z/(\mu_0 \sigma W B_y D)';


set(gca,'FontSize',16,'FontName','Times')
%axis([-lim(1) lim(1) -lim(2) lim(2) -1 1]*5)
axis([-1 1 -1 1 -1 1]*20)
[-max(abs(data),[],'all') max(abs(data),[],'all')]
caxis([-max(abs(data),[],'all') max(abs(data),[],'all')])
%caxis([-2 2])
title('B_z')
set(gcf,'Color','w')

view(2)
grid on
box on

%plot([-0.5 0.5],[0 0],'k-','LineWidth',3)
cleanfigure;

matlab2tikz('Figure1_field2.tex','width','2.5in','height','2.5in');
end

function plotDist(Xplot,Yplot, data)
%%
figure(1)

yyaxis left
cntr = 1;
clearvars p
inds = [15 18 21 24 28 31 52 73 76 80 83 86 89];
inds = inds(3:end-2);
cmap = cool(length(inds));
for idx = inds
    hold on
    if cntr == round(length(inds)/2)
        kk = '--';
    else
        kk='-';
    end
    %%p(cntr) = plot(Xplot(:,idx),mag_z(:,idx),kk,'Color',cmap(cntr,:)*0.8,'LineWidth',3);
    
    p(cntr) = plot(Xplot(:,idx),(data(:,idx)),'-','Color',cmap(cntr,:)*0.8,'LineWidth',2);
%     
%     p(cntr) = plot(diff([Xplot(1:end,idx)])./2+Xplot(1:end-1,idx),diff([data(:,idx)])./diff([Xplot(1:end,idx)])...
%         ,kk,'Color',cmap(cntr,:)*0.8,'LineWidth',2);
        
    %%plot(abs(Xplot(:,idx)),(data(:,idx)),kk,'Color',cmap(cntr,:)*0.8,'LineWidth',3);

    cntr = cntr+1;
end
hold off
set(gca,'XScale','log')
set(gca,'YScale','log')
legend(p,num2str(round(-Yplot(1,inds)',2)),'location','eastoutside')
grid on
box on
set(gcf,'Color','w')
set(gca,'FontSize',18)
xlabel('y/D')
ylabel('b_z/(\mu_0 \sigma W B_y D)')
%axis([1e-1 1e4 1e-6 2])
set(gca,'YColor','k')
matlab2tikz('Figure2_prof2.tex','width','2.5in','height','2.5in');

%%
figure(2)
clf

rbcm = redblue(101);
colormap(rbcm.^0.25)
%data =mag_z;
[c, p] = contourf(Xplot,-Yplot,data,10,'LineColor','k','LineWidth',0.5);
shading interp
xlabel('x/D');    ylabel('y/D');
cbar = colorbar; cbar.Label.String = 'b_z/(\mu_0 \sigma W B_y D)';
set(gca,'FontSize',16,'FontName','Times')
%axis([-lim(1) lim(1) -lim(2) lim(2) -1 1]*5)
axis([-1 1 -1 1 -1 1]*10)
caxis([-max(abs(data),[],'all') max(abs(data),[],'all')])

title('B_z')
set(gcf,'Color','w')
%
cntr = 1;
cmap = cool(length(inds));
for idx = inds
    hold on
    p(cntr) = plot(Xplot(:,idx),-Yplot(:,idx),'-','Color',cmap(cntr,:)*0.8,'LineWidth',2);
    cntr = cntr+1;
end
axis([-5 5 Yplot(1,inds(1))-1 Yplot(1,inds(end))+1 -1 1])
set(gcf,'Color','w')
end

function plotWandB(Xplot,Yplot, data,const,func)
%%
figure(1)
hold on
yyaxis right
cntr = 1;
clearvars p
inds = [15 18 21 24 28 31 52 73 76 80 83 86 89];
inds = inds(1:end);
cmap = cool(length(inds));
set(gca,'FontSize',18)

for idx = inds
    hold on
    if cntr == round(length(inds)/2)
        kk = '--';
    else
        kk='-.';
    end
    p(cntr) = plot(Xplot(:,idx),func.W(0, Xplot(:,idx), Yplot(1,idx)', ...
        const.sigma_0, const.U, const.D, const.k_w, const.z_offset)...
        ,kk,'Color',cmap(cntr,:)*0.8,'LineWidth',1);
    cntr = cntr+1;
end
hold off
set(gca,'XScale','log')
set(gca,'YScale','log')
legend(p,num2str(round(-Yplot(1,inds)',2)),'location','eastoutside')
grid on
box on
set(gcf,'Color','w')
set(gca,'FontSize',18)
axis([1e-1 1e4 1e-6 2])
xlabel('y/D')
ylabel('W/(\delta w)')
set(gca,'YColor','k')
end