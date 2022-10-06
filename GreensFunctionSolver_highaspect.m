clearvars
close all

gpud = 1;
for rep = 1:2
    tic
    clearvars -except rep gpud gf
    
    % GPU Activation
    if gpuDeviceCount>0 && rep>0
        gpud = 1;
    end
    
    % Gaussian Function
    gs = @(c, x, y, D) c*exp(-(x.^2+y.^2)/(2*D^2));
    D = 1;      % Diameter of jet
    c = 1;          % Amplitude of jet
    
    % Number of grid points
    %N = [1 1 8*4]*64*2^(rep-1);% [32 64 16]
    AR = 5;
    N = [1 1 AR]*64*2^(rep-1);

    % Limits of jet
    %lim = [1 1 8*4]*5*D; %[3 3 50];
    lim = [1 1 AR]*5*D; %[3 3 50];
    % Node spacing
    dxp = 2*lim./(N-1);
    
    if gpud == 1
        % GPU array for integration of volumes
        gpuDevice(gpuDeviceCount);
        X = gpuArray(single(linspace(-lim(1), lim(1), N(1))));
        Y = gpuArray(single(linspace(-lim(2), lim(2), N(2))));
        Z = gpuArray(single(linspace(-lim(3), lim(3), N(3))));
    else
        % CPU array for integration of volumes
        X = (linspace(-lim(1), lim(1), N(1)));
        Y = (linspace(-lim(2), lim(2), N(2)));
        Z = (linspace(-lim(3), lim(3), N(3)));
    end
    
    % Grid-mesh in the migration
    [xp, yp, zp] = ndgrid(  X, Y, Z);
    %     [FX,FY] = gradient( gs(c, x2d, y2d, sigma), dxp(1), dxp(2));
    
    %Optional Plotting of dw/dy and w
    if 0
        x2d = xp(:,:,1);  y2d(:,:,1) = yp(:,:,1);
        
        figure(5); clf
        p(1) = surf(x2d, y2d, 2*y2d.*gs(c, x2d, y2d, D));
        p(1).EdgeAlpha = 0.25;
        
        % Plot gaussian
        hold on
        p(2) = surf(x2d, y2d, gs(c, x2d, y2d, D));
        p(2).FaceAlpha = 0;
        p(2).EdgeAlpha = 0.1; p(2).EdgeColor = 'k';
        p(2).LineWidth = 1;
        xlabel('X');    ylabel('Y');    zlabel('C');
        cbar = colorbar; cbar.Label.String = 'C';
        axis([-lim(1) lim(1) -lim(2) lim(2) -2 2])
    end
    
    % Grid for r-points. Locations where we are solving for the magnetic
    % field in 1-D Arrays
    y_r = [-fliplr(logspace(-2,3,50)*D) 0 logspace(-2,3,50)*D]';
    x_r = [-fliplr(logspace(-2,3,49)*D) 0 logspace(-2,3,49)*D]';
    
    % Make the grid from the 1-D arrays
    [Xplot, Yplot] = ndgrid(x_r,y_r);
    r = zeros(numel(Xplot),3);
    r(:,1) = Xplot(:); r(:,2) = Yplot(:); r(:,3) = Xplot(:)*0;
    
    figure(2)
    hold on
    for idx = 1:size(r,1)
        r_rp = sqrt(    (r(idx,1)-xp).^2 +...
            (r(idx,2)-yp).^2 +...
            (r(idx,3)-zp).^2);
        if gpud == 1
            temp = gpuArray(single(yp.*gs(1,xp,yp,1)./r_rp./(4*pi)));
            % temp = gpuArray(single(FY./r_rp));
        else
            temp = (yp.*gs(1,xp,yp,1)./r_rp./(4*pi));
            % temp = (FY./r_rp);
        end
        temp = trapz(Z, temp, 3);
        temp = trapz(Y, temp, 2);
        temp = trapz(X, temp, 1);
        if gpud == 1
            gf(rep, idx) = gather(temp);
        else
            gf(rep, idx) = (temp);
        end
    end
    gather(X); gather(Y); gather(Z);
    %loglog(ray./sigma,gf(rep, :),'o-')
    %set(gca, 'YScale','log');
    %set(gca, 'XScale','log');
    grid on;
    box on;
    [rep  toc]

end
%% Plotting
% Grid convergence Error

%% Error plot between the two grid spacings. Spacing differ by 2x
figure(1)
clf
plot(abs(diff(gf,1)'./(abs(gf(2,:)'))))
hold on
plot(abs(gf(2,:)'))
hold on
plot(abs(gf(1,:)'))

        (std(diff(gf,1))./std(gf(rep,:)));
Xplot   =   reshape(Xplot, size(x_r,1), size(y_r,1));
Yplot   =   reshape(Yplot, size(x_r,1), size(y_r,1));
GF      =   reshape(gf(end,:), size(x_r,1), size(y_r,1));
GF_c    =   reshape(gf(end-1,:), size(x_r,1), size(y_r,1));

for ind = 1:size(gf,1)
gf_c{ind} =  reshape(gf(ind,:), size(x_r,1), size(y_r,1));
end
%%
figure(2)
clf
colormap(redbluecmap)

[c, p] = contourf(Xplot,Yplot,GF,41,'LineColor','none');
%p = surf(Xplot,Yplot,GF);
% p.AlphaData = abs(p.ZData)./max(p.ZData,[],'all');
% p.FaceAlpha= 'interp';
shading interp
%p.EdgeAlpha = 0.25;
xlabel('x/D');    ylabel('y/D');
cbar = colorbar; cbar.Label.String = 'b_z/(\mu_0 \sigma W B_y D)';

set(gca,'FontSize',16,'FontName','Times')
axis([-lim(1) lim(1) -lim(2) lim(2) -1 1]*2)
view(2)
grid on
box on
cleanfigure;
matlab2tikz('Figure1_field.tex','width','2.5in','height','2.5in');
%%
figure(5)
clf
colormap(redbluecmap)
x2d = xp(:,:,1);  y2d = yp(:,:,1);
[p] = surf(gather(x2d),gather(y2d),-gs(1,x2d,y2d,1));
p.Parent.CLim = [-1 1];
p.AlphaData = abs(p.ZData)./max(abs(p.ZData)+0.1,[],'all');
p.FaceAlpha= 'interp';
p.FaceColor = 'interp';
p.EdgeAlpha = 0.01;
p.EdgeColor ='interp';

xlabel('x/D');    ylabel('y/D');
cbar = colorbar; cbar.Label.String = 'b_z/(\mu_0 \sigma W B_y D)';
cbar.Limits=[-1 1];
set(gca,'FontSize',16,'FontName','Times')
axis([-lim(1) lim(1) -lim(2) lim(2) -1 0])
view(2)
grid on
box on
cleanfigure;
%matlab2tikz('Figure1_field.tex','width','2.5in','height','2.5in');

%%
figure(3)
cmap2 = lines(3);
clf
hold on

inds = (1:size(y_r))+length(y_r)*(0:size(y_r)-1);
% Y-axis NS plot
yyaxis left
% p(1) = plot(y_r,GF(floor(end/2)+1,1:end),'-o','Color',cmap2(1,:),'LineWidth',1,...
%     'MarkerSize',4);
% p(2)= plot(y_r,GF_c(floor(end/2)+1,1:end),'-x','Color',cmap2(1,:),'LineWidth',1,...
%     'MarkerSize',4);
p(1) = plot(y_r,GF(floor(end/2)+1,1:end),'-o','Color',cmap2(1,:),'LineWidth',1,...
    'MarkerSize',4);
p(2)= plot(y_r,GF_c(floor(end/2)+1,1:end),'-x','Color',cmap2(1,:),'LineWidth',1,...
    'MarkerSize',4);
y_0 = logspace(-2, log10(sqrt(2)));
y_1 = sqrt(2):lim(3);
y_2 = lim(3):1000;
plot(y_0,(y_0)/2,'--','Color','k','LineWidth',2,'MarkerSize',4);
text(0.03,0.5,'(r/D)','FontSize',18);

plot(y_1,(1./(y_1)),'--','Color','k','LineWidth',2,'MarkerSize',4);
text(50,.3,'(r/D)^{-1}','FontSize',18);

plot(y_2,(lim(3)./(y_2).^2),'--','Color','k','LineWidth',2,'MarkerSize',4);
text(500,0.01,'(r/D)^{-2}','FontSize',18);
set(gca, 'YScale','log');
set(gca, 'XScale','log');
ylabel('b_z/(\mu_0 \sigma W B_y D)')

axis([1e-2 3e3 1e-4 3])

% % X-axis EW plot
% % plot(x_r,GF(1:end,floor(end/2)+1),'-o','Color',cmap2(2,:),'LineWidth',1,...
% %             'MarkerSize',4)
% 
% 
% % Diagonal
% % p(3)= plot(sign(Xplot(inds)).*sqrt(Xplot(inds).^2+Yplot(inds).^2),GF(inds),'-o',...
% %     'Color',cmap2(3,:),'LineWidth',1,'MarkerSize',4);
% % % Diagonal
% % p(4) = plot(sign(Xplot(inds)).*sqrt(Xplot(inds).^2+Yplot(inds).^2),GF_c(inds),'-x',...
% %     'Color',cmap2(3,:),'LineWidth',1,'MarkerSize',4);
% %plot(y_1,(0.5./(y_1)),'--','Color',cmap2(3,:),'LineWidth',2,'MarkerSize',4)
% %plot(y_2,(3./(y_2)).^2,'--','Color',cmap2(3,:),'LineWidth',2,'MarkerSize',4)

yyaxis right

p(2)= plot(y_r,gs(1,0,y_r,1),'-','LineWidth',2,'Color',cmap2(2,:),'MarkerSize',4);

set(gca,'FontSize',18, 'FontName','Times')
xlabel('r/D')
ylabel('W')

grid on
box on
%legend(p, 'b_z : NS','b_z : SW-NE? ' )
set(gca, 'YScale','log');
set(gca, 'XScale','log');
axis([1e-2 3e3 1e-4 3])
matlab2tikz('Figure2_profile.tex','width','3in','height','2in');
%% Rankine Vortex
figure(4)
cmap2 = lines(3);
clf
hold on

inds = (1:size(y_r))+length(y_r)*(0:size(y_r)-1);
% Y-axis NS plot
p(1) = plot(y_r,GF(floor(end/2)+1,1:end),'-o','Color',cmap2(1,:),'LineWidth',1,...
    'MarkerSize',4);
p(2)= plot(y_r,GF_c(floor(end/2)+1,1:end),'-x','Color',cmap2(1,:),'LineWidth',1,...
    'MarkerSize',4);
y_0 = logspace(-2, log10(sqrt(2))); %y_0 = [fliplr(-y_0) y_0];
y_1 = logspace(log10(sqrt(2)),log10(lim(3)),100);
y_2 = lim(3):1000;
plot(y_0,(y_0)./2,'--','Color','k','LineWidth',2,'MarkerSize',4);

plot(y_1,(1./(y_1)),'--','Color','k','LineWidth',2,'MarkerSize',4);

plot(y_2,(sqrt(lim(3))./(y_2)).^2,'--','Color','k','LineWidth',2,'MarkerSize',4);
%set(gca, 'XScale','log');
ylabel('b_z/(\mu_0 \sigma W B_y D)')

xlabel('r/D')
grid on
box on
axis([0 10 0 1.5])
cleanfigure
matlab2tikz('Figure4_rankine.tex','width','3in','height','2in');
%% Sine function
F = scatteredInterpolant(Xplot(:),Yplot(:),double(GF(:)));
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
F = scatteredInterpolant(Xplot(:),Yplot(:),Yplot(:).*gs(1,Xplot(:),Yplot(:),1));
ff = F(cos(thetas), sin(thetas));
plot(thetas',ff./max(ff),'x','LineWidth',1)
legend('Sinusoid','Numerical','dw/dy')


matlab2tikz('Figure3_theta.tex','width','3in','height','2in');

%% H study
figure(3)
cmap2 = lines(3);
clf
hold on

inds = (1:size(y_r))+length(y_r)*(0:size(y_r)-1);
% Y-axis NS plot
yyaxis left
for ind = 1:length(gf_c)
p(1) = plot(y_r,gf_c{ind}(floor(end/2)+1,1:end),'-o','Color',cmap2(1,:),'LineWidth',1,...
    'MarkerSize',4);
end
plot(y_0,(y_0)/2,'--','Color','k','LineWidth',2,'MarkerSize',4);
text(0.03,0.5,'(r/D)','FontSize',18);

plot(y_1,(1./(y_1)),'--','Color','k','LineWidth',2,'MarkerSize',4);
text(50,.3,'(r/D)^{-1}','FontSize',18);

plot(y_2,(sqrt(1*lim(3))./(y_2)).^2,'--','Color','k','LineWidth',2,'MarkerSize',4);
text(500,0.01,'(r/D)^{-2}','FontSize',18);
set(gca, 'YScale','log');
set(gca, 'XScale','log');
ylabel('b_z/(\mu_0 \sigma W B_y D)')
Ys = logspace(-1,3);
plot(Ys,Ys.*(lim(3))./(Ys.^2.*sqrt(Ys.^2+lim(3).^2)),'-','Color','k','LineWidth',2,'MarkerSize',4);

axis([1e-2 3e3 1e-4 3])

% % X-axis EW plot
% % plot(x_r,GF(1:end,floor(end/2)+1),'-o','Color',cmap2(2,:),'LineWidth',1,...
% %             'MarkerSize',4)
% 
% 
% % Diagonal
% % p(3)= plot(sign(Xplot(inds)).*sqrt(Xplot(inds).^2+Yplot(inds).^2),GF(inds),'-o',...
% %     'Color',cmap2(3,:),'LineWidth',1,'MarkerSize',4);
% % % Diagonal
% % p(4) = plot(sign(Xplot(inds)).*sqrt(Xplot(inds).^2+Yplot(inds).^2),GF_c(inds),'-x',...
% %     'Color',cmap2(3,:),'LineWidth',1,'MarkerSize',4);
% %plot(y_1,(0.5./(y_1)),'--','Color',cmap2(3,:),'LineWidth',2,'MarkerSize',4)
% %plot(y_2,(3./(y_2)).^2,'--','Color',cmap2(3,:),'LineWidth',2,'MarkerSize',4)

yyaxis right

p(2)= plot(y_r,gs(1,0,y_r,1),'-','LineWidth',2,'Color',cmap2(2,:),'MarkerSize',4);

set(gca,'FontSize',18, 'FontName','Times')
xlabel('r/D')
ylabel('W')

grid on
box on
%legend(p, 'b_z : NS','b_z : SW-NE? ' )
set(gca, 'YScale','log');
set(gca, 'XScale','log');
axis([1e-2 3e3 1e-4 3])
matlab2tikz('Figure2_profile.tex','width','3in','height','2in');
