load 'recon_data_32x32x4.mat';
data_version = 1.0;
Nx = 32;
Ny = 32;
Nz = 4;

ncoils  = 4;
nslices = 1;

%%%%%%%%%%%%%%%%% No Change beyond this line unless you know what you are doing %%%%%%%%%%%%%%%

%% image data
image_file_size = (Nx*Ny*Nz); 
idata_r = single(zeros(size(mask)));
idata_i = single(zeros(size(mask)));
%coil_number argument in datawrite(.) is set to -1 if it is irrelevant to
%this file.
datawrite(data_version,Nx,Ny,Nz,-1,nslices,image_file_size,idata_r,'32x32x4/idata_r.dat');
datawrite(data_version,Nx,Ny,Nz,-1,nslices,image_file_size,idata_i,'32x32x4/idata_i.dat');

% mask
datawrite(data_version,Nx,Ny,Nz,-1,nslices,image_file_size,mask,'32x32x4/mask.dat');

%[iy,ix,iz] = meshgrid(1:Ny,1:Nx,1:Nz);
[ix,iz,iy] = meshgrid(1:Nx,1:Nz,1:Ny);
iy = (iy - Ny/2 - 1)./Ny;
ix = (ix - Nx/2 - 1)./Nx;
iz = (iz - Nz/2 - 1)./Nz;
iy = iy(:);
ix = ix(:);
iz = iz(:);
datawrite(data_version,Nx,Ny,Nz,-1,nslices,image_file_size,ix,'32x32x4/ix.dat');
datawrite(data_version,Nx,Ny,Nz,-1,nslices,image_file_size,iy,'32x32x4/iy.dat');
datawrite(data_version,Nx,Ny,Nz,-1,nslices,image_file_size,iz,'32x32x4/iz.dat');
%% field map
fm_3D = reshape(fm,Ny,Nx,Nz);%field map in col major order [y x z] to 2D
fm_3D_r_order = permute(fm_3D,[3,2,1]);% field map in row major order:[z x y]
fm_1D_r_order = fm_3D_r_order(:);
%fm_1D_r_order = zeros(size(fm_1D_r_order));%no field correction.
datawrite(data_version,Nx,Ny,Nz,-1,nslices,image_file_size,fm_1D_r_order,'32x32x4/fm.dat');

%% k-space data and time vector
num_rK = length(kdata_r) / ncoils;%nrows per coil
datawrite(data_version,Nx,Ny,Nz,ncoils,nslices,num_rK*ncoils,kdata_r,'32x32x4/kdata_r.dat');
datawrite(data_version,Nx,Ny,Nz,ncoils,nslices,num_rK*ncoils,kdata_i,'32x32x4/kdata_i.dat');

datawrite(data_version,Nx,Ny,Nz,-1,nslices,num_rK,kx,'32x32x4/kx.dat');
datawrite(data_version,Nx,Ny,Nz,-1,nslices,num_rK,ky,'32x32x4/ky.dat');
datawrite(data_version,Nx,Ny,Nz,-1,nslices,num_rK,kz,'32x32x4/kz.dat');

datawrite(data_version,Nx,Ny,Nz,-1,nslices,num_rK,t,'32x32x4/t.dat');
%% sensitivity map
for c=1:ncoils
   semp_r = sensi_r(1+(c-1)*Nx*Ny*Nz:c*Nx*Ny*Nz);
   semp_r_3D = reshape(semp_r,Ny,Nx,Nz);
   semp_r_3D_rorder = permute(semp_r_3D,[3,2,1]);
   semp_r_1D_rorder = semp_r_3D_rorder(:);
   sensi_r(1+(c-1)*Nx*Ny*Nz:c*Nx*Ny*Nz) = semp_r_1D_rorder;

   semp_i = sensi_i(1+(c-1)*Nx*Ny*Nz:c*Nx*Ny*Nz);
   semp_i_3D = reshape(semp_i,Ny,Nx,Nz);
   semp_i_3D_rorder = permute(semp_i_3D,[3,2,1]);
   semp_i_1D_rorder = semp_i_3D_rorder(:);
   sensi_i(1+(c-1)*Nx*Ny*Nz:c*Nx*Ny*Nz) = semp_i_1D_rorder;
end
datawrite(data_version,Nx,Ny,Nz,ncoils,nslices,image_file_size*ncoils,sensi_r,'32x32x4/sensi_r.dat');
datawrite(data_version,Nx,Ny,Nz,ncoils,nslices,image_file_size*ncoils,sensi_i,'32x32x4/sensi_i.dat');
