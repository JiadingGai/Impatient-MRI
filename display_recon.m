function dummy = display_recon(fname,Ny,Nx,Nz)
%[nRows,nCols,nDepths] = [Ny,Nx,Nz]
if(nargin<4)
    disp('Missing: the dimensions of the image.');
end

fp=fopen([fname '/out.file'],'r');
recon = fread(fp,'single');
len = length(recon);
recon_r = recon(1:len/2);
recon_i = recon(len/2+1:end);

if(Nz==1) % 2D
    N = (len/2) ^ 0.5;
    gpu_output = reshape(recon_r+1i*recon_i,[N,N]);
    figure;
    imagesc(abs(gpu_output));
    colormap(gray);
    colorbar;
    axis off;
    axis square;
    %caxis([0,0.8]);
    %save('GPUrecon.mat','gpu_output');
else
    output = zeros(Ny,Nx,Nz);
    for y=1:Ny
    for x=1:Nx
    for z=1:Nz
      lIndex = z + (x-1)*Nz + (y-1)*Nx*Nz;
      output(y,x,z) = abs(recon_r(lIndex)+1i*recon_i(lIndex));
    end
    end
    end

    figure;colormap(gray);
    for z=1:Nz
      imagesc(output(:,:,z));
      colormap(gray);
      colorbar;
      axis on;
      axis square;
      tmp = [int2str(z) '_Toe.png'];
      print('-dpng',tmp);
      pause(0.5);
    end
end



%[pathstr, name, ext] = fileparts(fname);
%pathstr(find(pathstr==filesep))='_';
%output_file_name = [pathstr(9:end) '_Toeplitz.png'];
%print('-dpng', output_file_name);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
