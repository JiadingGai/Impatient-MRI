function dummy = display_reconBF(fname,Ny,Nx,Nz)

if( ~exist([fname '/output/']) )
   error_msg = [fname '/output/' ' not found!'];
   disp(error_msg);
   return;
end


if( exist([fname '/output/output_gpu_r.dat']) && exist([fname '/output/output_gpu_i.dat']) )
   load([fname '/output/output_gpu_r.dat']);
   load([fname '/output/output_gpu_i.dat']);
 if(Nz==1) %2D

   len = length(output_gpu_r);
   size_x = len^0.5;
   size_y = size_x;
   size_z = 1;
   gpu_output=reshape(output_gpu_r+1i*output_gpu_i,[size_x size_y size_z]);
   figure;
   hgpu =imagesc(abs(gpu_output));%axis image off;caxis([0,5e4])
   title('Gpuslice');
   colormap(gray);
   colorbar;
   scale = -1;
   %print -dpng 'Gpu.png';
 else
    output = zeros(Ny,Nx,Nz);
    for y=1:Ny
    for x=1:Nx
    for z=1:Nz
      lIndex = z + (x-1)*Nz + (y-1)*Nx*Nz;
      output(y,x,z) = abs(output_gpu_r(lIndex)+1i*output_gpu_i(lIndex));
    end
    end
    end

    figure;colormap(gray);
    for z=1:Nz
      imagesc(output(:,:,z));
      title('Gpuslice');
      colormap(gray);
      colorbar;
      pause(0.5);
    end
 end
 
end




if( exist([fname '/output/output_cpu_r.dat']) && exist([fname '/output/output_cpu_i.dat']) )
   load([fname '/output/output_cpu_r.dat']);
   load([fname '/output/output_cpu_i.dat']);
 if(Nz==1) %2D
   len = length(output_cpu_r);
   size_x = len^0.5;
   size_y = size_x;
   size_z = 1;
   cpu_output=reshape(output_cpu_r+1i*output_cpu_i,[size_x size_y size_z]);
   figure;
   hcpu =imagesc(abs(cpu_output));%axis image off;caxis([0,5e4])
   title('Cpuslice');
   colormap(gray);
   colorbar;
   scale = -1;
   %print -dpng 'Gpu.png';
 else
    output_cpu = zeros(Ny,Nx,Nz);
    for y=1:Ny
    for x=1:Nx
    for z=1:Nz
      lIndex = z + (x-1)*Nz + (y-1)*Nx*Nz;
      output_cpu(y,x,z) = abs(output_cpu_r(lIndex)+1i*output_cpu_i(lIndex));
    end
    end
    end

    figure;colormap(gray);
    for z=1:Nz
      imagesc(output_cpu(:,:,z));
      title('Cpuslice');
      colormap(gray);
      colorbar;
      pause(0.5);
    end
 end
 
end

