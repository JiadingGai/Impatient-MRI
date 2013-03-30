function datawrite(version,xDimension,yDimension,zDimension,coil_number,slice_number,file_size,data,filename)
fp = fopen(filename,'w');

fprintf(fp,'version = %f\n',version);
fprintf(fp,'xDimension = %d\n',xDimension);
fprintf(fp,'yDimension = %d\n',yDimension);
fprintf(fp,'zDimension = %d\n',zDimension);
if(-1~=coil_number)
    %if coil_number is relevant to this file.
    fprintf(fp,'coil_number = %d\n',coil_number);
else
    % if coil_number is irrelevant, we can safely
    % pretend that coil_number = 1. Also, there's 
    % no need to specify coil_number in this file
    coil_number = 1;
end
fprintf(fp,'slice_number = %d\n',slice_number);
fprintf(fp,'file_size = %d\n',file_size);

file_size_per_coil = file_size / coil_number;
for i=1:coil_number  
  fprintf(fp,'// Coil No. %d\n',i);
  fprintf(fp,'Binary_Size = %d\n',file_size_per_coil);
  fprintf(fp,'Binary:\n');
  fwrite(fp,data(1+(i-1)*file_size_per_coil:i*file_size_per_coil),'single');
end

fprintf(fp,'EOF\n');% end of file

status = fclose(fp);
if (status == -1)
    error('file closing fails');
end

