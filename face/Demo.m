fid = fopen('rawdata/2099')

I = fread(fid)

imagesc(reshape(I, 128, 128)')