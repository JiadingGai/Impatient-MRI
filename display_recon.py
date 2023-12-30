import struct
import numpy as np
import sys

def read_binary(fileName):
    with open(fileName, 'rb') as fp:
        data = fp.read()

    FORMAT='f'
    recon = []
    for fields in struct.iter_unpack(FORMAT, data):
        float_value, = fields
        recon.append(float_value)
        # printf(f"Float: {float_value}")
    return recon

def read_text(fileName):
    recon = []
    with open(fileName, 'r') as fp:
        for data in fp:
            recon.append(float(data))
    return recon

def render_image(recon_r, recon_i, outfile='__output.png'):
    assert len(recon_r) == len(recon_i)
    Len = len(recon_r)
    N = int(np.sqrt(Len))
    out = np.empty(shape=[Len,], dtype=np.csingle)
    out.real = recon_r
    out.imag = recon_i
    out_image = np.absolute(out.reshape([N, N]))
    import matplotlib.pyplot as plt
    plt.imsave(outfile, out_image)

# compute and return the real and imaginary parts of recon separately.
def compute_reim(dir_format):
   # FIXME:
   # text format stores real/imag parts separately in text under output/
   # binary format combines real/imag parts in one file: out.file
   # so we use the heuristics of whether "output" is presented in the out
   # dir string to distinguish between text formatted output and binary
   # formatted output. very hacky!!
   format_is_text = True if 'output' in dir_format else False
   format_is_binary = not format_is_text 

   if format_is_text:
     recon_r = read_text(dir_format + "/" + "output_gpu_r.dat")
     recon_i = read_text(dir_format + "/" + "output_gpu_i.dat")
     return recon_r, recon_i

   assert format_is_binary == True
   recon = read_binary(dir_format + "/" + "out.file")
   Len = int(len(recon) / 2)
   recon_r = recon[:Len]
   recon_i = recon[Len:]
   return recon_r, recon_i

# text format: recon_dir = "mriData/64x64x1/output"
# binary format: recon_dir = "mriData/384x384x1-32coils/"

assert len(sys.argv) == 2
recon_dir = sys.argv[1]
recon_r, recon_i = compute_reim(recon_dir)
render_image(recon_r, recon_i, "__output_gpu.png")
