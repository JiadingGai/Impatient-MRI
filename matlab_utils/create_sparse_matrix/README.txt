This sub-folder contains the Matlab scripts that are used to generate 
the regularization sparse matrices (i.e., those c.rmtx files in each
data sub-directory, residing along with other input *.dat files). 
The sparse matrices are first generated in memory using Matlab script 
createDWithPeriodicBoundary.m, then written to hard disk via mmwrite.m. 
c.rmtx is a format invented by Matrix Market. 

For more details about *.rmtx, check out:

         http://math.nist.gov/MatrixMarket/formats.html

Section 1. 2D case:

The following example is how we create c.rmtx:

N1=512;% size of one dimension
N2=N1;% assume recon image is square.
[D,Dp] = createDWithPeriodicBoundary(N1,N2);
mmwrite('c.rmtx',D);

Section 2. 3D case:

The finite difference sparse matrices that createDWithPeriodicBoundary3D 
creates assume that image vectors those sparse matrices operate on are
vectorized in a column-major order. In contrast, IMPATIENT code follows
a row-major order convention. To make createDWithPeriodicBoundary3D produce
sparse matrices usable to IMPATIENt, we need to pick the right order of 
input argument list (i.e., image dimensions) to createDWithPeriodicBoundary3D.m.

Here's a 3D example of create a c.rmtx for IMPATIENT 
using createDWithPeriodicBoundary3D.m:

3D Image of size Nx=64, Ny=64, Nz=16;
IMPATIENT uses row-major order: Ny x Nx x Nz, with the z dimension contiguous.
createDWithPeriodicBoundary3D.m uses column-major order: Ny x Nx x Nz, with the y dimension contiguous. 

Ny=64;% size of one dimension
Nx=64;
Nz=16;
%Re-ordered dimensions as input to make it generate a 
%sparse matrix corresponding to a column-major order 
%vectorized vector
[D,Dp] = createDWithPeriodicBoundary3D(Nz,Nx,Ny);
mmwrite('c.rmtx',D);


