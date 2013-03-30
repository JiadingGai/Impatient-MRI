/*
(C) Copyright 2010 The Board of Trustees of the University of Illinois.
All rights reserved.

Developed by:

                     IMPACT & MRFIL Research Groups
                University of Illinois, Urbana Champaign

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal with the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimers.

Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimers in the documentation
and/or other materials provided with the distribution.

Neither the names of the IMPACT Research Group, MRFIL Research Group, the
University of Illinois, nor the names of its contributors may be used to
endorse or promote products derived from this Software without specific
prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
THE SOFTWARE.
*/

/*****************************************************************************

    File Name   [mmio.cpp]

    Synopsis    [Matrix Market I/O library for ANSI C. This library is used
        by sparse matrix-vector multiplication.
        See http://math.nist.gov/MatrixMarket for details.]

    Description []

    Revision    [0.1; Initial build; Xiao-Long Wu, ECE UIUC]
    Revision    [0.1.1; Revised to remove g++ warnings, Code cleaning;
                 Xiao-Long Wu, ECE UIUC]
    Date        [03/23/2010]

 *****************************************************************************/

#ifndef MMIO_H
#define MMIO_H

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <iostream>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Macro definitions                                                        */
/*---------------------------------------------------------------------------*/

#define MM_MAX_LINE_LENGTH      1025
#define MatrixMarketBanner      "%%MatrixMarket"  // Note: There are two "%%".
#define MM_MAX_TOKEN_LENGTH     64

typedef char MM_typecode[4];

/********************* Matrix Market error codes ***************************/

#define MM_COULD_NOT_READ_FILE  11
#define MM_PREMATURE_EOF        12
#define MM_NOT_MTX              13
#define MM_NO_HEADER            14
#define MM_UNSUPPORTED_TYPE     15
#define MM_LINE_TOO_LONG        16
#define MM_COULD_NOT_WRITE_FILE 17

/******************** Matrix Market internal definitions ********************

   MM_matrix_typecode: 4-character sequence

                    object     sparse/       data     storage
                    dense       type        scheme

   string position:	 [0]        [1]			[2]         [3]

   Matrix typecode:  M(atrix)  C(oord)		R(eal)      G(eneral)
                               A(array)     C(omplex)   H(ermitian)
                                            P(attern)   S(ymmetric)
                                            I(nteger)	K(kew)

 ***********************************************************************/

#define MM_MTX_STR          "matrix"
#define MM_ARRAY_STR        "array"
#define MM_DENSE_STR        "array"
#define MM_COORDINATE_STR   "coordinate"
#define MM_SPARSE_STR       "coordinate"
#define MM_COMPLEX_STR      "complex"
#define MM_REAL_STR         "real"
#define MM_INT_STR          "integer"
#define MM_GENERAL_STR      "general"
#define MM_SYMM_STR         "symmetric"
#define MM_HERM_STR         "hermitian"
#define MM_SKEW_STR         "skew-symmetric"
#define MM_PATTERN_STR      "pattern"

/*---------------------------------------------------------------------------*/
/*  Function prototypes                                                      */
/*---------------------------------------------------------------------------*/

    char *
mm_typecode_to_str(MM_typecode matcode);
    char *
mm_strdup(const char *s);
    int
mm_read_banner(FILE *f, MM_typecode *matcode);
    int
mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz);
    int
mm_read_mtx_array_size(FILE *f, int *M, int *N);
    int
mm_read_mtx_crd(
    char *fname, int *M, int *N, int *nz, int **I, int **J,
    FLOAT_T **val, MM_typecode *matcode);
    int
mm_write_banner(FILE *f, MM_typecode matcode);
    int
mm_write_mtx_crd_size(FILE *f, int M, int N, int nz);
    int
mm_write_mtx_array_size(FILE *f, int M, int N);

/********************* MM_typecode query fucntions ***************************/

#define mm_is_matrix(typecode)      ((typecode)[0] == 'M')

#define mm_is_sparse(typecode)      ((typecode)[1] == 'C')
#define mm_is_coordinate(typecode)  ((typecode)[1] == 'C')
#define mm_is_dense(typecode)       ((typecode)[1] == 'A')
#define mm_is_array(typecode)       ((typecode)[1] == 'A')

#define mm_is_complex(typecode)     ((typecode)[2] == 'C')
#define mm_is_real(typecode)        ((typecode)[2] == 'R')
#define mm_is_pattern(typecode)     ((typecode)[2] == 'P')
#define mm_is_integer(typecode)     ((typecode)[2] == 'I')

#define mm_is_symmetric(typecode)   ((typecode)[3] == 'S')
#define mm_is_general(typecode)     ((typecode)[3] == 'G')
#define mm_is_skew(typecode)        ((typecode)[3] == 'K')
#define mm_is_hermitian(typecode)   ((typecode)[3] == 'H')

int mm_is_valid(MM_typecode matcode); /* too complex for a macro */

/********************* MM_typecode modify fucntions ***************************/

#define mm_set_matrix(typecode)     ((*typecode)[0] = 'M')
#define mm_set_coordinate(typecode) ((*typecode)[1] = 'C')
#define mm_set_array(typecode)      ((*typecode)[1] = 'A')
#define mm_set_dense(typecode)      mm_set_array(typecode)
#define mm_set_sparse(typecode)     mm_set_coordinate(typecode)

#define mm_set_complex(typecode)    ((*typecode)[2] = 'C')
#define mm_set_real(typecode)       ((*typecode)[2] = 'R')
#define mm_set_pattern(typecode)    ((*typecode)[2] = 'P')
#define mm_set_integer(typecode)    ((*typecode)[2] = 'I')

#define mm_set_symmetric(typecode)  ((*typecode)[3] = 'S')
#define mm_set_general(typecode)    ((*typecode)[3] = 'G')
#define mm_set_skew(typecode)       ((*typecode)[3] = 'K')
#define mm_set_hermitian(typecode)  ((*typecode)[3] = 'H')

#define mm_clear_typecode(typecode) ((*typecode)[0] = (*typecode)[1] = \
                         (*typecode)[2] = ' ', (*typecode)[3] = 'G')

#define mm_initialize_typecode(typecode) mm_clear_typecode(typecode)

/*  high level routines */

int mm_write_mtx_crd(char fname[], int M, int N, int nz, int I[], int J[],
                     FLOAT_T val[], MM_typecode matcode);
int mm_read_mtx_crd_data(FILE * f, int M, int N, int nz, int I[], int J[],
                         FLOAT_T val[], MM_typecode matcode);
int mm_read_mtx_crd_entry(FILE *f, int *I, int *J, FLOAT_T *real, FLOAT_T *img,
                          MM_typecode matcode);

int mm_read_unsymmetric_sparse(const char *fname, int *M_, int *N_, int *nz_,
                               FLOAT_T **val_, int **I_, int **J_);

    void
readMtxFile(
    const char *mtx_fn,         // mtx file name
    int *AW,                    // matrix width
    int *AH,                    // matrix height
    int **AI,                   // row index array
    int **AJ,                   // column index array
    FLOAT_T **AVal,          // value array
    int *ele_num_A,             // number of nonzero elements in matrix A
    const int if_silent         // If not show messages
    );

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

#endif // MMIO_H

