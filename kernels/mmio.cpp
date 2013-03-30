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
    Revision    [0.1.1; Revised to remove compiler warnings, Code cleaning; 
                 Xiao-Long Wu, ECE UIUC]
    Date        [03/23/2010]

*****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <limits.h>
// Project header files
#include <xcpplib_process.h>
#include "tools.h"
#include "structures.h"
#include "mmio.h"

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Namespace used                                                           */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*  Function prototypes                                                      */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*  Function definitions                                                     */
/*---------------------------------------------------------------------------*/

    int 
mm_read_unsymmetric_sparse(
    const char *fname, int *M_, int *N_, int *nz_,
    FLOAT_T **val_, int **I_, int **J_)
{
    FILE *f;
    MM_typecode matcode;
    int M, N, nz;
    int i;
    FLOAT_T *val;
    int *I, *J;

    f = openFile(fname, "r", !DEBUG_MODE);

    if (mm_read_banner(f, &matcode) != 0) {
        printf("mm_read_unsymetric: Could not process Matrix Market banner ");
        printf(" in file [%s]\n", fname);
        return -1;
    }

    if ( !(mm_is_real(matcode) && mm_is_matrix(matcode) &&
           mm_is_sparse(matcode))) {
        fprintf(stderr, "Sorry, this application does not support ");
        fprintf(stderr, "Market Market type: [%s]\n",
                mm_typecode_to_str(matcode));
        return -1;
    }

    /* find out size of sparse matrix: M, N, nz .... */

    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
        fprintf(stderr, "*** Error at line %d of %s:\n", __LINE__, __FILE__);
        fprintf(stderr, "*** Error: Could not parse matrix size.\n");
        return -1;
    }

    *M_ = M;
    *N_ = N;
    *nz_ = nz;

    /* reseve memory for matrices */

    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    val = (FLOAT_T *) malloc(nz * sizeof(FLOAT_T));
    double val_temp; // always read in a double and convert later if necessary
    *val_ = val;
    *I_ = I;
    *J_ = J;

    /* NOTE: when reading in floats, ANSI C requires the use of the "l"   */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i = 0; i < nz; i++) {
        if (fscanf(f, "%d %d %lf", &I[i], &J[i], &val_temp) == 3) {
            val[i] = (FLOAT_T)val_temp;
            I[i]--;  /* adjust from 1-based to 0-based */
            J[i]--;
        } else {
            fprintf(stderr, "*** Error at line %d of %s:\n", 
                __LINE__, __FILE__);
            fprintf(stderr,
                "*** Error: Missing elements on reading data from %s.\n",
                fname);
            exit(1);
        }
    }
    fclose(f);

    return 0;
}

    int 
mm_is_valid(MM_typecode matcode)
{
    if (!mm_is_matrix(matcode)) return 0;
    if (mm_is_dense(matcode) && mm_is_pattern(matcode)) return 0;
    if (mm_is_real(matcode) && mm_is_hermitian(matcode)) return 0;
    if (mm_is_pattern(matcode) && (mm_is_hermitian(matcode) ||
                                   mm_is_skew(matcode))) return 0;
    return 1;
}

    int 
mm_read_banner(FILE *f, MM_typecode *matcode)
{
    char line[MM_MAX_LINE_LENGTH];
    char banner[MM_MAX_TOKEN_LENGTH];
    char mtx[MM_MAX_TOKEN_LENGTH];
    char crd[MM_MAX_TOKEN_LENGTH];
    char data_type[MM_MAX_TOKEN_LENGTH];
    char storage_scheme[MM_MAX_TOKEN_LENGTH];
    char *p;
    int error_code = 0;

    mm_clear_typecode(matcode);

    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL) {
        error_code = MM_PREMATURE_EOF;
        goto ERROR_mm_read_banner;
    }

    if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type,
               storage_scheme) != 5) {
        error_code = MM_PREMATURE_EOF;
        goto ERROR_mm_read_banner;
    }

    for (p = mtx; *p != '\0'; *p = tolower(*p), p++) ; /* convert to lower case */
    for (p = crd; *p != '\0'; *p = tolower(*p), p++) ;
    for (p = data_type; *p != '\0'; *p = tolower(*p), p++) ;
    for (p = storage_scheme; *p != '\0'; *p = tolower(*p), p++) ;

    /* check for banner */
    if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0) {
        error_code = MM_NO_HEADER;
        goto ERROR_mm_read_banner;
    }

    /* first field should be "mtx" */
    if (strcmp(mtx, MM_MTX_STR) != 0) {
        error_code = MM_UNSUPPORTED_TYPE;
        goto ERROR_mm_read_banner;
    }
    mm_set_matrix(matcode);

    /* second field describes whether this is a sparse matrix (in coordinate
       storgae) or a dense array */

    if (strcmp(crd, MM_SPARSE_STR) == 0)
        mm_set_sparse(matcode);
    else
    if (strcmp(crd, MM_DENSE_STR) == 0)
        mm_set_dense(matcode);
    else {
        error_code = MM_UNSUPPORTED_TYPE;
        goto ERROR_mm_read_banner;
    }

    /* third field */

    if (strcmp(data_type, MM_REAL_STR) == 0)
        mm_set_real(matcode);
    else
    if (strcmp(data_type, MM_COMPLEX_STR) == 0)
        mm_set_complex(matcode);
    else
    if (strcmp(data_type, MM_PATTERN_STR) == 0)
        mm_set_pattern(matcode);
    else
    if (strcmp(data_type, MM_INT_STR) == 0)
        mm_set_integer(matcode);
    else {
        error_code = MM_UNSUPPORTED_TYPE;
        goto ERROR_mm_read_banner;
    }

    /* fourth field */

    if (strcmp(storage_scheme, MM_GENERAL_STR) == 0)
        mm_set_general(matcode);
    else
    if (strcmp(storage_scheme, MM_SYMM_STR) == 0)
        mm_set_symmetric(matcode);
    else
    if (strcmp(storage_scheme, MM_HERM_STR) == 0)
        mm_set_hermitian(matcode);
    else
    if (strcmp(storage_scheme, MM_SKEW_STR) == 0)
        mm_set_skew(matcode);
    else {
        error_code = MM_UNSUPPORTED_TYPE;
        goto ERROR_mm_read_banner;
    }

    return 0;

ERROR_mm_read_banner:

    fprintf(stderr, "*** Error: Failed to read matrix file.\n");
    switch (error_code)
    {
    case MM_COULD_NOT_READ_FILE:
        fprintf(stderr, "Reason: Could not read file.\n");
        break;
    case MM_PREMATURE_EOF:
        fprintf(stderr, "Reason: Premature EOF.\n");
        break;
    case MM_NOT_MTX:
        fprintf(stderr, "Reason: Not MTX format.\n");
        break;
    case MM_NO_HEADER:
        fprintf(stderr, "Reason: Empty or wrong header is defined.\n");
        break;
    case MM_UNSUPPORTED_TYPE:
        fprintf(stderr, "Reason: Unsupported type.\n");
        break;
    case MM_LINE_TOO_LONG:
        fprintf(stderr, "Reason: Line is too long.\n");
        break;
    case MM_COULD_NOT_WRITE_FILE:
        fprintf(stderr, "Reason: Could not write to file.\n");
        break;
    default:
        fprintf(stderr,
                "*** Fatal error: Undefined error code at %d of %s.\n",
                __LINE__, __FILE__);
        exit(1);
    }

    return error_code;
}

    int 
mm_write_mtx_crd_size(FILE *f, int M, int N, int nz)
{
    // Modified by Xiao-Long Wu, 10/25/2008
//    if (fprintf(f, "%d %d %d\n", M, N, nz) != 3)
//        return MM_COULD_NOT_WRITE_FILE;

    if (fprintf(f, "%12d %12d %25d\n", M, N, nz) != 3)
        return MM_COULD_NOT_WRITE_FILE;
    else
        return 0;
}

    int 
mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz)
{
    char line[MM_MAX_LINE_LENGTH];
    int num_items_read;

    /* set return null parameter values, in case we exit with errors */
    *M = *N = *nz = 0;

    /* now continue scanning until you reach the end-of-comments */
    do
    {
        if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
            return MM_PREMATURE_EOF;
    }
    while (line[0] == '%');

    /* line[] is either blank or has M,N, nz */
    if (sscanf(line, "%d %d %d", M, N, nz) == 3)
        return 0;

    else
        do
        {
            num_items_read = fscanf(f, "%d %d %d", M, N, nz);
            if (num_items_read == EOF) return MM_PREMATURE_EOF;
        }
        while (num_items_read != 3);

    return 0;
}

    int 
mm_read_mtx_array_size(FILE *f, int *M, int *N)
{
    char line[MM_MAX_LINE_LENGTH];
    int num_items_read;
    /* set return null parameter values, in case we exit with errors */
    *M = *N = 0;

    /* now continue scanning until you reach the end-of-comments */
    do
    {
        if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
            return MM_PREMATURE_EOF;
    }
    while (line[0] == '%');

    /* line[] is either blank or has M,N, nz */
    if (sscanf(line, "%d %d", M, N) == 2)
        return 0;

    else /* we have a blank line */
        do
        {
            num_items_read = fscanf(f, "%d %d", M, N);
            if (num_items_read == EOF) return MM_PREMATURE_EOF;
        }
        while (num_items_read != 2);

    return 0;
}

    int 
mm_write_mtx_array_size(FILE *f, int M, int N)
{
    if (fprintf(f, "%d %d\n", M, N) != 2)
        return MM_COULD_NOT_WRITE_FILE;
    else
        return 0;
}

/*-------------------------------------------------------------------------*/

/******************************************************************/
/* use when I[], J[], and val[]J, and val[] are already allocated */
/******************************************************************/

    int 
mm_read_mtx_crd_data(
    FILE *f, int M, int N, int nz, int I[], int J[],
    FLOAT_T val[], MM_typecode matcode)
{
    int i;
    double val_temp1; // always read in a double and convert later if necessary
    double val_temp2; // always read in a double and convert later if necessary
    if (mm_is_complex(matcode)) {
        for (i = 0; i < nz; i++) {
            if (fscanf(f, "%d %d %lg %lg", &I[i], &J[i], &val_temp1, &val_temp2) != 4) {
                return MM_PREMATURE_EOF;
            }
            val[2 * i] = (FLOAT_T)val_temp1;
            val[2 * i + 1] = (FLOAT_T)val_temp2;
        }
    } else if (mm_is_real(matcode)) {
        for (i = 0; i < nz; i++) {
            if (fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val_temp1) != 3) {
                return MM_PREMATURE_EOF;
            }
            val[i] = val_temp1;
        }
    } else if (mm_is_pattern(matcode)) {
        for (i = 0; i < nz; i++)
            if (fscanf(f, "%d %d", &I[i], &J[i]) != 2) return MM_PREMATURE_EOF;
    } else
        return MM_UNSUPPORTED_TYPE;

    return 0;
}

    int 
mm_read_mtx_crd_entry(
    FILE *f, int *I, int *J,
    FLOAT_T *real, FLOAT_T *imag, MM_typecode matcode)
{
    double real_temp; // always read in a double and convert later if necessary
    double imag_temp; // always read in a double and convert later if necessary
    if (mm_is_complex(matcode)) {
        if (fscanf(f, "%d %d %lg %lg", I, J, &real_temp, &imag_temp) != 4) {
            return MM_PREMATURE_EOF;
        }
        *real = (FLOAT_T) real_temp;
        *imag = (FLOAT_T) imag_temp;
    } else if (mm_is_real(matcode)) {
        if (fscanf(f, "%d %d %lg\n", I, J, &real_temp) != 3) {
            return MM_PREMATURE_EOF;
        }
        *real = (FLOAT_T)real_temp;
    } else if (mm_is_pattern(matcode)) {
        if (fscanf(f, "%d %d", I, J) != 2) return MM_PREMATURE_EOF;
    } else
        return MM_UNSUPPORTED_TYPE;

    return 0;
}

/************************************************************************
    mm_read_mtx_crd()  fills M, N, nz, array of values, and return
                        type code, e.g. 'MCRS'

                        if matrix is complex, values[] is of size 2*nz,
                            (nz pairs of real/imaginary values)
************************************************************************/

    int 
mm_read_mtx_crd(
    char *fname, int *M, int *N, int *nz, int **I, int **J,
    FLOAT_T **val, MM_typecode *matcode)
{
    int ret_code;
    FILE *f;

    if (strcmp(fname, "stdin") == 0) f = stdin;
    else
    if ((f = fopen(fname, "r")) == NULL)
        return MM_COULD_NOT_READ_FILE;

    if ((ret_code = mm_read_banner(f, matcode)) != 0)
        return ret_code;

    if (!(mm_is_valid(*matcode) && mm_is_sparse(*matcode) &&
          mm_is_matrix(*matcode)))
        return MM_UNSUPPORTED_TYPE;

    if ((ret_code = mm_read_mtx_crd_size(f, M, N, nz)) != 0)
        return ret_code;

    *I = (int *)  malloc(*nz * sizeof(int));
    *J = (int *)  malloc(*nz * sizeof(int));
    *val = NULL;

    if (mm_is_complex(*matcode)) {
        *val = (FLOAT_T *) malloc(*nz * 2 * sizeof(FLOAT_T));
        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val,
                                        *matcode);
        if (ret_code != 0) return ret_code;
    } else if (mm_is_real(*matcode)) {
        *val = (FLOAT_T *) malloc(*nz * sizeof(FLOAT_T));
        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val,
                                        *matcode);
        if (ret_code != 0) return ret_code;
    } else if (mm_is_pattern(*matcode)) {
        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val,
                                        *matcode);
        if (ret_code != 0) return ret_code;
    }

    if (f != stdin) fclose(f);
    return 0;
}

    int 
mm_write_banner(FILE *f, MM_typecode matcode)
{
    char *str = mm_typecode_to_str(matcode);
    int ret_code;

    ret_code = fprintf(f, "%s %s\n", MatrixMarketBanner, str);
    free(str);
    if (ret_code != 2 )
        return MM_COULD_NOT_WRITE_FILE;
    else
        return 0;
}

    int
mm_write_mtx_crd(
    char fname[], int M, int N, int nz, int I[], int J[],
    FLOAT_T val[], MM_typecode matcode)
{
    FILE *f;
    int i;

    if (strcmp(fname, "stdout") == 0)
        f = stdout;
    else
    if ((f = fopen(fname, "w")) == NULL)
        return MM_COULD_NOT_WRITE_FILE;

    /* print banner followed by typecode */
    fprintf(f, "%s ", MatrixMarketBanner);
    fprintf(f, "%s\n", mm_typecode_to_str(matcode));

    /* print matrix sizes and nonzeros */
    fprintf(f, "%d %d %d\n", M, N, nz);

    /* print values */
    if (mm_is_pattern(matcode))
        for (i = 0; i < nz; i++)
            fprintf(f, "%d %d\n", I[i], J[i]);
    else
    if (mm_is_real(matcode))
        for (i = 0; i < nz; i++)
            fprintf(f, "%d %d %20.16g\n", I[i], J[i], val[i]);
    else
    if (mm_is_complex(matcode))
        for (i = 0; i < nz; i++)
            fprintf(f, "%d %d %20.16g %20.16g\n", I[i], J[i], val[2 * i],
                    val[2 * i + 1]);
    else {
        if (f != stdout) fclose(f);
        return MM_UNSUPPORTED_TYPE;
    }

    if (f != stdout) fclose(f);

    return 0;
}

/**
 *  Create a new copy of a string s.  mm_strdup() is a common routine, but
 *  not part of ANSI C, so it is included here.  Used by mm_typecode_to_str().
 *
 */
    char *
mm_strdup(const char *s)
{
    int len = strlen(s);
    char *s2 = (char *) malloc((len + 1) * sizeof(char));
    return strcpy(s2, s);
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Return the matrix data types according to the given         */
/*      MM_typecode.]                                                        */
/*                                                                           */
/*  Description [This function is revised by Xiao-Long to remove the         */
/*      warnings by g++.]                                                    */
/*                                                                           */
/*  Note        [The returned string must be freed by the caller function.]  */
/*                                                                           */
/*===========================================================================*/

    char *
mm_typecode_to_str(MM_typecode matcode)
{
    char buffer[MM_MAX_LINE_LENGTH];
    char *types[4] = { NULL};
    char *mm_strdup(const char *);
    //int error = 0;

    /* check for MTX type */
    if (mm_is_matrix(matcode)) {
        types[0] = mm_strdup(MM_MTX_STR);
    } else {
        //error = 1;
        goto ERROR_mm_typecode_to_str;
    }

    /* check for CRD or ARR matrix */
    if (mm_is_sparse(matcode))
        types[1] = mm_strdup(MM_SPARSE_STR);
    else
    if (mm_is_dense(matcode))
        types[1] = mm_strdup(MM_DENSE_STR);
    else
        goto ERROR_mm_typecode_to_str;

    /* check for element data type */
    if (mm_is_real(matcode))
        types[2] = mm_strdup(MM_REAL_STR);
    else
    if (mm_is_complex(matcode))
        types[2] = mm_strdup(MM_COMPLEX_STR);
    else
    if (mm_is_pattern(matcode))
        types[2] = mm_strdup(MM_PATTERN_STR);
    else
    if (mm_is_integer(matcode))
        types[2] = mm_strdup(MM_INT_STR);
    else
        goto ERROR_mm_typecode_to_str;

    /* check for symmetry type */
    if (mm_is_general(matcode))
        types[3] = mm_strdup(MM_GENERAL_STR);
    else
    if (mm_is_symmetric(matcode))
        types[3] = mm_strdup(MM_SYMM_STR);
    else
    if (mm_is_hermitian(matcode))
        types[3] = mm_strdup(MM_HERM_STR);
    else
    if (mm_is_skew(matcode))
        types[3] = mm_strdup(MM_SKEW_STR);
    else
        goto ERROR_mm_typecode_to_str;

    sprintf(buffer, "%s %s %s %s", types[0], types[1], types[2], types[3]);
    for (int i = 0; i < 4; i++) {
        free(types[i]);
    }
    return mm_strdup(buffer);

ERROR_mm_typecode_to_str:
    for (int i = 0; i < 4; i++) {
        if (types[i] != NULL) free(types[i]);
    }
    return NULL;
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Read MTX matrix file.]                                      */
/*  Description []                                                           */
/*  Note        [The allocated MTX matrix data structure must be freed by    */
/*      other functions.]                                                    */
/*                                                                           */
/*===========================================================================*/
    void
readMtxFile(
    const char *mtx_fn,         // mtx file name
    int *AW,                    // matrix width
    int *AH,                    // matrix height
    int **AI,                   // row index array
    int **AJ,                   // column index array
    FLOAT_T **AVal,               // value array
    int *ele_num_A,             // number of nonzero elements in matrix A
    const int if_silent         // If not show messages
    )
{
    const int MAX_WIDTH = INT_MAX; //16385 // Ideally this number should be ok.
    const int MAX_HEIGHT = INT_MAX;
    const int MAX_ELE_NUM = INT_MAX; // MAX_WIDTH * MAX_HEIGHT;
    // Max num of width to output computed matrix results for debugging
    const int OUTPUT_WIDTH = 0; // This should be as small as possible.

    int width, height;
    //int ele_num_org;    // number of original nonzero elements in matrix A
    bool warning = false, error = false;

    if (!if_silent) {
        printf("Read MTX file: %s\n", mtx_fn); fflush(stdout);
    }

    FILE *mtx_f = openFile(mtx_fn, "r", if_silent);
    MM_typecode matcode;
    if (mm_read_banner(mtx_f, &matcode) != 0) {
        printf("*** Error: Could not process Matrix Market banner.\n");
        exit(1);
    }
    if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
        mm_is_sparse(matcode)) {
        printf("*** Error: Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }
    int ret_code = mm_read_mtx_crd_size(mtx_f, &height, &width, ele_num_A);
    if (ret_code != 0) exit(1);
    //ele_num_org = *ele_num_A;

    // FIXME: Should modify programs for unlimited element numbers

    if (width > MAX_WIDTH) {
        printf("*** Warning: Matrix width (%d) exceeds maximum width (%d)\n",
            width, MAX_WIDTH);
        printf("         Matrix width will be trimmed to maximum width.\n");
        width = MAX_WIDTH;
        warning = true;
    }
    if (height > MAX_HEIGHT) {
        printf("*** Warning: Matrix height (%d) exceeds maximum height (%d)\n",
            height, MAX_HEIGHT);
        printf("         Matrix height will be trimmed to maximum height.\n");
        height = MAX_HEIGHT;
        warning = true;
    }
    if (*ele_num_A > MAX_ELE_NUM) {
        printf("*** Warning: Matrix element # (%d) exceeds maximum # (%d)\n",
            *ele_num_A, MAX_ELE_NUM);
        printf("         Matrix element # will be trimmed to maximum #.\n");
        *ele_num_A = MAX_ELE_NUM;
        warning = true;
    }

    if (!if_silent) {
        printf("Loading array (height x width = %dx%d)\n", height, width);
        fflush(stdout);
    }

    // Matrix C
    // --------

    *AW = width;
    *AH = height;
	// Jiading GAI:
    // *AI = new int[*ele_num_A];  // row index array
    // *AJ = new int[*ele_num_A];  // column index array
    // *AVal = new FLOAT_T[*ele_num_A]; // value array
    *AI   = mriNewCpu<int>(*ele_num_A);  // row index array
    *AJ   = mriNewCpu<int>(*ele_num_A);  // column index array
    *AVal = mriNewCpu<FLOAT_T>(*ele_num_A); // value array

    if (!if_silent && *ele_num_A < OUTPUT_WIDTH)
        printf("\nInput matrix contents:\n");

    char if_trim = 0;
    for (int i = 0; i < *ele_num_A; i++) {
        int ci = 0, cj = 0;
        double cval = 0.0; // always read in a double and convert later if necessary

        if (fscanf(mtx_f, "%d %d %lg\n", &ci, &cj, &cval) != 3)
            error = true;

        // FIXME: Should modify programs for unlimited element number?
        if (ci > MAX_WIDTH || cj > MAX_HEIGHT) {
            ele_num_A--;
            if (!if_trim) {
                printf("\n*** Warning: Elements at line %d are trimmed.\n",
                    i+3);
                fflush(stdout);
                if_trim = 1;
                warning = true;
            }
            continue;

        // adjust from 1-based Matrix Market format to 0-based data structure
        } else {
            (*AI)[i] = ci-1;
            (*AJ)[i] = cj-1;
            (*AVal)[i] = (FLOAT_T) cval;

            #if DEBUG_MODE
            // Error check
            if (ci >= height) {
                printf("\n*** Error: Row index value (%d) is greater than row size (%d).\n", ci, height);
                printf("    Line # in the file %s: %d\n", mtx_fn, i+3);
                fflush(stdout);
                error = true;
            }
            if (cj >= width) {
                printf("\n*** Error: Column index value (%d) is greater than column size (%d).\n", cj, width);
                printf("    Line # in the file %s: %d\n", mtx_fn, i+3);
                fflush(stdout);
                error = true;
            }
            #endif

            if (!if_silent && *ele_num_A < OUTPUT_WIDTH) {
                printf("    AI[%d]=%d, AJ[%d]=%d, value %lg, AVal[%d]=%lg\n",
                    i, (*AI)[i], i, (*AJ)[i], cval, i, (*AVal)[i]);
            }
        }
    }
    if (mtx_f != stdin) fclose(mtx_f); mtx_f = NULL;

    if (!if_silent) {
        if (!warning && !error) {
            printf("Data are successfully loaded.\n");
        } else {
            printf("Data are loaded with warnings or errors.\n");
        }
        fflush(stdout);
    }

    #if DEBUG_MODE
    // Error check
    printf("readMtxFile: AW: %d, AH: %d, ele_num_A: %d\n",
        *AW, *AH, *ele_num_A);
    for (int i = 0; i < *ele_num_A; i++) {
        if ((*AI)[i] >= *AH) {
            printf("*** Errors rows: %d\n", i);
            printf("    AI[%d]=%d, AJ[%d]=%d, AVal[%d]=%lg\n",
                i, (*AI)[i], i, (*AJ)[i], i, (*AVal)[i]);
        }
        if ((*AJ)[i] >= *AW) {
            printf("*** Errors cols: %d\n", i);
            printf("    AI[%d]=%d, AJ[%d]=%d, AVal[%d]=%lg\n",
                i, (*AI)[i], i, (*AJ)[i], i, (*AVal)[i]);
        }
    }
    #endif
}

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

