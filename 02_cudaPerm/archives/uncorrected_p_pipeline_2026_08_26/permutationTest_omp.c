#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <math.h>
#include <omp.h>

///
/// @brief returns number of lines a file has
/// @param filename input txt file
/// @return number of lines 
/// 
/// used to get number of voxels (not connections!) a subject data file has.
///
size_t getNumberLines(char* filename)
{
    FILE *stream;
    char *line = NULL;
    size_t len = 0;
    size_t nlines = 0;

    stream = fopen(filename, "r");
    if (stream == NULL) 
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    while( getline(&line, &len, stream) != -1 ) 
    {
        nlines++;
    }

    fclose(stream);
    free(line);

    return(nlines);
}

///
/// @brief returns the number of values a file has in the first line
/// @param filename input txt file
/// @return number of values in the first line
///
/// used to get number of voxels (not connections!) a subject data file has.
/// Sanity check getNumberLines() too as proper subject connectivity data should
/// have the same number of lines as values in the first line!
///
size_t getNumberValsFirstLine(char* filename)
{
    FILE *stream;
    char *line = NULL;
    char *linebuff;
    size_t len = 0;

    stream = fopen(filename, "r");
    if (stream == NULL) 
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    size_t line_length = 0;
    while( getline(&line, &len, stream) != -1 ) 
    {
        line_length = 0;
        linebuff = strtok(line, " ");
        while ( linebuff != NULL ) 
        {
            linebuff = strtok(NULL, " ");
            line_length++;
        }
        break;
    }

    fclose(stream);
    free(line);

    return(line_length);
}

///
/// @brief peek into file list and grabs the number of lines each subject has
/// @param filelist input file list 
/// @return number of lines each subject has
///
/// convenience routine to grab subject dimensions using the file list
/// references only
///
size_t peekFileList(char* filelist)
{
    FILE *fl;
    char* fl_line = NULL;
    size_t fl_len = 0;

    size_t output;

    fl = fopen(filelist, "r");
    if (fl == NULL) 
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    while( getline(&fl_line, &fl_len, fl) != -1 )
    {
        fl_line[strcspn(fl_line, "\n")] = 0;
        output = getNumberLines(fl_line);
    } 

    fclose(fl);
    free(fl_line);

    return(output);
}

///
/// @brief reads permutations file
/// @param filename generated permutations file from generatePermutations.py program
/// @param buffer array of ints to store the parsed permutations
/// @param nr_subs number of subjects in total for the test
///
/// note the expected input (as obtained from the output of
/// generatePermutations.py) is indices of a group (group A), and NOT one-hot
///
/// the output buffer contains one-hot labels of the subject permutations
///
void parsePermutations(char* filename, int* buffer, size_t nr_subs)
{
    /* variables to load permutations file */
    FILE *pt;
    char *pt_line = NULL;
    size_t pt_len = 0;
    int pt_lines;

    /* variables to parse file */
    char* line_buff;
    float val_buff;
    size_t line_idx;

    // opening permutations file
    pt = fopen(filename, "r");
    if (pt == NULL) 
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    line_idx = 0;
    while ((pt_lines = getline(&pt_line, &pt_len, pt)) != -1) 
    {
        if (line_idx == 0)
        {
            line_buff = (char*)malloc(sizeof(char)*pt_lines);
        }

        line_buff = strtok(pt_line, " ");
        while ( line_buff != NULL ) 
        {
            if (*line_buff == '\n') 
            {
                line_buff = strtok(NULL, " ");
                continue;
            } 
            else
            {
                val_buff = atof(line_buff);
                buffer[(int)((line_idx*nr_subs)+val_buff)] = 1;
                line_buff = strtok(NULL, " ");
            }
        }
        line_idx++;
    } 

    // cleanup
    free(line_buff);
}

///
/// @brief reads file list and parses all subjects from index N to M
/// @param filename input file list
/// @param nr_sub number of subjects in total in file list
/// @param N from (and including) index 
/// @param M to (and including) index
/// @param buffer output buffer of subject connection values
///
/// Output buffer contains index N as first row to index M as last row of
/// values. For each row subject order is the same as specified in the file
/// list.
///
void parseFileListNtoM(char* filename, int nr_sub, int N, int M, float* buffer)
{
    /* vars */
    /* variables to load file list */
    FILE *fl;
    char* fl_line = NULL;
    size_t fl_len = 0;

    /* variables to load file */
    FILE *stream;
    char *line = NULL;
    size_t len = 0;
    size_t nread;

    /* variables to parse file */
    char* line_buff;
    float buff;
    size_t line_idx;
    size_t k;
    size_t line_length;

    /* variables for device buffer */
    size_t row_counter = 0;
    /* /vars */

    fl = fopen(filename, "r");
    if (fl == NULL) 
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    size_t sub_idx = 0;
    while( getline(&fl_line, &fl_len, fl) != -1 )
    {
        /* get filename (remove trailing \n)*/
        fl_line[strcspn(fl_line, "\n")] = 0;

        /* load file */
        stream = fopen(fl_line, "r");
        if (stream == NULL) 
        {
            perror("fopen");
            exit(EXIT_FAILURE);
        }

        /* read file line by line */
        line_idx = 0;
        k = 0;
        row_counter = 0;
        while ((nread = getline(&line, &len, stream)) != -1) 
        {
            if (line_idx == 0) // firstline
            {
                line_buff = (char*)malloc(sizeof(char)*nread);
            }

            line_length = 0;
            line_buff = strtok(line, " ");
            while ( line_buff != NULL ) 
            {
                if (*line_buff == '\n') 
                {
                    line_buff = strtok(NULL, " ");
                    continue;
                } 
                else
                {
                    buff = atof(line_buff);
                    if (k >= N && k <= M)
                    {
                        buffer[(row_counter*nr_sub)+sub_idx] = buff;
                        row_counter++;
                    }
                    ++k;
                    line_length++;
                    line_buff = strtok(NULL, " ");
                }
            }

            line_idx++;
        } // end of reading a file line by line

        sub_idx++;
    } // end of parsing file list  

    /* cleanup */
    fclose(fl);
    free(fl_line);
    
    fclose(stream);
    free(line);

    free(line_buff);

    printf("done!\n");
}

///
/// @brief save results as upper triangular format
/// @param outputData data to be saved to file
/// @param nrows number of rows in the output file/nr voxels the output should have
/// @param fileName output file name
///
/// note that nrows specifies the number of voxels, not connections each subject
/// has.
///
void saveResToText(float *outputData, size_t nrows, char *fileName)
{
    FILE *output = fopen(fileName, "w");
    
    size_t c = 0;
    for (size_t i=0; i<nrows; ++i)
    {
        for (size_t j=i+1; j<nrows; ++j) 
        {
            fprintf(output, "%f ", outputData[c]);
            c++;
        }
        fprintf(output, "\n");
    }

    printf("[DBG] saved a total of %zu values\n", c);
    fclose(output);
}

///
/// @brief performs permutation test
/// @param input input buffer (subject values) 
/// @param idx index to calculate
/// @param onehot permutations buffer in one-hot format
/// @param nr_vals number of values to process (connections)
/// @param nr_sub number of subjects in total
/// @param nr_perm number of permutations
/// @param two_tailed is two-tailed test or not (1 = two-tailed, 0 = single-tailed)
/// @return permutation test p-value
///
/// main function to calculate permutation test p-value for a given index
///
float t_permute(float* input, int idx, int* onehot, 
                size_t nr_vals, size_t nr_sub, size_t nr_perm, 
                int two_tailed)
{
    float t_obs;
    float p_val = 0.;

    for (int i=0; i<nr_perm; ++i) 
    {
        // Welch's t-stat: t = (mean_A - mean_B) / sqrt(var_A/nA + var_B/nB)
        float a_sum = 0.f, b_sum = 0.f;
        float a_sq  = 0.f, b_sq  = 0.f;
        float nA = 0, nB = 0;
        float tstat = 0.f;

        for (int j=0; j<nr_sub; ++j)
        {
            float val = input[(idx*nr_sub)+j];
            if (onehot[(i*nr_sub)+j] == 0)
            {
                b_sum += val;
                b_sq  += val * val;
                nB++;
            } 
            else 
            {
                a_sum += val;
                a_sq  += val * val;
                nA++;
            }
        }
        float a_mean = (nA > 0) ? a_sum / nA : 0.f;
        float b_mean = (nB > 0) ? b_sum / nB : 0.f;
        float a_var  = (nA > 1) ? (a_sq - a_sum * a_sum / nA) / (nA - 1.f) : 0.f;
        float b_var  = (nB > 1) ? (b_sq - b_sum * b_sum / nB) / (nB - 1.f) : 0.f;
        float se     = sqrtf(a_var / fmaxf(nA, 1.f) + b_var / fmaxf(nB, 1.f));
        tstat = (se > 1e-12f) ? (a_mean - b_mean) / se : 0.f;

        if (i == 0) // first permutation, t_obs is tstat
        {
            t_obs = tstat;
        }

        if (i > 0)  // only count permutations, not the observed
        {
            if (two_tailed == 1) 
            {
                if (fabsf(tstat) >= fabsf(t_obs))
                    p_val++;
            }
            else
            {
                // One-tailed: test in the direction of the observed effect
                if (t_obs >= 0.f)
                {
                    if (tstat >= t_obs) p_val++;
                }
                else
                {
                    if (tstat <= t_obs) p_val++;
                }
            }
        }
    }
    p_val = (p_val + 1.0f) / (float)(nr_perm + 1);
    return p_val;
}

int
main(int argc, char *argv[])
{
    /* read arguments */    
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <file list> <permutations file> <output file>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    /* parse input arguments */
    char* filelist = argv[1];
    char* permutations = argv[2];
    char* outfile = argv[3];

    /* get dimensions */
    size_t nr_r1vals = peekFileList(filelist);
    size_t nr_vals = (nr_r1vals*(nr_r1vals-1))/2;
    size_t nr_perm = getNumberLines(permutations);
    size_t nr_subs = getNumberLines(filelist);
    printf("Number of permutations: %zd\n", nr_perm);
    printf("Number of subjects: %zu\n", nr_subs);
    printf("Number of connections in each subject: %zu\n", nr_vals);

    printf("allocating memory buffers ...\n");
    int* perm_buff = (int*)malloc(sizeof(int)*nr_perm*nr_subs);
    /* we have to zero out the permutation buffer */
    for (int i=0; i<nr_perm*nr_subs; ++i)
    {
        perm_buff[i] = 0; 
    }
    float* device_buff = (float*)malloc(sizeof(float)*nr_vals*nr_subs);
    
    printf("parsing input files ...\n");
    parsePermutations(permutations, perm_buff, nr_subs);
    parseFileListNtoM(filelist, nr_subs, 0, nr_vals-1, device_buff);

    float* perm_test_res;
    perm_test_res = (float*)malloc(sizeof(float)*nr_vals);

    printf("performing permutation tests ...\n");
    #pragma omp parallel num_threads(24)
    for (int i=0; i<nr_vals; ++i)
    {
        perm_test_res[i] = t_permute(device_buff, i, perm_buff, nr_vals, nr_subs, nr_perm, 0);
    }
    printf("... done!\n");

    printf("writing to file ...\n");
    saveResToText(perm_test_res, nr_r1vals, outfile);
    printf("done!\n");

    /* cleanup */
    free(device_buff);
    free(perm_buff);
    free(perm_test_res);

    return(EXIT_SUCCESS);
}