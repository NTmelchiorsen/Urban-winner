#include "stdio.h"
#include <stdlib.h>
#include "matrix.h"
#include <math.h>
#include "mex.h"

double utility(double c, double rho, double alpha, double l_hour, double lw)
{
    return pow(c, 1 - rho) / (1 - rho) -  lw * pow(l_hour, alpha) / alpha;
}

void upper_envelope(double *c_out, double *v_out, mwSize n,
                    double *c_raw, double *m_raw, double *v_plus_raw, double *a_raw, double *grid_m, 
                    double l_hour, double rho, double alpha, double beta, double lw)       
{
    // Allocate and initialize
    double inf;
    inf = mxGetInf();
    for (int ii = 0; ii < n; ++ii) {
        c_out[ii] = 0;
        v_out[ii] = -(inf);
    }
    double m_low, m_high, c_slope, a_low, a_high, v_plus_slope, m_now;
    double c_guess, a_guess, v_plus, v_guess, util;

    // Upper envelope
    for (int i = 0; i < n; ++i) {
        
        m_low = m_raw[i];
        m_high = m_raw[i+1];
        c_slope = (c_raw[i+1] - c_raw[i]) / (m_high - m_low);
    
        a_low = a_raw[i];
        a_high = a_raw[i+1];
        v_plus_slope = (v_plus_raw[i+1] - v_plus_raw[i]) / (a_high - a_low);
    
        for (int j = 0; j < n; ++j) {
      
            m_now = grid_m[j];
      
            if (((m_now >= m_low) && (m_now <= m_high)) || ((m_now > m_high) && (i == n-2))) {
          
                c_guess = c_raw[i] + c_slope * (m_now - m_low);
                a_guess = m_now - c_guess;
                v_plus = v_plus_raw[i] + v_plus_slope * (a_guess - a_low);
                util = utility(c_guess, rho, alpha, l_hour, lw);
                v_guess = util + beta * v_plus;
                
                if (v_guess > v_out[j]) {
                    v_out[j] = v_guess;
                    c_out[j] = c_guess;
                }
            }    
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    //check
    if(nrhs != 10) {
    mexErrMsgIdAndTxt("MyToolbox:upper_envelope:nrhs",
                      "Ten inputs required.");
    }

    if(nlhs != 2) {
    mexErrMsgIdAndTxt("MyToolbox:upper_envelope:nlhs",
                      "Two outputs required.");
    }    

    //input
    mwSize nrows;
    double *c_raw, *m_raw, *v_plus_raw, *a_raw, *grid_m;
    double l_hour, rho, alpha, beta, lw;
    
    //output
    double *c_out, *v_out;
    
    //get input
    c_raw = mxGetPr(prhs[0]);
    m_raw = mxGetPr(prhs[1]);
    v_plus_raw = mxGetPr(prhs[2]);
    a_raw = mxGetPr(prhs[3]);
    grid_m = mxGetPr(prhs[4]);
    nrows = mxGetM(prhs[4]);
    l_hour = mxGetScalar(prhs[5]);
    rho = mxGetScalar(prhs[6]);
    alpha = mxGetScalar(prhs[7]);
    beta = mxGetScalar(prhs[8]);
    lw = mxGetScalar(prhs[9]);    
    
    //get output
    plhs[0] = mxCreateDoubleMatrix(nrows, 1, mxREAL);  
    plhs[1] = mxCreateDoubleMatrix(nrows, 1, mxREAL);  
    c_out = mxGetPr(plhs[0]);
    v_out = mxGetPr(plhs[1]);
    
    //main routine
    upper_envelope(c_out, v_out, nrows, c_raw, m_raw, v_plus_raw, a_raw, grid_m, l_hour, rho, alpha, beta, lw);
}