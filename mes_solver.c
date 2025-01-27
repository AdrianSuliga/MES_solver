#include "mes_solver.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <math.h>

static double min(double a, double b);
static double max(double a, double b);

int main(int argc, char **argv) {
    int n;
    if (argc > 1) {
        n = atoi(argv[1]);
    } else {
        n = 20;
    }

    // The ultimate goal of this program is to solve the equation
    // L * W = R where 
    // L is a matrix of B(e_i, e_j) elements
    // W is a vector that we're looking for
    // R is a vector of L_tilde elements
    // For more details about this equation see Equation_Calculations.pdf
    gsl_matrix *L = create_left_matrix(n);
    gsl_vector *R = create_right_vector(n);
    gsl_vector *W = gsl_vector_alloc(n - 1);

    // gsl provides easy way to solve matrix equations
    // using the following functions
    gsl_permutation *p = gsl_permutation_alloc(n - 1);
    int signum;

    gsl_linalg_LU_decomp(L, p, &signum);
    gsl_linalg_LU_solve(L, p, R, W);

    // Now write a few thousands (x, Phi(x)) points to .csv file
    FILE *fpt;
    fpt = fopen("result.csv", "w+");
    fprintf(fpt, "X, Y\n");
    for (double x = 0.0; x < 3.0; x += 0.001) {
        fprintf(fpt, "%le, %le\n", x, Phi(x, W, n));
    }

    // Cleanup code
    fclose(fpt);
    gsl_matrix_free(L);
    gsl_vector_free(R);
    gsl_vector_free(W);
    gsl_permutation_free(p);
    return 0;
}

// Phi(x) = 5 - x/3 + sum of w_i * e_i values
// Note that most of the time w_i * e_i = 0 
double Phi(double x, gsl_vector *W, int n) {
    double suma = 0.0;
    for (int i = 1; i <= n - 1; i++) {
        suma += gsl_vector_get(W, i - 1) * e(x, i, n);
    }
    return (5 - x / 3 + suma);
}

// Fill the matrix with B(e_i, e_j) values
// Note that most of the time B(e_i, e_j) = 0
// See B() for more details
gsl_matrix* create_left_matrix(int n) {
    gsl_matrix *M = gsl_matrix_alloc(n - 1, n - 1);
    for (int i = 1; i <= n - 1; i++) {
        for (int j = 1; j <= n - 1; j++) {
            gsl_matrix_set(M, i - 1, j - 1, B(i, j, n));
        }
    }
    return M;
}

// Fill the vector with L_tilde values
// See L_tilde() for more details
gsl_vector* create_right_vector(int n) {
    gsl_vector *M = gsl_vector_alloc(n - 1);
    for (int i = 1; i <= n - 1; i++) {
        gsl_vector_set(M, i - 1, L_tilde(i, n));
    }
    return M;
}

// See .pdf file to know which integral we're using for B
// There are 3 distinct cases when calculating B
double B(int i, int j, int n) {
    double h = DOMAIN_WIDTH / n;
    double a = 0.0, b = 0.0; // We'll be integrating over [a,b] interval
    if (abs(j - i) > 1) { // 1. case - B = 0, see e() definition to find out why
        return 0.0;
    } else if (i == j) { // 2. case - B is an integral over interval of 
        a = (i - 1) * h; // [x_(i-1), x_(i+1)]
        b = (i + 1) * h;
    } else { // 3. case - i == j + 1 or j == i + 1, so we change
        a = min(i, j) * h; // interval accordingly
        b = max(i, j) * h;
    }

    // See Gaussian quadrature section in .pdf file for the proof
    // of why this formula results in an integral
    double x0 = (b - a) / (2 * sqrt(3)) + (a + b) / 2;
    double x1 = (a - b) / (2 * sqrt(3)) + (a + b) / 2;

    return (a - b) / 2.0 * (ee(x0, i, j, n) + ee(x1, i, j, n));
}

// See .pdf file to know which integral we're using for L
double L(int i, int n) {
    double h = DOMAIN_WIDTH / n;
    if (i * h < 1 || i * h > 2) { // According to L's definition
        return 0.0; // it's value is 0.0 except of interval [1,2]
    } else { // on which we need to calculate an integral over [a, b]
        double a = max(1.0, (i - 1) * h); // such that 1 <= a < b <= 2
        double b = min(2.0, (i + 1) * h);

        // Again, refer to gaussian quadrature in .pdf
        double x0 = (b - a) / (2 * sqrt(3)) + (a + b) / 2;
        double x1 = (a - b) / (2 * sqrt(3)) + (a + b) / 2;
        double integral = (a - b) / 2.0 * (e(x0, i, n) + e(x1, i, n));
        return 4.0 * PI * G * integral;
    }
}

// B_tilde = B(5 - x/3, e_i)
// Note that d(5-x/3)/dx = -1/3
double B_tilde(int i, int n) {
    double h = DOMAIN_WIDTH / n;
    double a = (i - 1) * h;
    double b = (i + 1) * h;

    // Guess what? Gaussian quadrature
    double integral = ((a - b) / 6) * (de((b - a) / (2 * sqrt(3)) + (a + b) / 2, i, n) + 
                    de((a - b) / (2 * sqrt(3)) + (a + b) / 2, i, n));
    return integral;
}


// All the functions below are defined in .pdf file
double L_tilde(int i, int n) {
    double fst = L(i, n);
    double snd = B_tilde(i, n);
    return fst - snd;
}

double e(double x, int k, int n) {
    double h = DOMAIN_WIDTH / n;
    if (x >= (k - 1) * h && x <= k * h) {
        return x / h - k + 1.0;
    } else if (x > k * h && x <= (k + 1) * h) {
        return k + 1.0 - x / h;
    } else {
        return 0.0;
    }
}

double de(double x, int k, int n) {
    double h = DOMAIN_WIDTH / n;
    if (x >= (k - 1) * h && x <= k * h) {
        return 1 / h;
    } else if (x > k * h && x < (k + 1) / h) {
        return -1 / h;
    } else {
        return 0.0;
    }
}

double ee(double x, int i, int j, int n) {
    return de(x, i, n) * de(x, j, n);
}

static double min(double a, double b) {
    return a < b ? a : b;
}
static double max(double a, double b) {
    return a > b ? a : b;
}
