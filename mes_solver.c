#include "mes_solver.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <math.h>

static double min(double a, double b);
static double max(double a, double b);
static void print_vector(const gsl_vector *v);
static void print_matrix(const gsl_matrix *v);

int main(int argc, char **argv) {
    int n;
    if (argc > 1) {
        n = atoi(argv[1]);
    } else {
        n = 20;
    }

    gsl_matrix *L = create_left_matrix(n);
    gsl_vector *R = create_right_vector(n);
    gsl_vector *W = gsl_vector_alloc(n - 1);

    gsl_permutation *p = gsl_permutation_alloc(n - 1);
    int signum;

    gsl_linalg_LU_decomp(L, p, &signum);
    gsl_linalg_LU_solve(L, p, R, W);

    FILE *fpt;
    fpt = fopen("result.csv", "w+");
    fprintf(fpt, "X, Y\n");
    for (double x = 0.0; x < 3.0; x += 0.001) {
        fprintf(fpt, "%le, %le\n", x, Phi(x, W, n));
    }

    fclose(fpt);
    gsl_matrix_free(L);
    gsl_vector_free(R);
    gsl_vector_free(W);
    gsl_permutation_free(p);
    return 0;
}

double Phi(double x, gsl_vector *W, int n) {
    double suma = 0.0;
    for (int i = 1; i <= n - 1; i++) {
        suma += gsl_vector_get(W, i - 1) * e(x, i, n);
    }
    return (5 - x / 3 + suma);
}

gsl_matrix* create_left_matrix(int n) {
    gsl_matrix *M = gsl_matrix_alloc(n - 1, n - 1);
    for (int i = 1; i <= n - 1; i++) {
        for (int j = 1; j <= n - 1; j++) {
            gsl_matrix_set(M, i - 1, j - 1, B(i, j, n));
        }
    }
    return M;
}

gsl_vector* create_right_vector(int n) {
    gsl_vector *M = gsl_vector_alloc(n - 1);
    for (int i = 1; i <= n - 1; i++) {
        gsl_vector_set(M, i - 1, L_tilde(i, n));
    }
    return M;
}

double B(int i, int j, int n) {
    double h = DOMAIN_WIDTH / n;
    double a = 0.0, b = 0.0;
    if (abs(j - i) > 1) {
        return 0.0;
    } else if (i == j) {
        a = (i - 1) * h;
        b = (i + 1) * h;
    } else {
        a = min(i, j) * h;
        b = max(i, j) * h;
    }

    double x0 = (b - a) / (2 * sqrt(3)) + (a + b) / 2;
    double x1 = (a - b) / (2 * sqrt(3)) + (a + b) / 2;

    return (a - b) / 2.0 * (ee(x0, i, j, n) + ee(x1, i, j, n));
}

double L(int i, int n) {
    double h = DOMAIN_WIDTH / n;
    if (i * h < 1 || i * h > 2) {
        return 0.0;
    } else {
        double a = max(1.0, (i - 1) * h);
        double b = min(2.0, (i + 1) * h);
        double x0 = (b - a) / (2 * sqrt(3)) + (a + b) / 2;
        double x1 = (a - b) / (2 * sqrt(3)) + (a + b) / 2;
        double integral = (a - b) / 2.0 * (e(x0, i, n) + e(x1, i, n));
        return 4.0 * PI * G * integral;
    }
}

double B_tilde(int i, int n) {
    double h = DOMAIN_WIDTH / n;
    double a = (i - 1) * h;
    double b = (i + 1) * h;

    double integral = ((a - b) / 6) * (de((b - a) / (2 * sqrt(3)) + (a + b) / 2, i, n) + 
                    de((a - b) / (2 * sqrt(3)) + (a + b) / 2, i, n));
    return integral;
}

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

static void print_vector(const gsl_vector *v) {
    printf("Printing vector with size %zu\n", v->size);
    for (size_t i = 0; i < v->size; i++) {
        printf("%le ", gsl_vector_get(v, i));
    }
    printf("\n");
}

static void print_matrix(const gsl_matrix *m) {
    printf("Printing square matrix with size %zu\n", m->size1);
    for (size_t i = 0; i < m->size1; i++) {
        for (size_t j = 0; j < m->size2; j++) {
            printf("%le", gsl_matrix_get(m, i, j));
        }
        printf("\n");
    }
}