#pragma once
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#define DOMAIN_WIDTH 3.0
#define PI 3.141592653589793
#define G 6.6743e-11

double e(double x, int k, int n);
double de(double x, int k, int n);
double ee(double x, int i, int j, int n);

double B(int i, int j, int n);
double L(int i, int n);
double B_tilde(int i, int n);
double L_tilde(int i, int n);

double Phi(double x, gsl_vector *W, int n);

gsl_matrix* create_left_matrix(int n);
gsl_vector* create_right_vector(int n);