/*
 * Random (fake) function for CS6643 Parallel Processing, 
 * Assignment 2: K-Means Algorithm (Sequential)
 *
 * To students: You should not modify this file
 * 
 * Author:
 *     Wei Wang <wei.wang@utsa.edu>
 */
#include <stdlib.h>

#include "k_means.h"

struct point randoms[] = {{-10.2, 11.20},
			  {11.2, -11.2},
			  {-111, 111},
			  {167, -67},
			  {0, 0},
			  {156, 45},
			  {-89, -11.245},
			  {22, 11.13},
			  {-12, -156},
			  {-45, -56}};

int counter = 0;
int max = 200;
int min = -200;

struct point random_center()
{
	//return randoms[(counter++)%10];
	struct point u;
	u.x = (max - min) * ( (double)rand() / (double)RAND_MAX ) + min;
	u.y = (max - min) * ( (double)rand() / (double)RAND_MAX ) + min;

	return u;
}
