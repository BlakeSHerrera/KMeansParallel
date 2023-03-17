/*
 * Skeleton function for Parallel Programming, 
 * Assignment 3: K-Means Algorithm (OpenMP)
 *
 * To students: You should finish the implementation of k_means function
 * 
 * Author:
 *     Wei Wang <wei.wang@utsa.edu>
 */
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <omp.h> /* OpenMP header */

#include "k_means.h"

/*
 * k_means: k_means clustering algorithm implementation.
 *
 * Input parameters:
 *     struct point p[]: array of data points
 *     int m           : number of data points in p[]
 *     int k           : number of clusters to find
 *     int iters       : number of clustering iterations to run
 *
 * Output parameters:   
 *     struct point u[]: array of cluster centers
 *     int c[]         : cluster id for each data points
 */
void k_means(struct point p[MAX_POINTS], 
	    int m, 
	    int k,
	    int iters,
	    struct point u[MAX_CENTERS],
	    int c[MAX_POINTS])
{
	 /* To Students: add your local variables */
	int i, j, iter, clusters_count[k];
	float distance, closest_distance;
	struct point next_centers[k];
	
	/* randomly initialized the centers */
	/* Note: DO NOT CHANGE THIS RANDOM GENERATOR! */
	/* Note: DO NOT PARALLELIZE THIS LOOP */
	/* Note: THE INTERFACE TO random_center HAS CHANGED */
	for(j = 0; j < k; j++)
		u[j] = random_center(p);
	
	/* 
	 * To students: please implment K-Means algorithm with OpenMP here
	 * Your K-means implementation should do "iters" rounds of clustering. After 
	 * all iterations finish, array u[MAX_CENTERS] should have the coordinations 
	 * of your centers, and array c[MAX_POINTS] should have the cluster assignment
	 * for each point.
	 */
	for(iter=0; iter<iters; iter++) {
		// Reset count of points for clusters and next centers
		#pragma omp parallel for
		for(i=0; i<k; i++) {
			clusters_count[i] = 0;
			next_centers[i].x = 0;
			next_centers[i].y = 0;
		}
		#pragma omp parallel private(closest_distance, distance)
		{
			// Loop through points
			#pragma omp for
			for(i=0; i<m; i++) {
				closest_distance = FLT_MAX;
				// Loop through clusters
				#pragma omp parallel for firstprivate(i)
				for(j=0; j<k; j++) {
					distance = pow(p[i].x - u[j].x, 2) + 
						   pow(p[i].y - u[j].y, 2);
					if(distance < closest_distance) {
						closest_distance = distance;
						c[i] = j;
					}
				}
				next_centers[c[i]].x += p[i].x;
				next_centers[c[i]].y += p[i].y;
				clusters_count[c[i]]++;
			}
			// Check for clusters without assigned points
			#pragma omp for
			for(i=0; i<k; i++) {
				if(!clusters_count[i]) {
					// No points, reassign center
					u[i] = random_center(p);
				} else { 
					u[i] = next_centers[i];
					u[i].x /= clusters_count[i];
					u[i].y /= clusters_count[i];
				}
			}
		}
	}
	
	return;
}
