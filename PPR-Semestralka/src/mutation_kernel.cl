__kernel void mutation_and_cross(
	__global const unsigned char **best, 
	__global const unsigned char **v1, 
	__global const unsigned char **v2, 
	__global const unsigned char **v3, 
	__global const unsigned char **v4,
	__global const unsigned char **individual,
	__global const double **cross_rand,
	__global const double *f,
	__global const double *cr,
	_global unsigned char **res) {
 
	unsigned char noise_vec_elem;
	int k = 0;
	
	
    // get the index of the current element to be processed
    int i = get_global_id(0);
	
	// process it
	for(k = 0; k < 10; k++) {
		// mutation
		noise_vec_elem = best[i % 10][k] + (*f)*(v1[i][k] + v2[i][k] - v3[i][k] - v4[i][k]);
		
		// binomic cross
		if (cross_rand[i][k] <= (*cr)) {
			res[i][k] = noise_vec_elem;
		} else {
			res[i][k] = individual[i][k];
		}
	}
}