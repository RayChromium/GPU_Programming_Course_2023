#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define totaldegrees 180
#define binsperdegree 4
#define threadsperblock 512
#define PI_F            3.141592654f


// data for the real galaxies will be read into these arrays
float *ra_real, *decl_real;

// data for the real galaxies, on GPU mem
float *ra_real_gm, *decl_real_gm;


// number of real galaxies
int NoofReal;

// data for the simulated random galaxies will be read into these arrays
float *ra_sim, *decl_sim;

// data for the simulated random galaxies read&copied into GPU mem
float *ra_sim_gm, *decl_sim_gm;

// number of simulated random galaxies
int    NoofSim;

const float binWidth = 0.25f;
const int numBins = 180 / binWidth;
// we already know the number of bins in the histogram, so in host memory no need to malloc:
// unsigned int *histogramDR, *histogramDD, *histogramRR;
unsigned int histogramDR[numBins] = {0};
unsigned int histogramDD[numBins] = {0};
unsigned int histogramRR[numBins] = {0};
float omega[numBins] = {0};

// but still need cudaMalloc on device memory:
unsigned int *histogramDR_gm, *histogramDD_gm, *histogramRR_gm;

__device__ inline float calculateAngularDistance(float g1_ra, float g1_dec, float g2_ra, float g2_dec) {
    // turning arc minutes to degree
    float ra1 = g1_ra/ 60.0f * PI_F / 180.0f;
    float dec1 = g1_dec/ 60.0f * PI_F / 180.0f;
    float ra2 = g2_ra/ 60.0f * PI_F / 180.0f;
    float dec2 = g2_dec/ 60.0f * PI_F / 180.0f;
    // float sin_ra1 = sinf(ra1);
    // float sin_ra2 = sinf(ra2);
    // float cos_ra1 = cosf(ra1);
    // float cos_ra2 = cosf(ra2);


    // calculate angular distance
    float delta_ra = ra2 - ra1;
    float cos_c = sinf(dec1) * sinf(dec2) + cosf(dec1) * cosf(dec2) * cosf(delta_ra);
    if (cos_c > 1.0f) {
        cos_c = 1.0f;
    } else if (cos_c < -1.0f) {
        cos_c = -1.0f;
    }
    float c = acosf(cos_c);

    if (isnan(c) || isinf(c)) {
        return 0.0f;
    } else {
        return c * 180.0f / PI_F;
    }

}

__global__ void calculateHistograms(float* d_ra_real, float * d_decl_real, float* r_ra_sim, float* r_decl_sim, unsigned int* dd, unsigned int* dr, unsigned int* rr, long int maxInputLength) {
    long int index = ((long int)threadIdx.x + (long int)blockIdx.x * (long int)blockDim.x);

    if (index >= (long int)maxInputLength * maxInputLength) {
        return;
    }

    int i = index / maxInputLength;
    int j = index % maxInputLength;

    int bin_dd = (int)(calculateAngularDistance(d_ra_real[i], d_decl_real[i], d_ra_real[j],d_decl_real[j]) * 4.0f);
    atomicAdd(&dd[bin_dd], 1);

    int bin_dr = (int)(calculateAngularDistance(d_ra_real[i], d_decl_real[i], r_ra_sim[j],r_decl_sim[j]) * 4.0f);
    atomicAdd(&dr[bin_dr], 1);

    int bin_rr = (int)(calculateAngularDistance(r_ra_sim[i], r_decl_sim[i], r_ra_sim[j], r_decl_sim[j]) * 4.0f);
    atomicAdd(&rr[bin_rr], 1);
}

__global__ void calculateSingleHistogram(float* ra1, float * decl1, float* ra2, float* decl2, unsigned int* histogram, long int maxInputLength) {
    long int index = ((long int)threadIdx.x + (long int)blockIdx.x * (long int)blockDim.x);

    if (index >= (long int)maxInputLength * maxInputLength) {
        return;
    }

    int i = index / maxInputLength;
    int j = index % maxInputLength;

    int bin = (int)(calculateAngularDistance(ra1[i], decl1[i], ra2[j], decl2[j]) * 4.0f);
    atomicAdd(&histogram[bin], 1);
}


void calculateOmega( long int* histogramDRsum, long int* histogramDDsum, long int* histogramRRsum ) {
    for( int i = 0; i < numBins; ++i ) {
        if( histogramRR[i] != 0 ) {
            omega[i] = (float)( histogramDD[i] - 2.0f * histogramDR[i] + histogramRR[i] ) / histogramRR[i];
            *histogramDDsum += histogramDD[i];
            *histogramDRsum += histogramDR[i];
            *histogramRRsum += histogramRR[i];
        }
    }
}

void printResult( FILE *outfil ) {
    fprintf( outfil, "bin start/deg\t\tomega\t\thist_DD\t\thist_DR\t\thist_RR\n");
    for( int i = 0; i < numBins; ++i ){
        if(histogramDD[i] == 0) {
            break;
        }
        if( i < 10 ) {
            printf("%.3f\t\t%.6f\t\t%u\t\t%u\t\t%u\n", i * binWidth, omega[i], histogramDD[i], histogramDR[i], histogramRR[i]);
        }
        fprintf( outfil, "%.3f\t\t%.6f\t\t%u\t\t%u\t\t%u\n", i * binWidth, omega[i], histogramDD[i], histogramDR[i], histogramRR[i] );
    }
}

unsigned int *d_histogram;


int main(int argc, char *argv[])
{
   int    i;
   long int    noofblocks;
   int    readdata(char *argv1, char *argv2);
   int    getDevice(int deviceno);
   long int histogramDRsum, histogramDDsum, histogramRRsum;
   double w;
   double start, end, kerneltime;
   struct timeval _ttime;
   struct timezone _tzone;
   cudaError_t myError;

   FILE *outfil;

   if ( argc != 4 ) {printf("Usage: a.out real_data random_data output_data\n");return(-1);}

//    if ( getDevice(0) != 0 ) return(-1);

   // start timing 
   kerneltime = 0.0;
   gettimeofday(&_ttime, &_tzone);
   start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;

   if ( readdata(argv[1], argv[2]) != 0 ) return(-1);

   // allocate mameory on the GPU --> Better use Managed
   const unsigned long long real_axis_size = NoofReal*sizeof(float);
   const unsigned long long sim_axis_size = NoofSim*sizeof(float);
   cudaMalloc( &ra_real_gm,     real_axis_size );
   cudaMalloc( &decl_real_gm,   real_axis_size );
   cudaMalloc( &ra_sim_gm,      sim_axis_size );
   cudaMalloc( &decl_sim_gm,    sim_axis_size );

   const int histogramSize = numBins*sizeof(unsigned int);

// Better to use cudaMallocManaged in order to allocate memory in Gpu and use it directly in the Gpu (we do not have to use Memcopy)
   cudaMalloc( &histogramDR_gm,    histogramSize );
   cudaMalloc( &histogramDD_gm,    histogramSize );
   cudaMalloc( &histogramRR_gm,    histogramSize );

   // copy data to the GPU
   cudaMemcpy( ra_real_gm, ra_real, real_axis_size, cudaMemcpyHostToDevice );
   cudaMemcpy( decl_real_gm, decl_real, real_axis_size, cudaMemcpyHostToDevice );
   cudaMemcpy( ra_sim_gm, ra_sim, sim_axis_size, cudaMemcpyHostToDevice );
   cudaMemcpy( decl_sim_gm, decl_sim, sim_axis_size, cudaMemcpyHostToDevice );
   cudaMemset( histogramDR_gm, 0, histogramSize );
   cudaMemset( histogramDD_gm, 0, histogramSize );
   cudaMemset( histogramRR_gm, 0, histogramSize );

   // check to see which array of coordinates are longer, use that longer length as range
   const long int maxInputLength = (NoofReal > NoofSim ? NoofReal : NoofSim);
   noofblocks = (long int)(( maxInputLength * maxInputLength + threadsperblock - 1) / threadsperblock);
   printf("size of threadblocks (threads per block): %d\n", threadsperblock);
   printf("number of threadblocks: %ld\n", noofblocks);
   printf("number of threads: %ld\n", noofblocks*threadsperblock);
   calculateHistograms<<< noofblocks, threadsperblock >>>( ra_real_gm, decl_real_gm, ra_sim_gm, decl_sim_gm, histogramDD_gm, histogramDR_gm, histogramRR_gm, maxInputLength );
//    calculateSingleHistogram<<< noofblocks, threadsperblock >>>( ra_real_gm, decl_real_gm, ra_real_gm, decl_real_gm, histogramDD_gm, maxInputLength );
//    calculateSingleHistogram<<< noofblocks, threadsperblock >>>( ra_real_gm, decl_real_gm, ra_sim_gm, decl_sim_gm, histogramDR_gm, maxInputLength );
//    calculateSingleHistogram<<< noofblocks, threadsperblock >>>( ra_sim_gm, decl_sim_gm, ra_sim_gm, decl_sim_gm, histogramRR_gm, maxInputLength );

   
   // copy the results back to the CPU
   cudaMemcpy( histogramDD, histogramDD_gm, histogramSize, cudaMemcpyDeviceToHost );
   cudaMemcpy( histogramDR, histogramDR_gm, histogramSize, cudaMemcpyDeviceToHost );
   cudaMemcpy( histogramRR, histogramRR_gm, histogramSize, cudaMemcpyDeviceToHost );

   // calculate omega values on the CPU
   // initializing them to zero:
   histogramDDsum = 0;
   histogramRRsum = 0;
   histogramDRsum = 0;
   calculateOmega( &histogramDRsum, &histogramDDsum, &histogramRRsum);

   outfil = fopen(argv[3], "w");
   printResult(outfil);
   printf("sums of histogram DD: %ld, histogram DR: %ld, histogram RR: %ld\n", histogramDDsum, histogramDRsum, histogramRRsum);
   fclose(outfil);

   cudaFree(ra_real_gm);
   cudaFree(decl_real_gm);
   cudaFree(ra_sim_gm);
   cudaFree(decl_sim_gm);
   cudaFree(histogramDD_gm);
   cudaFree(histogramDR_gm);
   cudaFree(histogramRR_gm);

   // end timing
   gettimeofday(&_ttime, &_tzone);
   end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
   kerneltime += end-start;

   printf("time: %f s\n", kerneltime);

   return(0);
}


int readdata(char *argv1, char *argv2)
{
  int i,linecount;
  char inbuf[180];
  double ra, dec, phi, theta, dpi;
  FILE *infil;
                                         
  printf("   Assuming input data is given in arc minutes!\n");
                          // spherical coordinates phi and theta:
                          // phi   = ra/60.0 * dpi/180.0;
                          // theta = (90.0-dec/60.0)*dpi/180.0;

  dpi = acos(-1.0);
  infil = fopen(argv1,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv1);return(-1);}

  // read the number of galaxies in the input file
  int announcednumber;
  if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv1);return(-1);}
  linecount =0;
  while ( fgets(inbuf,180,infil) != NULL ) ++linecount;
  rewind(infil);

  if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv1, linecount);
  else 
      {
      printf("   %s does not contain %d galaxies but %d\n",argv1, announcednumber,linecount);
      return(-1);
      }

  NoofReal = linecount;
  ra_real   = (float *)calloc(NoofReal,sizeof(float));
  decl_real = (float *)calloc(NoofReal,sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i = 0;
  while ( fgets(inbuf,80,infil) != NULL )
      {
      if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ) 
         {
         printf("   Cannot read line %d in %s\n",i+1,argv1);
         fclose(infil);
         return(-1);
         }
      ra_real[i]   = (float)ra;
      decl_real[i] = (float)dec;
      ++i;
      }

  fclose(infil);

  if ( i != NoofReal ) 
      {
      printf("   Cannot read %s correctly\n",argv1);
      return(-1);
      }

  infil = fopen(argv2,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv2);return(-1);}

  if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv2);return(-1);}
  linecount =0;
  while ( fgets(inbuf,80,infil) != NULL ) ++linecount;
  rewind(infil);

  if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv2, linecount);
  else
      {
      printf("   %s does not contain %d galaxies but %d\n",argv2, announcednumber,linecount);
      return(-1);
      }

  NoofSim = linecount;
  ra_sim   = (float *)calloc(NoofSim,sizeof(float));
  decl_sim = (float *)calloc(NoofSim,sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i =0;
  while ( fgets(inbuf,80,infil) != NULL )
      {
      if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ) 
         {
         printf("   Cannot read line %d in %s\n",i+1,argv2);
         fclose(infil);
         return(-1);
         }
      ra_sim[i]   = (float)ra;
      decl_sim[i] = (float)dec;
      ++i;
      }

  fclose(infil);

  if ( i != NoofSim ) 
      {
      printf("   Cannot read %s correctly\n",argv2);
      return(-1);
      }

  return(0);
} 


int getDevice(int deviceNo)
{

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("   Found %d CUDA devices\n",deviceCount);
  if ( deviceCount < 0 || deviceCount > 128 ) return(-1);
  int device;
  for (device = 0; device < deviceCount; ++device) {
       cudaDeviceProp deviceProp;
       cudaGetDeviceProperties(&deviceProp, device);
       printf("      Device %s                  device %d\n", deviceProp.name,device);
       printf("         compute capability            =        %d.%d\n", deviceProp.major, deviceProp.minor);
       printf("         totalGlobalMemory             =       %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
       printf("         l2CacheSize                   =   %8d B\n", deviceProp.l2CacheSize);
       printf("         regsPerBlock                  =   %8d\n", deviceProp.regsPerBlock);
       printf("         multiProcessorCount           =   %8d\n", deviceProp.multiProcessorCount);
       printf("         maxThreadsPerMultiprocessor   =   %8d\n", deviceProp.maxThreadsPerMultiProcessor);
       printf("         sharedMemPerBlock             =   %8d B\n", (int)deviceProp.sharedMemPerBlock);
       printf("         warpSize                      =   %8d\n", deviceProp.warpSize);
       printf("         clockRate                     =   %8.2lf MHz\n", deviceProp.clockRate/1000.0);
       printf("         maxThreadsPerBlock            =   %8d\n", deviceProp.maxThreadsPerBlock);
       printf("         asyncEngineCount              =   %8d\n", deviceProp.asyncEngineCount);
       printf("         f to lf performance ratio     =   %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
       printf("         maxGridSize                   =   %d x %d x %d\n",
                          deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
       printf("         maxThreadsDim in thread block =   %d x %d x %d\n",
                          deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
       printf("         concurrentKernels             =   ");
       if(deviceProp.concurrentKernels==1) printf("     yes\n"); else printf("    no\n");
       printf("         deviceOverlap                 =   %8d\n", deviceProp.deviceOverlap);
       if(deviceProp.deviceOverlap == 1)
       printf("            Concurrently copy memory/execute kernel\n");
       }

    cudaSetDevice(deviceNo);
    cudaGetDevice(&device);
    if ( device != 0 ) printf("   Unable to set device 0, using %d instead",device);
    else printf("   Using CUDA device %d\n\n", device);

return(0);
}

