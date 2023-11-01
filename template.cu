#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define totaldegrees 180
#define binsperdegree 4
#define threadsperblock 512


// data for the real galaxies will be read into these arrays
float *ra_real, *decl_real;
// data for the real galaxies, on GPU mem
float *ra_real_gm, *decl_real_gm;
// number of real galaxies
int    NoofReal;

// data for the simulated random galaxies will be read into these arrays
float *ra_sim, *decl_sim;
// data for the simulated random galaxies read&copied into GPU mem
float *ra_sim_gm, *decl_sim_gm;
// number of simulated random galaxies
int    NoofSim;

const float binWidth = 0.25;
const int numBins = 180 / binWidth;
// we already know the number of bins in the histogram, so in host memory no need to malloc:
// unsigned int *histogramDR, *histogramDD, *histogramRR;
unsigned int histogramDR[numBins] = {0};
unsigned int histogramDD[numBins] = {0};
unsigned int histogramRR[numBins] = {0};
float omega[numBins] = {0};
// but still need cudaMalloc on device memory:
unsigned int *histogramDR_gm, *histogramDD_gm, *histogramRR_gm;

__device__ float calculateAngularDistance(float g1_ra, float g1_dec, float g2_ra, float g2_dec) {
    // turning arc minutes to degree
    float ra1 = g1_ra/ 60 * M_PI / 180.0;
    float dec1 = g1_dec/ 60 * M_PI / 180.0;
    float ra2 = g2_ra/ 60 * M_PI / 180.0;
    float dec2 = g2_dec/ 60 * M_PI / 180.0;

    // calculate angular distance
    float delta_ra = ra2 - ra1;
    float cos_c = sin(dec1) * sin(dec2) + cos(dec1) * cos(dec2) * cos(delta_ra);
    if (cos_c > 1.0) {
        cos_c = 1.0;
    } else if (cos_c < -1.0) {
        cos_c = -1.0;
    }
    float c = acosf(cos_c);

    if (isnan(c) || isinf(c)) {
        return 0.0;
    } else {
        return c * 180 / M_PI;
    }

}

__global__ void calculateHistograms(float* d_ra_real, float * d_decl_real, float* r_ra_sim, float* r_decl_sim, unsigned int* dd, unsigned int* dr, unsigned int* rr, int numD, int numR) {
    long int index = (long int)(threadIdx.x + blockIdx.x * blockDim.x);

    if( index < numD ) {
        for (int j = index + 1; j < numD; j++) {
            int bin = (int)( calculateAngularDistance(d_ra_real[index], d_decl_real[index], d_ra_real[j], d_decl_real[j]) / 0.25 );
            atomicAdd(&dd[bin], 1);
        }

        for (int j = 0; j < numR; j++) {
            int bin = (int)( calculateAngularDistance(d_ra_real[index], d_decl_real[index], r_ra_sim[j], r_decl_sim[j]) / 0.25 );
            atomicAdd(&dr[bin], 1);
        }
    }

    if( index < numR )  {
        for (int j = index + 1; j < numR; j++) {
            int bin = (int)( calculateAngularDistance(r_ra_sim[index], r_decl_sim[index], r_ra_sim[j], r_decl_sim[j]) / 0.25 );
            atomicAdd(&rr[bin], 1);
        }
    }
}

void calculateOmega() {
    for( int i = 0; i < numBins; ++i ) {
        if( histogramRR[i] != 0 ) {
            omega[i] = (float)( histogramDD[i] - 2 * histogramDR[i] + histogramRR[i] ) / histogramRR[i];
        }
    }
}

void printResult() {
    printf("bin start/deg\t\tomega\t\thist_DD\t\thist_DR\t\thist_RR\n");
    for( int i = 0; i < numBins; ++i ){
        if(histogramDD[i] == 0) {
            break;
        }
        printf("%.3f\t\t%.6f\t\t%u\t\t%u\t\t%u\n", i * binWidth, omega[i], histogramDD[i], histogramDR[i], histogramRR[i] );
    }
}

unsigned int *d_histogram;


int main(int argc, char *argv[])
{
   int    i;
   int    noofblocks;
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
   cudaMalloc( &ra_real_gm,     NoofReal*sizeof(float) );
   cudaMalloc( &decl_real_gm,   NoofReal*sizeof(float) );
   cudaMalloc( &ra_sim_gm,      NoofSim*sizeof(float) );
   cudaMalloc( &decl_sim_gm,    NoofSim*sizeof(float) );

// Better to use cudaMallocManaged in order to allocate memory in Gpu and use it directly in the Gpu (we do not have to use Memcopy)
   cudaMalloc( &histogramDR_gm,    numBins*sizeof(unsigned int) );
   cudaMalloc( &histogramDD_gm,    numBins*sizeof(unsigned int) );
   cudaMalloc( &histogramRR_gm,    numBins*sizeof(unsigned int) );

   // copy data to the GPU
   cudaMemcpy( ra_real_gm, ra_real, NoofReal*sizeof(float), cudaMemcpyHostToDevice );
   cudaMemcpy( decl_real_gm, decl_real, NoofReal*sizeof(float), cudaMemcpyHostToDevice );
   cudaMemcpy( ra_sim_gm, ra_sim, NoofSim*sizeof(float), cudaMemcpyHostToDevice );
   cudaMemcpy( decl_sim_gm, decl_sim, NoofSim*sizeof(float), cudaMemcpyHostToDevice );
   cudaMemset( histogramDR_gm, 0, numBins*sizeof(unsigned int) );
   cudaMemset( histogramDD_gm, 0, numBins*sizeof(unsigned int) );
   cudaMemset( histogramRR_gm, 0, numBins*sizeof(unsigned int) );

   
   noofblocks = (int)(( (NoofReal > NoofSim ? NoofReal : NoofSim) + threadsperblock - 1) / threadsperblock);
   calculateHistograms<<< noofblocks, threadsperblock >>>( ra_real_gm, decl_real_gm, ra_sim_gm, decl_sim_gm, histogramDD_gm, histogramDR_gm, histogramRR_gm, NoofReal, NoofSim );

   // copy the results back to the CPU
   cudaMemcpy( histogramDD, histogramDD_gm, numBins*sizeof(unsigned int), cudaMemcpyDeviceToHost );
   cudaMemcpy( histogramDR, histogramDR_gm, numBins*sizeof(unsigned int), cudaMemcpyDeviceToHost );
   cudaMemcpy( histogramRR, histogramRR_gm, numBins*sizeof(unsigned int), cudaMemcpyDeviceToHost );

   // calculate omega values on the CPU
   calculateOmega();
   printResult();

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

