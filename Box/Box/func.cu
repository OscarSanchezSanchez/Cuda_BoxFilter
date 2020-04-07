//****************************************************************************
// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//****************************************************************************

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_profiler_api.h>

//anchura del filtro y declaracion de la matriz que usaremos como filtro en memoria constante, por defecto Laplace 5x5
__constant__ const int filterWidthConstant = 5;
__constant__ float filterMatrixConstant[filterWidthConstant * filterWidthConstant];

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

//comentar "const float* const filter," en la cabecera si se quiere usar memoria de constantes
__global__
void box_filter(const unsigned char* const inputChannel, unsigned char* const outputChannel, int numRows, int numCols,
    /*const float* const filter, */const int filterWidth)
{
    //acceso al identificardor del thread en el conjunto global de threads
    const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                         blockIdx.y * blockDim.y + threadIdx.y);

    //convertimos el identificador 2D del thread a uno en 1D para escribir en la memoria global
    const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

    //comprobamos que el indice obtenido no se sale de los limites de la imagen
    if ( thread_2D_pos.x >= numCols || thread_2D_pos.y>= numRows )
        return;

    //recorremos la matriz del filtro
    float outputPixel = 0.0f;
    for (int i = 0; i < filterWidth; i++)
    {
        for (int j = 0; j < filterWidth; j++)
        {

            int row = (int)(thread_2D_pos.y + (i - filterWidth / 2));
            //comprobamos que seguimos dentro de la imagen en las filas
            if (row < 0)
				row = 0;
			if (row > numRows - 1)
				row = numRows - 1;

            int column = (int)(thread_2D_pos.x + (j - filterWidth / 2));
            //comprobamos que seguimos dentro de la imagen en las columnas
            if (column < 0)
				column = 0;
			if (column > numCols - 1)
				column = numCols - 1;

            //Con memoria de constantes
			outputPixel += (float)filterMatrixConstant[i * filterWidth + j] * (float)(inputChannel[row * numCols + column]);

			//sin memoria de constantes
		    //outputPixel += (float)filter[i * filterWidth + j] * (float)(inputChannel[row * numCols + column]);

        }
    }

    //comprobamos que el color resultado no sea erroneo RGB --> 0-255
    if (outputPixel < 0.0f)
		outputPixel = 0.0f;
	if (outputPixel > 255.0f)
		outputPixel = 255.0f;

	outputChannel[thread_1D_pos] = outputPixel;
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA, int numRows, int numCols, unsigned char* const redChannel,
    unsigned char* const greenChannel,unsigned char* const blueChannel)
{
    //acceso al identificardor del thread en el conjunto global de threads
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		                                 blockIdx.y * blockDim.y + threadIdx.y);

    //convertimos el identificador 2D del thread a uno en 1D para escribir en la memoria global
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//comprobamos que el indice obtenido no se sale de los limites de la imagen
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	//Dividimos los 3 canales de la imagen (RGB)
	redChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].x;
	greenChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].y;
	blueChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].z;
}

//This kernel takes in three color channels and recombines them
//into one image. The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,const unsigned char* const greenChannel,const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,int numRows,int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  // ------------------------       MEMORIA CONSTANTE       ------------------------//
  //Si se quiere usar memoria de constante hay que comentar estas dos funciones.

  //Reservar memoria para el filtro en GPU: d_filter, la cual ya esta declarada
  //checkCudaErrors(cudaMalloc(&d_filter, filterWidth * filterWidth * sizeof(float)));

  // Copiar el filtro  (h_filter) a memoria global de la GPU (d_filter)
  //checkCudaErrors(cudaMemcpy(d_filter, h_filter, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));
}


void create_filter(float **h_filter, int *filterWidth)
{
    //Modificar el tamaño del filtro dependiendo de cual queremos usar.
    //const int KernelWidth = 5; //Memoria global
    const int KernelWidth = filterWidthConstant; //Memoria de constante
    *filterWidth = KernelWidth;

    //create and fill the filter we will convolve with
    *h_filter = new float[KernelWidth * KernelWidth];
  
    /*
    //Filtro gaussiano: blur
    const float KernelSigma = 2.;

    float filterSum = 0.f; //for normalization

    for (int r = -KernelWidth/2; r <= KernelWidth/2; ++r) 
    {
        for (int c = -KernelWidth/2; c <= KernelWidth/2; ++c) 
        {
            float filterValue = expf( -(float)(c * c + r * r) / (2.f * KernelSigma * KernelSigma));
            (*h_filter)[(r + KernelWidth/2) * KernelWidth + c + KernelWidth/2] = filterValue;
            filterSum += filterValue;
        }
    }

    float normalizationFactor = 1.f / filterSum;

    for (int r = -KernelWidth/2; r <= KernelWidth/2; ++r) 
    {
        for (int c = -KernelWidth/2; c <= KernelWidth/2; ++c) {
            (*h_filter)[(r + KernelWidth/2) * KernelWidth + c + KernelWidth/2] *= normalizationFactor;
        }
    }
    */

    //Laplaciano 5x5
    (*h_filter)[0] = 0;   (*h_filter)[1] = 0;    (*h_filter)[2] = -1.;  (*h_filter)[3] = 0;    (*h_filter)[4] = 0;
    (*h_filter)[5] = 1.;  (*h_filter)[6] = -1.;  (*h_filter)[7] = -2.;  (*h_filter)[8] = -1.;  (*h_filter)[9] = 0;
    (*h_filter)[10] = -1.;(*h_filter)[11] = -2.; (*h_filter)[12] = 17.; (*h_filter)[13] = -2.; (*h_filter)[14] = -1.;
    (*h_filter)[15] = 1.; (*h_filter)[16] = -1.; (*h_filter)[17] = -2.; (*h_filter)[18] = -1.; (*h_filter)[19] = 0;
    (*h_filter)[20] = 1.;  (*h_filter)[21] = 0;   (*h_filter)[22] = -1.; (*h_filter)[23] = 0;   (*h_filter)[24] = 0;
  
    //TODO: crear los filtros segun necesidad
    //NOTA: cuidado al establecer el tamaño del filtro a utilizar
    /*
    //Aumentar nitidez 3x3
	(*h_filter)[0] = -0.25;   (*h_filter)[1] = -0.25;    (*h_filter)[2] = -0.25;
	(*h_filter)[3] = -0.25;  (*h_filter)[4] = 3.;  (*h_filter)[5] = -0.25;
	(*h_filter)[6] = -0.25; (*h_filter)[7] = -0.25; (*h_filter)[8] = -0.25;
    */
    /*
	//Suavizado - 5x5
	(*h_filter)[0] = 1;   (*h_filter)[1] = 1;    (*h_filter)[2] = 1.;  (*h_filter)[3] = 1;    (*h_filter)[4] = 1.;
    (*h_filter)[5] = 1.;  (*h_filter)[6] = 4.;  (*h_filter)[7] = 4.;  (*h_filter)[8] = 4.;  (*h_filter)[9] = 1.;
    (*h_filter)[10] = 1.;(*h_filter)[11] = 4.; (*h_filter)[12] = 12.; (*h_filter)[13] = 4.; (*h_filter)[14] = 1.;
    (*h_filter)[15] = 1.; (*h_filter)[16] = 4.; (*h_filter)[17] = 4.; (*h_filter)[18] = 4.; (*h_filter)[19] = 1.;
    (*h_filter)[20] = 1.;  (*h_filter)[21] = 1.;   (*h_filter)[22] = 1.; (*h_filter)[23] = 1.;   (*h_filter)[24] = 1.;
	*/

    /*
	//Detectar bordes - 3x3
	(*h_filter)[0] = 0.;   (*h_filter)[1] = 1.;    (*h_filter)[2] = 0.;
	(*h_filter)[3] = 1.;  (*h_filter)[4] = -4.;  (*h_filter)[5] = 1.;
	(*h_filter)[6] = 0.; (*h_filter)[7] = 1.; (*h_filter)[8] = 0.;
	*/


    //Memoria de constantes(comentar si no se quiere usar), copia a la GPU
    cudaMemcpyToSymbol(filterMatrixConstant, *h_filter, sizeof(float) * KernelWidth * KernelWidth);
}


void convolution(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redFiltered, 
                        unsigned char *d_greenFiltered, 
                        unsigned char *d_blueFiltered,
                        const int filterWidth)
{
  //Calcular tamaños de bloque
    const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    const dim3 gridSize(ceil(1.0f*numCols / blockSize.x), ceil(1.0f*numRows / blockSize.y));

  //Lanzar kernel para separar imagenes RGBA en diferentes colores
    separateChannels << <gridSize, blockSize >> > (d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //Ejecutar convolución. Una por canal
  	//Box Filter
	{
        //comentar "d_filter," para memoria de constante
		box_filter <<<gridSize, blockSize >> > (d_red,   d_redFiltered,   numRows, numCols,/* d_filter, */filterWidth);
		box_filter <<<gridSize, blockSize >> > (d_green, d_greenFiltered, numRows, numCols,/* d_filter, */filterWidth);
		box_filter <<<gridSize, blockSize >> > (d_blue,  d_blueFiltered,  numRows, numCols,/* d_filter, */filterWidth);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}
  

  // Recombining the results. 
  recombineChannels<<<gridSize, blockSize>>>(d_redFiltered,
                                             d_greenFiltered,
                                             d_blueFiltered,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}


//Free all the memory that we allocated
//make sure you free any arrays that you allocated
void cleanup() {
    checkCudaErrors(cudaFree(d_red));
    checkCudaErrors(cudaFree(d_green));
    checkCudaErrors(cudaFree(d_blue));
    //comentar si memoria de constante
    //checkCudaErrors(cudaFree(d_filter));
}
