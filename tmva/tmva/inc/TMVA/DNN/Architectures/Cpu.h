// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 05/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 //////////////////////////////////////////////////////////////////
// Definition of the TCpu architecture, which provides a         //
 // multi-threaded CPU implementation of the low-level interface //
 // networks for Cpus using BLAS and Roots TThreadExecutor            //
 //////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_CPU
#define TMVA_DNN_ARCHITECTURES_CPU

#include "TMVA/DNN/Functions.h"

#include "Cpu/CpuBuffer.h"
#include "Cpu/CpuMatrix.h"
#include <vector>

class TRandom; 

namespace TMVA
{
namespace DNN
{
   //class EActivationFunction;

/** The TCpu architecture class.
 *
 * Low-level interface class for multi-threaded CPU architectures. Contains as
 * public types the declaration of the scalar, matrix and data loader types
 * for this architecture as well as the remaining functions in the low-level
 * interface in the form of static members.
 */
template<typename AReal = Real_t>
class TCpu
{
private:
   static TRandom * fgRandomGen;
public:

   using Scalar_t       = AReal;
   using Matrix_t       = TCpuMatrix<AReal>;
   using HostBuffer_t   = TCpuBuffer<AReal>;
   using DeviceBuffer_t = TCpuBuffer<AReal>;

   //____________________________________________________________________________
   //
   // Propagation
   //____________________________________________________________________________

   /** @name Forward Propagation
    * Low-level functions required for the forward propagation of activations
    * through the network.
    */
   ///@{
   /** Matrix-multiply \p input with the transpose of \pweights and
    *  write the results into \p output. */
   static void MultiplyTranspose(TCpuMatrix<Scalar_t> &output,
                                 const TCpuMatrix<Scalar_t> &input,
                                 const TCpuMatrix<Scalar_t> &weights);
   /** Add the vectors biases row-wise to the matrix output */
   static void AddRowWise(TCpuMatrix<Scalar_t> &output,
                          const TCpuMatrix<Scalar_t> &biases);
   ///@}

   /** @name Backward Propagation
    * Low-level functions required for the forward propagation of activations
    * through the network.
    */
   ///@{
   /** Perform the complete backward propagation step. If the provided
    *  \p activationGradientsBackward matrix is not empty, compute the
    *  gradients of the objective function with respect to the activations
    *  of the previous layer (backward direction).
    *  Also compute the weight and the bias gradients. Modifies the values
    *  in \p df and thus produces only a valid result, if it is applied the
    *  first time after the corresponding forward propagation has been per-
    *  formed. */
   static void Backward(TCpuMatrix<Scalar_t> & activationGradientsBackward,
                        TCpuMatrix<Scalar_t> & weightGradients,
                        TCpuMatrix<Scalar_t> & biasGradients,
                        TCpuMatrix<Scalar_t> & df,
                        const TCpuMatrix<Scalar_t> & activationGradients,
                        const TCpuMatrix<Scalar_t> & weights,
                        const TCpuMatrix<Scalar_t> & activationBackward);
   /** Backward pass for Recurrent Networks */
   static Matrix_t & RecurrentLayerBackward(TCpuMatrix<Scalar_t> & state_gradients_backward, // BxH
                                            TCpuMatrix<Scalar_t> & input_weight_gradients,
                                            TCpuMatrix<Scalar_t> & state_weight_gradients,
                                            TCpuMatrix<Scalar_t> & bias_gradients,
                                            TCpuMatrix<Scalar_t> & df, //DxH
                                            const TCpuMatrix<Scalar_t> & state, // BxH
                                            const TCpuMatrix<Scalar_t> & weights_input, // HxD 
                                            const TCpuMatrix<Scalar_t> & weights_state, // HxH
                                            const TCpuMatrix<Scalar_t> & input,  // BxD
                                            TCpuMatrix<Scalar_t> & input_gradient);
   /** Adds a the elements in matrix B scaled by c to the elements in
    *  the matrix A. This is required for the weight update in the gradient
    *  descent step.*/
   static void ScaleAdd(TCpuMatrix<Scalar_t> & A,
                        const TCpuMatrix<Scalar_t> & B,
                        Scalar_t beta = 1.0);

   static void Copy(TCpuMatrix<Scalar_t> & B,
                    const TCpuMatrix<Scalar_t> & A);

   // copy from another type of matrix
   template<typename AMatrix_t>
   static void CopyDiffArch(TCpuMatrix<Scalar_t> & B, const AMatrix_t & A); 


   /** Above functions extended to vectors */
   static void ScaleAdd(std::vector<TCpuMatrix<Scalar_t>> & A,
                        const std::vector<TCpuMatrix<Scalar_t>> & B,
                        Scalar_t beta = 1.0);

   static void Copy(std::vector<TCpuMatrix<Scalar_t>> & A,
                    const std::vector<TCpuMatrix<Scalar_t>> & B);

   // copy from another architecture
   template<typename AMatrix_t>
   static void CopyDiffArch(std::vector<TCpuMatrix<Scalar_t>> & A,
                    const std::vector<AMatrix_t> & B);

   ///@}

   //____________________________________________________________________________
   //
   // Activation Functions
   //____________________________________________________________________________

   /** @name Activation Functions
    * For each activation function, the low-level interface contains two routines.
    * One that applies the acitvation function to a matrix and one that evaluate
    * the derivatives of the activation function at the elements of a given matrix
    * and writes the results into the result matrix.
    */
   ///@{
   static void IdentityDerivative(TCpuMatrix<Scalar_t> & B,
                                  const TCpuMatrix<Scalar_t> &A);

   static void Relu(TCpuMatrix<Scalar_t> & B);
   static void ReluDerivative(TCpuMatrix<Scalar_t> & B,
                              const TCpuMatrix<Scalar_t> & A);

   static void Sigmoid(TCpuMatrix<Scalar_t> & B);
   static void SigmoidDerivative(TCpuMatrix<Scalar_t> & B,
                                 const TCpuMatrix<Scalar_t> & A);

   static void Tanh(TCpuMatrix<Scalar_t> & B);
   static void TanhDerivative(TCpuMatrix<Scalar_t> & B,
                              const TCpuMatrix<Scalar_t> & A);

   static void SymmetricRelu(TCpuMatrix<Scalar_t> & B);
   static void SymmetricReluDerivative(TCpuMatrix<Scalar_t> & B,
                                       const TCpuMatrix<Scalar_t> & A);

   static void SoftSign(TCpuMatrix<Scalar_t> & B);
   static void SoftSignDerivative(TCpuMatrix<Scalar_t> & B,
                                  const TCpuMatrix<Scalar_t> & A);

   static void Gauss(TCpuMatrix<Scalar_t> & B);
   static void GaussDerivative(TCpuMatrix<Scalar_t> & B,
                               const TCpuMatrix<Scalar_t> & A);
   ///@}

   //____________________________________________________________________________
   //
   // Loss Functions
   //____________________________________________________________________________

   /** @name Loss Functions
    * Loss functions compute a scalar value given the \p output of the network
    * for a given training input and the expected network prediction \p Y that
    * quantifies the quality of the prediction. For each function also a routing
    * that computes the gradients (suffixed by Gradients) must be provided for
    * the starting of the backpropagation algorithm.
    */
   ///@{

   static Scalar_t MeanSquaredError(const TCpuMatrix<Scalar_t> &Y, const TCpuMatrix<Scalar_t> &output,
                                    const TCpuMatrix<Scalar_t> &weights);
   static void MeanSquaredErrorGradients(TCpuMatrix<Scalar_t> &dY, const TCpuMatrix<Scalar_t> &Y,
                                         const TCpuMatrix<Scalar_t> &output, const TCpuMatrix<Scalar_t> &weights);

   /** Sigmoid transformation is implicitly applied, thus \p output should
    *  hold the linear activations of the last layer in the net. */
   static Scalar_t CrossEntropy(const TCpuMatrix<Scalar_t> &Y, const TCpuMatrix<Scalar_t> &output,
                                const TCpuMatrix<Scalar_t> &weights);

   static void CrossEntropyGradients(TCpuMatrix<Scalar_t> &dY, const TCpuMatrix<Scalar_t> &Y,
                                     const TCpuMatrix<Scalar_t> &output, const TCpuMatrix<Scalar_t> &weights);

   /** Softmax transformation is implicitly applied, thus \p output should
    *  hold the linear activations of the last layer in the net. */
   static Scalar_t SoftmaxCrossEntropy(const TCpuMatrix<Scalar_t> &Y, const TCpuMatrix<Scalar_t> &output,
                                       const TCpuMatrix<Scalar_t> &weights);
   static void SoftmaxCrossEntropyGradients(TCpuMatrix<Scalar_t> &dY, const TCpuMatrix<Scalar_t> &Y,
                                            const TCpuMatrix<Scalar_t> &output, const TCpuMatrix<Scalar_t> &weights);
   ///@}

   //____________________________________________________________________________
   //
   // Output Functions
   //____________________________________________________________________________

   /** @name Output Functions
    * Output functions transform the activations \p output of the
    * output layer in the network to a valid prediction \p YHat for
    * the desired usage of the network, e.g.  the identity function
    * for regression or the sigmoid transformation for two-class
    * classification.
    */
   ///@{
   static void Sigmoid(TCpuMatrix<Scalar_t> &YHat,
                        const TCpuMatrix<Scalar_t> & );
   static void Softmax(TCpuMatrix<Scalar_t> &YHat,
                       const TCpuMatrix<Scalar_t> & );
   ///@}

   //____________________________________________________________________________
   //
   // Regularization
   //____________________________________________________________________________

   /** @name Regularization
    * For each regularization type two functions are required, one named
    * <tt><Type>Regularization</tt> that evaluates the corresponding
    * regularization functional for a given weight matrix and the
    * <tt>Add<Type>RegularizationGradients</tt>, that adds the regularization
    * component in the gradients to the provided matrix.
    */
   ///@{

   static Scalar_t L1Regularization(const TCpuMatrix<Scalar_t> & W);
   static void AddL1RegularizationGradients(TCpuMatrix<Scalar_t> & A,
                                            const TCpuMatrix<Scalar_t> & W,
                                            Scalar_t weightDecay);

   static Scalar_t L2Regularization(const TCpuMatrix<Scalar_t> & W);
   static void AddL2RegularizationGradients(TCpuMatrix<Scalar_t> & A,
                                            const TCpuMatrix<Scalar_t> & W,
                                            Scalar_t weightDecay);
   ///@}

   //____________________________________________________________________________
   //
   // Initialization
   //____________________________________________________________________________

   /** @name Initialization
    * For each initialization method, one function in the low-level interface
    * is provided. The naming scheme is <p>Initialize<Type></p> for a given
    * initialization method Type.
    */
   ///@{

   static void InitializeGauss(TCpuMatrix<Scalar_t> & A);
   static void InitializeUniform(TCpuMatrix<Scalar_t> & A);
   static void InitializeIdentity(TCpuMatrix<Scalar_t> & A);
   static void InitializeZero(TCpuMatrix<Scalar_t> & A);
   static void InitializeGlorotNormal(TCpuMatrix<Scalar_t> & A);
   static void InitializeGlorotUniform(TCpuMatrix<Scalar_t> & A);

   // return static instance of random generator used for initialization
   // if generator does not exist it is created the first time with a random seed (e.g. seed = 0)
   static TRandom & GetRandomGenerator(); 
   // set random seed for the static geenrator
   // if the static geneerator does not exists it is created
   static void SetRandomSeed(size_t seed); 
   ///@}

   //____________________________________________________________________________
   //
   // Dropout
   //____________________________________________________________________________

   /** @name Dropout
    */
   ///@{

   /** Apply dropout with activation probability \p p to the given
    *  matrix \p A and scale the result by reciprocal of \p p. */
   static void Dropout(TCpuMatrix<Scalar_t> & A, Scalar_t p);

   ///@}

   //____________________________________________________________________________
   //
   //  Convolutional Layer Propagation
   //____________________________________________________________________________

   /** @name Forward Propagation in Convolutional Layer
    */
   ///@{

   /** Transform the matrix B in local view format, suitable for
    *  convolution, and store it in matrix A */
   static void Im2col(TCpuMatrix<AReal> &A, const TCpuMatrix<AReal> &B, size_t imgHeight, size_t imgWidth, size_t fltHeight,
                      size_t fltWidth, size_t strideRows, size_t strideCols, size_t zeroPaddingHeight,
                      size_t zeroPaddingWidth);
   static void Im2colIndices(std::vector<int> &V, const TCpuMatrix<AReal> &B, size_t nLocalViews, size_t imgHeight, size_t imgWidth, size_t fltHeight,
                      size_t fltWidth, size_t strideRows, size_t strideCols, size_t zeroPaddingHeight,
                      size_t zeroPaddingWidth);
   static void Im2colFast(TCpuMatrix<AReal> &A, const TCpuMatrix<AReal> &B, const std::vector<int> & V); 

   /** Rotates the matrix \p B, which is representing a weights,
    *  and stores them in the matrix \p A. */
   static void RotateWeights(TCpuMatrix<AReal> &A, const TCpuMatrix<AReal> &B, size_t filterDepth, size_t filterHeight,
                             size_t filterWidth, size_t numFilters);

   /** Add the biases in the Convolutional Layer.  */
   static void AddConvBiases(TCpuMatrix<Scalar_t> &output, const TCpuMatrix<Scalar_t> &biases);
   ///@}

   /** Forward propagation in the Convolutional layer */
   static void ConvLayerForward(std::vector<TCpuMatrix<Scalar_t>> & output, std::vector<TCpuMatrix<Scalar_t>> & derivatives,
                                const std::vector<TCpuMatrix<Scalar_t>> &input,
                                const TCpuMatrix<Scalar_t> & weights, const TCpuMatrix<Scalar_t> & biases,
                                EActivationFunction func, const std::vector<int> & vIndices,
                                size_t nlocalViews, size_t nlocalViewPixels,
                                Scalar_t dropoutProbability, bool applyDropout);

   /** @name Backward Propagation in Convolutional Layer
    */
   ///@{

   /** Perform the complete backward propagation step in a Convolutional Layer.
    *  If the provided \p activationGradientsBackward matrix is not empty, compute the
    *  gradients of the objective function with respect to the activations
    *  of the previous layer (backward direction).
    *  Also compute the weight and the bias gradients. Modifies the values
    *  in \p df and thus produces only a valid result, if it is applied the
    *  first time after the corresponding forward propagation has been per-
    *  formed. */
   static void ConvLayerBackward(std::vector<TCpuMatrix<Scalar_t>> &activationGradientsBackward,
                                 TCpuMatrix<Scalar_t> &weightGradients, TCpuMatrix<Scalar_t> &biasGradients,
                                 std::vector<TCpuMatrix<Scalar_t>> &df,
                                 const std::vector<TCpuMatrix<Scalar_t>> &activationGradients,
                                 const TCpuMatrix<Scalar_t> &weights,
                                 const std::vector<TCpuMatrix<Scalar_t>> &activationBackward, size_t batchSize,
                                 size_t inputHeight, size_t inputWidth, size_t depth, size_t height, size_t width,
                                 size_t filterDepth, size_t filterHeight, size_t filterWidth, size_t nLocalViews);

   /** Utility function for calculating the activation gradients of the layer
    *  before the convolutional layer. */
   static void CalculateConvActivationGradients(std::vector<TCpuMatrix<Scalar_t>> &activationGradientsBackward,
                                                const std::vector<TCpuMatrix<Scalar_t>> &df,
                                                const TCpuMatrix<Scalar_t> &weights, size_t batchSize,
                                                size_t inputHeight, size_t inputWidth, size_t depth, size_t height,
                                                size_t width, size_t filterDepth, size_t filterHeight,
                                                size_t filterWidth);

   /** Utility function for calculating the weight gradients of the convolutional
    * layer. */
   static void CalculateConvWeightGradients(TCpuMatrix<Scalar_t> &weightGradients,
                                            const std::vector<TCpuMatrix<Scalar_t>> &df,
                                            const std::vector<TCpuMatrix<Scalar_t>> &activations_backward,
                                            size_t batchSize, size_t inputHeight, size_t inputWidth, size_t depth,
                                            size_t height, size_t width, size_t filterDepth, size_t filterHeight,
                                            size_t filterWidth, size_t nLocalViews);

   /** Utility function for calculating the bias gradients of the convolutional
    *  layer */
   static void CalculateConvBiasGradients(TCpuMatrix<Scalar_t> &biasGradients, const std::vector<TCpuMatrix<Scalar_t>> &df,
                                          size_t batchSize, size_t depth, size_t nLocalViews);
   ///@}

   //____________________________________________________________________________
   //
   //  Max Pooling Layer Propagation
   //____________________________________________________________________________
   /** @name Forward Propagation in Max Pooling Layer
    */
   ///@{

   /** Downsample the matrix \p C to the matrix \p A, using max
    * operation, such that the winning indices are stored in matrix
    * \p B. */
   static void Downsample(TCpuMatrix<AReal> &A, TCpuMatrix<AReal> &B, const TCpuMatrix<AReal> &C, size_t imgHeight,
                          size_t imgWidth, size_t fltHeight, size_t fltWidth, size_t strideRows, size_t strideCols);

   ///@}

   /** @name Backward Propagation in Max Pooling Layer
    */
   ///@{
   /** Perform the complete backward propagation step in a Pooling Layer. Based on the
    *  winning idices stored in the index matrix, it just forwards the actiovation
    *  gradients to the previous layer. */
   static void MaxPoolLayerBackward(std::vector<TCpuMatrix<AReal>> &activationGradientsBackward,
                                    const std::vector<TCpuMatrix<AReal>> &activationGradients,
                                    const std::vector<TCpuMatrix<AReal>> &indexMatrix, size_t batchSize, size_t depth,
                                    size_t nLocalViews);

   ///@}

   //____________________________________________________________________________
   //
   //  Zero Padding Layer Propagation
   //____________________________________________________________________________
   /** @name Forward Propagation in Zero Padding Layer
    */
   ///@{

  /** Zero Pad the matrix \p B to the matrix \p A, using the
    *  padding dimensions specified.
    *   */
   static void ZeroPad2DForward(TCpuMatrix<AReal> &A, const TCpuMatrix<AReal> &B, 
                                size_t topPad, size_t bottomPad, size_t leftPad,
                                size_t rightPad, size_t outputHeight, size_t outputWidth);

   ///@}

   /** @name Backward Propagation in Zero Padding Layer
    */
   ///@{

   /** Perform the complete backward propagation step in a Zero Padding Layer. The gradients
    *  at the padded positions get discarded. */
   static void ZeroPad2DBackward(std::vector<TCpuMatrix<AReal>> &activationGradientsBackward,
                                 const std::vector<TCpuMatrix<AReal>> &activationGradients,
                                 size_t topPad, size_t bottomPad, size_t leftPad,
                                 size_t rightPad, size_t outputHeight, size_t outputWidth,
                                 size_t batchSize, size_t depth);
   ///@}

   //____________________________________________________________________________
   //
   //  Reshape Layer Propagation
   //____________________________________________________________________________
   /** @name Forward and Backward Propagation in Reshape Layer
    */
   ///@{

   /** Transform the matrix \p B to a matrix with different dimensions \p A */
   static void Reshape(TCpuMatrix<AReal> &A, const TCpuMatrix<AReal> &B);

   /** Flattens the tensor \p B, such that each matrix, is stretched in
    *  one row, resulting with a matrix \p A. */
   static void Flatten(TCpuMatrix<AReal> &A, const std::vector<TCpuMatrix<AReal>> &B, size_t size, size_t nRows,
                       size_t nCols);

   /** Transforms each row of \p B to a matrix and stores it in the
    *  tensor \p B. */
   static void Deflatten(std::vector<TCpuMatrix<AReal>> &A, const TCpuMatrix<AReal> &B, size_t index, size_t nRows,
                         size_t nCols);
   /** Rearrage data accoring to time fill B x T x D out with T x B x D matrix in*/
   static void Rearrange(std::vector<TCpuMatrix<AReal>> &out, const std::vector<TCpuMatrix<AReal>> &in); 


   ///@}

   //____________________________________________________________________________
   //
   // Additional Arithmetic Functions
   //____________________________________________________________________________

   /** @name Additional Arithmetic Functions
    *
    * Additional arithmetic on CUDA matrices  used to implement the low-level
    * interface.
    */
   ///@{

   /** Standard multiplication of two matrices \p A and \p B with the result being
    *  written into C.
    */
   static void Multiply(TCpuMatrix<Scalar_t> &C,
                        const TCpuMatrix<Scalar_t> &A,
                        const TCpuMatrix<Scalar_t> &B);
   /** Matrix multiplication of two matrices \p A and \p B^T (transposed) with the
    *  result being written into C.
    */
   static void TransposeMultiply(TCpuMatrix<Scalar_t> &output,
                                 const TCpuMatrix<Scalar_t> &input,
                                 const TCpuMatrix<Scalar_t> &Weights,
                                 Scalar_t alpha = 1.0, Scalar_t beta = 0.);
   /** In-place Hadamard (element-wise) product of matrices \p A and \p B
    *  with the result being written into \p A.
    */
   static void Hadamard(TCpuMatrix<Scalar_t> &A,
                        const TCpuMatrix<Scalar_t> &B);

   /** Sum columns of (m x n) matrixx \p A and write the results into the first
    * m elements in \p A.
    */
   static void SumColumns(TCpuMatrix<Scalar_t> &B,
                          const TCpuMatrix<Scalar_t> &A,
                          Scalar_t alpha = 1.0, Scalar_t beta = 0.);

   /** Compute the sum of all elements in \p A */
   static Scalar_t Sum(const TCpuMatrix<Scalar_t> &A);

};

//____________________________________________________________________________
template <typename Real_t>
template <typename AMatrix_t>
void TCpu<Real_t>::CopyDiffArch(TCpuMatrix<Real_t> &B,
                        const AMatrix_t &A)
{
   // copy from another architecture using the reference one
   // this is not very efficient since creates temporary objects
   TMatrixT<Real_t> tmp = A;
   Copy(B, TCpuMatrix<Real_t>(tmp) ); 
}

//____________________________________________________________________________
template <typename Real_t>
template <typename AMatrix_t>
void TCpu<Real_t>::CopyDiffArch(std::vector<TCpuMatrix<Real_t>> &B,
                            const std::vector<AMatrix_t> &A)
{
   for (size_t i = 0; i < B.size(); ++i) {
      CopyDiffArch(B[i], A[i]);
   }
}


} // namespace DNN
} // namespace TMVA

#endif
