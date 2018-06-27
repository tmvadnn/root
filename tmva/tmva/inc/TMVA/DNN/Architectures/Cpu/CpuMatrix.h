// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 20/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////
// Definition of the CpuMatrix class used to represent  //
// weight and bias matrices in neural nets.             //
//////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_CPU_CPUMATRIX
#define TMVA_DNN_ARCHITECTURES_CPU_CPUMATRIX

#include <cstddef>
#include <vector>

#include "TMatrix.h"
#include "TMVA/Config.h"
#include "CpuBuffer.h"
#include <TMVA/Config.h>

//#define DEBUG_TMVA_TCPUMATRIX
#if defined(DEBUG_TMVA_TCPUMATRIX)
#define PrintMatrix(mat, text)                                                                             \
   {                                                                                                       \
      auto _dpointer = mat.GetRawDataPointer();                                                            \
      if (_dpointer == NULL) {                                                                             \
         std::cout << #mat << " is null pointer" << std::endl;                                             \
         exit(1);                                                                                          \
      }                                                                                                    \
      auto _nrows = mat.GetNrows();                                                                        \
      auto _ncols = mat.GetNcols();                                                                        \
      std::cout << "---------------------" << text << " " << #mat << "(" << _nrows << "," << _ncols << ")" \
                << "--------------------" << std::endl;                                                    \
      for (size_t _i = 0; _i < _nrows; _i++) {                                                               \
         for (size_t _j = 0; _j < _ncols; _j++) {                                                            \
            std::cout << mat(_i, _j);                                                                      \
            if (_j < _ncols - 1) std::cout << ",";                                                         \
         }                                                                                                 \
         std::cout << std::endl;                                                                           \
      }                                                                                                    \
   }
#else
#define PrintMatrix(mat, text)
#endif

namespace TMVA
{
namespace DNN
{

/** The TCpuMatrix class.
 *
 * Matrix class for multi-threaded CPU architectures. Uses the TCpuBuffer
 * class to store the matrices in column-major format for compatibility with
 * BLAS. Provides Map and MapFrom member functions to simplify the application of
 * activation functions and derivatives to matrices.
 *
 * Copying and assignment of TCpuMatrix objects only performs shallow copies, i.e.
 * copying is fast and the resulting objects share the element data.
 *
 * \tparam AFloat The floating point type used to represent the matrix elements.
 */
//______________________________________________________________________________
template<typename AFloat>
class TCpuMatrix
{
private:
   static std::vector<AFloat> fOnes;  ///< Vector filled with ones used for BLAS calls.

   TCpuBuffer<AFloat> fBuffer; ///< The buffer holding the matrix elements
                               ///< in column-major format.
   size_t     fNCols;
   size_t     fNRows;

public:

   /** Returns pointer to a vector holding only ones with a guaranteed length
    *  of the number of columns of every instantiated CpuMatrix object. */
   static const AFloat * GetOnePointer() {return fOnes.data();}

   static size_t GetOnePointerSize() { return fOnes.size(); }

   static void InitializeOneVector( size_t n); 

   /** Construct matrix and allocate space for its elements. */
   TCpuMatrix(size_t nRows, size_t nCols);
   /** Construct a TCpuMatrix object by (deeply) copying from a
    *  TMatrixT<Double_t> matrix. */
   TCpuMatrix(const TMatrixT<Double_t> &);
   /** Construct a m-times-n matrix from the given buffer. The size must of
    *  course match. */
   TCpuMatrix(const TCpuBuffer<AFloat> &buffer, size_t m, size_t n);

   //N.B the default copy constructor does a shallow copy (NOT a deep one) !
   TCpuMatrix(const TCpuMatrix  &)             = default;
   TCpuMatrix(      TCpuMatrix &&)             = default;
   TCpuMatrix & operator=(const TCpuMatrix &)  = default;
   TCpuMatrix & operator=(TCpuMatrix &&)       = default;
   ~TCpuMatrix()                               = default;

   /** Clear content of the matrix and initialize to zero elements
    */
   void Zero();

   /** Convert to a TMatrixT<Double_t> object. Performs a deep copy of the matrix
    *  elements. */
   operator TMatrixT<Double_t>() const;

   /** Map the given function over the matrix elements. Executed in parallel
    *  using TThreadExecutor. */
   template <typename Function_t>
   void Map(Function_t &f);

   /** Same as maps but takes the input values from the matrix \p A and writes
    *  the results in this matrix. */
   template <typename Function_t>
   void MapFrom(Function_t &f, const TCpuMatrix & A);

   size_t GetNrows() const {return fNRows;}
   size_t GetNcols() const {return fNCols;}
   size_t GetNElements() const {return fNRows * fNCols;}

   /** Return matrix element in row \p i and column \p j. */
   AFloat   operator()(size_t i, size_t j) const {return fBuffer[j * fNRows + i];}
   AFloat & operator()(size_t i, size_t j)       {return fBuffer[j * fNRows + i];}

   /** Return raw pointer to the elements stored contiguously in column-major
    *  order. */
   AFloat *       GetRawDataPointer()        {return fBuffer;}
   const AFloat * GetRawDataPointer()  const {return fBuffer;}

   static ROOT::TThreadExecutor &GetThreadExecutor() { return TMVA::Config::Instance().GetThreadExecutor(); }

    // static function to get the number of elements for task
   static size_t GetNWorkItems(size_t nelements);

   // print matrix
   void Print() const {
      TCpuMatrix cpuMatrix = *this; 
      PrintMatrix(cpuMatrix,"CpuMatrix");
   }
   

private:

   void Initialize();

};

template<typename AFloat>
std::vector<AFloat> TCpuMatrix<AFloat>::fOnes {};


// Inline Functions.
//______________________________________________________________________________
template<typename AFloat>
size_t TCpuMatrix<AFloat>::GetNWorkItems(size_t nElements) 
{
   // const size_t nWorkers = TMVA::Config::Instance().GetNCpu();
   // return  (nElements > nWorkers) ?  (int) nElements/nWorkers : 1;
   const size_t nCpu = TMVA::Config::Instance().GetNCpu();
   if (nElements <= nCpu) return 1;
   if (nElements < nCpu*20) return nElements/nCpu;
   return nElements/(nCpu*10); 
}

   
//______________________________________________________________________________
template<typename AFloat>
template<typename Function_t>
inline void TCpuMatrix<AFloat>::Map(Function_t &f)
{
   AFloat  *data = GetRawDataPointer();
   size_t nelements =  GetNElements();
   size_t nsteps = TCpuMatrix<AFloat>::GetNWorkItems(nelements);

   auto ff = [data, &nsteps, &nelements, &f](UInt_t workerID)
   {
      for (size_t j = 0; j < nsteps; ++j) {
         size_t idx = workerID+j;
         if (idx >= nelements) break; 
         data[idx] = f(data[idx]);
      }
      return 0;
   };

#ifdef DL_USE_MTE
   TMVA::Config::Instance().GetThreadExecutor().Foreach(ff, ROOT::TSeqI(0,nelements,nsteps));
#else
   for (size_t i = 0;  i < nelements; i+=nsteps)
      ff(i);
#endif
}

//______________________________________________________________________________
template<typename AFloat>
template<typename Function_t>
inline void TCpuMatrix<AFloat>::MapFrom(Function_t &f, const TCpuMatrix &A)
{
         AFloat  *dataB = GetRawDataPointer();
   const AFloat  *dataA = A.GetRawDataPointer();

   size_t nelements =  GetNElements();
   R__ASSERT(nelements == A.GetNElements() );
   size_t nsteps = TCpuMatrix<AFloat>::GetNWorkItems(nelements);

   auto ff = [&dataB, &dataA,  &nsteps, &nelements, &f](UInt_t workerID)
   {
      for (size_t j = 0; j < nsteps; ++j) {
         size_t idx = workerID+j;
         if (idx >= nelements) break; 
         dataB[idx] = f(dataA[idx]);
      }
      return 0;
   };
#ifdef DL_USE_MTE
   TMVA::Config::Instance().GetThreadExecutor().Foreach(ff, ROOT::TSeqI(0,nelements,nsteps));
#else
   for (size_t i = 0;  i < nelements; i+=nsteps)
      ff(i);
#endif
}

//______________________________________________________________________________
template<typename AFloat>
void TCpuMatrix<AFloat>::Zero()  
{
   for (size_t j = 0; j < fNCols; j++) {
      for (size_t i = 0; i < fNRows; i++) {
         (*this)(i, j) = 0;
      }
   }
}


} // namespace DNN
} // namespace TMVA

#endif
