// @(#)root/tmva/tmva/dnn:$Id$
// Author: Ravi Kiran S

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TSGD                                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Stochastic Batch Gradient Descent Optimizer Class                         *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Ravi Kiran S      <sravikiran0606@gmail.com>  - CERN, Switzerland         *
 *                                                                                *
 * Copyright (c) 2005-2018:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef TMVA_DNN_SGD
#define TMVA_DNN_SGD

#include "TMatrix.h"
#include "TMVA/DNN/Optimizer.h"
#include "TMVA/DNN/Functions.h"

namespace TMVA {
namespace DNN {

/** \class TSGD
    Stochastic Batch Gradient Descent Optimizer class

    This class represents the Stochastic Batch Gradient Descent Optimizer with options for applying momentum
    and nesterov momentum.
 */
template <typename Architecture_t, typename Layer_t = VGeneralLayer<Architecture_t>,
          typename DeepNet_t = TDeepNet<Architecture_t, Layer_t>>
class TSGD : public VOptimizer<Architecture_t, Layer_t, DeepNet_t> {
public:
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;

protected:
   Scalar_t fMomentum; ///< The momentum used for training.
   std::vector<std::vector<Matrix_t>>
      fPastWeightGradients; ///< The sum of the past weight gradients associated with the deep net.
   std::vector<std::vector<Matrix_t>>
      fPastBiasGradients; ///< The sum of the past bias gradients associated with the deep net.

   /*! Update the weights, given the current weight gradients. */
   void UpdateWeights(size_t layerIndex, std::vector<Matrix_t> &weights, const std::vector<Matrix_t> &weightGradients);

   /*! Update the biases, given the current bias gradients. */
   void UpdateBiases(size_t layerIndex, std::vector<Matrix_t> &biases, const std::vector<Matrix_t> &biasGradients);

public:
   /*! Constructor. */
   TSGD(Scalar_t learningRate, DeepNet_t &deepNet, Scalar_t momentum);

   /*! Destructor. */
   ~TSGD() = default;

   /*! Getters */
   Scalar_t GetMomentum() const { return fMomentum; }

   std::vector<std::vector<Matrix_t>> &GetPastWeightGradients() { return fPastWeightGradients; }
   std::vector<Matrix_t> &GetPastWeightGradientsAt(size_t i) { return fPastWeightGradients[i]; }

   std::vector<std::vector<Matrix_t>> &GetPastBiasGradients() { return fPastBiasGradients; }
   std::vector<Matrix_t> &GetPastBiasGradientsAt(size_t i) { return fPastBiasGradients[i]; }
};

//
//
//  The Stochastic Gradient Descent Optimizer Class - Implementation
//_________________________________________________________________________________________________
template <typename Architecture_t, typename Layer_t, typename DeepNet_t>
TSGD<Architecture_t, Layer_t, DeepNet_t>::TSGD(Scalar_t learningRate, DeepNet_t &deepNet, Scalar_t momentum)
   : VOptimizer<Architecture_t, Layer_t, DeepNet_t>(learningRate, deepNet), fMomentum(momentum)
{
   std::vector<Layer_t *> &layers = deepNet.GetLayers();
   size_t layersNSlices = layers.size();
   fPastWeightGradients.resize(layersNSlices);
   fPastBiasGradients.resize(layersNSlices);

   for (size_t i = 0; i < layersNSlices; i++) {
      size_t weightsNSlices = (layers[i]->GetWeights()).size();

      for (size_t j = 0; j < weightsNSlices; j++) {
         Matrix_t &currentWeights = layers[i]->GetWeightsAt(j);
         size_t weightsNRows = currentWeights.GetNrows();
         size_t weightsNCols = currentWeights.GetNcols();

         fPastWeightGradients[i].emplace_back(weightsNRows, weightsNCols);
         initialize<Architecture_t>(fPastWeightGradients[i][j], EInitialization::kZero);
      }

      size_t biasesNSlices = (layers[i]->GetBiases()).size();

      for (size_t j = 0; j < biasesNSlices; j++) {
         Matrix_t &currentBiases = layers[i]->GetBiasesAt(j);
         size_t biasesNRows = currentBiases.GetNrows();
         size_t biasesNCols = currentBiases.GetNcols();

         fPastBiasGradients[i].emplace_back(biasesNRows, biasesNCols);
         initialize<Architecture_t>(fPastBiasGradients[i][j], EInitialization::kZero);
      }
   }
}

// //_________________________________________________________________________________________________
// template <typename Architecture_t, typename Layer_t, typename DeepNet_t>
// TSGD<Architecture_t, Layer_t, DeepNet_t>::~TSGD()
// {
// }

//_________________________________________________________________________________________________
template <typename Architecture_t, typename Layer_t, typename DeepNet_t>
auto TSGD<Architecture_t, Layer_t, DeepNet_t>::UpdateWeights(size_t layerIndex, std::vector<Matrix_t> &weights,
                                                             const std::vector<Matrix_t> &weightGradients) -> void
{
   // accumulating the current layer past weight gradients to include the current weight gradients.
   // Vt = momentum * Vt-1 + currentGradients
   std::vector<Matrix_t> &currentLayerPastWeightGradients = this->GetPastWeightGradientsAt(layerIndex);
   for (size_t k = 0; k < currentLayerPastWeightGradients.size(); k++) {
      Matrix_t accumulation(currentLayerPastWeightGradients[k].GetNrows(),
                            currentLayerPastWeightGradients[k].GetNcols());
      initialize<Architecture_t>(accumulation, EInitialization::kZero);
      Architecture_t::ScaleAdd(accumulation, currentLayerPastWeightGradients[k], this->GetMomentum());
      Architecture_t::ScaleAdd(accumulation, weightGradients[k], 1.0);
      Architecture_t::Copy(currentLayerPastWeightGradients[k], accumulation);
   }

   // updating the weights.
   // theta = theta - learningRate * Vt
   for (size_t i = 0; i < weights.size(); i++) {
      Architecture_t::ScaleAdd(weights[i], currentLayerPastWeightGradients[i], -this->GetLearningRate());
   }
}

//_________________________________________________________________________________________________
template <typename Architecture_t, typename Layer_t, typename DeepNet_t>
auto TSGD<Architecture_t, Layer_t, DeepNet_t>::UpdateBiases(size_t layerIndex, std::vector<Matrix_t> &biases,
                                                            const std::vector<Matrix_t> &biasGradients) -> void
{
   // accumulating the current layer past bias gradients to include the current bias gradients.
   // Vt = momentum * Vt-1 + currentGradients
   std::vector<Matrix_t> &currentLayerPastBiasGradients = this->GetPastBiasGradientsAt(layerIndex);
   for (size_t k = 0; k < currentLayerPastBiasGradients.size(); k++) {
      Matrix_t accumulation(currentLayerPastBiasGradients[k].GetNrows(), currentLayerPastBiasGradients[k].GetNcols());
      initialize<Architecture_t>(accumulation, EInitialization::kZero);
      Architecture_t::ScaleAdd(accumulation, currentLayerPastBiasGradients[k], this->GetMomentum());
      Architecture_t::ScaleAdd(accumulation, biasGradients[k], 1.0);
      Architecture_t::Copy(currentLayerPastBiasGradients[k], accumulation);
   }

   // updating the biases
   // theta = theta - learningRate * Vt
   for (size_t i = 0; i < biases.size(); i++) {
      Architecture_t::ScaleAdd(biases[i], currentLayerPastBiasGradients[i], -this->GetLearningRate());
   }
}

} // namespace DNN
} // namespace TMVA

#endif