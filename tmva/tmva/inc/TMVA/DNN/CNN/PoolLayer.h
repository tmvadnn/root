// @(#)root/tmva/tmva/dnn:$Id$
// Author: Emmanouil Stergiadis

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TAvgPoolLayer                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Pooling Deep Neural Network Layer                                        *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Emmanouil Stergiadis      <em.stergiadis@gmail.com>  - JADS, Netherlands  *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef POOLLAYER_H_
#define POOLLAYER_H_

#include "TMatrix.h"

#include "TMVA/DNN/GeneralLayer.h"
#include "TMVA/DNN/Functions.h"

#include <iostream>

namespace TMVA {
namespace DNN {
namespace CNN {

/** \class TAvgPoolLayer

Generic Pooling Layer class.

This generic Pooling Layer Class represents a pooling layer of
a CNN. It inherits all of the properties of the generic virtual base class
VGeneralLayer.

The height and width of the weights and biases is set to 0, since this
layer does not contain any weights.

The supported pooling methods are currently max (default) and average.

*/
template <typename Architecture_t>
class TPoolLayer : public VGeneralLayer<Architecture_t> {
public:
    using Matrix_t = typename Architecture_t::Matrix_t;
    using Scalar_t = typename Architecture_t::Scalar_t;

private:
    std::vector<Matrix_t> indexMatrix; ///< Matrix of indices for the backward pass in case MaxPooling is used.

    size_t fFrameHeight; ///< The height of the frame.
    size_t fFrameWidth;  ///< The width of the frame.

    size_t fStrideRows; ///< The number of row pixels to slid the filter each step.
    size_t fStrideCols; ///< The number of column pixels to slid the filter each step.

    size_t fNLocalViewPixels; ///< The number of pixels in one local image view.
    size_t fNLocalViews;      ///< The number of local views in one image.

    Scalar_t fDropoutProbability; ///< Probability that an input is active.

    std::string fMethod; ///< Method to be used for pooling.

public:
    /*! Constructor. */
    TPoolLayer(size_t BatchSize, size_t InputDepth, size_t InputHeight, size_t InputWidth, size_t Height,
                  size_t Width, size_t OutputNSlices, size_t OutputNRows, size_t OutputNCols, size_t FrameHeight,
                  size_t FrameWidth, size_t StrideRows, size_t StrideCols, Scalar_t DropoutProbability, std::string Method = "max");

    /*! Copy the max pooling layer provided as a pointer */
    TPoolLayer(TPoolLayer<Architecture_t> *layer);

    /*! Copy constructor. */
    TPoolLayer(const TPoolLayer &);

    /*! Destructor. */
    ~TPoolLayer();

    /*! Computes activation of the layer for the given input. The input
     *  must be in 3D tensor form with the different matrices corresponding to
     *  different events in the batch. It spatially downsamples the input
     *  matrices. */
    void Forward(std::vector<Matrix_t> &input, bool applyDropout = false);

    /*! Depending on the winning units determined during the Forward pass,
     *  it only forwards the derivatives to the right units in the previous
     *  layer. Must only be called directly at the corresponding call
     *  to Forward(...). */
    void Backward(std::vector<Matrix_t> &gradients_backward, const std::vector<Matrix_t> &activations_backward,
                  std::vector<Matrix_t> &inp1, std::vector<Matrix_t> &inp2);

    /*! Writes the information and the weights about the layer in an XML node. */
    virtual void AddWeightsXMLTo(void *parent);

    /*! Read the information and the weights about the layer from XML node. */
    virtual void ReadWeightsFromXML(void *parent);


    /*! Prints the info about the layer. */
    void Print() const;

    /*! Getters */
    const std::vector<Matrix_t> &GetIndexMatrix() const
    {
        if(method == "max") {
            return indexMatrix;
        }
        throw std::invalid_argument("Average Pooling does not include an IndexMatrix.");

    }
    std::vector<Matrix_t> &GetIndexMatrix()
    {
        if (method == "max") {
            return indexMatrix;
        }
        throw std::invalid_argument("Average Pooling does not include an IndexMatrix.");
    }

    size_t GetFrameHeight() const { return fFrameHeight; }
    size_t GetFrameWidth() const { return fFrameWidth; }

    size_t GetStrideRows() const { return fStrideRows; }
    size_t GetStrideCols() const { return fStrideCols; }

    size_t GetNLocalViewPixels() const { return fNLocalViewPixels; }
    size_t GetNLocalViews() const { return fNLocalViews; }

    Scalar_t GetDropoutProbability() const { return fDropoutProbability; }
    std::string GetMethod() const {return fMethod; }
};

//______________________________________________________________________________
template <typename Architecture_t>
TPoolLayer<Architecture_t>::TPoolLayer(size_t batchSize, size_t inputDepth, size_t inputHeight, size_t inputWidth,
                                             size_t height, size_t width, size_t outputNSlices, size_t outputNRows,
                                             size_t outputNCols, size_t frameHeight, size_t frameWidth,
                                             size_t strideRows, size_t strideCols, Scalar_t dropoutProbability, std::string method)
        : VGeneralLayer<Architecture_t>(batchSize, inputDepth, inputHeight, inputWidth, inputDepth, height, width, 0, 0, 0,
                                        0, 0, 0, outputNSlices, outputNRows, outputNCols, EInitialization::kZero),
          fFrameHeight(frameHeight), fFrameWidth(frameWidth), fStrideRows(strideRows),
          fStrideCols(strideCols), fNLocalViewPixels(inputDepth * frameHeight * frameWidth), fNLocalViews(height * width),
          fDropoutProbability(dropoutProbability), fMethod(method)
{
    if(fMethod == "max") {
        for (size_t i = 0; i < this->GetBatchSize(); i++) {
            indexMatrix.emplace_back(this->GetDepth(), this->GetNLocalViews());
        }
    }
}


//______________________________________________________________________________
template <typename Architecture_t>
TPoolLayer<Architecture_t>::TPoolLayer(TPoolLayer<Architecture_t> *layer)
        : VGeneralLayer<Architecture_t>(layer), fFrameHeight(layer->GetFrameHeight()),
          fFrameWidth(layer->GetFrameWidth()), fStrideRows(layer->GetStrideRows()), fStrideCols(layer->GetStrideCols()),
          fNLocalViewPixels(layer->GetNLocalViewPixels()), fNLocalViews(layer->GetNLocalViews()),
          fDropoutProbability(layer->GetDropoutProbability()), fMethod(layer->method)
{
    if(fMethod == "max") {
        for (size_t i = 0; i < layer->GetBatchSize(); i++) {
            indexMatrix.emplace_back(layer->GetDepth(), layer->GetNLocalViews());
        }
    }
}

//______________________________________________________________________________
template <typename Architecture_t>
TPoolLayer<Architecture_t>::TPoolLayer(const TPoolLayer &layer)
        : VGeneralLayer<Architecture_t>(layer), fFrameHeight(layer.fFrameHeight),
          fFrameWidth(layer.fFrameWidth), fStrideRows(layer.fStrideRows), fStrideCols(layer.fStrideCols),
          fNLocalViewPixels(layer.fNLocalViewPixels), fNLocalViews(layer.fNLocalViews),
          fDropoutProbability(layer.fDropoutProbability), fMethod(layer.method)
{
    if(fMethod == "max") {
        for (size_t i = 0; i < layer.fBatchSize; i++) {
            indexMatrix.emplace_back(layer.fDepth, layer.fNLocalViews);
        }
    }
}

//______________________________________________________________________________
template <typename Architecture_t>
TPoolLayer<Architecture_t>::~TPoolLayer()
{
}

//______________________________________________________________________________
template <typename Architecture_t>
auto TPoolLayer<Architecture_t>::Forward(std::vector<Matrix_t> &input, bool applyDropout) -> void
{
    for (size_t i = 0; i < this->GetBatchSize(); i++) {

        if (applyDropout && (this->GetDropoutProbability() != 1.0)) {
            Architecture_t::Dropout(input[i], this->GetDropoutProbability());
        }

        Architecture_t::Downsample(this, input[i], i);
    }
}

//______________________________________________________________________________
template <typename Architecture_t>
auto TPoolLayer<Architecture_t>::Backward(std::vector<Matrix_t> &gradients_backward,
                                             const std::vector<Matrix_t> & /*activations_backward*/,
                                             std::vector<Matrix_t> & /*inp1*/, std::vector<Matrix_t> &
        /*inp2*/) -> void
{
    Architecture_t::PoolLayerBackward(gradients_backward, this);
}

//______________________________________________________________________________
template <typename Architecture_t>
auto TPoolLayer<Architecture_t>::Print() const -> void
{
    std::cout << "\t\t POOL LAYER: " << std::endl;
    std::cout << "\t\t\t Width = " << this->GetWidth() << std::endl;
    std::cout << "\t\t\t Height = " << this->GetHeight() << std::endl;
    std::cout << "\t\t\t Depth = " << this->GetDepth() << std::endl;

    std::cout << "\t\t\t Frame Width = " << this->GetFrameWidth() << std::endl;
    std::cout << "\t\t\t Frame Height = " << this->GetFrameHeight() << std::endl;
}

//______________________________________________________________________________
template <typename Architecture_t>
void TPoolLayer<Architecture_t>::AddWeightsXMLTo(void *parent)
{
    auto layerxml = gTools().xmlengine().NewChild(parent, 0, "PoolLayer");

    // write  maxpool layer info
    gTools().xmlengine().NewAttr(layerxml, 0, "FrameHeight", gTools().StringFromInt(this->GetFrameHeight()));
    gTools().xmlengine().NewAttr(layerxml, 0, "FrameWidth", gTools().StringFromInt(this->GetFrameWidth()));
    gTools().xmlengine().NewAttr(layerxml, 0, "StrideRows", gTools().StringFromInt(this->GetStrideRows()));
    gTools().xmlengine().NewAttr(layerxml, 0, "StrideCols", gTools().StringFromInt(this->GetStrideCols()));
    gTools().xmlengine().NewAttr(layerxml, 0, "Method", this->GetMethod().c_str());

}

//______________________________________________________________________________
template <typename Architecture_t>
void TPoolLayer<Architecture_t>::ReadWeightsFromXML(void * /*parent */)
{
    // all info is read before - nothing to do
}

} // namespace CNN
} // namespace DNN
} // namespace TMVA

#endif
