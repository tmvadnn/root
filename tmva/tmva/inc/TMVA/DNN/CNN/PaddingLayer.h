// @(#)root/tmva/tmva/dnn:$Id$
// Author: Siddhartha Rao Kamalakara

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TConvLayer                                                            *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Padding Layer                                                             *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Siddhartha Rao Kamalakara        <srk97c@gmail.com>  - CERN, Switzerland  *
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

#ifndef TMVA_CNN_PADDINGLAYER
#define TMVA_CNN_PADDINGLAYER

#include "TMatrix.h"

#include "TMVA/DNN/GeneralLayer.h"
#include "TMVA/DNN/Functions.h"

#include <vector>
#include <iostream>

namespace TMVA {
namespace DNN {
namespace CNN {

template <typename Architecture_t>
class TPaddingLayer : public VGeneralLayer<Architecture_t>
{

public:
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;

private:
	size_t fTopPad;
	size_t fBottomPad;
	size_t fLeftPad;
	size_t fRightPad;
	size_t outputWidth;
	size_t outputHeight;

	size_t calculateDimension(size_t imgHeight, size_t imgWidth, size_t pad_left, size_t pad_right, size_t pad_top, size_t pad_bottom);

public:
	/*! Constructor. */
	TPaddingLayer(size_t BatchSize, size_t inputDepth, size_t inputHeight, size_t inputWidth, size_t depth, size_t height, size_t width, size_t TopPad, size_t BottomPad, size_t LeftPad, size_t RightPad);

	/*! Copy the conv layer provided as a pointer */
	TPaddingLayer(TPaddingLayer<Architecture_t> *layer);

	/*! Copy constructor. */
	TPaddingLayer(const TPaddingLayer &);

	/*! Destructor. */
	~TPaddingLayer();

	/*! Pads the input array with the dimensions given by
	 *  the user. Padding is done in two dimensions for each
	 *  example in the batch */
	void Forward(std::vector<Matrix_t> &input, bool applyDropout = false);

	/*! Discards the gradients through the padded inputs
	 *  since they are zero padded */
	void Backward(std::vector<Matrix_t> &gradients_backward,
	                                             const std::vector<Matrix_t> & /*activations_backward*/,
	                                             std::vector<Matrix_t> & /*inp1*/, std::vector<Matrix_t> &
	                                             /*inp2*/);

  /*! Writes the information and the weights about the layer in an XML node. */
  virtual void AddWeightsXMLTo(void *parent);

  /*! Read the information and the weights about the layer from XML node. */
  virtual void ReadWeightsFromXML(void *parent);

  /*! Prints the info about the layer. */
  void Print() const;		

  size_t GetTopPadding() const {return fTopPad;}

  size_t GetBottomPadding() const {return fBottomPad;}

  size_t GetLeftPadding() const {return fLeftPad;}

  size_t GetRightPadding() const {return fRightPad;}

  size_t GetOutputHeight() const {return outputHeight;}

  size_t GetOutputWidth() const {return outputWidth;}


};

template <typename Architecture_t>
TPaddingLayer<Architecture_t>::TPaddingLayer(size_t batchSize, size_t inputDepth, size_t inputHeight, size_t inputWidth,
                                             size_t depth, size_t height, size_t width,
                                             size_t topPad, size_t bottomPad, size_t leftPad, size_t rightPad)
   : VGeneralLayer<Architecture_t>(batchSize, inputDepth, inputHeight, inputWidth, depth, height, width, 0, 0, 0, 0, 0,
                                   0, batchSize, inputDepth, calculateDimension(inputHeight, inputWidth, leftPad, rightPad, topPad, bottomPad), EInitialization::kZero),
     fTopPad(topPad), fBottomPad(bottomPad), fLeftPad(leftPad), fRightPad(rightPad)
{

	this->outputHeight = inputHeight + topPad + bottomPad;
	this->outputWidth = inputWidth + leftPad + rightPad;	
}


//_________________________________________________________________________________________________
template <typename Architecture_t>
TPaddingLayer<Architecture_t>::TPaddingLayer(TPaddingLayer<Architecture_t> *layer)
   : VGeneralLayer<Architecture_t>(layer), fTopPad(layer->GetTopPadding()), fBottomPad(layer->GetBottomPadding()),
   	fLeftPad(layer->GetLeftPadding()), fRightPad(layer->GetRightPadding())
{
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
TPaddingLayer<Architecture_t>::TPaddingLayer(const TPaddingLayer &layer)
   : VGeneralLayer<Architecture_t>(layer), fTopPad(layer.fTopPad), fBottomPad(layer.fBottomPad),
   	fLeftPad(layer.fLeftPad), fRightPad(layer.fRightPad)
{
   // Nothing to do here.
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
TPaddingLayer<Architecture_t>::~TPaddingLayer()
{
   // Nothing to do here.
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
auto TPaddingLayer<Architecture_t>::Forward(std::vector<Matrix_t> &input, bool /*applyDropout*/) -> void
{

  for (size_t i = 0; i < this->GetBatchSize(); i++) {
     Architecture_t::ZeroPad2DForward(this->GetOutputAt(i), input[i], fTopPad, fBottomPad, fLeftPad, fRightPad, this->GetOutputHeight(), this->GetOutputWidth());
  }

}

//_________________________________________________________________________________________________
template <typename Architecture_t>
auto TPaddingLayer<Architecture_t>::Backward(std::vector<Matrix_t> &gradients_backward,
                                             const std::vector<Matrix_t> & /*activations_backward*/,
                                             std::vector<Matrix_t> & /*inp1*/, std::vector<Matrix_t> &
                                             /*inp2*/) -> void
{
	Architecture_t::ZeroPad2DBackward(gradients_backward, this->GetActivationGradients(), fTopPad, fBottomPad, fLeftPad,
									  fRightPad, this->GetOutputHeight(), this->GetOutputWidth(), this->GetBatchSize(),
									  this->GetDepth());
}

//_________________________________________________________________________________________________
template <typename Architecture_t>
auto TPaddingLayer<Architecture_t>::Print() const -> void
{
   std::cout << " PADDING Layer \t ";
   std::cout << "Input = ( " << this->GetInputDepth() << " , " <<  this->GetInputHeight() << " , " << this->GetInputWidth() << " ) ";
   if (this->GetOutput().size() > 0) {
      std::cout << "\tOutput = ( " << this->GetOutput().size() << " , " << this->GetOutput()[0].GetNrows() << " , " << this->GetOutput()[0].GetNcols() << " ) ";
   }
   std::cout << std::endl;
}

template <typename Architecture_t>
auto TPaddingLayer<Architecture_t>::AddWeightsXMLTo(void *parent) -> void
{
   auto layerxml = gTools().xmlengine().NewChild(parent, 0, "PaddingLayer");

   // write info for padding layer
   gTools().xmlengine().NewAttr(layerxml, 0, "LeftPad", gTools().StringFromInt(this->GetLeftPadding()));
   gTools().xmlengine().NewAttr(layerxml, 0, "RightPad", gTools().StringFromInt(this->GetRightPadding()));
   gTools().xmlengine().NewAttr(layerxml, 0, "TopPad", gTools().StringFromInt(this->GetTopPadding()));
   gTools().xmlengine().NewAttr(layerxml, 0, "BottomPad", gTools().StringFromInt(this->GetBottomPadding()));


}

//______________________________________________________________________________
template <typename Architecture_t>
void TPaddingLayer<Architecture_t>::ReadWeightsFromXML(void * /*parent*/)
{
   // no info to read
}


template <typename Architecture_t>
size_t TPaddingLayer<Architecture_t>::calculateDimension(size_t imgHeight, size_t imgWidth, size_t pad_left, size_t pad_right, size_t pad_top, size_t pad_bottom){

	size_t height = imgHeight + pad_top + pad_bottom;
	size_t width  = imgWidth + pad_left + pad_right;

	return height*width;
}


} // namespace DNN
} // namespace TMVA
}

#endif