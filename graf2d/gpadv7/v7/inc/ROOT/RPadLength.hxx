/// \file ROOT/RPadLength.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-07-06
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPadLength
#define ROOT7_RPadLength

#include <string>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RPadLength
  A coordinate in a `RPad`.
  */

class RPadLength {
public:
   template <class DERIVED>
   struct CoordSysBase {
      double fVal = 0.; ///< Coordinate value

      CoordSysBase() = default;
      CoordSysBase(double val): fVal(val) {}
      DERIVED &ToDerived() { return static_cast<DERIVED &>(*this); }

      DERIVED operator-() { return DERIVED(-fVal); }

      friend DERIVED operator+(DERIVED lhs, DERIVED rhs) { return DERIVED{lhs.fVal + rhs.fVal}; }
      friend DERIVED operator-(DERIVED lhs, DERIVED rhs) { return DERIVED{lhs.fVal - rhs.fVal}; }
      friend double operator/(DERIVED lhs, DERIVED rhs) { return lhs.fVal / rhs.fVal; }
      DERIVED &operator+=(const DERIVED &rhs)
      {
         fVal += rhs.fVal;
         return ToDerived();
      }
      DERIVED &operator-=(const DERIVED &rhs)
      {
         fVal -= rhs.fVal;
         return ToDerived();
      }
      DERIVED &operator*=(double scale)
      {
         fVal *= scale;
         return ToDerived();
      }
      friend DERIVED operator*(const DERIVED &lhs, double rhs) { return DERIVED(lhs.fVal * rhs); }
      friend DERIVED operator*(double lhs, const DERIVED &rhs) { return DERIVED(lhs * rhs.fVal); }
      friend DERIVED operator/(const DERIVED &lhs, double rhs) { return DERIVED(lhs.fVal * rhs); }
      friend bool operator<(const DERIVED &lhs, const DERIVED &rhs) { return lhs.fVal < rhs.fVal; }
      friend bool operator>(const DERIVED &lhs, const DERIVED &rhs) { return lhs.fVal > rhs.fVal; }
      friend bool operator<=(const DERIVED &lhs, const DERIVED &rhs) { return lhs.fVal <= rhs.fVal; }
      friend bool operator>=(const DERIVED &lhs, const DERIVED &rhs) { return lhs.fVal >= rhs.fVal; }
      // no ==, !=
   };

   /// \defgroup PadCoordSystems RPad coordinate systems
   /// These define typesafe coordinates used by RPad to identify which coordinate system a coordinate is referring to.
   /// The origin (0,0) is in the `RPad`'s bottom left corner for all of them.
   /// \{

   /** \class Normal
     A normalized coordinate: 0 in the left, bottom corner, 1 in the top, right corner of the `RPad`. Resizing the pad
     will resize the objects with it.
    */
   struct Normal: CoordSysBase<Normal> {
      using CoordSysBase<Normal>::CoordSysBase;
   };

   /** \class Pixel
     A pixel coordinate: 0 in the left, bottom corner, 1 in the top, right corner of the `RPad`. Resizing the pad will
     keep the pixel-position of the objects positioned in `Pixel` coordinates.
    */
   struct Pixel: CoordSysBase<Pixel> {
      using CoordSysBase<Pixel>::CoordSysBase;
   };

   /** \class User
     A user coordinate, as defined by the EUserCoordSystem parameter of the `RPad`.
    */
   struct User: CoordSysBase<User> {
      using CoordSysBase<User>::CoordSysBase;
   };
   /// \}

   /// The normalized coordinate summand.
   Normal fNormal;

   /// The pixel coordinate summand.
   Pixel fPixel;

   /// The user coordinate summand.
   User fUser;

   /// Default constructor, initializing all coordinate parts to `0.`.
   RPadLength() = default;

   /// Constructor from a `Normal` coordinate.
   RPadLength(Normal normal): fNormal(normal) {}

   /// Constructor from a `Pixel` coordinate.
   RPadLength(Pixel px): fPixel(px) {}

   /// Constructor from a `User` coordinate.
   RPadLength(User user): fUser(user) {}

   /// Sort-of aggregate initialization constructor taking normal, pixel and user parts.
   RPadLength(Normal normal, Pixel px, User user): fNormal(normal), fPixel(px), fUser(user) {}

   /// Add two `RPadLength`s.
   friend RPadLength operator+(RPadLength lhs, const RPadLength &rhs)
   {
      return RPadLength{lhs.fNormal + rhs.fNormal, lhs.fPixel + rhs.fPixel, lhs.fUser + rhs.fUser};
   }

   /// Subtract two `RPadLength`s.
   friend RPadLength operator-(RPadLength lhs, const RPadLength &rhs)
   {
      return RPadLength{lhs.fNormal - rhs.fNormal, lhs.fPixel - rhs.fPixel, lhs.fUser - rhs.fUser};
   }

   /// Unary -.
   RPadLength operator-() {
      return RPadLength(-fNormal, -fPixel, -fUser);
   }

   /// Add a `RPadLength`.
   RPadLength &operator+=(const RPadLength &rhs)
   {
      fNormal += rhs.fNormal;
      fPixel += rhs.fPixel;
      fUser += rhs.fUser;
      return *this;
   };

   /// Subtract a `RPadLength`.
   RPadLength &operator-=(const RPadLength &rhs)
   {
      fNormal -= rhs.fNormal;
      fPixel -= rhs.fPixel;
      fUser -= rhs.fUser;
      return *this;
   };

   RPadLength &operator*=(double scale)
   {
      fNormal *= scale;
      fPixel *= scale;
      fUser *= scale;
      return *this;
   }

   void SetFromAttrString(const std::string &name, const std::string &attrStrVal);
};

/// User-defined literal for `RPadLength::Normal`
///
/// Use as
/// ```
/// using namespace ROOT::Experimental;
/// RLine(0.1_normal, 0.0_normal, RLineExtent(0.2_normal, 0.5_normal));
/// ```
inline RPadLength::Normal operator"" _normal(long double val)
{
   return RPadLength::Normal{(double)val};
}
inline RPadLength::Normal operator"" _normal(unsigned long long int val)
{
   return RPadLength::Normal{(double)val};
}

/// User-defined literal for `RPadLength::Pixel`
///
/// Use as
/// ```
/// using namespace ROOT::Experimental;
/// RLine(100_px, 0_px, RLineExtent(20_px, 50_px));
/// ```
inline RPadLength::Pixel operator"" _px(long double val)
{
   return RPadLength::Pixel{(double)val};
}
inline RPadLength::Pixel operator"" _px(unsigned long long int val)
{
   return RPadLength::Pixel{(double)val};
}

/// User-defined literal for `RPadLength::User`
///
/// Use as
/// ```
/// using namespace ROOT::Experimental;
/// RLine(0.1_user, 0.0_user, RLineExtent(0.2_user, 0.5_user));
/// ```
inline RPadLength::User operator"" _user(long double val)
{
   return RPadLength::User{(double)val};
}
inline RPadLength::User operator"" _user(unsigned long long int val)
{
   return RPadLength::User{(double)val};
}

} // namespace Experimental
} // namespace ROOT

#endif
