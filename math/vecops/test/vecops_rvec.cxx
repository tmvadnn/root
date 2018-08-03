#include <gtest/gtest.h>
#include <ROOT/RVec.hxx>
#include <ROOT/TSeq.hxx>
#include <TFile.h>
#include <TTree.h>
#include <TSystem.h>
#include <vector>
#include <sstream>

using namespace ROOT::VecOps;

void CheckEqual(const RVec<float> &a, const RVec<float> &b, std::string_view msg = "")
{
   const auto asize = a.size();
   const auto bsize = b.size();
   EXPECT_EQ(asize, bsize);
   for (unsigned int i = 0; i < asize; ++i) {
      EXPECT_FLOAT_EQ(a[i], b[i]) << msg;
   }
}

void CheckEqual(const RVec<double> &a, const RVec<double> &b, std::string_view msg = "")
{
   const auto asize = a.size();
   const auto bsize = b.size();
   EXPECT_EQ(asize, bsize);
   for (unsigned int i = 0; i < asize; ++i) {
      EXPECT_DOUBLE_EQ(a[i], b[i]) << msg;
   }
}

template <typename T, typename V>
void CheckEqual(const T &a, const V &b, std::string_view msg = "")
{
   const auto asize = a.size();
   const auto bsize = b.size();
   EXPECT_EQ(asize, bsize);
   for (unsigned int i = 0; i < asize; ++i) {
      EXPECT_EQ(a[i], b[i]) << msg;
   }
}

TEST(VecOps, DefaultCtor)
{
   ROOT::VecOps::RVec<int> v;
   EXPECT_EQ(v.size(), 0u);
}

TEST(VecOps, InitListCtor)
{
   ROOT::VecOps::RVec<int> v{1, 2, 3};
   EXPECT_EQ(v.size(), 3u);
   EXPECT_EQ(v[0], 1);
   EXPECT_EQ(v[1], 2);
   EXPECT_EQ(v[2], 3);
}

TEST(VecOps, CopyCtor)
{
   ROOT::VecOps::RVec<int> v1{1, 2, 3};
   ROOT::VecOps::RVec<int> v2(v1);
   EXPECT_EQ(v1.size(), 3u);
   EXPECT_EQ(v2.size(), 3u);
   EXPECT_EQ(v2[0], 1);
   EXPECT_EQ(v2[1], 2);
   EXPECT_EQ(v2[2], 3);
}

class TLeakChecker {
public:
   static bool fgDestroyed;
   ~TLeakChecker(){
      fgDestroyed = true;
   }
};
bool TLeakChecker::fgDestroyed = false;

TEST(VecOps, CopyCtorCheckNoLeak)
{
   ROOT::VecOps::RVec<TLeakChecker> ref;
   ref.emplace_back(TLeakChecker());
   ROOT::VecOps::RVec<TLeakChecker> proxy(ref.data(), ref.size());
   TLeakChecker::fgDestroyed = false;
   {
      auto v = proxy;
   }
   EXPECT_TRUE(TLeakChecker::fgDestroyed);

   TLeakChecker::fgDestroyed = false;
   ref.clear();
   EXPECT_TRUE(TLeakChecker::fgDestroyed);

}

TEST(VecOps, MoveCtor)
{
   ROOT::VecOps::RVec<int> v1{1, 2, 3};
   ROOT::VecOps::RVec<int> v2(std::move(v1));
   EXPECT_EQ(v1.size(), 0u);
   EXPECT_EQ(v2.size(), 3u);
}

TEST(VecOps, Conversion)
{
   ROOT::VecOps::RVec<float> fvec{1.0f, 2.0f, 3.0f};
   ROOT::VecOps::RVec<unsigned> uvec{1u, 2u, 3u};

   ROOT::VecOps::RVec<int>  ivec = uvec;
   ROOT::VecOps::RVec<long> lvec = ivec;

   EXPECT_EQ(1, ivec[0]);
   EXPECT_EQ(2, ivec[1]);
   EXPECT_EQ(3, ivec[2]);
   EXPECT_EQ(3u, ivec.size());
   EXPECT_EQ(1l, lvec[0]);
   EXPECT_EQ(2l, lvec[1]);
   EXPECT_EQ(3l, lvec[2]);
   EXPECT_EQ(3u, lvec.size());

   auto dvec1 = ROOT::VecOps::RVec<double>(fvec);
   auto dvec2 = ROOT::VecOps::RVec<double>(uvec);

   EXPECT_EQ(1.0, dvec1[0]);
   EXPECT_EQ(2.0, dvec1[1]);
   EXPECT_EQ(3.0, dvec1[2]);
   EXPECT_EQ(3u, dvec1.size());
   EXPECT_EQ(1.0, dvec2[0]);
   EXPECT_EQ(2.0, dvec2[1]);
   EXPECT_EQ(3.0, dvec2[2]);
   EXPECT_EQ(3u, dvec2.size());
}

TEST(VecOps, ArithmeticsUnary)
{
   ROOT::VecOps::RVec<int> ivec{1, 2, 3};
   ROOT::VecOps::RVec<int> pvec = +ivec;
   ROOT::VecOps::RVec<int> nvec = -ivec;
   ROOT::VecOps::RVec<int> tvec = ~ivec;

   EXPECT_EQ(1, pvec[0]);
   EXPECT_EQ(2, pvec[1]);
   EXPECT_EQ(3, pvec[2]);
   EXPECT_EQ(3u, pvec.size());

   EXPECT_EQ(-1, nvec[0]);
   EXPECT_EQ(-2, nvec[1]);
   EXPECT_EQ(-3, nvec[2]);
   EXPECT_EQ(3u, nvec.size());

   EXPECT_EQ(-2, tvec[0]);
   EXPECT_EQ(-3, tvec[1]);
   EXPECT_EQ(-4, tvec[2]);
   EXPECT_EQ(3u, tvec.size());
}

TEST(VecOps, MathScalar)
{
   ROOT::VecOps::RVec<double> ref{1, 2, 3};
   ROOT::VecOps::RVec<double> v(ref);
   int scalar = 3;
   auto plus = v + scalar;
   auto minus = v - scalar;
   auto mult = v * scalar;
   auto div = v / scalar;

   CheckEqual(plus, ref + scalar);
   CheckEqual(minus, ref - scalar);
   CheckEqual(mult, ref * scalar);
   CheckEqual(div, ref / scalar);

   // The same with views
   ROOT::VecOps::RVec<double> w(ref.data(), ref.size());
   plus = w + scalar;
   minus = w - scalar;
   mult = w * scalar;
   div = w / scalar;

   CheckEqual(plus, ref + scalar);
   CheckEqual(minus, ref - scalar);
   CheckEqual(mult, ref * scalar);
   CheckEqual(div, ref / scalar);
}

TEST(VecOps, MathScalarInPlace)
{
   ROOT::VecOps::RVec<double> ref{1, 2, 3};
   const ROOT::VecOps::RVec<double> v(ref);
   int scalar = 3;
   auto plus = v;
   plus += scalar;
   auto minus = v;
   minus -= scalar;
   auto mult = v;
   mult *= scalar;
   auto div = v;
   div /= scalar;

   CheckEqual(plus, ref + scalar);
   CheckEqual(minus, ref - scalar);
   CheckEqual(mult, ref * scalar);
   CheckEqual(div, ref / scalar);
}

TEST(VecOps, MathVector)
{
   ROOT::VecOps::RVec<double> ref{1, 2, 3};
   ROOT::VecOps::RVec<double> vec{3, 4, 5};
   ROOT::VecOps::RVec<double> v(ref);
   auto plus = v + vec;
   auto minus = v - vec;
   auto mult = v * vec;
   auto div = v / vec;

   CheckEqual(plus, ref + vec);
   CheckEqual(minus, ref - vec);
   CheckEqual(mult, ref * vec);
   CheckEqual(div, ref / vec);

   // The same with 1 view
   ROOT::VecOps::RVec<double> w(ref.data(), ref.size());
   plus = w + vec;
   minus = w - vec;
   mult = w * vec;
   div = w / vec;

   CheckEqual(plus, ref + vec);
   CheckEqual(minus, ref - vec);
   CheckEqual(mult, ref * vec);
   CheckEqual(div, ref / vec);

   // The same with 2 views
   ROOT::VecOps::RVec<double> w2(ref.data(), ref.size());
   plus = w + w2;
   minus = w - w2;
   mult = w * w2;
   div = w / w2;

   CheckEqual(plus, ref + w2);
   CheckEqual(minus, ref - w2);
   CheckEqual(mult, ref * w2);
   CheckEqual(div, ref / w2);
}

TEST(VecOps, MathVectorInPlace)
{
   ROOT::VecOps::RVec<double> ref{1, 2, 3};
   ROOT::VecOps::RVec<double> vec{3, 4, 5};
   ROOT::VecOps::RVec<double> v(ref);
   auto plus = v;
   plus += vec;
   auto minus = v;
   minus -= vec;
   auto mult = v;
   mult *= vec;
   auto div = v;
   div /= vec;

   CheckEqual(plus, ref + vec);
   CheckEqual(minus, ref - vec);
   CheckEqual(mult, ref * vec);
   CheckEqual(div, ref / vec);
}

TEST(VecOps, Filter)
{
   ROOT::VecOps::RVec<int> v{0, 1, 2, 3, 4, 5};
   const std::vector<int> vEvenRef{0, 2, 4};
   const std::vector<int> vOddRef{1, 3, 5};
   auto vEven = v[v % 2 == 0];
   auto vOdd = v[v % 2 == 1];
   CheckEqual(vEven, vEvenRef, "Even check");
   CheckEqual(vOdd, vOddRef, "Odd check");

   // now with the helper function
   vEven = Filter(v, [](int i) { return 0 == i % 2; });
   vOdd = Filter(v, [](int i) { return 1 == i % 2; });
   CheckEqual(vEven, vEvenRef, "Even check");
   CheckEqual(vOdd, vOddRef, "Odd check");
}

template <typename T, typename V>
std::string PrintRVec(ROOT::VecOps::RVec<T> v, V w)
{
   std::stringstream ss;
   ss << v << " " << w << std::endl;
   ss << v + w << std::endl;
   ss << v - w << std::endl;
   ss << v * w << std::endl;
   ss << v / w << std::endl;
   ss << (v > w) << std::endl;
   ss << (v >= w) << std::endl;
   ss << (v == w) << std::endl;
   ss << (v <= w) << std::endl;
   ss << (v < w) << std::endl;
   ss << w + v << std::endl;
   ss << w - v << std::endl;
   ss << w * v << std::endl;
   ss << w / v << std::endl;
   ss << (w > v) << std::endl;
   ss << (w >= v) << std::endl;
   ss << (w == v) << std::endl;
   ss << (w <= v) << std::endl;
   ss << (w < v) << std::endl;
   return ss.str();
}

TEST(VecOps, PrintOps)
{
   ROOT::VecOps::RVec<int> ref{1, 2, 3};
   ROOT::VecOps::RVec<int> v(ref);

   auto ref0 = R"ref0({ 1, 2, 3 } 2
{ 3, 4, 5 }
{ -1, 0, 1 }
{ 2, 4, 6 }
{ 0.5, 1, 1.5 }
{ 0, 0, 1 }
{ 0, 1, 1 }
{ 0, 1, 0 }
{ 1, 1, 0 }
{ 1, 0, 0 }
{ 3, 4, 5 }
{ 1, 0, -1 }
{ 2, 4, 6 }
{ 2, 1, 0.666667 }
{ 1, 0, 0 }
{ 1, 1, 0 }
{ 0, 1, 0 }
{ 0, 1, 1 }
{ 0, 0, 1 }
)ref0";
   auto t0 = PrintRVec(v, 2.);
   EXPECT_STREQ(t0.c_str(), ref0);
   auto ref1 = R"ref1({ 1, 2, 3 } { 3, 4, 5 }
{ 4, 6, 8 }
{ -2, -2, -2 }
{ 3, 8, 15 }
{ 0, 0, 0 }
{ 0, 0, 0 }
{ 0, 0, 0 }
{ 0, 0, 0 }
{ 1, 1, 1 }
{ 1, 1, 1 }
{ 4, 6, 8 }
{ 2, 2, 2 }
{ 3, 8, 15 }
{ 3, 2, 1 }
{ 1, 1, 1 }
{ 1, 1, 1 }
{ 0, 0, 0 }
{ 0, 0, 0 }
{ 0, 0, 0 }
)ref1";
   auto t1 = PrintRVec(v, ref + 2);
   EXPECT_STREQ(t1.c_str(), ref1);

   ROOT::VecOps::RVec<int> w(ref.data(), ref.size());

   auto ref2 = R"ref2({ 1, 2, 3 } 2
{ 3, 4, 5 }
{ -1, 0, 1 }
{ 2, 4, 6 }
{ 0.5, 1, 1.5 }
{ 0, 0, 1 }
{ 0, 1, 1 }
{ 0, 1, 0 }
{ 1, 1, 0 }
{ 1, 0, 0 }
{ 3, 4, 5 }
{ 1, 0, -1 }
{ 2, 4, 6 }
{ 2, 1, 0.666667 }
{ 1, 0, 0 }
{ 1, 1, 0 }
{ 0, 1, 0 }
{ 0, 1, 1 }
{ 0, 0, 1 }
)ref2";
   auto t2 = PrintRVec(v, 2.);
   EXPECT_STREQ(t2.c_str(), ref2);

   auto ref3 = R"ref3({ 1, 2, 3 } { 3, 4, 5 }
{ 4, 6, 8 }
{ -2, -2, -2 }
{ 3, 8, 15 }
{ 0, 0, 0 }
{ 0, 0, 0 }
{ 0, 0, 0 }
{ 0, 0, 0 }
{ 1, 1, 1 }
{ 1, 1, 1 }
{ 4, 6, 8 }
{ 2, 2, 2 }
{ 3, 8, 15 }
{ 3, 2, 1 }
{ 1, 1, 1 }
{ 1, 1, 1 }
{ 0, 0, 0 }
{ 0, 0, 0 }
{ 0, 0, 0 }
)ref3";
   auto t3 = PrintRVec(v, ref + 2);
   EXPECT_STREQ(t3.c_str(), ref3);
}

#ifdef R__HAS_VDT
#include <vdt/vdtMath.h>
#endif

TEST(VecOps, MathFuncs)
{
   ROOT::VecOps::RVec<double> u{1, 1, 1};
   ROOT::VecOps::RVec<double> v{1, 2, 3};
   ROOT::VecOps::RVec<double> w{1, 4, 27};

   CheckEqual(pow(1,v), u, " error checking math function pow");
   CheckEqual(pow(v,1), v, " error checking math function pow");
   CheckEqual(pow(v,v), w, " error checking math function pow");

   CheckEqual(sqrt(v), Map(v, [](double x) { return std::sqrt(x); }), " error checking math function sqrt");
   CheckEqual(log(v), Map(v, [](double x) { return std::log(x); }), " error checking math function log");
   CheckEqual(sin(v), Map(v, [](double x) { return std::sin(x); }), " error checking math function sin");
   CheckEqual(cos(v), Map(v, [](double x) { return std::cos(x); }), " error checking math function cos");
   CheckEqual(tan(v), Map(v, [](double x) { return std::tan(x); }), " error checking math function tan");
   CheckEqual(atan(v), Map(v, [](double x) { return std::atan(x); }), " error checking math function atan");
   CheckEqual(sinh(v), Map(v, [](double x) { return std::sinh(x); }), " error checking math function sinh");
   CheckEqual(cosh(v), Map(v, [](double x) { return std::cosh(x); }), " error checking math function cosh");
   CheckEqual(tanh(v), Map(v, [](double x) { return std::tanh(x); }), " error checking math function tanh");
   CheckEqual(asinh(v), Map(v, [](double x) { return std::asinh(x); }), " error checking math function asinh");
   CheckEqual(acosh(v), Map(v, [](double x) { return std::acosh(x); }), " error checking math function acosh");
   v /= 10.;
   CheckEqual(asin(v), Map(v, [](double x) { return std::asin(x); }), " error checking math function asin");
   CheckEqual(acos(v), Map(v, [](double x) { return std::acos(x); }), " error checking math function acos");
   CheckEqual(atanh(v), Map(v, [](double x) { return std::atanh(x); }), " error checking math function atanh");

#ifdef R__HAS_VDT
   #define CHECK_VDT_FUNC(F) \
   CheckEqual(fast_##F(v), Map(v, [](double x) { return vdt::fast_##F(x); }), "error checking vdt function " #F);

   CHECK_VDT_FUNC(exp)
   CHECK_VDT_FUNC(log)

   CHECK_VDT_FUNC(sin)
   CHECK_VDT_FUNC(sin)
   CHECK_VDT_FUNC(cos)
   CHECK_VDT_FUNC(atan)
   CHECK_VDT_FUNC(acos)
   CHECK_VDT_FUNC(atan)
#endif
}

TEST(VecOps, PhysicsSelections)
{
   // We emulate 8 muons
   ROOT::VecOps::RVec<short> mu_charge{1, 1, -1, -1, -1, 1, 1, -1};
   ROOT::VecOps::RVec<float> mu_pt{56.f, 45.f, 32.f, 24.f, 12.f, 8.f, 7.f, 6.2f};
   ROOT::VecOps::RVec<float> mu_eta{3.1f, -.2f, -1.1f, 1.f, 4.1f, 1.6f, 2.4f, -.5f};

   // Pick the pt of the muons with a pt greater than 10, an eta between -2 and 2 and a negative charge
   // or the ones with a pt > 20, outside the eta range -2:2 and with positive charge
   auto goodMuons_pt = mu_pt[(mu_pt > 10.f && abs(mu_eta) <= 2.f && mu_charge == -1) ||
                             (mu_pt > 15.f && abs(mu_eta) > 2.f && mu_charge == 1)];
   ROOT::VecOps::RVec<float> goodMuons_pt_ref = {56.f, 32.f, 24.f};
   CheckEqual(goodMuons_pt, goodMuons_pt_ref, "Muons quality cut");
}

template<typename T0>
void CheckEq(const T0 &v, const T0 &ref)
{
   auto vsize = v.size();
   auto refsize = ref.size();
   EXPECT_EQ(vsize, refsize) << "Sizes are: " << vsize << " " << refsize << std::endl;
   for (auto i : ROOT::TSeqI(vsize)) {
      EXPECT_EQ(v[i], ref[i]) << "RVecs differ" << std::endl;
   }
}

TEST(VecOps, inputOutput)
{
   auto filename = "vecops_inputoutput.root";
   auto treename = "t";

   const ROOT::VecOps::RVec<double>::Impl_t dref {1., 2., 3.};
   const ROOT::VecOps::RVec<float>::Impl_t fref {1.f, 2.f, 3.f};
   const ROOT::VecOps::RVec<UInt_t>::Impl_t uiref {1, 2, 3};
   const ROOT::VecOps::RVec<ULong_t>::Impl_t ulref {1UL, 2UL, 3UL};
   const ROOT::VecOps::RVec<ULong64_t>::Impl_t ullref {1ULL, 2ULL, 3ULL};
   const ROOT::VecOps::RVec<UShort_t>::Impl_t usref {1, 2, 3};
   const ROOT::VecOps::RVec<UChar_t>::Impl_t ucref {1, 2, 3};
   const ROOT::VecOps::RVec<Int_t>::Impl_t iref {1, 2, 3};;
   const ROOT::VecOps::RVec<Long_t>::Impl_t lref {1UL, 2UL, 3UL};;
   const ROOT::VecOps::RVec<Long64_t>::Impl_t llref {1ULL, 2ULL, 3ULL};
   const ROOT::VecOps::RVec<Short_t>::Impl_t sref {1, 2, 3};
   const ROOT::VecOps::RVec<Char_t>::Impl_t cref {1, 2, 3};

   {
      auto d = dref;
      auto f = fref;
      auto ui = uiref;
      auto ul = ulref;
      auto ull = ullref;
      auto us = usref;
      auto uc = ucref;
      auto i = iref;
      auto l = lref;
      auto ll = llref;
      auto s = sref;
      auto c = cref;
      TFile file(filename, "RECREATE");
      TTree t(treename, treename);
      t.Branch("d", &d);
      t.Branch("f", &f);
      t.Branch("ui", &ui);
      t.Branch("ul", &ul);
      t.Branch("ull", &ull);
      t.Branch("us", &us);
      t.Branch("uc", &uc);
      t.Branch("i", &i);
      t.Branch("l", &l);
      t.Branch("ll", &ll);
      t.Branch("s", &s);
      t.Branch("c", &c);
      t.Fill();
      t.Write();
   }

   auto d = new ROOT::VecOps::RVec<double>::Impl_t();
   auto f = new ROOT::VecOps::RVec<float>::Impl_t;
   auto ui = new ROOT::VecOps::RVec<UInt_t>::Impl_t();
   auto ul = new ROOT::VecOps::RVec<ULong_t>::Impl_t();
   auto ull = new ROOT::VecOps::RVec<ULong64_t>::Impl_t();
   auto us = new ROOT::VecOps::RVec<UShort_t>::Impl_t();
   auto uc = new ROOT::VecOps::RVec<UChar_t>::Impl_t();
   auto i = new ROOT::VecOps::RVec<Int_t>::Impl_t();
   auto l = new ROOT::VecOps::RVec<Long_t>::Impl_t();
   auto ll = new ROOT::VecOps::RVec<Long64_t>::Impl_t();
   auto s = new ROOT::VecOps::RVec<Short_t>::Impl_t();
   auto c = new ROOT::VecOps::RVec<Char_t>::Impl_t();

   TFile file(filename);
   TTree *tp;
   file.GetObject(treename, tp);
   auto &t = *tp;

   t.SetBranchAddress("d", &d);
   t.SetBranchAddress("f", &f);
   t.SetBranchAddress("ui", &ui);
   t.SetBranchAddress("ul", &ul);
   t.SetBranchAddress("ull", &ull);
   t.SetBranchAddress("us", &us);
   t.SetBranchAddress("uc", &uc);
   t.SetBranchAddress("i", &i);
   t.SetBranchAddress("l", &l);
   t.SetBranchAddress("ll", &ll);
   t.SetBranchAddress("s", &s);
   t.SetBranchAddress("c", &c);

   t.GetEntry(0);
   CheckEq(*d, dref);
   CheckEq(*f, fref);
   CheckEq(*d, dref);
   CheckEq(*f, fref);
   CheckEq(*d, dref);
   CheckEq(*f, fref);
   CheckEq(*d, dref);
   CheckEq(*f, fref);
   CheckEq(*d, dref);
   CheckEq(*f, fref);

   gSystem->Unlink(filename);

}

TEST(VecOps, SimpleStatOps)
{
   ROOT::VecOps::RVec<double> v0 {};
   ASSERT_DOUBLE_EQ(Sum(v0), 0.);
   ASSERT_DOUBLE_EQ(Mean(v0), 0.);
   ASSERT_DOUBLE_EQ(StdDev(v0), 0.);
   ASSERT_DOUBLE_EQ(Var(v0), 0.);

   ROOT::VecOps::RVec<double> v1 {42.};
   ASSERT_DOUBLE_EQ(Sum(v1), 42.);
   ASSERT_DOUBLE_EQ(Mean(v1), 42.);
   ASSERT_DOUBLE_EQ(StdDev(v1), 0.);
   ASSERT_DOUBLE_EQ(Var(v1), 0.);

   ROOT::VecOps::RVec<double> v2 {1., 2., 3.};
   ASSERT_DOUBLE_EQ(Sum(v2), 6.);
   ASSERT_DOUBLE_EQ(Mean(v2), 2.);
   ASSERT_DOUBLE_EQ(Var(v2), 1.);
   ASSERT_DOUBLE_EQ(StdDev(v2), 1.);

   ROOT::VecOps::RVec<double> v3 {10., 20., 32.};
   ASSERT_DOUBLE_EQ(Sum(v3), 62.);
   ASSERT_DOUBLE_EQ(Mean(v3), 20.666666666666668);
   ASSERT_DOUBLE_EQ(Var(v3), 121.33333333333337);
   ASSERT_DOUBLE_EQ(StdDev(v3), 11.015141094572206);
}

TEST(VecOps, Any)
{
   ROOT::VecOps::RVec<int> vi {0, 1, 2};
   EXPECT_TRUE(Any(vi));
   vi = {0, 0, 0};
   EXPECT_FALSE(Any(vi));
   vi = {1, 1};
   EXPECT_TRUE(Any(vi));
}

TEST(VecOps, All)
{
   ROOT::VecOps::RVec<int> vi {0, 1, 2};
   EXPECT_FALSE(All(vi));
   vi = {0, 0, 0};
   EXPECT_FALSE(All(vi));
   vi = {1, 1};
   EXPECT_TRUE(All(vi));
}

TEST(VecOps, Argsort)
{
   ROOT::VecOps::RVec<int> v{2, 0, 1};
   using size_type = typename ROOT::VecOps::RVec<int>::size_type;
   auto i = Argsort(v);
   ROOT::VecOps::RVec<size_type> ref{1, 2, 0};
   CheckEqual(i, ref);
}

TEST(VecOps, ByIndices)
{
   ROOT::VecOps::RVec<int> v0{2, 0, 1};
   ROOT::VecOps::RVec<typename ROOT::VecOps::RVec<int>::size_type> i{1, 2, 0};
   auto v1 = ByIndices(v0, i);
   ROOT::VecOps::RVec<int> ref{0, 1, 2};
   CheckEqual(v1, ref);
}
