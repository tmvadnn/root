#include "ROOT/RDataFrame.hxx"
#include "TROOT.h"

#include "gtest/gtest.h"

namespace TEST_CATEGORY {

void FillTree(const char *filename, const char *treeName, int nevents = 0)
{
   TFile f(filename, "RECREATE");
   TTree t(treeName, treeName);
   t.SetAutoFlush(1); // yes, one event per cluster: to make MT more meaningful
   int b;
   t.Branch("b1", &b);
   for (int i = 0; i < nevents; ++i) {
      b = i;
      t.Fill();
   }
   t.Write();
   f.Close();
}
}

TEST(TEST_CATEGORY, InvalidRef)
{
   auto getFilterNode = []() {
      ROOT::RDataFrame d(5);
      return d.Filter([]() { return true; });
   };
   int ret(1);
   auto f = getFilterNode();
   try {
      f.Filter([]() { return true; });
   } catch (const std::runtime_error &) {
      ret = 0;
   }
   EXPECT_EQ(0, ret) << "No exception thrown when the original tdf went out of scope.";
}

TEST(TEST_CATEGORY, MultipleTriggerRun)
{
   auto fileName = "dataframe_regression_0.root";
   auto treeName = "t";
#ifndef dataframe_regression_0_CREATED
#define dataframe_regression_0_CREATED
   {
      ROOT::RDataFrame tdf(1);
      tdf.Define("b1", []() { return 1U; }).Snapshot<unsigned int>(treeName, fileName, {"b1"});
   }
#endif

   ROOT::RDataFrame d(treeName, fileName, {"b1"});
   int i = 0;
   auto sentinel = [&i]() {
      ++i;
      return true;
   };
   auto f1 = d.Filter(sentinel);
   auto m1 = f1.Min();
   *m1; // this should cause i to be incremented
   EXPECT_EQ(1, i) << "The filter was not correctly executed.";

   auto f2 = d.Filter(sentinel);
   auto dummy = f2.Max();
   *m1; // this should NOT cause i to be incremented
   EXPECT_EQ(1, i) << "The filter was executed also when it was not supposed to.";

   *dummy; // this should cause i to be incremented
   EXPECT_EQ(2, i) << "The filter was not correctly executed for the second time.";
}

TEST(TEST_CATEGORY, EmptyTree)
{
   auto fileName = "dataframe_regression_2.root";
   auto treeName = "t";
#ifndef dataframe_regression_2_CREATED
#define dataframe_regression_2_CREATED
   {
      TFile wf(fileName, "RECREATE");
      TTree t(treeName, treeName);
      int a;
      t.Branch("a", &a);
      t.Write();
   }
#endif
   ROOT::RDataFrame d(treeName, fileName, {"a"});
   auto min = d.Min<int>();
   auto max = d.Max<int>();
   auto mean = d.Mean<int>();
   auto h = d.Histo1D<int>();
   auto c = d.Count();
   auto g = d.Take<int>();
   std::atomic_int fc(0);
   d.Foreach([&fc]() { ++fc; });

   EXPECT_DOUBLE_EQ(*min, std::numeric_limits<int>::max());
   EXPECT_DOUBLE_EQ(*max, std::numeric_limits<int>::lowest());
   EXPECT_DOUBLE_EQ(*mean, 0);
   EXPECT_EQ(h->GetEntries(), 0);
   EXPECT_EQ(*c, 0U);
   EXPECT_EQ(g->size(), 0U);
   EXPECT_EQ(fc.load(), 0);
}
