#include "ROOT/RDataFrame.hxx"
#include "ROOT/RTrivialDS.hxx"
#include "TMemFile.h"
#include "TSystem.h"
#include "TTree.h"

#include "gtest/gtest.h"

using namespace ROOT;
using namespace ROOT::RDF;

TEST(RDataFrameInterface, CreateFromCStrings)
{
   RDataFrame tdf("t", "file");
}

TEST(RDataFrameInterface, CreateFromStrings)
{
   std::string t("t"), f("file");
   RDataFrame tdf(t, f);
}

TEST(RDataFrameInterface, CreateFromContainer)
{
   std::string t("t");
   std::vector<std::string> f({"f1", "f2"});
   RDataFrame tdf(t, f);
}

TEST(RDataFrameInterface, CreateFromInitList)
{
   RDataFrame tdf("t", {"f1", "f2"});
}

TEST(RDataFrameInterface, CreateFromNullTDirectory)
{
   int ret = 1;
   try {
      RDataFrame tdf("t", nullptr);
   } catch (const std::runtime_error &) {
      ret = 0;
   }
   EXPECT_EQ(0, ret);
}

TEST(RDataFrameInterface, CreateFromNonExistingTree)
{
   int ret = 1;
   try {
      RDataFrame tdf("theTreeWhichDoesNotExist", gDirectory);
   } catch (const std::runtime_error &) {
      ret = 0;
   }
   EXPECT_EQ(0, ret);
}

TEST(RDataFrameInterface, CreateFromTree)
{
   TMemFile f("dataframe_interfaceAndUtils_0.root", "RECREATE");
   TTree t("t", "t");
   RDataFrame tdf(t);
   auto c = tdf.Count();
   EXPECT_EQ(0U, *c);
}

TEST(RDataFrameInterface, CreateAliases)
{
   RDataFrame tdf(1);
   auto aliased_tdf = tdf.Define("c0", []() { return 0; }).Alias("c1", "c0").Alias("c2", "c0").Alias("c3", "c1");
   auto c = aliased_tdf.Count();
   EXPECT_EQ(1U, *c);

   int ret(1);
   try {
      aliased_tdf.Alias("c4", "c");
   } catch (const std::runtime_error &) {
      ret = 0;
   }
   EXPECT_EQ(0, ret) << "No exception thrown when trying to alias a non-existing column.";

   ret = 1;
   try {
      aliased_tdf.Alias("c0", "c2");
   } catch (const std::runtime_error &) {
      ret = 0;
   }
   EXPECT_EQ(0, ret) << "No exception thrown when specifying an alias name which is the name of a column.";

   ret = 1;
   try {
      aliased_tdf.Alias("c2", "c1");
   } catch (const std::runtime_error &) {
      ret = 0;
   }
   EXPECT_EQ(0, ret) << "No exception thrown when re-using an alias for a different column.";
}

TEST(RDataFrameInterface, CheckAliasesPerChain)
{
   RDataFrame tdf(1);
   auto d = tdf.Define("c0", []() { return 0; });
   // Now branch the graph
   auto ok = []() { return true; };
   auto f0 = d.Filter(ok);
   auto f1 = d.Filter(ok);
   auto f0a = f0.Alias("c1", "c0");
   // must work
   auto f0aa = f0a.Alias("c2", "c1");
   // must fail
   auto ret = 1;
   try {
      auto f1a = f1.Alias("c2", "c1");
   } catch (const std::runtime_error &) {
      ret = 0;
   }
   EXPECT_EQ(0, ret) << "No exception thrown when trying to alias a non-existing column.";
}

TEST(RDataFrameInterface, GetColumnNamesFromScratch)
{
   RDataFrame f(1);
   auto dummyGen = []() { return 1; };
   auto names = f.Define("a", dummyGen).Define("b", dummyGen).Define("tdfDummy_", dummyGen).GetColumnNames();
   EXPECT_STREQ("a", names[0].c_str());
   EXPECT_STREQ("b", names[1].c_str());
   EXPECT_EQ(2U, names.size());
}

TEST(RDataFrameInterface, GetColumnNamesFromTree)
{
   TTree t("t", "t");
   int a, b;
   t.Branch("a", &a);
   t.Branch("b", &b);
   RDataFrame tdf(t);
   auto names = tdf.GetColumnNames();
   EXPECT_STREQ("a", names[0].c_str());
   EXPECT_STREQ("a.a", names[1].c_str());
   EXPECT_STREQ("b", names[2].c_str());
   EXPECT_STREQ("b.b", names[3].c_str());
   EXPECT_EQ(4U, names.size());
}

TEST(RDataFrameInterface, GetColumnNamesFromOrdering)
{
   TTree t("t", "t");
   int a, b;
   t.Branch("zzz", &a);
   t.Branch("aaa", &b);
   RDataFrame tdf(t);
   auto names = tdf.GetColumnNames();
   EXPECT_STREQ("zzz", names[0].c_str());
   EXPECT_STREQ("zzz.zzz", names[1].c_str());
   EXPECT_STREQ("aaa", names[2].c_str());
   EXPECT_STREQ("aaa.aaa", names[3].c_str());
   EXPECT_EQ(4U, names.size());
}

TEST(RDataFrameInterface, GetColumnNamesFromSource)
{
   std::unique_ptr<RDataSource> tds(new RTrivialDS(1));
   RDataFrame tdf(std::move(tds));
   auto names = tdf.Define("b", []() { return 1; }).GetColumnNames();
   EXPECT_STREQ("b", names[0].c_str());
   EXPECT_STREQ("col0", names[1].c_str());
   EXPECT_EQ(2U, names.size());
}

TEST(RDataFrameInterface, GetFilterNamesFromNode)
{
   RDataFrame f(1);
   auto dummyGen = []() { return 1; };
   auto dummyFilter = [](int val) { return val > 0; };
   auto names = f.Define("a", dummyGen)
                   .Define("b", dummyGen)
                   .Filter("a>0")
                   .Range(30)
                   .Filter(dummyFilter, {"a"})
                   .Define("d", dummyGen)
                   .Range(30)
                   .Filter("a>0", "filt_a_jit")
                   .Filter(dummyFilter, {"a"}, "filt_a")
                   .Filter("a>0")
                   .Filter(dummyFilter, {"a"})
                   .GetFilterNames();

   std::vector<std::string> comparison(
      {"Unnamed Filter", "Unnamed Filter", "filt_a_jit", "filt_a", "Unnamed Filter", "Unnamed Filter"});
   EXPECT_EQ(comparison, names);
}

TEST(RDataFrameInterface, GetFilterNamesFromLoopManager)
{
   RDataFrame f(1);
   auto dummyGen = []() { return 1; };
   auto dummyFilter = [](int val) { return val > 0; };
   auto names_one = f.Define("a", dummyGen)
                       .Define("b", dummyGen)
                       .Filter("a>0")
                       .Range(30)
                       .Filter(dummyFilter, {"a"})
                       .Define("c", dummyGen)
                       .Range(30)
                       .Filter("a>0", "filt_a_jit")
                       .Filter(dummyFilter, {"b"}, "filt_b")
                       .Filter("a>0")
                       .Filter(dummyFilter, {"a"});
   auto names_two = f.Define("d", dummyGen)
                       .Define("e", dummyGen)
                       .Filter("d>0")
                       .Range(30)
                       .Filter(dummyFilter, {"d"})
                       .Define("f", dummyGen)
                       .Range(30)
                       .Filter("d>0", "filt_d_jit")
                       .Filter(dummyFilter, {"e"}, "filt_e")
                       .Filter("e>0")
                       .Filter(dummyFilter, {"e"});

   std::vector<std::string> comparison({"Unnamed Filter", "Unnamed Filter", "filt_a_jit", "filt_b", "Unnamed Filter",
                                        "Unnamed Filter", "Unnamed Filter", "Unnamed Filter", "filt_d_jit", "filt_e",
                                        "Unnamed Filter", "Unnamed Filter"});
   auto names = f.GetFilterNames();
   EXPECT_EQ(comparison, names);
}

TEST(RDataFrameInterface, GetFilterNamesFromNodeNoFilters)
{
   RDataFrame f(1);
   auto dummyGen = []() { return 1; };
   auto names =
      f.Define("a", dummyGen).Define("b", dummyGen).Range(30).Define("d", dummyGen).Range(30).GetFilterNames();

   std::vector<std::string> comparison({});
   EXPECT_EQ(comparison, names);
}

TEST(RDataFrameInterface, GetFilterNamesFromLoopManagerNoFilters)
{
   RDataFrame f(1);
   auto dummyGen = []() { return 1; };
   auto names_one = f.Define("a", dummyGen).Define("b", dummyGen).Range(30).Define("c", dummyGen).Range(30);
   auto names_two = f.Define("d", dummyGen).Define("e", dummyGen).Range(30).Define("f", dummyGen).Range(30);

   std::vector<std::string> comparison({});
   auto names = f.GetFilterNames();
   EXPECT_EQ(comparison, names);
}

TEST(RDataFrameInterface, DefaultColumns)
{
   RDataFrame tdf(8);
   ULong64_t i(0ULL);
   auto checkSlotAndEntries = [&i](unsigned int slot, ULong64_t entry) {
      EXPECT_EQ(entry, i);
      EXPECT_EQ(slot, 0U);
      i++;
   };
   tdf.Foreach(checkSlotAndEntries, {"tdfslot_", "tdfentry_"});
}

TEST(RDataFrameInterface, JitDefaultColumns)
{
   RDataFrame tdf(8);
   auto f = tdf.Filter("tdfslot_ + tdfentry_ == 3");
   auto maxEntry = f.Max("tdfentry_");
   auto minEntry = f.Min("tdfentry_");
   EXPECT_EQ(*maxEntry, *minEntry);
}

TEST(RDataFrameInterface, InvalidDefine)
{
   RDataFrame df(1);
   try {
      df.Define("1", [] { return true; });
   } catch (const std::runtime_error &e) {
      EXPECT_STREQ("Cannot define column \"1\": not a valid C++ variable name.", e.what());
   }
   try {
      df.Define("a-b", "true");
   } catch (const std::runtime_error &e) {
      EXPECT_STREQ("Cannot define column \"a-b\": not a valid C++ variable name.", e.what());
   }
}

struct S {
   int a;
   int b;
};

TEST(RDataFrameInterface, GetColumnType)
{
   const auto fname = "tdf_getcolumntype.root";
   TFile f(fname, "recreate");
   TTree t("t", "t");   
   S s{1,2};
   int x = 42;
   t.Branch("s", &s, "a/I:b/I");
   t.Branch("x", &x);
   t.Fill();
   t.Write();
   f.Close();

   auto df = RDataFrame("t", fname).Define("y", [] { return std::vector<int>{}; }).Define("z", "double(x)");
   EXPECT_EQ(df.GetColumnType("x"), "Int_t");
   EXPECT_EQ(df.GetColumnType("y"), "vector<int>");
   EXPECT_EQ(df.GetColumnType("z"), "double");
   EXPECT_EQ(df.GetColumnType("s.a"), "Int_t");

   gSystem->Unlink(fname);
}
