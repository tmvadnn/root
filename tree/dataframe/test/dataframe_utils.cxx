#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDFUtils.hxx"
#include "TTree.h"

#include "gtest/gtest.h"

namespace RDFInt = ROOT::Internal::RDF;

// Thanks clang-format...
TEST(RDataFrameUtils, DeduceAllPODsFromTmpColumns)
{
   ROOT::RDataFrame tdf(1);
   auto d = tdf.Define("char_tmp", []() { return char(0); })
               .Define("uchar_tmp", []() { return (unsigned char)(0u); })
               .Define("int_tmp", []() { return int(0); })
               .Define("uint_tmp", []() { return (unsigned int)(0u); })
               .Define("short_tmp", []() { return short(0); })
               .Define("ushort_tmp", []() { return (unsigned short)(0u); })
               .Define("double_tmp", []() { return double(0.); })
               .Define("float_tmp", []() { return float(0.f); })
               .Define("Long64_t_tmp", []() { return Long64_t(0ll); })
               .Define("ULong64_t_tmp", []() { return ULong64_t(0ull); })
               .Define("bool_tmp", []() { return bool(false); });
   auto c = d.Snapshot<char, unsigned char, int, unsigned int, short, unsigned short, double, float, Long64_t,
                       ULong64_t, bool>("t", "dataframe_interfaceAndUtils_1.root",
                                        {"char_tmp", "uchar_tmp", "int_tmp", "uint_tmp", "short_tmp", "ushort_tmp",
                                         "double_tmp", "float_tmp", "Long64_t_tmp", "ULong64_t_tmp", "bool_tmp"});
}

TEST(RDataFrameUtils, DeduceAllPODsFromColumns)
{
   char c;
   unsigned char uc;
   int i;
   unsigned int ui;
   short s;
   unsigned short us;
   double d;
   float f;
   Long64_t l;
   ULong64_t ul;
   bool b;
   int a[2];

   TTree t("t", "t");
   t.Branch("char", &c);
   t.Branch("uchar", &uc);
   t.Branch("i", &i);
   t.Branch("uint", &ui);
   t.Branch("short", &s);
   t.Branch("ushort", &us);
   t.Branch("double", &d);
   t.Branch("float", &f);
   t.Branch("Long64_t", &l);
   t.Branch("ULong64_t", &ul);
   t.Branch("bool", &b);
   t.Branch("arrint", &a, "a[2]/I");
   t.Branch("vararrint", &a, "a[i]/I");

   std::map<const char *, const char *> nameTypes = {{"char", "Char_t"},
                                                     {"uchar", "UChar_t"},
                                                     {"i", "Int_t"},
                                                     {"uint", "UInt_t"},
                                                     {"short", "Short_t"},
                                                     {"ushort", "UShort_t"},
                                                     {"double", "Double_t"},
                                                     {"float", "Float_t"},
                                                     {"Long64_t", "Long64_t"},
                                                     {"ULong64_t", "ULong64_t"},
                                                     {"bool", "Bool_t"},
                                                     {"arrint.a", "ROOT::VecOps::RVec<Int_t>"},
                                                     {"vararrint.a", "ROOT::VecOps::RVec<Int_t>"}};

   for (auto &nameType : nameTypes) {
      auto typeName = RDFInt::ColumnName2ColumnTypeName(nameType.first, /*nsID=*/0, &t, /*ds=*/nullptr,
                                                                     /*custom=*/false);
      EXPECT_STREQ(nameType.second, typeName.c_str());
   }
}

TEST(RDataFrameUtils, DeduceTypeOfBranchesWithCustomTitle)
{
   int i;
   float f;
   int a[2];

   TTree t("t", "t");
   auto b = t.Branch("float", &f);
   b->SetTitle("custom title");
   b = t.Branch("i", &i);
   b->SetTitle("custom title");
   b = t.Branch("arrint", &a, "a[2]/I");
   b->SetTitle("custom title");
   b = t.Branch("vararrint", &a, "a[i]/I");
   b->SetTitle("custom title");

   std::map<const char *, const char *> nameTypes = {{"float", "Float_t"},
                                                     {"i", "Int_t"},
                                                     {"arrint.a", "ROOT::VecOps::RVec<Int_t>"},
                                                     {"vararrint.a", "ROOT::VecOps::RVec<Int_t>"}};

   for (auto &nameType : nameTypes) {
      auto typeName = RDFInt::ColumnName2ColumnTypeName(nameType.first, /*nsID=*/0, &t, /*ds=*/nullptr,
                                                                     /*custom=*/false);
      EXPECT_STREQ(nameType.second, typeName.c_str());
   }
}

TEST(RDataFrameUtils, CheckNonExistingCustomColumnNullTree)
{
   // CheckCustomColumn(std::string_view definedCol, TTree *treePtr, const ColumnNames_t &customCols,
   //                   const ColumnNames_t &dataSourceColumns)
   RDFInt::CheckCustomColumn("Bla", nullptr, {"a", "b"}, {});
}

TEST(RDataFrameUtils, CheckExistingCustomColumnNullTree)
{
   int ret = 1;
   try {
      RDFInt::CheckCustomColumn("a", nullptr, {"a", "b"}, {});
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret);
}

TEST(RDataFrameUtils, CheckExistingCustomColumn)
{
   int i;
   TTree t("t", "t");
   t.Branch("a", &i);

   int ret = 1;
   try {
      RDFInt::CheckCustomColumn("a", &t, {"b"}, {});
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret);
}

TEST(RDataFrameUtils, CheckExistingCustomColumnDataSource)
{
   int i;
   TTree t("t", "t");
   t.Branch("a", &i);

   int ret = 1;
   try {
      RDFInt::CheckCustomColumn("c", &t, {"b"}, {"c"});
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret);
}

TEST(RDataFrameUtils, CheckTypesAndPars)
{
   int ret = 1;
   try {
      RDFInt::CheckTypesAndPars(5, 4);
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret);
}

TEST(RDataFrameUtils, SelectColumnsNNamesDiffersRequiredNames)
{
   int ret = 1;
   try {
      RDFInt::SelectColumns(3, {"a", "b"}, {});
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret);
}

TEST(RDataFrameUtils, SelectColumnsTooFewRequiredNames)
{
   int ret = 1;
   try {
      RDFInt::SelectColumns(3, {}, {"bla"});
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret);
}

TEST(RDataFrameUtils, SelectColumnsCheckNames)
{
   RDFInt::ColumnNames_t cols{"a", "b", "c"};
   auto ncols = RDFInt::SelectColumns(2, {}, cols);
   EXPECT_STREQ("a", ncols[0].c_str());
   EXPECT_STREQ("b", ncols[1].c_str());
}

TEST(RDataFrameUtils, FindUnknownColumns)
{
   int i;
   TTree t("t", "t");
   t.Branch("a", &i);

   auto ncols = RDFInt::FindUnknownColumns({"a", "b", "c", "d"}, RDFInt::GetBranchNames(t), {"b"}, {});
   EXPECT_STREQ("c", ncols[0].c_str());
   EXPECT_STREQ("d", ncols[1].c_str());
}

TEST(RDataFrameUtils, FindUnknownColumnsWithDataSource)
{
   int i;
   TTree t("t", "t");
   t.Branch("a", &i);

   auto ncols = RDFInt::FindUnknownColumns({"a", "b", "c", "d"}, RDFInt::GetBranchNames(t), {"b"}, {"c"});
   EXPECT_EQ(ncols.size(), 1u);
   EXPECT_STREQ("d", ncols[0].c_str());
}

struct DummyStruct {
   int a, b;
};

TEST(RDataFrameUtils, FindUnknownColumnsNestedNames)
{
   // check we recognize column names of the form "branch.leaf"
   TTree t("t", "t");
   DummyStruct s{1, 2};
   t.Branch("s", &s, "a/I:b/I");

   auto unknownCols = RDFInt::FindUnknownColumns({"s.a", "s.b", "s", "s.", ".s", "_asd_"}, RDFInt::GetBranchNames(t), {}, {});
   const auto trueUnknownCols = std::vector<std::string>({"s", "s.", ".s", "_asd_"});
   EXPECT_EQ(unknownCols, trueUnknownCols);
}

TEST(RDataFrameUtils, FindUnknownColumnsFriendTrees)
{
   int i;

   TTree t1("t1", "t1");
   t1.Branch("c1", &i);

   TTree t2("t2", "t2");
   t2.Branch("c2", &i);

   TTree t3("t3", "t3");
   t3.Branch("c3", &i);

   // Circular
   TTree t4("t4", "t4");
   t4.Branch("c4", &i);
   t4.AddFriend(&t1);

   t2.AddFriend(&t3);
   t1.AddFriend(&t2);
   t1.AddFriend(&t4);

   auto ncols = RDFInt::FindUnknownColumns({"c2", "c3", "c4"}, RDFInt::GetBranchNames(t1), {}, {});
   EXPECT_EQ(ncols.size(), 0u) << "Cannot find column in friend trees.";
}
