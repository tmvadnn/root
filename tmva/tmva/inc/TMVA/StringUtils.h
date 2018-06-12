
#ifndef ROOT_TMVA_StringUtils
#define ROOT_TMVA_StringUtils

#include "TString.h"
#include "TObjString.h"
#include <sstream>

namespace TMVA{

////////////////////////////////////////////////////////////////////////////////
inline TString fetchValueUtils(const std::map<TString, TString> &keyValueMap, TString key)
{
   key.ToUpper();
   std::map<TString, TString>::const_iterator it = keyValueMap.find(key);
   if (it == keyValueMap.end()) {
      return TString("");
   }
   return it->second;
}

////////////////////////////////////////////////////////////////////////////////
template <typename T> inline
T fetchValueUtils(const std::map<TString, TString> &keyValueMap, TString key, T defaultValue);

////////////////////////////////////////////////////////////////////////////////
template <> inline
int fetchValueUtils(const std::map<TString, TString> &keyValueMap, TString key, int defaultValue)
{
   TString value(fetchValueUtils(keyValueMap, key));
   if (value == "") {
      return defaultValue;
   }
   return value.Atoi();
}

////////////////////////////////////////////////////////////////////////////////
template <> inline
double fetchValueUtils(const std::map<TString, TString> &keyValueMap, TString key, double defaultValue)
{
   TString value(fetchValueUtils(keyValueMap, key));
   if (value == "") {
      return defaultValue;
   }
   return value.Atof();
}

////////////////////////////////////////////////////////////////////////////////
template <> inline
TString fetchValueUtils(const std::map<TString, TString> &keyValueMap, TString key, TString defaultValue)
{
   TString value(fetchValueUtils(keyValueMap, key));
   if (value == "") {
      return defaultValue;
   }
   return value;
}

////////////////////////////////////////////////////////////////////////////////
template <> inline
bool fetchValueUtils(const std::map<TString, TString> &keyValueMap, TString key, bool defaultValue)
{
   TString value(fetchValueUtils(keyValueMap, key));
   if (value == "") {
      return defaultValue;
   }

   value.ToUpper();
   if (value == "TRUE" || value == "T" || value == "1") {
      return true;
   }

   return false;
}

////////////////////////////////////////////////////////////////////////////////
template <> inline
std::vector<double> fetchValueUtils(const std::map<TString, TString> &keyValueMap, TString key,
                                  std::vector<double> defaultValue)
{
   TString parseString(fetchValueUtils(keyValueMap, key));
   if (parseString == "") {
      return defaultValue;
   }

   parseString.ToUpper();
   std::vector<double> values;

   const TString tokenDelim("+");
   TObjArray *tokenStrings = parseString.Tokenize(tokenDelim);
   TIter nextToken(tokenStrings);
   TObjString *tokenString = (TObjString *)nextToken();
   for (; tokenString != NULL; tokenString = (TObjString *)nextToken()) {
      std::stringstream sstr;
      double currentValue;
      sstr << tokenString->GetString().Data();
      sstr >> currentValue;
      values.push_back(currentValue);
   }
   return values;
}

}

#endif