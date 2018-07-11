// Author: Enrico Guiraud, Danilo Piparo CERN  12/2016

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <algorithm>
#include <stdexcept>

#include "ROOT/RDataFrame.hxx"
#include "TChain.h"
#include "TDirectory.h"

// clang-format off
/**
* \class ROOT::RDataFrame
* \ingroup dataframe
* \brief ROOT's RDataFrame offers a high level interface for analyses of data stored in `TTree`s, CSV's and other data formats.

In addition, multi-threading and other low-level optimisations allow users to exploit all the resources available
on their machines completely transparently.<br>
Skip to the [class reference](#reference) or keep reading for the user guide.

In a nutshell:
~~~{.cpp}
ROOT::EnableImplicitMT(); // Tell ROOT you want to go parallel
ROOT::RDataFrame d("myTree", "file_*.root"); // Interface to TTree and TChain
auto myHisto = d.Histo1D("Branch_A"); // This happens in parallel!
myHisto->Draw();
~~~

Calculations are expressed in terms of a type-safe *functional chain of actions and transformations*, `RDataFrame` takes
care of their execution. The implementation automatically puts in place several low level optimisations such as
multi-thread parallelisation and caching.

\htmlonly
<a href="https://doi.org/10.5281/zenodo.260230"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.260230.svg"
alt="DOI"></a>
\endhtmlonly

## Table of Contents
- [Cheat sheet](#cheatsheet)
- [Introduction](#introduction)
- [Crash course](#crash-course)
- [More features](#more-features)
- [Transformations](#transformations) -- manipulating data
- [Actions](#actions) -- getting results
- [Parallel execution](#parallel-execution) -- how to use it and common pitfalls
- [Class reference](#reference) -- most methods are implemented in the RInterface base class

## <a name="cheatsheet"></a>Cheat sheet
These are the operations which can be performed with RDataFrame

### Transformations
Transformations are a way to manipulated the data.

| **Transformation** | **Description** |
|------------------|-----------------|
| [Define](classROOT_1_1RDF_1_1RInterface.html#a7d48eb23b4378e99ebccb35e94ad025a) | Creates a new column in the dataset. |
| [DefineSlot](classROOT_1_1RDF_1_1RInterface.html#acaacf727b8a41d27c6bb4513348ac892) | Same as `Define`, but the user-defined function must take an extra `unsigned int slot` as its first parameter. `slot` will take a different value, `0` to `nThreads - 1`, for each thread of execution. This is meant as a helper in writing thread-safe `Define` transformation when using `RDataFrame` after `ROOT::EnableImplicitMT()`. `DefineSlot` works just as well with single-thread execution: in that case `slot` will always be `0`.  |
| [DefineSlotEntry](classROOT_1_1RDF_1_1RInterface.html#a4f17074d5771916e3df18f8458186de7) | Same as `DefineSlot`, but the entry number is passed in addition to the slot number. This is meant as a helper in case some dependency on the entry number needs to be honoured. |
| [Filter](classROOT_1_1RDF_1_1RInterface.html#a70284a3bedc72b19610aaa91b5007ebd) | Filter the rows of the dataset. |
| [Range](classROOT_1_1RDF_1_1RInterface.html#a1b36b7868831de2375e061bb06cfc225) | Creates a node that filters entries based on range of entries |

### Actions
Actions are a way to produce a result out of the data. Each one is described in more detail in the reference guide.

In the following, whenever we say an action "returns" something, we always mean it returns a smart pointer to it. Also
note that all actions are only executed for events that pass all preceding filters.

Lazy actions only trigger the event loop when one of the results is accessed for the first time, making it easy to
produce several different results in one event loop. Instant actions trigger the event loop instantly.


| **Lazy action** | **Description** |
|------------------|-----------------|
| [Aggregate](classROOT_1_1RDF_1_1RInterface.html#ae540b00addc441f9b504cbae0ef0a24d) | Execute a user-defined accumulation operation on the processed column values. |
| [Book](classROOT_1_1RDF_1_1RInterface.html#a9b2f61f3333d1669e57055b9ae8be9d9) | Book execution of a custom action using a user-defined helper object. |
| [Cache](classROOT_1_1RDF_1_1RInterface.html#aaaa0a7bb8eb21315d8daa08c3e25f6c9) | Caches in contiguous memory columns' entries. Custom columns can be cached as well, filtered entries are not cached. Users can specify which columns to save (default is all). |
| [Count](classROOT_1_1RDF_1_1RInterface.html#a37f9e00c2ece7f53fae50b740adc1456) | Return the number of events processed. |
| [Fill](classROOT_1_1RDF_1_1RInterface.html#a0cac4d08297c23d16de81ff25545440a) | Fill a user-defined object with the values of the specified branches, as if by calling `Obj.Fill(branch1, branch2, ...). |
| [Histo{1D,2D,3D}](classROOT_1_1RDF_1_1RInterface.html#a247ca3aeb7ce5b95015b7fae72983055) | Fill a {one,two,three}-dimensional histogram with the processed branch values. |
| [Max](classROOT_1_1RDF_1_1RInterface.html#a057179b1e77599466a0b02200d5cd8c3) | Return the maximum of processed branch values. If the type of the column is inferred, the return type is `double`, the type of the column otherwise.|
| [Mean](classROOT_1_1RDF_1_1RInterface.html#ade6b020284f2f4fe9d3b09246b5f376a) | Return the mean of processed branch values.|
| [Min](classROOT_1_1RDF_1_1RInterface.html#a7005702189e601972b6d19ecebcdc80c) | Return the minimum of processed branch values. If the type of the column is inferred, the return type is `double`, the type of the column otherwise.|
| [Profile{1D,2D}](classROOT_1_1RDF_1_1RInterface.html#a8ef7dc16b0e9f7bc9cfbe2d9e5de0cef) | Fill a {one,two}-dimensional profile with the branch values that passed all filters. |
| [Reduce](classROOT_1_1RDF_1_1RInterface.html#a118e723ae29834df8f2a992ded347354) | Reduce (e.g. sum, merge) entries using the function (lambda, functor...) passed as argument. The function must have signature `T(T,T)` where `T` is the type of the branch. Return the final result of the reduction operation. An optional parameter allows initialization of the result object to non-default values. |
| [Report](classROOT_1_1RDF_1_1RInterface.html#a94f322531dcb25beb8f53a602e5d6332) | Obtains statistics on how many entries have been accepted and rejected by the filters. See the section on [named filters](#named-filters-and-cutflow-reports) for a more detailed explanation. The method returns a RCutFlowReport instance which can be queried programmatically to get information about the effects of the individual cuts. |
| [Sum](classROOT_1_1RDF_1_1RInterface.html#a61d03407459120df6749af43ed506891) | Return the sum of the values in the column. If the type of the column is inferred, the return type is `double`, the type of the column otherwise. |
| [Take](classROOT_1_1RDF_1_1RInterface.html#a4fd694773a2931b6b07737ddcd1e73b4) | Extract a column from the dataset as a collection of values. If the type of the column is a C-style array, the type stored in the return container is a `ROOT::VecOps::RVec<T>` to guarantee the lifetime of the data involved. |

| **Instant action** | **Description** |
|---------------------|-----------------|
| [Foreach](classROOT_1_1RDF_1_1RInterface.html#ad2822a7ccb8a9afdf3e5b2ea321886ca) | Execute a user-defined function on each entry. Users are responsible for the thread-safety of this lambda when executing with implicit multi-threading enabled. |
| [ForeachSlot](classROOT_1_1RDF_1_1RInterface.html#a3650ca30aae1ccd0d92bf3d680314129) | Same as `Foreach`, but the user-defined function must take an extra `unsigned int slot` as its first parameter. `slot` will take a different value, `0` to `nThreads - 1`, for each thread of execution. This is meant as a helper in writing thread-safe `Foreach` actions when using `RDataFrame` after `ROOT::EnableImplicitMT()`. `ForeachSlot` works just as well with single-thread execution: in that case `slot` will always be `0`. |
| [Snapshot](classROOT_1_1RDF_1_1RInterface.html#a233b7723e498967f4340705d2c4db7f8) | Writes processed data-set to disk, in a new `TTree` and `TFile`. Custom columns can be saved as well, filtered entries are not saved. Users can specify which columns to save (default is all). Snapshot, by default, overwrites the output file if it already exists. `Snapshot` can be made *lazy* setting the appropriate flage in the snapshot options.|


### Other Operations

| **Operation** | **Description** |
|---------------------|-----------------|
| [Alias](classROOT_1_1RDF_1_1RInterface.html#a31ca327e4a192dcc05a4aac240e1a725) | Introduce an alias for a particular column name |
| [GetColumnNames](classROOT_1_1RDF_1_1RInterface.html#a951fe60b74d3a9fda37df59fd1dac186) | Get all the available columns of the dataset |


## <a name="introduction"></a>Introduction
Users define their analysis as a sequence of operations to be performed on the data-frame object; the framework
takes care of the management of the loop over entries as well as low-level details such as I/O and parallelisation.
`RDataFrame` provides methods to perform most common operations required by ROOT analyses;
at the same time, users can just as easily specify custom code that will be executed in the event loop.

`RDataFrame` is built with a *modular* and *flexible* workflow in mind, summarised as follows:

1. **build a data-frame** object by specifying your data-set
2. **apply a series of transformations** to your data
   1.  **filter** (e.g. apply some cuts) or
   2.  **define** a new column (e.g. the result of an expensive computation on branches)
3. **apply actions** to the transformed data to produce results (e.g. fill a histogram)

The following table shows how analyses based on `TTreeReader` and `TTree::Draw` translate to `RDataFrame`. Follow the
[crash course](#crash-course) to discover more idiomatic and flexible ways to express analyses with `RDataFrame`.
<table>
<tr>
   <td>
      <b>TTreeReader</b>
   </td>
   <td>
      <b>ROOT::RDataFrame</b>
   </td>
</tr>
<tr>
   <td>
~~~{.cpp}
TTreeReader reader("myTree", file);
TTreeReaderValue<A_t> a(reader, "A");
TTreeReaderValue<B_t> b(reader, "B");
TTreeReaderValue<C_t> c(reader, "C");
while(reader.Next()) {
   if(IsGoodEvent(a, b, c))
      DoStuff(a, b, c);
}
~~~
   </td>
   <td>
~~~{.cpp}
ROOT::RDataFrame d("myTree", file, {"A", "B", "C"});
d.Filter(IsGoodEvent).Foreach(DoStuff);
~~~
   </td>
</tr>
<tr>
   <td>
      <b>TTree::Draw</b>
   </td>
   <td>
      <b>ROOT::RDataFrame</b>
   </td>
</tr>
<tr>
   <td>
~~~{.cpp}
TTree *t = static_cast<TTree*>(
   file->Get("myTree")
);
t->Draw("var", "var > 2");
~~~
   </td>
   <td>
~~~{.cpp}
ROOT::RDataFrame d("myTree", file);
auto h = d.Filter("var > 2").Histo1D("var");
~~~
   </td>
</tr>
</table>

## <a name="crash-course"></a> Crash course
All snippets of code presented in the crash course can be executed in the ROOT interpreter. Simply precede them with
~~~{.cpp}
using namespace ROOT; // RDataFrame's namespace
~~~
which is omitted for brevity. The terms "column" and "branch" are used interchangeably.

### Creating a RDataFrame
RDataFrame's constructor is where the user specifies the dataset and, optionally, a default set of columns that
operations should work with. Here are the most common methods to construct a RDataFrame object:
~~~{.cpp}
// single file -- all ctors are equivalent
RDataFrame d1("treeName", "file.root");
TFile *f = TFile::Open("file.root");
RDataFrame d2("treeName", f); // same as TTreeReader
TTree *t = nullptr;
f.GetObject("treeName", t);
RDataFrame d3(*t); // TTreeReader takes a pointer, RDF takes a reference

// multiple files -- all ctors are equivalent
RDataFrame d3("myTree", {"file1.root", "file2.root"});
std::vector<std::string> files = {"file1.root", "file2.root"};
RDataFrame d3("myTree", files);
RDataFrame d4("myTree", "file*.root"); // see TRegexp's documentation for a list of valid regexes
TChain chain("myTree");
chain.Add("file1.root");
chain.Add("file2.root");
RDataFrame d3(chain);
~~~
Additionally, users can construct a RDataFrame specifying just an integer number. This is the number of "events" that
will be generated by this RDataFrame.
~~~{.cpp}
RDataFrame d(10); // a RDF with 10 entries (and no columns/branches, for now)
d.Foreach([] { static int i = 0; std::cout << i++ << std::endl; }); // silly example usage: count to ten
~~~
This is useful to generate simple data-sets on the fly: the contents of each event can be specified via the `Define`
transformation (explained below). For example, we have used this method to generate Pythia events (with a `Define`
transformation) and write them to disk in parallel (with the `Snapshot` action).

### Programmatically get the list of column names
The list of column names available in the dataset can be obtained with the `GetColumnsNames` method:
~~~{.cpp}
RDataFrame d("myTree", "file.root");
auto colNames = d.GetColumnNames();
for (auto &&colName : colNames) {
   std::cout << colName << std::endl;
   }
~~~

### Filling a histogram
Let's now tackle a very common task, filling a histogram:
~~~{.cpp}
// Fill a TH1D with the "MET" branch
RDataFrame d("myTree", "file.root");
auto h = d.Histo1D("MET");
h->Draw();
~~~
The first line creates a `RDataFrame` associated to the `TTree` "myTree". This tree has a branch named "MET".

`Histo1D` is an *action*; it returns a smart pointer (a `RResultPtr` to be precise) to a `TH1D` histogram filled
with the `MET` of all events. If the quantity stored in the branch is a collection (e.g. a vector or an array), the
histogram is filled with its elements.

You can use the objects returned by actions as if they were pointers to the desired results. There are many other
possible [actions](#overview), and all their results are wrapped in smart pointers; we'll see why in a minute.

### Applying a filter
Let's say we want to cut over the value of branch "MET" and count how many events pass this cut. This is one way to do it:
~~~{.cpp}
RDataFrame d("myTree", "file.root");
auto c = d.Filter("MET > 4.").Count();
std::cout << *c << std::endl;
~~~
The filter string (which must contain a valid c++ expression) is applied to the specified branches for each event;
the name and types of the columns are inferred automatically. The string expression is required to return a `bool`
which signals whether the event passes the filter (`true`) or not (`false`).

You can think of your data as "flowing" through the chain of calls, being transformed, filtered and finally used to
perform actions. Multiple `Filter` calls can be chained one after another.

Using string filters is nice for simple things, but they are limited to specifying the equivalent of a single return
statement or the body of a lambda, so it's cumbersome to use strings with more complex filters. They also add a small
runtime overhead, as ROOT needs to just-in-time compile the string into C++ code. When more freedom is required or
runtime performance is very important, a C++ callable can be specified instead (a lambda in the following snippet,
but it can be any kind of function or even a functor class), together with a list of branch names.
This snippet is analogous to the one above:
~~~{.cpp}
RDataFrame d("myTree", "file.root");
auto metCut = [](double x) { return x > 4.; }; // a c++11 lambda function checking "x > 4"
auto c = d.Filter(metCut, {"MET"}).Count();
std::cout << *c << std::endl;
~~~

An example of a more complex filter expressed as a string containing C++ code is shown below

~~~{.cpp}
RDataFrame d("myTree", "file.root");
auto df = d.Define("p", "std::array<double, 4> p{px, py, pz}; return p;")
           .Filter("double p2 = 0.0; for (auto&& x : p) p2 += x*x; return sqrt(p2) < 10.0;");
~~~

The code snippet above defines a column `p` that is a fixed-size array using the component column names and then
filters on its magnitude by looping over its elements. It must be noted that the usage of strings to define columns
like the one above is a major advantage when using PyROOT. However, only constants and data coming from other columns
in the dataset can be involved in the code passed as a string. Local variables and functions cannot be used, since
the interpreter will not know how to find them. When capturing local state is necessary, a C++ callable can be used.

More information on filters and how to use them to automatically generate cutflow reports can be found [below](#Filters).

### Defining custom columns
Let's now consider the case in which "myTree" contains two quantities "x" and "y", but our analysis relies on a derived
quantity `z = sqrt(x*x + y*y)`. Using the `Define` transformation, we can create a new column in the data-set containing
the variable "z":
~~~{.cpp}
RDataFrame d("myTree", "file.root");
auto sqrtSum = [](double x, double y) { return sqrt(x*x + y*y); };
auto zMean = d.Define("z", sqrtSum, {"x","y"}).Mean("z");
std::cout << *zMean << std::endl;
~~~
`Define` creates the variable "z" by applying `sqrtSum` to "x" and "y". Later in the chain of calls we refer to
variables created with `Define` as if they were actual tree branches/columns, but they are evaluated on demand, at most
once per event. As with filters, `Define` calls can be chained with other transformations to create multiple custom
columns. `Define` and `Filter` transformations can be concatenated and intermixed at will.

As with filters, it is possible to specify new columns as string expressions. This snippet is analogous to the one above:
~~~{.cpp}
RDataFrame d("myTree", "file.root");
auto zMean = d.Define("z", "sqrt(x*x + y*y)").Mean("z");
std::cout << *zMean << std::endl;
~~~
Again the names of the branches used in the expression and their types are inferred automatically. The string must be
valid c++ and is just-in-time compiled by the ROOT interpreter, cling -- the process has a small runtime overhead.

Previously, when showing the different ways a RDataFrame can be created, we showed a constructor that only takes a
number of entries a parameter. In the following example we show how to combine such an "empty" `RDataFrame` with `Define`
transformations to create a data-set on the fly. We then save the generated data on disk using the `Snapshot` action.
~~~{.cpp}
RDataFrame d(100); // a RDF that will generate 100 entries (currently empty)
int x = -1;
auto d_with_columns = d.Define("x", [&x] { return ++x; })
                       .Define("xx", [&x] { return x*x; });
d_with_columns.Snapshot("myNewTree", "newfile.root");
~~~
This example is slightly more advanced than what we have seen so far: for starters, it makes use of lambda captures (a
simple way to make external variables available inside the body of c++ lambdas) to act on the same variable `x` from
both `Define` transformations. Secondly we have *stored* the transformed data-frame in a variable. This is always
possible: at each point of the transformation chain, users can store the status of the data-frame for further use (more
on this [below](#callgraphs)).

You can read more about defining new columns [here](#custom-columns).

\image html RDF_Graph.png "A graph composed of two branches, one starting with a filter and one with a define. The end point of a branch is always an action."

### Running on a range of entries
It is sometimes necessary to limit the processing of the dataset to a range of entries. For this reason, the RDataFrame
offers the concept of ranges as a node of the RDataFrame chain of transformations; this means that filters, columns and
actions can be concatenated to and intermixed with `Range`s. If a range is specified after a filter, the range will act
exclusively on the entries passing the filter -- it will not even count the other entries! The same goes for a `Range`
hanging from another `Range`. Here are some commented examples:
~~~{.cpp}
RDataFrame d("myTree", "file.root");
// Here we store a data-frame that loops over only the first 30 entries in a variable
auto d30 = d.Range(30);
// This is how you pick all entries from 15 onwards
auto d15on = d.Range(15, 0);
// We can specify a stride too, in this case we pick an event every 3
auto d15each3 = d.Range(0, 15, 3);
~~~
Note that ranges are not available when multi-threading is enabled. More information on ranges is available
[here](#ranges).

### Executing multiple actions in the same event loop
As a final example let us apply two different cuts on branch "MET" and fill two different histograms with the "pt\_v" of
the filtered events.
By now, you should be able to easily understand what's happening:
~~~{.cpp}
RDataFrame d("treeName", "file.root");
auto h1 = d.Filter("MET > 10").Histo1D("pt_v");
auto h2 = d.Histo1D("pt_v");
h1->Draw();       // event loop is run once here
h2->Draw("SAME"); // no need to run the event loop again
~~~
`RDataFrame` executes all above actions by **running the event-loop only once**. The trick is that actions are not
executed at the moment they are called, but they are **lazy**, i.e. delayed until the moment one of their results is
accessed through the smart pointer. At that time, the event loop is triggered and *all* results are produced
simultaneously.

It is therefore good practice to declare all your transformations and actions *before* accessing their results, allowing
`RDataFrame` to run the loop once and produce all results in one go.

### Going parallel
Let's say we would like to run the previous examples in parallel on several cores, dividing events fairly between cores.
The only modification required to the snippets would be the addition of this line *before* constructing the main
data-frame object:
~~~{.cpp}
ROOT::EnableImplicitMT();
~~~
Simple as that. More details are given [below](#parallel-execution).

##  <a name="more-features"></a>More features
Here is a list of the most important features that have been omitted in the "Crash course" for brevity.
You don't need to read all these to start using `RDataFrame`, but they are useful to save typing time and runtime.

### Treatment of columns holding collections
When using RDataFrame to read data from a ROOT file, users can specify that the type of a branch is `RVec<T>`
to indicate the branch is a c-style array, an STL array or any other collection type associated to a
contiguous storage in memory.

Column values of type `RVec<T>` perform no copy of the underlying array data, it's in some sense a view,
and offer a rich interface to operate on the array elements in a vectorised fashion.

The `RVec<T>` type signals to RDataFrame that a special behaviour needs to be adopted when snapshotting
a dataset on disk. Indeed, if columns which are variable size C arrays are treated via the `RVec<T>`,
RDataFrame will correctly persistify them - if anything else is adopted, for example `std::span`, only
the first element of the array will be written.

### Callbacks
Acting on a RResultPtr, it is possible to register a callback that RDataFrame will execute "everyNEvents"
on a partial result.

The callback must be a callable that takes a reference to the result type as argument and returns nothing.
RDataFrame, acting as a full fledged data processing framework, will invoke registered callbacks passing
partial action results as arguments to them (e.g. a histogram filled with a part of the selected events).

Callbacks can be used e.g. to inspect partial results of the analysis while the event loop is running. For
example one can draw an up-to-date version of a result histogram every 100 entries like this:
~~~{.cpp}
auto h = tdf.Histo1D("x");
TCanvas c("c","x hist");
h.OnPartialResult(100, [&c](TH1D &h_) { c.cd(); h_.Draw(); c.Update(); });
h->Draw(); // event loop runs here, this `Draw` is executed after the event loop is finished
~~~

### Default branch lists
When constructing a `RDataFrame` object, it is possible to specify a **default column list** for your analysis, in the
usual form of a list of strings representing branch/column names. The default column list will be used as a fallback
whenever a list specific to the transformation/action is not present. RDataFrame will take as many of these columns as
needed, ignoring trailing extra names if present.
~~~{.cpp}
// use "b1" and "b2" as default branches
RDataFrame d1("myTree", "file.root", {"b1","b2"});
auto h = d1.Filter([](int b1, int b2) { return b1 > b2; }) // will act on "b1" and "b2"
           .Histo1D(); // will act on "b1"

// just one default branch this time
RDataFrame d2("myTree", "file.root", {"b1"});
auto min = d2.Filter([](double b2) { return b2 > 0; }, {"b2"}) // we can still specify non-default branch lists
             .Min(); // returns the minimum value of "b1" for the filtered entries
~~~

### Branch type guessing and explicit declaration of branch types
C++ is a statically typed language: all types must be known at compile-time. This includes the types of the `TTree`
branches we want to work on. For filters, temporary columns and some of the actions, **branch types are deduced from the
signature** of the relevant filter function/temporary column expression/action function:
~~~{.cpp}
// here b1 is deduced to be `int` and b2 to be `double`
dataFrame.Filter([](int x, double y) { return x > 0 && y < 0.; }, {"b1", "b2"});
~~~
If we specify an incorrect type for one of the branches, an exception with an informative message will be thrown at
runtime, when the branch value is actually read from the `TTree`: `RDataFrame` detects type mismatches. The same would
happen if we swapped the order of "b1" and "b2" in the branch list passed to `Filter`.

Certain actions, on the other hand, do not take a function as argument (e.g. `Histo1D`), so we cannot deduce the type of
the branch at compile-time. In this case **`RDataFrame` infers the type of the branch** from the `TTree` itself. This
is why we never needed to specify the branch types for all actions in the above snippets.

When the branch type is not a common one such as `int`, `double`, `char` or `float` it is nonetheless good practice to
specify it as a template parameter to the action itself, like this:
~~~{.cpp}
dataFrame.Histo1D("b1"); // OK, the type of "b1" is deduced at runtime
dataFrame.Min<MyNumber_t>("myObject"); // OK, "myObject" is deduced to be of type `MyNumber_t`
~~~

Deducing types at runtime requires the just-in-time compilation of the relevant actions, which has a small runtime
overhead, so specifying the type of the columns as template parameters to the action is good practice when performance is a goal.

### Generic actions
`RDataFrame` strives to offer a comprehensive set of standard actions that can be performed on each event. At the same
time, it **allows users to execute arbitrary code (i.e. a generic action) inside the event loop** through the `Foreach`
and `ForeachSlot` actions.

`Foreach(f, columnList)` takes a function `f` (lambda expression, free function, functor...) and a list of columns, and
executes `f` on those columns for each event. The function passed must return nothing (i.e. `void`). It can be used to
perform actions that are not already available in the interface. For example, the following snippet evaluates the root
mean square of column "b":
~~~{.cpp}
// Single-thread evaluation of RMS of column "b" using Foreach
double sumSq = 0.;
unsigned int n = 0;
RDataFrame d("bTree", bFilePtr);
d.Foreach([&sumSq, &n](double b) { ++n; sumSq += b*b; }, {"b"});
std::cout << "rms of b: " << std::sqrt(sumSq / n) << std::endl;
~~~
When executing on multiple threads, users are responsible for the thread-safety of the expression passed to `Foreach`.
The code above would need to employ some resource protection mechanism to ensure non-concurrent writing of `rms`; but
this is probably too much head-scratch for such a simple operation.

`ForeachSlot` can help in this situation. It is an alternative version of `Foreach` for which the function takes an
additional parameter besides the columns it should be applied to: an `unsigned int slot` parameter, where `slot` is a
number indicating which thread (0, 1, 2 , ..., poolSize - 1) the function is being run in. We can take advantage of
`ForeachSlot` to evaluate a thread-safe root mean square of branch "b":
~~~{.cpp}
// Thread-safe evaluation of RMS of branch "b" using ForeachSlot
ROOT::EnableImplicitMT();
const unsigned int nSlots = ROOT::GetImplicitMTPoolSize();
std::vector<double> sumSqs(nSlots, 0.);
std::vector<unsigned int> ns(nSlots, 0);

RDataFrame d("bTree", bFilePtr);
d.ForeachSlot([&sumSqs, &ns](unsigned int slot, double b) { sumSqs[slot] += b*b; ns[slot] += 1; }, {"b"});
double sumSq = std::accumulate(sumSqs.begin(), sumSqs.end(), 0.); // sum all squares
unsigned int n = std::accumulate(ns.begin(), ns.end(), 0); // sum all counts
std::cout << "rms of b: " << std::sqrt(sumSq / n) << std::endl;
~~~
You see how we created one `double` variable for each thread in the pool, and later merged their results via
`std::accumulate`.

### Friend trees
Friend trees are supported by RDataFrame.
In order to deal with friend trees with RDataFrame, the user is required to build
the tree and its friends and instantiate a RDataFrame with it.
Two caveats are presents when using jitted `Define`s and `Filter`s:
1) the only columns which can be used in the strings passed to the aforementioned transformations are the top level branches of the friend trees.
2) the "friend columns" cannot be written with the notation involving a dot. For example, if a tree is created like this:
~~~{.cpp}
TTree t([...]);
TTree ft([...]);
t.AddFriend(t,"myFriend");
~~~
in order to access a certain column `col` of the tree ft, it will be necessary to alias it before. To continue the example:
~~~{.cpp}
RDataFrame d(t);
d.Alias("myFriend_MyCol", "myFriend.MyCol");
auto f = d.Filter("myFriend_MyCol == 42");
~~~

### Reading file formats different from ROOT's
RDataFrame can be interfaced with RDataSources. The RDataSource interface defines an API that RDataFrame can use to read arbitrary data formats.

A concrete RDataSource implementation (i.e. a class that inherits from RDataSource and implements all of its pure
methods) provides an adaptor that RDataFrame can leverage to read any kind of tabular data formats.
RDataFrame calls into RDataSource to retrieve information about the data, retrieve (thread-local) readers or "cursors" for selected columns
and to advance the readers to the desired data entry.
Some predefined RDataSources are natively provided by ROOT such as the `RCsvDS` which allows to read comma separated files:
~~~{.cpp}
auto tdf = ROOT::RDF::MakeCsvDataFrame("MuRun2010B.csv");
auto filteredEvents =
   tdf.Filter("Q1 * Q2 == -1")
      .Define("m", "sqrt(pow(E1 + E2, 2) - (pow(px1 + px2, 2) + pow(py1 + py2, 2) + pow(pz1 + pz2, 2)))");
auto h = filteredEvents.Histo1D("m");
h->Draw();
~~~

### <a name="callgraphs"></a>Call graphs (storing and reusing sets of transformations)
**Sets of transformations can be stored as variables** and reused multiple times to create **call graphs** in which
several paths of filtering/creation of columns are executed simultaneously; we often refer to this as "storing the
state of the chain".

This feature can be used, for example, to create a temporary column once and use it in several subsequent filters or
actions, or to apply a strict filter to the data-set *before* executing several other transformations and actions,
effectively reducing the amount of events processed.

Let's try to make this clearer with a commented example:
~~~{.cpp}
// build the data-frame and specify a default column list
RDataFrame d(treeName, filePtr, {"var1", "var2", "var3"});

// apply a cut and save the state of the chain
auto filtered = d.Filter(myBigCut);

// plot branch "var1" at this point of the chain
auto h1 = filtered.Histo1D("var1");

// create a new branch "vec" with a vector extracted from a complex object (only for filtered entries)
// and save the state of the chain
auto newBranchFiltered = filtered.Define("vec", [](const Obj& o) { return o.getVector(); }, {"obj"});

// apply a cut and fill a histogram with "vec"
auto h2 = newBranchFiltered.Filter(cut1).Histo1D("vec");

// apply a different cut and fill a new histogram
auto h3 = newBranchFiltered.Filter(cut2).Histo1D("vec");

// Inspect results
h2->Draw(); // first access to an action result: run event-loop!
h3->Draw("SAME"); // event loop does not need to be run again here..
std::cout << "Entries in h1: " << h1->GetEntries() << std::endl; // ..or here
~~~
`RDataFrame` detects when several actions use the same filter or the same temporary column, and **only evaluates each
filter or temporary column once per event**, regardless of how many times that result is used down the call graph.
Objects read from each column are **built once and never copied**, for maximum efficiency.
When "upstream" filters are not passed, subsequent filters, temporary column expressions and actions are not evaluated,
so it might be advisable to put the strictest filters first in the chain.

##  <a name="transformations"></a>Transformations
### <a name="Filters"></a> Filters
A filter is defined through a call to `Filter(f, columnList)`. `f` can be a function, a lambda expression, a functor
class, or any other callable object. It must return a `bool` signalling whether the event has passed the selection
(`true`) or not (`false`). It must perform "read-only" actions on the columns, and should not have side-effects (e.g.
modification of an external or static variable) to ensure correct results when implicit multi-threading is active.

`RDataFrame` only evaluates filters when necessary: if multiple filters are chained one after another, they are executed
in order and the first one returning `false` causes the event to be discarded and triggers the processing of the next
entry. If multiple actions or transformations depend on the same filter, that filter is not executed multiple times for
each entry: after the first access it simply serves a cached result.

#### <a name="named-filters-and-cutflow-reports"></a>Named filters and cutflow reports
An optional string parameter `name` can be passed to the `Filter` method to create a **named filter**. Named filters
work as usual, but also keep track of how many entries they accept and reject.

Statistics are retrieved through a call to the `Report` method:

- when `Report` is called on the main `RDataFrame` object, it returns a RResultPtr<RCutFlowReport> relative to all
named filters declared up to that point
- when called on a specific node (e.g. the result of a `Define` or `Filter`), it returns a RResultPtr<RCutFlowReport>
relative all named filters in the section of the chain between the main `RDataFrame` and that node (included).

Stats are stored in the same order as named filters have been added to the graph, and *refer to the latest event-loop*
that has been run using the relevant `RDataFrame`.

### <a name="ranges"></a>Ranges
When `RDataFrame` is not being used in a multi-thread environment (i.e. no call to `EnableImplicitMT` was made),
`Range` transformations are available. These act very much like filters but instead of basing their decision on
a filter expression, they rely on `begin`,`end` and `stride` parameters.

- `begin`: initial entry number considered for this range.
- `end`: final entry number (excluded) considered for this range. 0 means that the range goes until the end of the dataset.
- `stride`: process one entry of the [begin, end) range every `stride` entries. Must be strictly greater than 0.

The actual number of entries processed downstream of a `Range` node will be `(end - begin)/stride` (or less if less
entries than that are available).

Note that ranges act "locally", not based on the global entry count: `Range(10,50)` means "skip the first 10 entries
*that reach this node*, let the next 40 entries pass, then stop processing". If a range node hangs from a filter node,
and the range has a `begin` parameter of 10, that means the range will skip the first 10 entries *that pass the
preceding filter*.

Ranges allow "early quitting": if all branches of execution of a functional graph reached their `end` value of
processed entries, the event-loop is immediately interrupted. This is useful for debugging and quick data explorations.

### <a name="custom-columns"></a> Custom columns
Custom columns are created by invoking `Define(name, f, columnList)`. As usual, `f` can be any callable object
(function, lambda expression, functor class...); it takes the values of the columns listed in `columnList` (a list of
strings) as parameters, in the same order as they are listed in `columnList`. `f` must return the value that will be
assigned to the temporary column.

A new variable is created called `name`, accessible as if it was contained in the dataset from subsequent
transformations/actions.

Use cases include:
- caching the results of complex calculations for easy and efficient multiple access
- extraction of quantities of interest from complex objects
- branch aliasing, i.e. changing the name of a branch

An exception is thrown if the `name` of the new column/branch is already in use for another branch in the `TTree`.

It is also possible to specify the quantity to be stored in the new temporary column as a C++ expression with the method
`Define(name, expression)`. For example this invocation

~~~{.cpp}
tdf.Define("pt", "sqrt(px*px + py*py)");
~~~

will create a new column called "pt" the value of which is calculated starting from the columns px and py. The system
builds a just-in-time compiled function starting from the expression after having deduced the list of necessary branches
from the names of the variables specified by the user.

#### Custom columns as function of slot and entry number

It is possible to create custom columns also as a function of the processing slot and entry numbers. The methods that can
be invoked are:
- `DefineSlot(name, f, columnList)`. In this case the callable f has this signature `R(unsigned int, T1, T2, ...)`: the
first parameter is the slot number which ranges from 0 to ROOT::GetImplicitMTPoolSize() - 1.
- `DefineSlotEntry(name, f, columnList)`. In this case the callable f has this signature `R(unsigned int, ULong64_t,
T1, T2, ...)`: the first parameter is the slot number while the second one the number of the entry being processed.

##  <a name="actions"></a>Actions
### Instant and lazy actions
Actions can be **instant** or **lazy**. Instant actions are executed as soon as they are called, while lazy actions are
executed whenever the object they return is accessed for the first time. As a rule of thumb, actions with a return value
are lazy, the others are instant.

##  <a name="parallel-execution"></a>Parallel execution
As pointed out before in this document, `RDataFrame` can transparently perform multi-threaded event loops to speed up
the execution of its actions. Users only have to call `ROOT::EnableImplicitMT()` *before* constructing the `RDataFrame`
object to indicate that it should take advantage of a pool of worker threads. **Each worker thread processes a distinct
subset of entries**, and their partial results are merged before returning the final values to the user.

### Thread safety
`Filter` and `Define` transformations should be inherently thread-safe: they have no side-effects and are not
dependent on global state.
Most `Filter`/`Define` functions will in fact be pure in the functional programming sense.
All actions are built to be thread-safe with the exception of `Foreach`, in which case users are responsible of
thread-safety, see [here](#generic-actions).

<a name="reference"></a>
*/
// clang-format on
namespace ROOT {

////////////////////////////////////////////////////////////////////////////
/// \brief Build the dataframe
/// \param[in] treeName Name of the tree contained in the directory
/// \param[in] dirPtr TDirectory where the tree is stored, e.g. a TFile.
/// \param[in] defaultBranches Collection of default branches.
///
/// The default branches are looked at in case no branch is specified in the
/// booking of actions or transformations.
/// See RInterface for the documentation of the
/// methods available.
RDataFrame::RDataFrame(std::string_view treeName, TDirectory *dirPtr, const ColumnNames_t &defaultBranches)
   : RInterface<RDFDetail::RLoopManager>(std::make_shared<RDFDetail::RLoopManager>(nullptr, defaultBranches))
{
   if (!dirPtr) {
      auto msg = "Invalid TDirectory!";
      throw std::runtime_error(msg);
   }
   const std::string treeNameInt(treeName);
   auto tree = static_cast<TTree *>(dirPtr->Get(treeNameInt.c_str()));
   if (!tree) {
      auto msg = "Tree \"" + treeNameInt + "\" cannot be found!";
      throw std::runtime_error(msg);
   }
   GetProxiedPtr()->SetTree(std::shared_ptr<TTree>(tree, [](TTree *) {}));
}

////////////////////////////////////////////////////////////////////////////
/// \brief Build the dataframe
/// \param[in] treeName Name of the tree contained in the directory
/// \param[in] filenameglob TDirectory where the tree is stored, e.g. a TFile.
/// \param[in] defaultBranches Collection of default branches.
///
/// The default branches are looked at in case no branch is specified in the
/// booking of actions or transformations.
/// See RInterface for the documentation of the
/// methods available.
RDataFrame::RDataFrame(std::string_view treeName, std::string_view filenameglob, const ColumnNames_t &defaultBranches)
   : RInterface<RDFDetail::RLoopManager>(std::make_shared<RDFDetail::RLoopManager>(nullptr, defaultBranches))
{
   const std::string treeNameInt(treeName);
   const std::string filenameglobInt(filenameglob);
   auto chain = std::make_shared<TChain>(treeNameInt.c_str());
   chain->Add(filenameglobInt.c_str());
   GetProxiedPtr()->SetTree(chain);
}

////////////////////////////////////////////////////////////////////////////
/// \brief Build the dataframe
/// \param[in] treeName Name of the tree contained in the directory
/// \param[in] filenames Collection of file names
/// \param[in] defaultBranches Collection of default branches.
///
/// The default branches are looked at in case no branch is specified in the booking of actions or transformations.
/// See RInterface for the documentation of the methods available.
RDataFrame::RDataFrame(std::string_view treeName, const std::vector<std::string> &filenames,
                       const ColumnNames_t &defaultBranches)
   : RDF::RInterface<RDFDetail::RLoopManager>(std::make_shared<RDFDetail::RLoopManager>(nullptr, defaultBranches))
{
   std::string treeNameInt(treeName);
   auto chain = std::make_shared<TChain>(treeNameInt.c_str());
   for (auto &fileName : filenames)
      chain->Add(fileName.c_str());
   GetProxiedPtr()->SetTree(chain);
}

////////////////////////////////////////////////////////////////////////////
/// \brief Build the dataframe
/// \param[in] tree The tree or chain to be studied.
/// \param[in] defaultBranches Collection of default column names to fall back to when none is specified.
///
/// The default branches are looked at in case no branch is specified in the
/// booking of actions or transformations.
/// See RInterface for the documentation of the
/// methods available.
RDataFrame::RDataFrame(TTree &tree, const ColumnNames_t &defaultBranches)
   : RInterface<RDFDetail::RLoopManager>(std::make_shared<RDFDetail::RLoopManager>(&tree, defaultBranches))
{
}

//////////////////////////////////////////////////////////////////////////
/// \brief Build a dataframe that generates numEntries entries.
/// \param[in] numEntries The number of entries to generate.
///
/// An empty-source dataframe constructed with a number of entries will
/// generate those entries on the fly when some action is triggered,
/// and it will do so for all the previously-defined temporary branches.
RDataFrame::RDataFrame(ULong64_t numEntries)
   : RInterface<RDFDetail::RLoopManager>(std::make_shared<RDFDetail::RLoopManager>(numEntries))
{
}

//////////////////////////////////////////////////////////////////////////
/// \brief Build dataframe associated to datasource.
/// \param[in] ds The data-source object.
/// \param[in] defaultBranches Collection of default column names to fall back to when none is specified.
///
/// A dataframe associated to a datasource will query it to access column values.
RDataFrame::RDataFrame(std::unique_ptr<RDataSource> ds, const ColumnNames_t &defaultBranches)
   : RInterface<RDFDetail::RLoopManager>(std::make_shared<RDFDetail::RLoopManager>(std::move(ds), defaultBranches))
{
}

} // namespace ROOT

namespace cling {
//////////////////////////////////////////////////////////////////////////
/// Print a RDataFrame at the prompt
std::string printValue(ROOT::RDataFrame *tdf)
{
   auto df = tdf->GetLoopManager();
   auto *tree = df->GetTree();
   auto defBranches = df->GetDefaultColumnNames();

   std::ostringstream ret;
   if (tree) {
      ret << "A data frame built on top of the " << tree->GetName() << " dataset.";
      if (!defBranches.empty()) {
         if (defBranches.size() == 1)
            ret << "\nDefault branch: " << defBranches[0];
         else {
            ret << "\nDefault branches:\n";
            for (auto &&branch : defBranches) {
               ret << " - " << branch << "\n";
            }
         }
      }
   } else {
      ret << "A data frame that will create " << df->GetNEmptyEntries() << " entries\n";
   }

   return ret.str();
}
} // namespace cling

