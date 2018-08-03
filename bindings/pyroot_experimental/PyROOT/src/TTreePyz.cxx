
// Bindings
#include "CPyCppyy.h"
#include "PyROOTPythonize.h"
#include "CPPInstance.h"
#include "ProxyWrappers.h"
#include "Converters.h"
#include "Utility.h"

// ROOT
#include "TClass.h"
#include "TTree.h"
#include "TBranch.h"
#include "TBranchElement.h"
#include "TBranchObject.h"
#include "TLeaf.h"
#include "TLeafElement.h"
#include "TLeafObject.h"
#include "TStreamerElement.h"
#include "TStreamerInfo.h"

using namespace CPyCppyy;

static TClass *GetClass(const CPPInstance *pyobj)
{
   return TClass::GetClass(Cppyy::GetFinalName(pyobj->ObjectIsA()).c_str());
}

static TBranch *SearchForBranch(TTree *tree, const char *name)
{
   TBranch *branch = tree->GetBranch(name);
   if (!branch) {
      // for benefit of naming of sub-branches, the actual name may have a trailing '.'
      branch = tree->GetBranch((std::string(name) + '.').c_str());
   }
   return branch;
}

static TLeaf *SearchForLeaf(TTree *tree, const char *name, TBranch *branch)
{
   TLeaf *leaf = tree->GetLeaf(name);
   if (branch && !leaf) {
      leaf = branch->GetLeaf(name);
      if (!leaf) {
         TObjArray *leaves = branch->GetListOfLeaves();
         if (leaves->GetSize() && (leaves->First() == leaves->Last())) {
            // i.e., if unambiguously only this one
            leaf = (TLeaf *)leaves->At(0);
         }
      }
   }
   return leaf;
}

static PyObject *BindBranchToProxy(TTree *tree, const char *name, TBranch *branch)
{
   // for partial return of a split object
   if (branch->InheritsFrom(TBranchElement::Class())) {
      TBranchElement *be = (TBranchElement *)branch;
      if (be->GetCurrentClass() && (be->GetCurrentClass() != be->GetTargetClass()) && (0 <= be->GetID())) {
         Long_t offset = ((TStreamerElement *)be->GetInfo()->GetElements()->At(be->GetID()))->GetOffset();
         return BindCppObjectNoCast(be->GetObject() + offset, Cppyy::GetScope(be->GetCurrentClass()->GetName()));
      }
   }

   // for return of a full object
   if (branch->IsA() == TBranchElement::Class() || branch->IsA() == TBranchObject::Class()) {
      TClass *klass = TClass::GetClass(branch->GetClassName());
      if (klass && branch->GetAddress())
         return BindCppObjectNoCast(*(void **)branch->GetAddress(), Cppyy::GetScope(branch->GetClassName()));

      // try leaf, otherwise indicate failure by returning a typed null-object
      TObjArray *leaves = branch->GetListOfLeaves();
      if (klass && !tree->GetLeaf(name) && !(leaves->GetSize() && (leaves->First() == leaves->Last())))
         return BindCppObjectNoCast(nullptr, Cppyy::GetScope(branch->GetClassName()));
   }

   return nullptr;
}

static PyObject *WrapLeaf(TLeaf *leaf)
{
   if (1 < leaf->GetLenStatic() || leaf->GetLeafCount()) {
      // array types
      std::string typeName = leaf->GetTypeName();
      Converter *pcnv = CreateConverter(typeName + '*', leaf->GetNdata());

      void *address = 0;
      if (leaf->GetBranch())
         address = (void *)leaf->GetBranch()->GetAddress();
      if (!address)
         address = (void *)leaf->GetValuePointer();

      PyObject *value = pcnv->FromMemory(&address);
      delete pcnv;

      return value;
   } else if (leaf->GetValuePointer()) {
      // value types
      Converter *pcnv = CreateConverter(leaf->GetTypeName());
      PyObject *value = 0;
      if (leaf->IsA() == TLeafElement::Class() || leaf->IsA() == TLeafObject::Class())
         value = pcnv->FromMemory((void *)*(void **)leaf->GetValuePointer());
      else
         value = pcnv->FromMemory((void *)leaf->GetValuePointer());
      delete pcnv;

      return value;
   }

   return nullptr;
}

// Allow access to branches/leaves as if they were data members
PyObject *GetAttr(const CPPInstance *self, PyObject *pyname)
{
   const char *name_possibly_alias = CPyCppyy_PyUnicode_AsString(pyname);
   if (!name_possibly_alias)
      return 0;

   // get hold of actual tree
   TTree *tree = (TTree *)GetClass(self)->DynamicCast(TTree::Class(), self->GetObject());

   if (!tree) {
      PyErr_SetString(PyExc_ReferenceError, "attempt to access a null-pointer");
      return 0;
   }

   // deal with possible aliasing
   const char *name = tree->GetAlias(name_possibly_alias);
   if (!name)
      name = name_possibly_alias;

   // search for branch first (typical for objects)
   TBranch *branch = SearchForBranch(tree, name);

   if (branch) {
      // found a branched object, wrap its address for the object it represents
      auto proxy = BindBranchToProxy(tree, name, branch);
      if (proxy != nullptr)
         return proxy;
   }

   // if not, try leaf
   TLeaf *leaf = SearchForLeaf(tree, name, branch);

   if (leaf) {
      // found a leaf, extract value and wrap with a Python object according to its type
      auto wrapper = WrapLeaf(leaf);
      if (wrapper != nullptr)
         return wrapper;
   }

   // confused
   PyErr_Format(PyExc_AttributeError, "\'%s\' object has no attribute \'%s\'", tree->IsA()->GetName(), name);
   return 0;
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add pythonizations to the TTree class.
/// \param[in] self Always null, since this is a module function.
/// \param[in] args Pointer to a Python tuple object containing the arguments
/// received from Python.
///
/// Inject new behaviour into the TTree class so that it can be used from
/// Python in a simpler or more pythonic way.
///
/// This function adds the following pythonizations:
/// - Allow access to branches/leaves as if they were data members (e.g. mytree.branch)
PyObject *PyROOT::PythonizeTTree(PyObject */* self */, PyObject *args)
{
   PyObject *pyclass = PyTuple_GetItem(args, 0);
   Utility::AddToClass(pyclass, "__getattr__", (PyCFunction)GetAttr, METH_O);
   Py_RETURN_NONE;
}
